"""Tests for benchmark infrastructure — all LLM and embedding calls mocked."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem
from benchmark.systems.engram_system import EngramSystem
from benchmark.systems.naive_rag_system import NaiveRagSystem
from benchmark.systems.no_memory_system import NoMemorySystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(text: str = "mocked answer") -> MagicMock:
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def _embed_response(vec: list[float] | None = None) -> SimpleNamespace:
    return SimpleNamespace(embedding=vec or [0.1] * 768)


# ---------------------------------------------------------------------------
# BaseSystem
# ---------------------------------------------------------------------------

class TestBaseSystem:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSystem()

    def test_concrete_subclass_must_implement_all_methods(self):
        class Incomplete(BaseSystem):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(BaseSystem):
            @property
            def name(self) -> str:
                return "complete"
            def ingest(self, session_id, content, tags):
                pass
            def query(self, question):
                return ""
            def reset(self):
                pass

        assert Complete().name == "complete"


# ---------------------------------------------------------------------------
# NoMemorySystem
# ---------------------------------------------------------------------------

class TestNoMemorySystem:
    def test_name(self):
        sys = NoMemorySystem()
        assert sys.name == "no_memory"

    def test_ingest_does_nothing(self):
        sys = NoMemorySystem()
        sys.ingest("s1", "some fact", ["tag"])  # must not raise

    def test_reset_does_nothing(self):
        sys = NoMemorySystem()
        sys.reset()  # must not raise

    def test_query_makes_llm_call_with_no_context(self):
        sys = NoMemorySystem()
        with patch.object(sys._client, "chat", return_value=_llm_response("direct answer")) as mock_chat:
            result = sys.query("What is Python?")

        assert result == "direct answer"
        call_messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        # Must be exactly one user message — no system context injected
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert "What is Python?" in call_messages[0]["content"]

    def test_query_returns_empty_string_on_llm_failure(self):
        sys = NoMemorySystem()
        with patch.object(sys._client, "chat", side_effect=RuntimeError("down")):
            assert sys.query("anything") == ""

    def test_query_does_not_include_previous_ingest(self):
        sys = NoMemorySystem()
        sys.ingest("s1", "The answer is 42", ["fact"])
        with patch.object(sys._client, "chat", return_value=_llm_response("I don't know")) as mock_chat:
            sys.query("What is the answer?")

        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        full_text = " ".join(m["content"] for m in messages)
        assert "42" not in full_text


# ---------------------------------------------------------------------------
# NaiveRagSystem
# ---------------------------------------------------------------------------

class TestNaiveRagSystem:
    def _make_system(self, tmp_path) -> NaiveRagSystem:
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = False
        mock_result = MagicMock()
        mock_result.points = []
        mock_qdrant.query_points.return_value = mock_result

        with patch("benchmark.systems.naive_rag_system.QdrantClient", return_value=mock_qdrant):
            sys = NaiveRagSystem()
        return sys

    def test_name(self, tmp_path):
        sys = self._make_system(tmp_path)
        assert sys.name == "naive_rag"

    def test_ingest_without_error(self, tmp_path):
        sys = self._make_system(tmp_path)
        with patch("benchmark.systems.naive_rag_system.ollama.embeddings",
                   return_value=_embed_response()):
            sys.ingest("s1", "Python is great", ["fact"])  # must not raise

    def test_ingest_skips_on_empty_embedding(self, tmp_path):
        sys = self._make_system(tmp_path)
        with patch("benchmark.systems.naive_rag_system.ollama.embeddings",
                   side_effect=RuntimeError("no ollama")):
            sys.ingest("s1", "content", ["tag"])  # must not raise
        sys._qdrant.upsert.assert_not_called()

    def test_reset_deletes_and_recreates_collection(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys._qdrant.collection_exists.return_value = True
        sys.reset()
        sys._qdrant.delete_collection.assert_called_once()
        assert sys._qdrant.create_collection.call_count >= 1

    def test_query_returns_llm_response(self, tmp_path):
        sys = self._make_system(tmp_path)
        with patch("benchmark.systems.naive_rag_system.ollama.embeddings",
                   return_value=_embed_response()):
            with patch.object(sys._client, "chat", return_value=_llm_response("rag answer")):
                result = sys.query("What is Python?")
        assert result == "rag answer"

    def test_query_includes_context_in_prompt(self, tmp_path):
        sys = self._make_system(tmp_path)
        point = MagicMock()
        point.payload = {"content": "Python is interpreted", "tags": []}
        sys._qdrant.query_points.return_value.points = [point]

        with patch("benchmark.systems.naive_rag_system.ollama.embeddings",
                   return_value=_embed_response()):
            with patch.object(sys._client, "chat", return_value=_llm_response()) as mock_chat:
                sys.query("Tell me about Python")

        messages = mock_chat.call_args.kwargs.get("messages") or mock_chat.call_args.args[1]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "Python is interpreted" in prompt_text


# ---------------------------------------------------------------------------
# EngramSystem
# ---------------------------------------------------------------------------

class TestEngramSystem:
    def _make_system(self, tmp_path) -> EngramSystem:
        mock_qdrant = MagicMock()
        mock_qdrant.collection_exists.return_value = True
        mock_result = MagicMock()
        mock_result.points = []
        mock_qdrant.query_points.return_value = mock_result

        with patch("hcma.memory.ltm_store.QdrantClient", return_value=mock_qdrant):
            sys = EngramSystem()
        return sys

    def test_name(self, tmp_path):
        sys = self._make_system(tmp_path)
        assert sys.name == "engram"

    def test_ingest_writes_to_episodic_buffer(self, tmp_path):
        sys = self._make_system(tmp_path)
        assert sys._buf.get_count() == 0
        sys.ingest("sess_1", "Python uses indentation", ["fact"])
        assert sys._buf.get_count() == 1

    def test_ingest_sets_correct_tags(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("sess_1", "Some content", ["debug", "error_pattern"])
        entries = sys._buf.read_all_raw()
        assert "debug" in entries[0].tags

    def test_ingest_sets_importance_07(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("sess_1", "content", [])
        entries = sys._buf.read_all_raw()
        assert entries[0].importance == 0.7

    def test_reset_clears_buffer(self, tmp_path):
        sys = self._make_system(tmp_path)
        sys.ingest("s1", "fact one", [])
        sys.ingest("s2", "fact two", [])
        sys.reset()
        assert sys._buf.get_count() == 0

    def test_query_calls_llm(self, tmp_path):
        sys = self._make_system(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            with patch.object(sys._consolidation_agent._client, "chat",
                              return_value=_llm_response()):
                with patch.object(sys._llm, "chat", return_value=_llm_response("engram answer")) as mock_chat:
                    result = sys.query("What is Python?")
        assert result == "engram answer"

    def test_multiple_ingests_accumulate(self, tmp_path):
        sys = self._make_system(tmp_path)
        for i in range(5):
            sys.ingest(f"s{i}", f"fact {i}", ["fact"])
        assert sys._buf.get_count() == 5


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------

class TestScenarioResult:
    def test_fields_are_correct_types(self):
        result = ScenarioResult(
            scenario_name="retention",
            system_name="engram",
            score=0.8,
            max_score=1.0,
            details=["PASS | Q: 'x' | got: 'y'"],
            duration_seconds=1.23,
        )
        assert isinstance(result.scenario_name, str)
        assert isinstance(result.system_name, str)
        assert isinstance(result.score, float)
        assert isinstance(result.max_score, float)
        assert isinstance(result.details, list)
        assert isinstance(result.duration_seconds, float)

    def test_score_range(self):
        r = ScenarioResult("s", "sys", 0.6, 1.0, [], 0.1)
        assert 0.0 <= r.score <= 1.0

    def test_details_is_list_of_strings(self):
        r = ScenarioResult("s", "sys", 1.0, 1.0, ["detail one", "detail two"], 0.5)
        assert all(isinstance(d, str) for d in r.details)


# ---------------------------------------------------------------------------
# BenchmarkEvaluator
# ---------------------------------------------------------------------------

def _make_results() -> list[ScenarioResult]:
    """Three systems × two scenarios with known scores."""
    rows = [
        # scenario,       system,      score, max
        ("retention",    "engram",     0.8,   1.0),
        ("retention",    "naive_rag",  0.4,   1.0),
        ("retention",    "no_memory",  0.2,   1.0),
        ("interference", "engram",     1.0,   1.0),
        ("interference", "naive_rag",  0.6,   1.0),
        ("interference", "no_memory",  0.0,   1.0),
    ]
    return [
        ScenarioResult(
            scenario_name=s, system_name=sys, score=sc, max_score=mx,
            details=[], duration_seconds=0.1,
        )
        for s, sys, sc, mx in rows
    ]


class TestBenchmarkEvaluator:
    def _ev(self) -> "BenchmarkEvaluator":
        from benchmark.evaluator import BenchmarkEvaluator
        return BenchmarkEvaluator(_make_results())

    def test_compute_summary_returns_correct_structure(self):
        summary = self._ev().compute_summary()
        assert "systems" in summary
        assert "winner" in summary
        assert "scenario_winners" in summary
        for sys_name in ("engram", "naive_rag", "no_memory"):
            assert sys_name in summary["systems"]
            sys_data = summary["systems"][sys_name]
            assert "total_score" in sys_data
            assert "max_score" in sys_data
            assert "percentage" in sys_data
            assert "per_scenario" in sys_data

    def test_compute_summary_correct_totals(self):
        summary = self._ev().compute_summary()
        # engram: 0.8 + 1.0 = 1.8 / 2.0 = 90%
        assert summary["systems"]["engram"]["total_score"] == pytest.approx(1.8)
        assert summary["systems"]["engram"]["max_score"] == pytest.approx(2.0)
        assert summary["systems"]["engram"]["percentage"] == pytest.approx(90.0)

    def test_winner_is_correctly_identified(self):
        summary = self._ev().compute_summary()
        assert summary["winner"] == "engram"

    def test_winner_with_tied_scores(self):
        from benchmark.evaluator import BenchmarkEvaluator
        results = [
            ScenarioResult("s1", "a", 1.0, 1.0, [], 0.1),
            ScenarioResult("s1", "b", 1.0, 1.0, [], 0.1),
        ]
        summary = BenchmarkEvaluator(results).compute_summary()
        # Both tied — winner is one of them (deterministic per dict ordering)
        assert summary["winner"] in ("a", "b")

    def test_scenario_winners_correct(self):
        summary = self._ev().compute_summary()
        assert summary["scenario_winners"]["retention"] == "engram"
        assert summary["scenario_winners"]["interference"] == "engram"

    def test_per_scenario_scores(self):
        summary = self._ev().compute_summary()
        assert summary["systems"]["naive_rag"]["per_scenario"]["retention"] == pytest.approx(0.4)
        assert summary["systems"]["no_memory"]["per_scenario"]["interference"] == pytest.approx(0.0)

    def test_print_report_does_not_raise(self, capsys):
        self._ev().print_report()
        out = capsys.readouterr().out
        assert "LHMBench" in out
        assert "engram" in out

    def test_print_report_contains_winner_line(self, capsys):
        self._ev().print_report()
        out = capsys.readouterr().out
        assert "Winner: engram" in out

    def test_save_results_writes_valid_json(self, tmp_path):
        import json as _json
        ev = self._ev()
        ev.save_results(str(tmp_path))
        json_files = list(tmp_path.glob("lhmbench_*.json"))
        assert len(json_files) == 1
        data = _json.loads(json_files[0].read_text())
        assert "results" in data
        assert "summary" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 6  # 3 systems × 2 scenarios

    def test_save_results_json_has_all_fields(self, tmp_path):
        import json as _json
        self._ev().save_results(str(tmp_path))
        json_files = list(tmp_path.glob("lhmbench_*.json"))
        data = _json.loads(json_files[0].read_text())
        first = data["results"][0]
        for field in ("scenario_name", "system_name", "score", "max_score",
                      "details", "duration_seconds"):
            assert field in first, f"Missing field: {field}"

    def test_save_results_filename_format(self, tmp_path):
        self._ev().save_results(str(tmp_path))
        json_files = list(tmp_path.glob("lhmbench_*.json"))
        assert len(json_files) == 1
        assert json_files[0].name.startswith("lhmbench_")
        assert json_files[0].suffix == ".json"


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def _make_runner_mocked(self) -> "BenchmarkRunner":
        """Return a BenchmarkRunner with all systems replaced by mocked stubs."""
        from benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner.__new__(BenchmarkRunner)
        runner.results = []

        # Replace real systems with lightweight stubs
        class _StubSystem:
            def __init__(self, sys_name: str, score: float = 0.5):
                self._name = sys_name
                self._score = score
                self.reset_called = 0

            @property
            def name(self):
                return self._name

            def reset(self):
                self.reset_called += 1

            def ingest(self, *a, **kw):
                pass

            def query(self, q):
                return "mocked answer"

        # Replace real scenarios with a stub that returns a fixed ScenarioResult
        class _StubScenario:
            def __init__(self, scenario_name: str):
                self._name = scenario_name

            @property
            def name(self):
                return self._name

            def run(self, system) -> ScenarioResult:
                return ScenarioResult(
                    scenario_name=self._name,
                    system_name=system.name,
                    score=0.75,
                    max_score=1.0,
                    details=["stub result"],
                    duration_seconds=0.01,
                )

        runner.systems = [
            _StubSystem("engram"),
            _StubSystem("naive_rag"),
            _StubSystem("no_memory"),
        ]
        runner.scenarios = [
            _StubScenario("retention"),
            _StubScenario("interference"),
        ]
        return runner

    def test_run_all_returns_results(self, capsys):
        runner = self._make_runner_mocked()
        results = runner.run_all()
        # 3 systems × 2 scenarios = 6 results
        assert len(results) == 6

    def test_run_all_calls_reset_on_each_system(self, capsys):
        runner = self._make_runner_mocked()
        runner.run_all()
        for system in runner.systems:
            assert system.reset_called == 1

    def test_run_all_prints_progress(self, capsys):
        runner = self._make_runner_mocked()
        runner.run_all()
        out = capsys.readouterr().out
        assert "engram" in out
        assert "retention" in out

    def test_run_all_results_appended_to_self_results(self, capsys):
        runner = self._make_runner_mocked()
        runner.run_all()
        assert len(runner.results) == 6

    def test_run_scenario_returns_per_system_results(self, capsys):
        runner = self._make_runner_mocked()
        results = runner.run_scenario("retention")
        assert len(results) == 3  # one per system
        assert all(r.scenario_name == "retention" for r in results)

    def test_run_scenario_unknown_raises(self):
        from benchmark.runner import BenchmarkRunner
        runner = self._make_runner_mocked()
        with pytest.raises(ValueError, match="Unknown scenario"):
            runner.run_scenario("nonexistent_scenario")
