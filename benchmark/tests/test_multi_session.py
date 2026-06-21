"""Tests for multi-session harness — all LLM/embedding calls mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark.multi_session_runner import MultiSessionRunner
from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.scenarios.multi_session_lifecycle import (
    MultiSessionLifecycleScenario,
    SessionSpec,
)
from benchmark.systems.base_system import BaseSystem
from benchmark.systems.engram_system import EngramSystem
from benchmark.systems.naive_rag_system import NaiveRagSystem
from benchmark.systems.no_memory_system import NoMemorySystem
from hcma.schemas.memory_types import ConsolidationResult


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


def _make_engram(tmp_path) -> EngramSystem:
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_result = MagicMock()
    mock_result.points = []
    mock_qdrant.query_points.return_value = mock_result
    with patch("hcma.memory.ltm_store.QdrantClient", return_value=mock_qdrant):
        sys = EngramSystem()
    return sys


def _make_naive_rag() -> NaiveRagSystem:
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False
    mock_result = MagicMock()
    mock_result.points = []
    mock_qdrant.query_points.return_value = mock_result
    with patch("benchmark.systems.naive_rag_system.QdrantClient", return_value=mock_qdrant):
        sys = NaiveRagSystem()
    return sys


def _small_scenario() -> MultiSessionLifecycleScenario:
    """Returns the real scenario (used to validate its structure)."""
    return MultiSessionLifecycleScenario()


# ---------------------------------------------------------------------------
# SessionSpec
# ---------------------------------------------------------------------------

class TestSessionSpec:
    def test_fields_accessible(self):
        spec = SessionSpec(
            session_id="s1",
            label="Test session",
            facts=[("fact content", ["tag1", "tag2"])],
        )
        assert spec.session_id == "s1"
        assert spec.label == "Test session"
        assert spec.facts[0][0] == "fact content"
        assert spec.facts[0][1] == ["tag1", "tag2"]

    def test_default_facts_is_empty_list(self):
        spec = SessionSpec(session_id="s", label="l")
        assert spec.facts == []


# ---------------------------------------------------------------------------
# MultiSessionLifecycleScenario structure
# ---------------------------------------------------------------------------

class TestMultiSessionLifecycleScenario:
    def test_name(self):
        assert _small_scenario().name == "multi_session_lifecycle"

    def test_has_seven_ingestion_sessions(self):
        assert len(_small_scenario().sessions) == 7

    def test_has_at_least_four_queries(self):
        assert len(_small_scenario().queries) >= 4

    def test_session_ids_are_unique(self):
        ids = [s.session_id for s in _small_scenario().sessions]
        assert len(ids) == len(set(ids))

    def test_all_facts_are_non_empty_strings(self):
        for session in _small_scenario().sessions:
            for content, tags in session.facts:
                assert isinstance(content, str) and len(content) > 0

    def test_all_facts_have_list_of_tags(self):
        for session in _small_scenario().sessions:
            for _content, tags in session.facts:
                assert isinstance(tags, list)

    def test_all_queries_have_question_and_keyword(self):
        for question, keyword in _small_scenario().queries:
            assert isinstance(question, str) and len(question) > 0
            assert isinstance(keyword, str) and len(keyword) > 0

    def test_sessions_have_labels(self):
        for session in _small_scenario().sessions:
            assert len(session.label) > 0

    def test_misleading_session_exists(self):
        """Session 5 should introduce the ORM hypothesis (misleading path)."""
        sessions = _small_scenario().sessions
        orms = [s for s in sessions if "orm" in " ".join(c for c, _ in s.facts).lower()]
        assert len(orms) >= 1

    def test_correction_session_exists(self):
        """A session should correct the ORM hypothesis."""
        sessions = _small_scenario().sessions
        corrections = [
            s for s in sessions
            if any("ruled out" in c.lower() or "disproved" in c.lower() for c, _ in s.facts)
        ]
        assert len(corrections) >= 1

    def test_postgresql_in_facts(self):
        all_facts = [c for s in _small_scenario().sessions for c, _ in s.facts]
        assert any("postgresql" in f.lower() for f in all_facts)

    def test_websocket_root_cause_in_facts(self):
        all_facts = [c for s in _small_scenario().sessions for c, _ in s.facts]
        assert any("websocket" in f.lower() and "root cause" in f.lower() for f in all_facts)


# ---------------------------------------------------------------------------
# BaseSystem extensions
# ---------------------------------------------------------------------------

class TestBaseSystemExtensions:
    def _make_concrete(self) -> BaseSystem:
        class Concrete(BaseSystem):
            @property
            def name(self):
                return "concrete"
            def ingest(self, session_id, content, tags):
                pass
            def query(self, question):
                return ""
            def reset(self):
                pass

        return Concrete()

    def test_after_session_default_does_not_raise(self):
        sys = self._make_concrete()
        sys.after_session()  # must be a no-op

    def test_get_session_state_default_returns_empty_dict(self):
        sys = self._make_concrete()
        state = sys.get_session_state()
        assert isinstance(state, dict)
        assert state == {}


# ---------------------------------------------------------------------------
# EngramSystem extensions
# ---------------------------------------------------------------------------

class TestEngramSystemExtensions:
    def test_after_session_triggers_consolidation(self, tmp_path):
        sys = _make_engram(tmp_path)
        with patch.object(
            sys._consolidation_agent,
            "run",
            return_value=ConsolidationResult(),
        ) as mock_run:
            sys.after_session()
        mock_run.assert_called_once()

    def test_after_session_survives_consolidation_failure(self, tmp_path):
        sys = _make_engram(tmp_path)
        with patch.object(
            sys._consolidation_agent, "run", side_effect=RuntimeError("down")
        ):
            sys.after_session()  # must not propagate

    def test_get_session_state_returns_required_keys(self, tmp_path):
        sys = _make_engram(tmp_path)
        state = sys.get_session_state()
        assert "buffer_raw" in state
        assert "ltm_memories" in state
        assert "contradictions" in state

    def test_get_session_state_buffer_raw_counts_only_raw_entries(self, tmp_path):
        sys = _make_engram(tmp_path)
        sys.ingest("s1", "fact one", ["fact"])
        sys.ingest("s1", "fact two", ["fact"])
        state = sys.get_session_state()
        assert state["buffer_raw"] == 2

    def test_get_session_state_buffer_raw_zero_after_consolidation(self, tmp_path):
        sys = _make_engram(tmp_path)
        sys.ingest("s1", "some content", ["fact"])

        # Mock consolidation to mark entries promoted without touching Qdrant/LLM
        def fake_consolidation():
            from hcma.schemas.memory_types import ConsolidationResult
            for entry in sys._buf.read_all_raw():
                sys._buf.update_status(entry.id, "promoted")
            return ConsolidationResult(promoted=1, total_processed=1)

        with patch.object(sys._consolidation_agent, "run", side_effect=fake_consolidation):
            sys.after_session()

        state = sys.get_session_state()
        assert state["buffer_raw"] == 0

    def test_get_session_state_initial_ltm_is_zero(self, tmp_path):
        sys = _make_engram(tmp_path)
        state = sys.get_session_state()
        assert state["ltm_memories"] == 0
        assert state["contradictions"] == 0


# ---------------------------------------------------------------------------
# NaiveRagSystem extensions
# ---------------------------------------------------------------------------

class TestNaiveRagSystemExtensions:
    def test_get_session_state_returns_stored_vectors_key(self):
        sys = _make_naive_rag()
        state = sys.get_session_state()
        assert "stored_vectors" in state

    def test_stored_count_zero_initially(self):
        sys = _make_naive_rag()
        assert sys.get_session_state()["stored_vectors"] == 0

    def test_stored_count_increments_on_successful_ingest(self):
        sys = _make_naive_rag()
        with patch(
            "benchmark.systems.naive_rag_system.ollama.embeddings",
            return_value=_embed_response(),
        ):
            sys.ingest("s1", "fact one", [])
            sys.ingest("s1", "fact two", [])
        assert sys.get_session_state()["stored_vectors"] == 2

    def test_stored_count_not_incremented_on_empty_embedding(self):
        sys = _make_naive_rag()
        with patch(
            "benchmark.systems.naive_rag_system.ollama.embeddings",
            side_effect=RuntimeError("no ollama"),
        ):
            sys.ingest("s1", "fact", [])
        assert sys.get_session_state()["stored_vectors"] == 0

    def test_reset_clears_stored_count(self):
        sys = _make_naive_rag()
        with patch(
            "benchmark.systems.naive_rag_system.ollama.embeddings",
            return_value=_embed_response(),
        ):
            sys.ingest("s1", "fact", [])
        assert sys.get_session_state()["stored_vectors"] == 1
        sys.reset()
        assert sys.get_session_state()["stored_vectors"] == 0

    def test_after_session_default_does_not_raise(self):
        sys = _make_naive_rag()
        sys.after_session()  # no-op


# ---------------------------------------------------------------------------
# NoMemorySystem extensions
# ---------------------------------------------------------------------------

class TestNoMemorySystemExtensions:
    def test_get_session_state_returns_empty_dict(self):
        sys = NoMemorySystem()
        assert sys.get_session_state() == {}

    def test_after_session_does_not_raise(self):
        NoMemorySystem().after_session()


# ---------------------------------------------------------------------------
# MultiSessionRunner
# ---------------------------------------------------------------------------

class _StubSystem:
    """Minimal system stub for runner tests."""

    def __init__(self, name: str = "stub", keyword: str = "answer") -> None:
        self._name = name
        self._keyword = keyword
        self.reset_called = 0
        self.after_session_calls: list[str] = []
        self.ingest_calls: list[tuple] = []

    @property
    def name(self) -> str:
        return self._name

    def ingest(self, session_id: str, content: str, tags: list[str]) -> None:
        self.ingest_calls.append((session_id, content, tags))

    def query(self, question: str) -> str:
        return f"The {self._keyword} is here."

    def reset(self) -> None:
        self.reset_called += 1

    def after_session(self) -> None:
        # record which session triggered this (last ingested session_id)
        if self.ingest_calls:
            self.after_session_calls.append(self.ingest_calls[-1][0])

    def get_session_state(self) -> dict:
        return {"stored": len(self.ingest_calls)}


def _small_runner(keyword: str = "postgresql") -> MultiSessionRunner:
    scenario = MultiSessionLifecycleScenario()
    systems = [_StubSystem("stub", keyword)]
    return MultiSessionRunner(scenario, systems)


class TestMultiSessionRunnerBasics:
    def test_run_returns_tuple_of_results_and_snapshots(self):
        runner = _small_runner()
        results, snapshots = runner.run()
        assert isinstance(results, list)
        assert isinstance(snapshots, dict)

    def test_run_returns_one_result_per_system(self):
        scenario = MultiSessionLifecycleScenario()
        systems = [_StubSystem("a"), _StubSystem("b")]
        runner = MultiSessionRunner(scenario, systems)
        results, _ = runner.run()
        assert len(results) == 2

    def test_result_scenario_name_correct(self):
        results, _ = _small_runner().run()
        assert results[0].scenario_name == "multi_session_lifecycle"

    def test_result_system_name_correct(self):
        results, _ = _small_runner().run()
        assert results[0].system_name == "stub"

    def test_score_is_float_between_0_and_1(self):
        results, _ = _small_runner("postgresql").run()
        assert 0.0 <= results[0].score <= 1.0

    def test_perfect_score_when_keyword_matches_all_queries(self):
        """Stub returns 'postgresql' in every response; not all queries expect that — partial."""
        results, _ = _small_runner("postgresql").run()
        # Only the "What database" query expects 'postgresql', others expect different keywords
        assert results[0].score < 1.0  # not all queries pass

    def test_details_list_length_matches_query_count(self):
        scenario = MultiSessionLifecycleScenario()
        results, _ = _small_runner().run()
        assert len(results[0].details) == len(scenario.queries)

    def test_details_start_with_pass_or_fail(self):
        results, _ = _small_runner().run()
        for detail in results[0].details:
            assert detail.startswith("PASS") or detail.startswith("FAIL")


class TestMultiSessionRunnerSessionHandling:
    def test_reset_called_once_per_system(self):
        scenario = MultiSessionLifecycleScenario()
        sys_a = _StubSystem("a")
        sys_b = _StubSystem("b")
        runner = MultiSessionRunner(scenario, [sys_a, sys_b])
        runner.run()
        assert sys_a.reset_called == 1
        assert sys_b.reset_called == 1

    def test_after_session_called_once_per_session(self):
        scenario = MultiSessionLifecycleScenario()
        sys = _StubSystem()
        runner = MultiSessionRunner(scenario, [sys])
        runner.run()
        assert len(sys.after_session_calls) == len(scenario.sessions)

    def test_all_facts_ingested(self):
        scenario = MultiSessionLifecycleScenario()
        total_facts = sum(len(s.facts) for s in scenario.sessions)
        sys = _StubSystem()
        runner = MultiSessionRunner(scenario, [sys])
        runner.run()
        assert len(sys.ingest_calls) == total_facts

    def test_ingest_receives_correct_session_id(self):
        scenario = MultiSessionLifecycleScenario()
        sys = _StubSystem()
        runner = MultiSessionRunner(scenario, [sys])
        runner.run()
        for call in sys.ingest_calls:
            session_id = call[0]
            known_ids = {s.session_id for s in scenario.sessions}
            assert session_id in known_ids


class TestMultiSessionRunnerSnapshots:
    def test_snapshots_keyed_by_system_name(self):
        _, snapshots = _small_runner().run()
        assert "stub" in snapshots

    def test_snapshots_length_equals_session_count(self):
        scenario = MultiSessionLifecycleScenario()
        _, snapshots = _small_runner().run()
        assert len(snapshots["stub"]) == len(scenario.sessions)

    def test_each_snapshot_has_session_id_label_and_state(self):
        _, snapshots = _small_runner().run()
        for snap in snapshots["stub"]:
            assert "session_id" in snap
            assert "label" in snap
            assert "state" in snap

    def test_snapshot_session_ids_match_scenario(self):
        scenario = MultiSessionLifecycleScenario()
        _, snapshots = _small_runner().run()
        expected_ids = [s.session_id for s in scenario.sessions]
        actual_ids = [snap["session_id"] for snap in snapshots["stub"]]
        assert actual_ids == expected_ids

    def test_snapshot_state_from_get_session_state(self):
        _, snapshots = _small_runner().run()
        # _StubSystem.get_session_state returns {"stored": <ingest_count>}
        for snap in snapshots["stub"]:
            assert "stored" in snap["state"]


class TestMultiSessionRunnerTrace:
    def test_print_trace_before_run_does_not_raise(self, capsys):
        runner = _small_runner()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "run()" in out  # warns that run() hasn't been called

    def test_print_trace_after_run_does_not_raise(self, capsys):
        runner = _small_runner()
        runner.run()
        runner.print_trace()
        capsys.readouterr()

    def test_print_trace_includes_system_name(self, capsys):
        runner = _small_runner()
        runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "stub" in out

    def test_print_trace_includes_scenario_name(self, capsys):
        runner = _small_runner()
        runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "multi_session_lifecycle" in out

    def test_print_trace_includes_all_session_labels(self, capsys):
        runner = _small_runner()
        runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        for session in MultiSessionLifecycleScenario().sessions:
            assert session.label in out

    def test_print_trace_includes_query_results(self, capsys):
        runner = _small_runner()
        runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "PASS" in out or "FAIL" in out

    def test_print_trace_no_memory_shows_stateless(self, capsys):
        scenario = MultiSessionLifecycleScenario()
        sys = NoMemorySystem()
        runner = MultiSessionRunner(scenario, [sys])
        with patch.object(sys._client, "chat", return_value=_llm_response()):
            runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "stateless" in out

    def test_print_trace_naive_rag_shows_stored_vectors(self, capsys):
        sys = _make_naive_rag()
        runner = MultiSessionRunner(MultiSessionLifecycleScenario(), [sys])
        with patch("benchmark.systems.naive_rag_system.ollama.embeddings",
                   return_value=_embed_response()):
            with patch.object(sys._client, "chat", return_value=_llm_response()):
                runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "stored_vectors" in out

    def test_print_trace_engram_shows_buffer_and_ltm(self, capsys, tmp_path):
        sys = _make_engram(tmp_path)
        runner = MultiSessionRunner(MultiSessionLifecycleScenario(), [sys])
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            with patch.object(
                sys._consolidation_agent._client,
                "chat",
                return_value=_llm_response("ACTION: promote\nEXISTING_ID: \nREASONING: useful"),
            ):
                with patch.object(sys._llm, "chat", return_value=_llm_response()):
                    runner.run()
        runner.print_trace()
        out = capsys.readouterr().out
        assert "buffer_raw" in out
        assert "ltm" in out


# ---------------------------------------------------------------------------
# End-to-end with stubs (score calculation)
# ---------------------------------------------------------------------------

class TestRunnerScoreCalculation:
    def test_zero_score_when_no_keyword_matches(self):
        scenario = MultiSessionLifecycleScenario()
        # query keywords are: postgresql, fastapi, react, websocket, orm
        # response never contains any of them
        sys = _StubSystem("stub", "xyzzy-nomatch")
        runner = MultiSessionRunner(scenario, [sys])
        results, _ = runner.run()
        assert results[0].score == 0.0

    def test_partial_score_when_one_keyword_matches(self):
        scenario = MultiSessionLifecycleScenario()
        sys = _StubSystem("stub", "postgresql")  # only 1 query expects "postgresql"
        runner = MultiSessionRunner(scenario, [sys])
        results, _ = runner.run()
        n_queries = len(scenario.queries)
        assert results[0].score == pytest.approx(1.0 / n_queries)

    def test_duration_seconds_is_positive(self):
        results, _ = _small_runner().run()
        assert results[0].duration_seconds >= 0.0
