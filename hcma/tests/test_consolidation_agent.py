"""Unit tests for ConsolidationAgent — all LLM and LTM calls are mocked."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from hcma.agents.consolidation_agent import ConsolidationAgent
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.schemas.memory_types import (
    ConsolidationResult,
    ContradictionFlag,
    Decision,
    EpisodicEntry,
    LTMMemory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf() -> EpisodicBuffer:
    return EpisodicBuffer(":memory:", capacity=50)


def _entry(
    content: str = "Python dicts preserve insertion order since 3.7",
    tags: list[str] | None = None,
    importance: float = 0.6,
    status: str = "raw",
) -> EpisodicEntry:
    return EpisodicEntry(
        content=content,
        source_task="test_task",
        session_id="sess_test",
        tags=tags or [],
        importance=importance,
        status=status,
    )


def _ltm_memory(content: str = "Some fact") -> LTMMemory:
    now = time.time()
    return LTMMemory(
        content=content,
        source_episode_ids=["ep_x"],
        created_at=now,
        last_accessed=now,
    )


def _mock_ltm() -> MagicMock:
    ltm = MagicMock()
    ltm.get_all.return_value = []
    ltm.read.return_value = None
    ltm.write.return_value = True
    return ltm


def _agent(buf: EpisodicBuffer | None = None, ltm: MagicMock | None = None) -> ConsolidationAgent:
    agent = ConsolidationAgent(buf or _buf(), ltm or _mock_ltm())
    return agent


def _llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def _promote_response() -> MagicMock:
    return _llm_response("ACTION: promote\nEXISTING_ID: \nREASONING: New useful info.")


def _discard_response() -> MagicMock:
    return _llm_response("ACTION: discard\nEXISTING_ID: \nREASONING: Trivial noise.")


# ---------------------------------------------------------------------------
# Schema tests (new dataclasses)
# ---------------------------------------------------------------------------

class TestNewSchemas:
    def test_decision_valid(self):
        d = Decision(action="promote", reasoning="useful")
        assert d.action == "promote"
        assert d.existing_memory_id == ""

    def test_decision_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            Decision(action="ignore", reasoning="x")

    def test_consolidation_result_defaults(self):
        r = ConsolidationResult()
        assert r.promoted == 0
        assert r.compressed == 0
        assert r.discarded == 0
        assert r.total_processed == 0

    def test_contradiction_flag_valid(self):
        f = ContradictionFlag(memory_id_a="a", memory_id_b="b", reason="r")
        assert f.severity == "low"

    def test_contradiction_flag_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            ContradictionFlag(memory_id_a="a", memory_id_b="b", reason="r", severity="critical")


# ---------------------------------------------------------------------------
# run() — empty buffer
# ---------------------------------------------------------------------------

class TestRunEmptyBuffer:
    def test_empty_buffer_returns_zero_result(self):
        agent = _agent()
        result = agent.run()
        assert isinstance(result, ConsolidationResult)
        assert result.promoted == 0
        assert result.compressed == 0
        assert result.discarded == 0
        assert result.total_processed == 0

    def test_empty_buffer_skips_llm(self):
        agent = _agent()
        with patch.object(agent._client, "chat") as mock_chat:
            agent.run()
        mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# run() — counts
# ---------------------------------------------------------------------------

class TestRunCounts:
    def test_promoted_count(self):
        buf = _buf()
        e1, e2 = _entry("fact one"), _entry("fact two")
        buf.write(e1)
        buf.write(e2)

        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_promote_response()):
            result = agent.run()

        assert result.promoted == 2
        assert result.total_processed == 2
        assert result.discarded == 0

    def test_discarded_count(self):
        buf = _buf()
        buf.write(_entry("trivial noise"))

        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_discard_response()):
            result = agent.run()

        assert result.discarded == 1
        assert result.promoted == 0

    def test_mixed_actions(self):
        buf = _buf()
        e_promote = _entry("useful fact about Python")
        e_discard = _entry("noise entry")
        buf.write(e_promote)
        buf.write(e_discard)

        agent = _agent(buf)
        responses = [_promote_response(), _discard_response()]
        with patch.object(agent._client, "chat", side_effect=responses):
            result = agent.run()

        assert result.promoted == 1
        assert result.discarded == 1
        assert result.total_processed == 2

    def test_duration_is_positive(self):
        buf = _buf()
        buf.write(_entry())
        agent = _agent(buf)
        with patch.object(agent._client, "chat", return_value=_promote_response()):
            result = agent.run()
        assert result.duration_seconds > 0.0


# ---------------------------------------------------------------------------
# _decide()
# ---------------------------------------------------------------------------

class TestDecide:
    def test_returns_promote_for_new_content(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_promote_response()):
            decision = agent._decide(_entry())
        assert decision.action == "promote"
        assert decision.reasoning == "New useful info."

    def test_returns_discard(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_discard_response()):
            decision = agent._decide(_entry())
        assert decision.action == "discard"

    def test_returns_compress_with_existing_id(self):
        agent = _agent()
        mem_id = "abcd1234-ffff-ffff-ffff-000000000000"
        resp = _llm_response(
            f"ACTION: compress\nEXISTING_ID: {mem_id}\nREASONING: Similar to existing."
        )
        with patch.object(agent._client, "chat", return_value=resp):
            decision = agent._decide(_entry())
        assert decision.action == "compress"
        assert decision.existing_memory_id == mem_id

    def test_fallback_on_llm_failure(self):
        agent = _agent()
        with patch.object(agent._client, "chat", side_effect=RuntimeError("down")):
            decision = agent._decide(_entry())
        assert decision.action == "promote"
        assert "fallback" in decision.reasoning

    def test_fallback_on_bad_parse(self):
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_llm_response("GARBAGE OUTPUT")):
            decision = agent._decide(_entry())
        # action defaults to promote when no ACTION: line found
        assert decision.action == "promote"

    def test_ltm_summary_injected_into_prompt(self):
        ltm = _mock_ltm()
        mem = _ltm_memory("Existing fact in LTM")
        mem.id = "aaaabbbb-0000-0000-0000-000000000000"
        ltm.get_all.return_value = [mem]

        agent = _agent(ltm=ltm)
        with patch.object(agent._client, "chat", return_value=_promote_response()) as mock_chat:
            agent._decide(_entry())

        system_msg = mock_chat.call_args.kwargs.get("messages")[0]["content"]
        assert "aaaabbbb" in system_msg

    def test_empty_ltm_uses_placeholder(self):
        agent = _agent()  # ltm.get_all returns []
        with patch.object(agent._client, "chat", return_value=_promote_response()) as mock_chat:
            agent._decide(_entry())

        system_msg = mock_chat.call_args.kwargs.get("messages")[0]["content"]
        assert "LTM is currently empty." in system_msg


# ---------------------------------------------------------------------------
# _promote()
# ---------------------------------------------------------------------------

class TestPromote:
    def test_creates_ltm_memory_with_correct_fields(self):
        buf = _buf()
        ltm = _mock_ltm()
        agent = _agent(buf, ltm)

        e = _entry(content="Generators use yield", tags=["pattern"], importance=0.8)
        buf.write(e)

        agent._promote(e)

        written: LTMMemory = ltm.write.call_args[0][0]
        assert written.content == e.content
        assert written.confidence == e.importance
        assert e.id in written.source_episode_ids
        assert written.memory_type == "pattern"

    def test_updates_entry_status_to_promoted(self):
        buf = _buf()
        ltm = _mock_ltm()
        agent = _agent(buf, ltm)

        e = _entry()
        buf.write(e)
        agent._promote(e)

        assert buf.read(e.id).status == "promoted"

    def test_returns_true_on_success(self):
        buf = _buf()
        ltm = _mock_ltm()
        agent = _agent(buf, ltm)
        e = _entry()
        buf.write(e)
        assert agent._promote(e) is True

    def test_returns_false_when_ltm_write_fails(self):
        buf = _buf()
        ltm = _mock_ltm()
        ltm.write.return_value = False
        agent = _agent(buf, ltm)
        e = _entry()
        buf.write(e)
        assert agent._promote(e) is False


# ---------------------------------------------------------------------------
# _infer_memory_type()
# ---------------------------------------------------------------------------

class TestInferMemoryType:
    def test_debug_tag_maps_to_error(self):
        agent = _agent()
        assert agent._infer_memory_type(["debug"]) == "error"

    def test_error_pattern_tag_maps_to_error(self):
        agent = _agent()
        assert agent._infer_memory_type(["error_pattern"]) == "error"

    def test_preference_tag(self):
        agent = _agent()
        assert agent._infer_memory_type(["preference"]) == "preference"

    def test_pattern_tag(self):
        agent = _agent()
        assert agent._infer_memory_type(["pattern"]) == "pattern"

    def test_default_is_fact(self):
        agent = _agent()
        assert agent._infer_memory_type([]) == "fact"
        assert agent._infer_memory_type(["user_query"]) == "fact"
        assert agent._infer_memory_type(["assistant_response"]) == "fact"

    def test_debug_takes_priority_over_pattern(self):
        agent = _agent()
        assert agent._infer_memory_type(["debug", "pattern"]) == "error"


# ---------------------------------------------------------------------------
# _detect_contradictions()
# ---------------------------------------------------------------------------

class TestDetectContradictions:
    def test_returns_empty_list_with_no_memories(self):
        ltm = _mock_ltm()
        ltm.get_all.return_value = []
        agent = _agent(ltm=ltm)
        assert agent._detect_contradictions() == []

    def test_returns_empty_list_with_one_memory(self):
        ltm = _mock_ltm()
        ltm.get_all.return_value = [_ltm_memory()]
        agent = _agent(ltm=ltm)
        assert agent._detect_contradictions() == []

    def test_returns_empty_list_with_two_memories(self):
        ltm = _mock_ltm()
        ltm.get_all.return_value = [_ltm_memory(), _ltm_memory()]
        agent = _agent(ltm=ltm)
        # Only groups with 3+ trigger LLM; two memories → no LLM call
        with patch.object(agent._client, "chat") as mock_chat:
            result = agent._detect_contradictions()
        mock_chat.assert_not_called()
        assert result == []

    def test_skips_groups_with_fewer_than_3(self):
        ltm = _mock_ltm()
        m1, m2 = _ltm_memory("fact A"), _ltm_memory("fact B")
        ltm.get_all.return_value = [m1, m2]
        agent = _agent(ltm=ltm)
        with patch.object(agent._client, "chat") as mock_chat:
            agent._detect_contradictions()
        mock_chat.assert_not_called()

    def test_calls_llm_for_group_with_3_plus(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(3)]
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        with patch.object(agent._client, "chat", return_value=_llm_response("NONE")) as mock_chat:
            agent._detect_contradictions()

        mock_chat.assert_called_once()

    def test_parses_contradiction_response(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(3)]
        for m in members:
            m.memory_type = "fact"
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        id_a = members[0].id[:8]
        id_b = members[1].id[:8]
        contradiction_text = (
            f"CONTRADICTION: {id_a} {id_b} | SEVERITY: high | REASON: They conflict."
        )
        with patch.object(agent._client, "chat", return_value=_llm_response(contradiction_text)):
            flags = agent._detect_contradictions()

        assert len(flags) == 1
        assert flags[0].severity == "high"
        assert flags[0].reason == "They conflict."

    def test_llm_failure_returns_empty_list(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(3)]
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        with patch.object(agent._client, "chat", side_effect=RuntimeError("timeout")):
            flags = agent._detect_contradictions()

        assert flags == []

    def test_save_contradiction_called_for_each_flag(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(3)]
        for m in members:
            m.memory_type = "fact"
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        id_a = members[0].id[:8]
        id_b = members[1].id[:8]
        contradiction_text = (
            f"CONTRADICTION: {id_a} {id_b} | SEVERITY: low | REASON: Minor conflict."
        )
        with patch.object(agent._client, "chat", return_value=_llm_response(contradiction_text)):
            flags = agent._detect_contradictions()

        assert len(flags) == 1
        ltm.save_contradiction.assert_called_once()
        saved_flag = ltm.save_contradiction.call_args[0][0]
        assert saved_flag.reason == "Minor conflict."

    def test_save_contradiction_not_called_when_no_flags(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(3)]
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        with patch.object(agent._client, "chat", return_value=_llm_response("NONE")):
            agent._detect_contradictions()

        ltm.save_contradiction.assert_not_called()

    def test_save_contradiction_called_multiple_times_for_multiple_flags(self):
        ltm = _mock_ltm()
        members = [_ltm_memory(f"fact {i}") for i in range(4)]
        for m in members:
            m.memory_type = "fact"
        ltm.get_all.return_value = members
        agent = _agent(ltm=ltm)

        id_a, id_b, id_c, id_d = [m.id[:8] for m in members]
        contradiction_text = (
            f"CONTRADICTION: {id_a} {id_b} | SEVERITY: low | REASON: First conflict.\n"
            f"CONTRADICTION: {id_c} {id_d} | SEVERITY: high | REASON: Second conflict."
        )
        with patch.object(agent._client, "chat", return_value=_llm_response(contradiction_text)):
            flags = agent._detect_contradictions()

        assert len(flags) == 2
        assert ltm.save_contradiction.call_count == 2


# ---------------------------------------------------------------------------
# _compress()
# ---------------------------------------------------------------------------

class TestCompress:
    def _real_ltm_store(self, tmp_path):
        from hcma.memory.ltm_store import LTMStore
        from unittest.mock import patch as _patch
        with _patch("hcma.memory.ltm_store.QdrantClient") as mock_qc:
            mock_qc.return_value.collection_exists.return_value = True
            store = LTMStore(
                db_path=":memory:",
                qdrant_storage_path=str(tmp_path),
                collection_name="test",
            )
        return store

    def test_compress_merges_content(self, tmp_path):
        ltm = self._real_ltm_store(tmp_path)
        existing = _ltm_memory("Original content about Python")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        e = _entry("Additional Python info")
        buf.write(e)

        agent = ConsolidationAgent(buf, ltm)
        agent._compress(e, existing.id)

        updated = ltm.read(existing.id)
        assert "Additional context:" in updated.content
        assert "Additional Python info" in updated.content

    def test_compress_increases_confidence(self, tmp_path):
        ltm = self._real_ltm_store(tmp_path)
        existing = _ltm_memory("some fact")
        existing.confidence = 0.7
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        e = _entry()
        buf.write(e)

        agent = ConsolidationAgent(buf, ltm)
        agent._compress(e, existing.id)

        assert ltm.read(existing.id).confidence == pytest.approx(0.75)

    def test_compress_falls_back_to_promote_if_not_found(self, tmp_path):
        ltm = self._real_ltm_store(tmp_path)
        buf = _buf()
        e = _entry()
        buf.write(e)

        agent = ConsolidationAgent(buf, ltm)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(e, "nonexistent-id")

        assert result is True
        assert buf.read(e.id).status == "promoted"

    def test_compress_appends_episode_id(self, tmp_path):
        ltm = self._real_ltm_store(tmp_path)
        existing = _ltm_memory("some fact")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        e = _entry()
        buf.write(e)

        agent = ConsolidationAgent(buf, ltm)
        agent._compress(e, existing.id)

        updated = ltm.read(existing.id)
        assert e.id in updated.source_episode_ids
