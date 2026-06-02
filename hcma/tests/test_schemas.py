"""Unit tests for HCMA memory type schemas."""

import time
import pytest

from hcma.schemas.memory_types import EpisodicEntry, LTMMemory


def _make_episode(**kwargs) -> EpisodicEntry:
    defaults = dict(content="test content", source_task="task_1", session_id="s_001")
    defaults.update(kwargs)
    return EpisodicEntry(**defaults)


def _make_ltm(**kwargs) -> LTMMemory:
    now = time.time()
    defaults = dict(
        content="test fact",
        source_episode_ids=["ep_1"],
        created_at=now,
        last_accessed=now,
    )
    defaults.update(kwargs)
    return LTMMemory(**defaults)


class TestEpisodicEntryDefaults:
    def test_id_is_generated(self):
        e = _make_episode()
        assert isinstance(e.id, str) and len(e.id) == 36

    def test_timestamp_is_recent(self):
        before = time.time()
        e = _make_episode()
        after = time.time()
        assert before <= e.timestamp <= after

    def test_importance_default(self):
        assert _make_episode().importance == 0.5

    def test_status_default(self):
        assert _make_episode().status == "raw"

    def test_tags_default_empty(self):
        e = _make_episode()
        assert e.tags == []

    def test_tags_not_shared_between_instances(self):
        a, b = _make_episode(), _make_episode()
        a.tags.append("x")
        assert b.tags == []


class TestEpisodicEntryValidation:
    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status"):
            _make_episode(status="unknown")

    def test_valid_statuses_accepted(self):
        for s in ("raw", "promoted", "discarded"):
            e = _make_episode(status=s)
            assert e.status == s

    def test_importance_below_range_raises(self):
        with pytest.raises(ValueError, match="importance"):
            _make_episode(importance=-0.1)

    def test_importance_above_range_raises(self):
        with pytest.raises(ValueError, match="importance"):
            _make_episode(importance=1.1)

    def test_importance_boundary_values_accepted(self):
        assert _make_episode(importance=0.0).importance == 0.0
        assert _make_episode(importance=1.0).importance == 1.0


class TestLTMMemoryDefaults:
    def test_id_is_generated(self):
        m = _make_ltm()
        assert isinstance(m.id, str) and len(m.id) == 36

    def test_confidence_default(self):
        assert _make_ltm().confidence == 0.7

    def test_access_count_default(self):
        assert _make_ltm().access_count == 0

    def test_vector_id_default_empty(self):
        assert _make_ltm().vector_id == ""

    def test_memory_type_default(self):
        assert _make_ltm().memory_type == "fact"


class TestLTMMemoryValidation:
    def test_invalid_memory_type_raises(self):
        with pytest.raises(ValueError, match="memory_type"):
            _make_ltm(memory_type="hallucination")

    def test_valid_memory_types_accepted(self):
        for t in ("fact", "preference", "pattern", "error"):
            m = _make_ltm(memory_type=t)
            assert m.memory_type == t

    def test_confidence_below_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_ltm(confidence=-0.1)

    def test_confidence_above_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            _make_ltm(confidence=1.01)

    def test_confidence_boundary_values_accepted(self):
        assert _make_ltm(confidence=0.0).confidence == 0.0
        assert _make_ltm(confidence=1.0).confidence == 1.0
