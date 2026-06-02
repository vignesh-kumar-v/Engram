"""Unit tests for EpisodicBuffer."""

from __future__ import annotations

import time

import pytest

from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.schemas.memory_types import EpisodicEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf(capacity: int = 10) -> EpisodicBuffer:
    """Return a fresh in-memory buffer for each test."""
    return EpisodicBuffer(":memory:", capacity=capacity)


def _entry(
    content: str = "test content",
    source_task: str = "task_1",
    session_id: str = "sess_1",
    importance: float = 0.5,
    status: str = "raw",
    tags: list[str] | None = None,
) -> EpisodicEntry:
    return EpisodicEntry(
        content=content,
        source_task=source_task,
        session_id=session_id,
        importance=importance,
        status=status,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# write / read round-trip
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_write_returns_true(self):
        buf = _buf()
        assert buf.write(_entry()) is True

    def test_read_round_trip(self):
        buf = _buf()
        e = _entry(content="hello world", tags=["python", "bug"])
        buf.write(e)
        result = buf.read(e.id)
        assert result is not None
        assert result.id == e.id
        assert result.content == e.content
        assert result.source_task == e.source_task
        assert result.session_id == e.session_id
        assert result.importance == e.importance
        assert result.status == e.status
        assert result.tags == e.tags

    def test_read_unknown_id_returns_none(self):
        buf = _buf()
        assert buf.read("nonexistent-id") is None

    def test_write_duplicate_id_returns_false(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        assert buf.write(e) is False

    def test_timestamp_preserved(self):
        buf = _buf()
        e = _entry()
        known_ts = 1_700_000_000.0
        e.timestamp = known_ts
        buf.write(e)
        assert buf.read(e.id).timestamp == known_ts


# ---------------------------------------------------------------------------
# read_all_raw
# ---------------------------------------------------------------------------

class TestReadAllRaw:
    def test_returns_only_raw_entries(self):
        buf = _buf()
        raw = _entry(content="raw", status="raw")
        promoted = _entry(content="promoted", status="promoted")
        discarded = _entry(content="discarded", status="discarded")
        for e in (raw, promoted, discarded):
            buf.write(e)
        results = buf.read_all_raw()
        assert len(results) == 1
        assert results[0].id == raw.id

    def test_ordered_by_timestamp_ascending(self):
        buf = _buf()
        now = time.time()
        e1 = _entry(content="first")
        e2 = _entry(content="second")
        e1.timestamp = now - 10
        e2.timestamp = now
        buf.write(e1)
        buf.write(e2)
        results = buf.read_all_raw()
        assert results[0].id == e1.id
        assert results[1].id == e2.id

    def test_empty_buffer_returns_empty_list(self):
        assert _buf().read_all_raw() == []


# ---------------------------------------------------------------------------
# get_count / is_at_capacity
# ---------------------------------------------------------------------------

class TestCapacity:
    def test_get_count_empty(self):
        assert _buf().get_count() == 0

    def test_get_count_increments(self):
        buf = _buf()
        buf.write(_entry())
        buf.write(_entry())
        assert buf.get_count() == 2

    def test_is_at_capacity_false_when_under(self):
        buf = _buf(capacity=3)
        buf.write(_entry())
        buf.write(_entry())
        assert buf.is_at_capacity() is False

    def test_is_at_capacity_true_at_limit(self):
        buf = _buf(capacity=2)
        buf.write(_entry())
        buf.write(_entry())
        assert buf.is_at_capacity() is True

    def test_is_at_capacity_true_when_over(self):
        # capacity check is read-only; nothing prevents over-filling via write()
        buf = _buf(capacity=1)
        buf.write(_entry())
        buf.write(_entry())
        assert buf.is_at_capacity() is True


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

class TestUpdateStatus:
    def test_update_to_promoted(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        assert buf.update_status(e.id, "promoted") is True
        assert buf.read(e.id).status == "promoted"

    def test_update_to_discarded(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        assert buf.update_status(e.id, "discarded") is True
        assert buf.read(e.id).status == "discarded"

    def test_invalid_status_returns_false(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        assert buf.update_status(e.id, "archived") is False
        # Original status unchanged
        assert buf.read(e.id).status == "raw"

    def test_nonexistent_id_returns_false(self):
        assert _buf().update_status("no-such-id", "promoted") is False


# ---------------------------------------------------------------------------
# delete_entry
# ---------------------------------------------------------------------------

class TestDeleteEntry:
    def test_delete_existing_entry(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        assert buf.delete_entry(e.id) is True
        assert buf.read(e.id) is None

    def test_delete_reduces_count(self):
        buf = _buf()
        e = _entry()
        buf.write(e)
        buf.delete_entry(e.id)
        assert buf.get_count() == 0

    def test_delete_nonexistent_returns_false(self):
        assert _buf().delete_entry("ghost-id") is False


# ---------------------------------------------------------------------------
# clear_non_raw
# ---------------------------------------------------------------------------

class TestClearNonRaw:
    def test_clears_promoted_and_discarded(self):
        buf = _buf()
        raw = _entry(status="raw")
        p = _entry(status="promoted")
        d = _entry(status="discarded")
        for e in (raw, p, d):
            buf.write(e)

        deleted = buf.clear_non_raw()
        assert deleted == 2
        assert buf.get_count() == 1
        assert buf.read(raw.id) is not None

    def test_returns_zero_when_nothing_to_clear(self):
        buf = _buf()
        buf.write(_entry(status="raw"))
        assert buf.clear_non_raw() == 0

    def test_empty_buffer_returns_zero(self):
        assert _buf().clear_non_raw() == 0

    def test_raw_entries_untouched(self):
        buf = _buf()
        raw = _entry(content="keep me", status="raw")
        buf.write(raw)
        buf.write(_entry(status="promoted"))
        buf.clear_non_raw()
        result = buf.read(raw.id)
        assert result is not None
        assert result.content == "keep me"


# ---------------------------------------------------------------------------
# search_by_tag
# ---------------------------------------------------------------------------

class TestSearchByTag:
    def test_finds_entry_with_matching_tag(self):
        buf = _buf()
        e = _entry(tags=["python", "bug"])
        buf.write(e)
        results = buf.search_by_tag("python")
        assert any(r.id == e.id for r in results)

    def test_does_not_return_entry_without_tag(self):
        buf = _buf()
        e = _entry(tags=["refactor"])
        buf.write(e)
        results = buf.search_by_tag("bug")
        assert all(r.id != e.id for r in results)

    def test_no_partial_tag_match(self):
        buf = _buf()
        e = _entry(tags=["python"])
        buf.write(e)
        # "py" should not match "python"
        assert buf.search_by_tag("py") == []

    def test_ordered_by_timestamp_descending(self):
        buf = _buf()
        now = time.time()
        e_old = _entry(tags=["shared"])
        e_new = _entry(tags=["shared"])
        e_old.timestamp = now - 100
        e_new.timestamp = now
        buf.write(e_old)
        buf.write(e_new)
        results = buf.search_by_tag("shared")
        assert results[0].id == e_new.id
        assert results[1].id == e_old.id

    def test_empty_tag_list_entry_not_returned(self):
        buf = _buf()
        buf.write(_entry(tags=[]))
        assert buf.search_by_tag("anything") == []

    def test_tags_preserved_on_round_trip(self):
        buf = _buf()
        e = _entry(tags=["a", "b", "c"])
        buf.write(e)
        result = buf.search_by_tag("b")
        assert result[0].tags == ["a", "b", "c"]
