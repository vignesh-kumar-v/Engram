"""Tests for LTMStore — unit tests (default) and integration tests (marked)."""

from __future__ import annotations

import tempfile
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hcma.schemas.memory_types import ContradictionFlag

from hcma.memory.ltm_store import LTMStore
from hcma.schemas.memory_types import LTMMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memory(
    content: str = "Python lists are ordered collections",
    memory_type: str = "fact",
    source_ids: list[str] | None = None,
) -> LTMMemory:
    now = time.time()
    return LTMMemory(
        content=content,
        source_episode_ids=source_ids or ["ep_001"],
        created_at=now,
        last_accessed=now,
        memory_type=memory_type,
    )


def _unit_store(tmp_path) -> tuple[LTMStore, MagicMock]:
    """Return an LTMStore with mocked Qdrant and mocked ollama embeddings."""
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True

    # query_points returns an object with .points list
    mock_result = MagicMock()
    mock_result.points = []
    mock_qdrant.query_points.return_value = mock_result

    with patch("hcma.memory.ltm_store.QdrantClient", return_value=mock_qdrant):
        store = LTMStore(
            db_path=":memory:",
            qdrant_storage_path=str(tmp_path),
            collection_name="test_ltm",
        )
    return store, mock_qdrant


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestWriteUnit:
    def test_write_success_with_valid_embedding(self, tmp_path):
        store, mock_qdrant = _unit_store(tmp_path)
        mem = _memory()
        fake_vec = [0.1] * 768

        with patch("hcma.memory.ltm_store.ollama.embeddings") as mock_emb:
            mock_emb.return_value = SimpleNamespace(embedding=fake_vec)
            result = store.write(mem)

        assert result is True
        mock_qdrant.upsert.assert_called_once()
        assert mem.vector_id != ""

    def test_write_falls_back_when_embedding_fails(self, tmp_path):
        store, mock_qdrant = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError("no ollama")):
            result = store.write(mem)

        assert result is True
        mock_qdrant.upsert.assert_not_called()
        assert mem.vector_id == ""

    def test_write_fallback_still_persists_to_sqlite(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError("no ollama")):
            store.write(mem)

        assert store.read(mem.id) is not None

    def test_write_returns_false_on_sqlite_failure(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", return_value=MagicMock(embedding=[0.1]*768)):
            store._conn.close()  # force SQLite failure
            result = store.write(mem)

        assert result is False


class TestReadUnit:
    def test_read_returns_none_for_unknown_id(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.read("nonexistent-id") is None

    def test_read_round_trip(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="Decorators wrap functions")

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        result = store.read(mem.id)
        assert result is not None
        assert result.id == mem.id
        assert result.content == mem.content
        assert result.memory_type == mem.memory_type
        assert result.source_episode_ids == mem.source_episode_ids

    def test_read_preserves_source_episode_ids(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory(source_ids=["ep_a", "ep_b", "ep_c"])

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        assert store.read(mem.id).source_episode_ids == ["ep_a", "ep_b", "ep_c"]


class TestReadByPrefixUnit:
    def test_returns_none_for_empty_prefix(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.read_by_prefix("") is None

    def test_returns_none_when_no_match(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(_memory())
        assert store.read_by_prefix("00000000") is None

    def test_finds_memory_by_8char_prefix(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="Python uses indentation")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)
        prefix = mem.id[:8]
        result = store.read_by_prefix(prefix)
        assert result is not None
        assert result.id == mem.id

    def test_finds_memory_by_bracketed_prefix(self, tmp_path):
        """Simulates the bug: LLM echoes [82c97b40] — caller must strip brackets first."""
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="FastAPI handles routing")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)
        # The fix in _parse_decision strips brackets before calling read_by_prefix
        clean_prefix = mem.id[:8]
        result = store.read_by_prefix(clean_prefix)
        assert result is not None and result.id == mem.id

    def test_returns_first_match_when_prefix_collides(self, tmp_path):
        """Prefix collision is astronomically unlikely with UUIDs but must not crash."""
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="Some fact")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)
        result = store.read_by_prefix(mem.id[:4])
        assert result is not None  # finds at least the one we wrote

    def test_returns_full_memory_object(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="Generators are lazy", memory_type="pattern")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)
        result = store.read_by_prefix(mem.id[:8])
        assert result.content == mem.content
        assert result.memory_type == "pattern"


class TestSearchByContentUnit:
    def test_returns_matching_entries(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        m1 = _memory(content="Python decorators modify functions")
        m2 = _memory(content="List comprehensions are concise loops")

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(m1)
            store.write(m2)

        results = store.search_by_content("decorator")
        assert len(results) == 1
        assert results[0].id == m1.id

    def test_returns_empty_for_no_match(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(_memory(content="Python lists"))

        assert store.search_by_content("javascript") == []

    def test_respects_top_k(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            for i in range(5):
                store.write(_memory(content=f"Python fact number {i}"))

        results = store.search_by_content("Python", top_k=3)
        assert len(results) == 3

    def test_ordered_by_access_count(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        m1 = _memory(content="Python generators are lazy iterators")
        m2 = _memory(content="Python sets have no duplicates")

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(m1)
            store.write(m2)

        store.update_access(m2.id)
        store.update_access(m2.id)
        store.update_access(m2.id)

        results = store.search_by_content("Python")
        assert results[0].id == m2.id


class TestUpdateAccessUnit:
    def test_increments_access_count(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        store.update_access(mem.id)
        store.update_access(mem.id)

        result = store.read(mem.id)
        assert result.access_count == 2

    def test_updates_last_accessed_timestamp(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory()
        original_ts = mem.last_accessed

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        time.sleep(0.01)
        store.update_access(mem.id)
        assert store.read(mem.id).last_accessed > original_ts

    def test_returns_false_for_unknown_id(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.update_access("ghost-id") is False


class TestDeleteUnit:
    def test_delete_removes_from_sqlite(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        assert store.delete(mem.id) is True
        assert store.read(mem.id) is None

    def test_delete_skips_qdrant_when_vector_id_empty(self, tmp_path):
        store, mock_qdrant = _unit_store(tmp_path)
        mem = _memory()

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

        assert mem.vector_id == ""
        store.delete(mem.id)
        mock_qdrant.delete.assert_not_called()

    def test_delete_calls_qdrant_when_vector_id_set(self, tmp_path):
        store, mock_qdrant = _unit_store(tmp_path)
        mem = _memory()
        fake_vec = [0.1] * 768

        with patch("hcma.memory.ltm_store.ollama.embeddings", return_value=SimpleNamespace(embedding=fake_vec)):
            store.write(mem)

        assert mem.vector_id != ""
        store.delete(mem.id)
        mock_qdrant.delete.assert_called_once()

    def test_delete_returns_false_for_unknown_id(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.delete("nonexistent") is False


class TestGetAllUnit:
    def test_returns_all_entries_ordered_by_created_at_desc(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        now = time.time()
        m1 = _memory(content="oldest fact")
        m2 = _memory(content="newest fact")
        m1.created_at = now - 100
        m2.created_at = now

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(m1)
            store.write(m2)

        results = store.get_all()
        assert results[0].id == m2.id
        assert results[1].id == m1.id

    def test_returns_empty_list_for_empty_store(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.get_all() == []


class TestSearchSemanticFallbackUnit:
    def test_falls_back_to_search_by_content_on_embedding_failure(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        mem = _memory(content="Python generators explained")

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)
            results = store.search_semantic("generator")

        assert any(r.id == mem.id for r in results)


# ---------------------------------------------------------------------------
# Contradiction persistence
# ---------------------------------------------------------------------------

class TestContradictionPersistence:
    def _flag(
        self,
        id_a: str = "aaaa-0000",
        id_b: str = "bbbb-0000",
        reason: str = "They conflict.",
        severity: str = "medium",
    ) -> ContradictionFlag:
        return ContradictionFlag(
            memory_id_a=id_a,
            memory_id_b=id_b,
            reason=reason,
            severity=severity,
        )

    def test_save_contradiction_returns_true(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.save_contradiction(self._flag()) is True

    def test_save_contradiction_persists_correctly(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        flag = self._flag(id_a="mem-aaa", id_b="mem-bbb", reason="Conflict!", severity="high")
        store.save_contradiction(flag)

        rows = store._conn.execute("SELECT * FROM contradictions").fetchall()
        assert len(rows) == 1
        assert rows[0]["memory_id_a"] == "mem-aaa"
        assert rows[0]["memory_id_b"] == "mem-bbb"
        assert rows[0]["reason"] == "Conflict!"
        assert rows[0]["severity"] == "high"
        assert rows[0]["resolved"] == 0
        assert rows[0]["detected_at"] > 0

    def test_save_contradiction_sets_detected_at(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        before = time.time()
        store.save_contradiction(self._flag())
        after = time.time()

        row = store._conn.execute("SELECT detected_at FROM contradictions").fetchone()
        assert before <= row["detected_at"] <= after

    def test_get_unresolved_returns_only_unresolved(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        f1 = self._flag("a1", "b1", "conflict one")
        f2 = self._flag("a2", "b2", "conflict two")
        store.save_contradiction(f1)
        store.save_contradiction(f2)

        # Manually resolve f2
        store._conn.execute(
            "UPDATE contradictions SET resolved = 1 WHERE memory_id_a = ?", ("a2",)
        )
        store._conn.commit()

        unresolved = store.get_unresolved_contradictions()
        assert len(unresolved) == 1
        assert unresolved[0].memory_id_a == "a1"

    def test_get_unresolved_returns_correct_fields(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        flag = self._flag(id_a="x1", id_b="x2", reason="Mismatch", severity="high")
        store.save_contradiction(flag)

        results = store.get_unresolved_contradictions()
        assert len(results) == 1
        r = results[0]
        assert r.memory_id_a == "x1"
        assert r.memory_id_b == "x2"
        assert r.reason == "Mismatch"
        assert r.severity == "high"

    def test_get_unresolved_ordered_by_detected_at_desc(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        store.save_contradiction(self._flag("aa", "bb", "first"))
        time.sleep(0.01)
        store.save_contradiction(self._flag("cc", "dd", "second"))

        results = store.get_unresolved_contradictions()
        assert results[0].memory_id_a == "cc"  # newest first
        assert results[1].memory_id_a == "aa"

    def test_get_unresolved_empty_when_none(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.get_unresolved_contradictions() == []

    def test_resolve_contradiction_marks_row(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        flag = self._flag("p1", "p2")
        store.save_contradiction(flag)

        assert store.resolve_contradiction("p1", "p2") is True
        assert store.get_unresolved_contradictions() == []

    def test_resolve_contradiction_returns_false_when_not_found(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        assert store.resolve_contradiction("no-such-a", "no-such-b") is False

    def test_resolve_contradiction_returns_false_when_already_resolved(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        flag = self._flag("q1", "q2")
        store.save_contradiction(flag)
        store.resolve_contradiction("q1", "q2")
        # second resolve call — already resolved
        assert store.resolve_contradiction("q1", "q2") is False

    def test_multiple_contradictions_resolve_only_correct_one(self, tmp_path):
        store, _ = _unit_store(tmp_path)
        store.save_contradiction(self._flag("m1", "m2"))
        store.save_contradiction(self._flag("m3", "m4"))

        store.resolve_contradiction("m1", "m2")

        unresolved = store.get_unresolved_contradictions()
        assert len(unresolved) == 1
        assert unresolved[0].memory_id_a == "m3"


# ---------------------------------------------------------------------------
# Integration tests (require Ollama + nomic-embed-text)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationLTMStore:
    """Requires: ollama serve && ollama pull nomic-embed-text"""

    def _store(self, tmp_path) -> LTMStore:
        return LTMStore(
            db_path=str(tmp_path / "ltm.db"),
            qdrant_storage_path=str(tmp_path / "qdrant"),
            collection_name="test_ltm_integration",
        )

    def test_write_and_search_semantic_round_trip(self, tmp_path):
        store = self._store(tmp_path)
        mem = _memory(content="Python decorators are higher-order functions")
        result = store.write(mem)

        assert result is True
        assert mem.vector_id != "", "Expected a vector_id to be set"

        hits = store.search_semantic("higher-order functions in Python", top_k=3)
        assert any(h.id == mem.id for h in hits), "Written memory not found via semantic search"

    def test_search_semantic_returns_most_relevant_first(self, tmp_path):
        store = self._store(tmp_path)

        m_relevant = _memory(content="Python list comprehensions create lists concisely")
        m_irrelevant = _memory(content="SQL JOIN combines rows from two tables")

        store.write(m_relevant)
        store.write(m_irrelevant)

        hits = store.search_semantic("Python list comprehension syntax", top_k=2)
        assert len(hits) >= 1
        assert hits[0].id == m_relevant.id, (
            f"Expected most relevant first, got: {[h.content for h in hits]}"
        )

    def test_delete_removes_from_both_backends(self, tmp_path):
        store = self._store(tmp_path)
        mem = _memory(content="Generators use yield to produce values lazily")
        store.write(mem)

        vector_id = mem.vector_id
        assert vector_id != ""

        assert store.delete(mem.id) is True
        assert store.read(mem.id) is None

        # Confirm vector gone from Qdrant by checking semantic search no longer returns it
        hits = store.search_semantic("yield generators lazy", top_k=5)
        assert all(h.id != mem.id for h in hits)
