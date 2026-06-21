"""
Production-grade regression and edge-case tests for the compress-ID-prefix fix.

Three bugs were present before the fix:
  1. _parse_decision stored EXISTING_ID verbatim, including LLM-echoed brackets
     e.g. "[82c97b40]" instead of "82c97b40"
  2. _compress called ltm.read() with that bracketed / 8-char prefix value,
     which is an exact UUID match — always fails
  3. Every compress decision silently fell back to promote, so the LTM
     accumulated unmerged duplicate entries

These tests verify:
  A. read_by_prefix — correct resolution in all cases
  B. _parse_decision — bracket stripping across all input shapes
  C. _compress — all resolution paths: full UUID / 8-char prefix / nonexistent
  D. Warning suppression — WARNING fires ONLY on genuine misses, not on prefix hits
  E. LTM deduplication — compress actually merges, not duplicates
  F. Pipeline integrity — run() with compress decisions reduces LTM count
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from hcma.agents.consolidation_agent import ConsolidationAgent
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore
from hcma.schemas.memory_types import (
    ConsolidationResult,
    Decision,
    EpisodicEntry,
    LTMMemory,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _buf(capacity: int = 50) -> EpisodicBuffer:
    return EpisodicBuffer(":memory:", capacity=capacity)


def _entry(content: str = "Python is great", tags: list[str] | None = None) -> EpisodicEntry:
    return EpisodicEntry(
        content=content,
        source_task="t",
        session_id="s",
        importance=0.6,
        tags=tags or ["fact"],
    )


def _ltm_memory(content: str = "Existing LTM fact") -> LTMMemory:
    now = time.time()
    return LTMMemory(
        content=content,
        source_episode_ids=["ep_x"],
        created_at=now,
        last_accessed=now,
    )


def _real_ltm_store(tmp_path) -> LTMStore:
    """LTMStore with real SQLite but mocked Qdrant."""
    with patch("hcma.memory.ltm_store.QdrantClient") as mock_qc:
        mock_qc.return_value.collection_exists.return_value = True
        store = LTMStore(
            db_path=":memory:",
            qdrant_storage_path=str(tmp_path),
            collection_name="test",
        )
    return store


def _mock_ltm() -> MagicMock:
    ltm = MagicMock()
    ltm.get_all.return_value = []
    ltm.read.return_value = None
    ltm.write.return_value = True
    return ltm


def _agent(buf: EpisodicBuffer | None = None, ltm=None) -> ConsolidationAgent:
    return ConsolidationAgent(buf or _buf(), ltm or _mock_ltm())


def _llm_text(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def _promote_resp() -> MagicMock:
    return _llm_text("ACTION: promote\nEXISTING_ID: \nREASONING: New info.")


def _none_resp() -> MagicMock:
    return _llm_text("NONE")


# ===========================================================================
# A. read_by_prefix — comprehensive edge-case coverage
# ===========================================================================

class TestReadByPrefixComprehensive:
    """Every meaningful input shape for LTMStore.read_by_prefix."""

    def _store(self, tmp_path) -> LTMStore:
        return _real_ltm_store(tmp_path)

    def _write(self, store, mem):
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            store.write(mem)

    # --- boundary / empty inputs ---

    def test_empty_prefix_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.read_by_prefix("") is None

    def test_single_char_prefix_finds_match(self, tmp_path):
        store = self._store(tmp_path)
        mem = _ltm_memory("single char test")
        self._write(store, mem)
        # Any UUID starts with a hex char; using the first char of this UUID
        result = store.read_by_prefix(mem.id[0])
        assert result is not None  # must find at least this memory

    def test_8char_hex_prefix_finds_correct_memory(self, tmp_path):
        store = self._store(tmp_path)
        mem = _ltm_memory("target memory")
        self._write(store, mem)
        result = store.read_by_prefix(mem.id[:8])
        assert result is not None
        assert result.id == mem.id
        assert result.content == "target memory"

    def test_full_uuid_prefix_finds_memory(self, tmp_path):
        """Passing the full UUID as prefix must still work (LIKE 'full-uuid%')."""
        store = self._store(tmp_path)
        mem = _ltm_memory("full uuid test")
        self._write(store, mem)
        result = store.read_by_prefix(mem.id)
        assert result is not None and result.id == mem.id

    def test_non_matching_prefix_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        mem = _ltm_memory("some fact")
        self._write(store, mem)
        # 8 zeros is astronomically unlikely to match any real UUID
        result = store.read_by_prefix("00000000")
        assert result is None

    def test_empty_store_returns_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.read_by_prefix("abcd1234") is None

    # --- correctness among multiple memories ---

    def test_finds_correct_memory_among_many(self, tmp_path):
        store = self._store(tmp_path)
        mems = [_ltm_memory(f"fact {i}") for i in range(5)]
        for m in mems:
            self._write(store, m)

        target = mems[2]
        result = store.read_by_prefix(target.id[:8])
        assert result is not None
        assert result.id == target.id

    def test_does_not_return_wrong_memory(self, tmp_path):
        store = self._store(tmp_path)
        m1 = _ltm_memory("memory A")
        m2 = _ltm_memory("memory B")
        self._write(store, m1)
        self._write(store, m2)

        result = store.read_by_prefix(m1.id[:8])
        assert result is not None and result.id == m1.id
        assert result.id != m2.id

    # --- returned object completeness ---

    def test_returned_object_has_all_fields(self, tmp_path):
        store = self._store(tmp_path)
        mem = _ltm_memory("Generators use yield")
        mem.memory_type = "pattern"
        mem.confidence = 0.9
        mem.access_count = 3
        self._write(store, mem)

        result = store.read_by_prefix(mem.id[:8])
        assert result.content == "Generators use yield"
        assert result.memory_type == "pattern"
        assert result.confidence == pytest.approx(0.9)
        assert result.source_episode_ids == ["ep_x"]


# ===========================================================================
# B. _parse_decision — bracket stripping across all input shapes
# ===========================================================================

class TestParseDecisionBracketStripping:
    """EXISTING_ID can arrive from the LLM in several forms; all must be handled."""

    def _parse(self, text: str) -> Decision:
        agent = _agent()
        with patch.object(agent._client, "chat", return_value=_llm_text(text)):
            return agent._decide(_entry())

    # --- the bug scenario: bracketed 8-char prefix ---

    def test_both_brackets_stripped(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: [abcd1234]\nREASONING: dup.")
        assert d.existing_memory_id == "abcd1234"
        assert "[" not in d.existing_memory_id
        assert "]" not in d.existing_memory_id

    def test_only_opening_bracket_stripped(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: [abcd1234\nREASONING: dup.")
        assert "[" not in d.existing_memory_id
        assert d.existing_memory_id == "abcd1234"

    def test_only_closing_bracket_stripped(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: abcd1234]\nREASONING: dup.")
        assert "]" not in d.existing_memory_id
        assert d.existing_memory_id == "abcd1234"

    # --- regression: inputs that must pass through unchanged ---

    def test_no_brackets_unchanged(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: abcd1234\nREASONING: dup.")
        assert d.existing_memory_id == "abcd1234"

    def test_full_uuid_no_brackets_unchanged(self):
        full = "abcd1234-ffff-ffff-ffff-000000000000"
        d = self._parse(f"ACTION: compress\nEXISTING_ID: {full}\nREASONING: dup.")
        assert d.existing_memory_id == full

    # --- edge: empty or degenerate EXISTING_ID ---

    def test_empty_existing_id(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: \nREASONING: dup.")
        assert d.existing_memory_id == ""

    def test_empty_brackets_existing_id(self):
        d = self._parse("ACTION: compress\nEXISTING_ID: []\nREASONING: dup.")
        assert d.existing_memory_id == ""

    # --- whitespace handling ---

    def test_whitespace_around_bracketed_id_stripped(self):
        d = self._parse("ACTION: compress\nEXISTING_ID:  [abcd1234] \nREASONING: dup.")
        assert d.existing_memory_id == "abcd1234"

    # --- action field unaffected by bracket fix ---

    def test_action_still_parsed_correctly_with_bracketed_id(self):
        d = self._parse("ACTION: discard\nEXISTING_ID: [ignored]\nREASONING: noise.")
        assert d.action == "discard"

    def test_promote_action_with_empty_existing_id(self):
        d = self._parse("ACTION: promote\nEXISTING_ID: \nREASONING: new info.")
        assert d.action == "promote"
        assert d.existing_memory_id == ""


# ===========================================================================
# C. _compress — all resolution paths
# ===========================================================================

class TestCompressResolutionPaths:
    """Every code path through the fixed _compress method."""

    def _setup(self, tmp_path):
        ltm = _real_ltm_store(tmp_path)
        existing = _ltm_memory("Original fact")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)
        buf = _buf()
        entry = _entry("Additional info")
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)
        return agent, buf, ltm, existing, entry

    # --- Path A: full UUID — direct read succeeds, no prefix lookup needed ---

    def test_path_a_full_uuid_direct_read_succeeds(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(entry, existing.id)  # full UUID
        assert result is True
        updated = ltm.read(existing.id)
        assert "Additional info" in updated.content

    def test_path_a_does_not_call_read_by_prefix_when_direct_read_succeeds(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with patch.object(ltm, "read_by_prefix", wraps=ltm.read_by_prefix) as spy:
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, existing.id)
        spy.assert_not_called()

    # --- Path B: 8-char prefix — direct read fails, prefix resolves ---

    def test_path_b_8char_prefix_resolves_and_merges(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        prefix = existing.id[:8]
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(entry, prefix)
        assert result is True
        updated = ltm.read(existing.id)
        assert "Additional info" in updated.content

    def test_path_b_returns_true_on_successful_prefix_resolution(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        prefix = existing.id[:8]
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            assert agent._compress(entry, prefix) is True

    def test_path_b_entry_status_becomes_promoted(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        prefix = existing.id[:8]
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, prefix)
        assert buf.read(entry.id).status == "promoted"

    # --- Path C: nonexistent ID — both reads fail, fallback to promote ---

    def test_path_c_nonexistent_falls_back_to_promote(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(entry, "00000000deadbeef")
        assert result is True  # promote fallback still succeeds
        assert buf.read(entry.id).status == "promoted"

    def test_path_c_fallback_creates_new_ltm_entry(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        before = len(ltm.get_all())
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, "00000000deadbeef")
        after = len(ltm.get_all())
        assert after == before + 1  # new entry added (promote fallback)

    # --- Path D: empty ID ---

    def test_path_d_empty_id_falls_back_to_promote(self, tmp_path):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(entry, "")
        assert result is True
        assert buf.read(entry.id).status == "promoted"

    # --- Path E: end-to-end with bracket stripping (the original bug scenario) ---

    def test_path_e_bracketed_prefix_full_pipeline(self, tmp_path):
        """Simulate what happens when LLM returns '[82c97b40]':
        _parse_decision strips brackets → '82c97b40' → _compress resolves by prefix."""
        agent, buf, ltm, existing, entry = self._setup(tmp_path)

        bracketed_response = _llm_text(
            f"ACTION: compress\nEXISTING_ID: [{existing.id[:8]}]\nREASONING: Similar."
        )
        with patch.object(agent._client, "chat", return_value=bracketed_response):
            decision = agent._decide(entry)

        # Bracket must be stripped
        assert "[" not in decision.existing_memory_id
        # Prefix must still resolve
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            result = agent._compress(entry, decision.existing_memory_id)

        assert result is True
        updated = ltm.read(existing.id)
        assert "Additional info" in updated.content


# ===========================================================================
# D. Warning suppression — WARNING fires only on genuine misses
# ===========================================================================

COMPRESS_LOGGER = "hcma.agents.consolidation_agent"
WARNING_TEXT = "not found, falling back to promote"


class TestWarningSuppression:
    """The 'not found, falling back to promote' WARNING must not fire when
    the prefix resolves. It must still fire on genuine misses."""

    def _setup(self, tmp_path):
        ltm = _real_ltm_store(tmp_path)
        existing = _ltm_memory("Some fact")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)
        buf = _buf()
        entry = _entry()
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)
        return agent, buf, ltm, existing, entry

    def test_no_warning_when_full_uuid_resolves(self, tmp_path, caplog):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with caplog.at_level(logging.WARNING, logger=COMPRESS_LOGGER):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, existing.id)
        assert WARNING_TEXT not in caplog.text

    def test_no_warning_when_8char_prefix_resolves(self, tmp_path, caplog):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        prefix = existing.id[:8]
        with caplog.at_level(logging.WARNING, logger=COMPRESS_LOGGER):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, prefix)
        assert WARNING_TEXT not in caplog.text

    def test_no_warning_for_bracketed_id_after_strip(self, tmp_path, caplog):
        """End-to-end: brackets stripped before compress → no warning."""
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        bracketed = f"[{existing.id[:8]}]"
        # Simulate what _parse_decision does: strip brackets
        clean_prefix = bracketed.strip("[]")
        with caplog.at_level(logging.WARNING, logger=COMPRESS_LOGGER):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, clean_prefix)
        assert WARNING_TEXT not in caplog.text

    def test_warning_still_fires_on_genuine_miss(self, tmp_path, caplog):
        """The warning must not be silenced — it fires when ID genuinely doesn't exist."""
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with caplog.at_level(logging.WARNING, logger=COMPRESS_LOGGER):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, "00000000deadbeef")
        assert WARNING_TEXT in caplog.text

    def test_warning_fires_for_empty_id(self, tmp_path, caplog):
        agent, buf, ltm, existing, entry = self._setup(tmp_path)
        with caplog.at_level(logging.WARNING, logger=COMPRESS_LOGGER):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                agent._compress(entry, "")
        assert WARNING_TEXT in caplog.text


# ===========================================================================
# E. LTM deduplication — compress merges, not duplicates
# ===========================================================================

class TestLTMDeduplication:
    """Before the fix, every compress fell back to promote, adding a new LTM
    entry instead of merging. These tests verify the correct post-fix behaviour."""

    def _store(self, tmp_path) -> LTMStore:
        return _real_ltm_store(tmp_path)

    def test_compress_with_prefix_does_not_create_duplicate_entry(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Original content")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        assert len(ltm.get_all()) == 1  # pre-condition

        buf = _buf()
        entry = _entry("New content to merge")
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)

        prefix = existing.id[:8]
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, prefix)

        all_mems = ltm.get_all()
        assert len(all_mems) == 1, (
            f"Expected 1 merged entry, got {len(all_mems)}. "
            f"The compress fix is not preventing duplicate creation."
        )

    def test_compress_with_prefix_merges_content_into_existing(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Original fact about Python")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entry = _entry("Python supports decorators")
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, existing.id[:8])

        merged = ltm.read(existing.id)
        assert "Python supports decorators" in merged.content
        assert "Original fact about Python" in merged.content

    def test_three_compresses_with_same_prefix_yield_one_ltm_entry(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Base Python knowledge")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        entries = [_entry(f"Python fact {i}") for i in range(3)]
        buf = _buf()
        for e in entries:
            buf.write(e)
        agent = ConsolidationAgent(buf, ltm)

        prefix = existing.id[:8]
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            for e in entries:
                agent._compress(e, prefix)

        all_mems = ltm.get_all()
        assert len(all_mems) == 1, (
            f"Three compresses into same target should leave 1 LTM entry, got {len(all_mems)}"
        )

    def test_promote_fallback_creates_new_entry_not_merge(self, tmp_path):
        """Promote (fallback) must still create a new entry — don't over-correct."""
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Existing knowledge")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entry = _entry("New separate fact")
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)

        # Pass a nonexistent ID → fallback to promote → new entry
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, "deadbeef00000000")

        all_mems = ltm.get_all()
        assert len(all_mems) == 2  # original + new promoted entry

    def test_confidence_boosted_on_prefix_compress(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("some fact")
        existing.confidence = 0.7
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entry = _entry()
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, existing.id[:8])

        updated = ltm.read(existing.id)
        assert updated.confidence == pytest.approx(0.75)

    def test_episode_id_appended_on_prefix_compress(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory()
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entry = _entry()
        buf.write(entry)
        agent = ConsolidationAgent(buf, ltm)

        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            agent._compress(entry, existing.id[:8])

        updated = ltm.read(existing.id)
        assert entry.id in updated.source_episode_ids


# ===========================================================================
# F. Pipeline integrity — run() with compress decisions reduces LTM count
# ===========================================================================

class TestPipelineIntegrity:
    """Full consolidation run() with the fix produces correct LTM structure."""

    def _store(self, tmp_path) -> LTMStore:
        return _real_ltm_store(tmp_path)

    def test_run_with_all_compress_decisions_yields_single_ltm_entry(self, tmp_path):
        """When LLM compresses every entry into an existing one, LTM stays at 1."""
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Base Python knowledge")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entries = [_entry(f"Python fact {i}") for i in range(3)]
        for e in entries:
            buf.write(e)

        agent = ConsolidationAgent(buf, ltm)
        prefix = existing.id[:8]

        compress_resps = [
            _llm_text(
                f"ACTION: compress\nEXISTING_ID: {prefix}\nREASONING: Same topic."
            )
            for _ in entries
        ]
        # Contradiction check returns NONE
        all_resps = compress_resps + [_none_resp()]

        with patch.object(agent._client, "chat", side_effect=all_resps):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                result = agent.run()

        assert result.compressed == 3
        assert result.promoted == 0
        assert len(ltm.get_all()) == 1, (
            f"Expected 1 LTM entry after 3 compresses into same target, "
            f"got {len(ltm.get_all())}"
        )

    def test_run_with_bracketed_compress_id_resolves_correctly(self, tmp_path):
        """The full path: LLM returns [xxxxxxxx] → bracket strip → prefix resolve → merge."""
        ltm = self._store(tmp_path)
        existing = _ltm_memory("FastAPI routing")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        entry = _entry("FastAPI dependency injection")
        buf.write(entry)

        agent = ConsolidationAgent(buf, ltm)

        bracketed_resp = _llm_text(
            f"ACTION: compress\nEXISTING_ID: [{existing.id[:8]}]\nREASONING: Same topic."
        )
        with patch.object(agent._client, "chat", side_effect=[bracketed_resp, _none_resp()]):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                result = agent.run()

        assert result.compressed == 1
        assert result.promoted == 0
        all_mems = ltm.get_all()
        assert len(all_mems) == 1
        assert "FastAPI dependency injection" in all_mems[0].content

    def test_run_mixed_promote_and_compress_correct_counts(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Existing Python fact")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        e_compress = _entry("Similar Python info")
        e_promote = _entry("Completely new topic: Rust ownership")
        buf.write(e_compress)
        buf.write(e_promote)

        agent = ConsolidationAgent(buf, ltm)
        prefix = existing.id[:8]

        resps = [
            _llm_text(f"ACTION: compress\nEXISTING_ID: {prefix}\nREASONING: Similar."),
            _llm_text("ACTION: promote\nEXISTING_ID: \nREASONING: New topic."),
            _none_resp(),  # contradiction check
        ]

        with patch.object(agent._client, "chat", side_effect=resps):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                result = agent.run()

        assert result.compressed == 1
        assert result.promoted == 1
        # LTM: 1 original (merged) + 1 new = 2
        assert len(ltm.get_all()) == 2

    def test_run_idempotent_on_empty_buffer(self, tmp_path):
        """Running consolidation twice with no new entries is a no-op."""
        ltm = self._store(tmp_path)
        existing = _ltm_memory("Some knowledge")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()  # empty buffer
        agent = ConsolidationAgent(buf, ltm)

        result1 = agent.run()
        result2 = agent.run()

        assert result1.total_processed == 0
        assert result2.total_processed == 0
        assert len(ltm.get_all()) == 1  # unchanged

    def test_result_compressed_count_is_accurate(self, tmp_path):
        ltm = self._store(tmp_path)
        existing = _ltm_memory("base")
        with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
            ltm.write(existing)

        buf = _buf()
        for i in range(4):
            buf.write(_entry(f"fact {i}"))

        agent = ConsolidationAgent(buf, ltm)
        prefix = existing.id[:8]

        # 2 compress, 2 promote
        resps = [
            _llm_text(f"ACTION: compress\nEXISTING_ID: {prefix}\nREASONING: Similar."),
            _llm_text("ACTION: promote\nEXISTING_ID: \nREASONING: New."),
            _llm_text(f"ACTION: compress\nEXISTING_ID: {prefix}\nREASONING: Similar."),
            _llm_text("ACTION: promote\nEXISTING_ID: \nREASONING: New."),
            _none_resp(),
        ]

        with patch.object(agent._client, "chat", side_effect=resps):
            with patch("hcma.memory.ltm_store.ollama.embeddings", side_effect=RuntimeError):
                result = agent.run()

        assert result.compressed == 2
        assert result.promoted == 2
        assert result.total_processed == 4
        # 1 original (merged twice) + 2 promoted = 3
        assert len(ltm.get_all()) == 3
