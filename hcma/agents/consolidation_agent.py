"""ConsolidationAgent: promotes/compresses/discards episodic entries into LTM."""

from __future__ import annotations

import logging
import time
from typing import List

import re
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from hcma.config import settings
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore
from hcma.schemas.memory_types import (
    ConsolidationResult,
    ContradictionFlag,
    Decision,
    EpisodicEntry,
    LTMMemory,
)

logger = logging.getLogger(__name__)

_DECISION_SYSTEM = (
    "You are a memory consolidation agent for a coding assistant.\n"
    "You will be given a single memory entry and the current long-term "
    "memory store summary. Decide what to do with this entry.\n\n"
    "Respond in this EXACT format and nothing else:\n"
    "ACTION: <promote|compress|discard>\n"
    "EXISTING_ID: <memory_id_if_compress_else_empty>\n"
    "REASONING: <one sentence max>\n\n"
    "Rules:\n"
    "- promote: entry contains new, useful information not in LTM\n"
    "- compress: entry is similar to an existing LTM memory (provide its id)\n"
    "- discard: entry is noise, trivial, or redundant with no value\n\n"
    "Current LTM summary (most recent 5 memories):\n"
    "{ltm_summary}"
)

_CONTRADICTION_SYSTEM = (
    "You are checking a coding assistant's memory for contradictions. "
    "Given a list of memory entries of the same type, identify any pair "
    "that directly contradicts each other.\n\n"
    "Respond in this EXACT format for each contradiction found, "
    "one per line. If none found respond with NONE:\n"
    "CONTRADICTION: <id_a[:8]> <id_b[:8]> | SEVERITY: <low|medium|high> "
    "| REASON: <one sentence>"
)


class ConsolidationAgent:
    def __init__(self, buffer: EpisodicBuffer, ltm: LTMStore) -> None:
        self.buffer = buffer
        self.ltm = ltm
        self._client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> ConsolidationResult:
        start = time.time()
        entries = self.buffer.read_all_raw()

        if not entries:
            logger.info("ConsolidationAgent.run: no raw entries, skipping")
            return ConsolidationResult()

        result = ConsolidationResult(total_processed=len(entries))

        for entry in entries:
            decision = self._decide(entry)
                
            logger.debug(
                "Entry %s → action=%s reason=%s",
                entry.id[:8], decision.action, decision.reasoning,
            )

            if decision.action == "promote":
                if self._promote(entry):
                    result.promoted += 1
            elif decision.action == "compress":
                if self._compress(entry, decision.existing_memory_id):
                    result.compressed += 1
            elif decision.action == "discard":
                self.buffer.update_status(entry.id, "discarded")
                result.discarded += 1

        flags = self._detect_contradictions()
        result.contradictions_found = len(flags)
        result.duration_seconds = time.time() - start

        logger.info(
            "Consolidation complete: promoted=%d compressed=%d discarded=%d "
            "contradictions=%d duration=%.2fs",
            result.promoted, result.compressed, result.discarded,
            result.contradictions_found, result.duration_seconds,
        )
        return result

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _do_chat(self, system_prompt, user_message):
        return self._client.chat(
            model=settings.CONSOLIDATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

    def _decide(self, entry: EpisodicEntry) -> Decision:
        ltm_memories = self.ltm.get_all()[:5]
        if ltm_memories:
            ltm_summary = "\n".join(
                f"- [{m.id[:8]}] {m.content[:100]}" for m in ltm_memories
            )
        else:
            ltm_summary = "LTM is currently empty."

        system_prompt = _DECISION_SYSTEM.format(ltm_summary=ltm_summary)
        user_message = (
            f"Entry to evaluate:\n"
            f"Content: {entry.content}\n"
            f"Tags: {entry.tags}\n"
            f"Importance: {entry.importance}\n\n"
            "Decide: promote, compress, or discard?"
        )

        try:
            response = self._do_chat(system_prompt, user_message)
            if response.message.content is None:
                return Decision(action="promote", reasoning="fallback: empty LLM response")
            return self._parse_decision(response.message.content)
        except Exception:
            logger.exception("_decide: LLM call failed for entry %s", entry.id[:8])
            return Decision(action="promote", reasoning="fallback: LLM call failed")

    def _parse_decision(self, text: str) -> Decision:
        action = "promote"
        existing_id = ""
        reasoning = "fallback: parse failed"

        try:
            for line in text.strip().splitlines():
                line = line.strip()
                if line.startswith("ACTION:"):
                    raw = line.split("ACTION:", 1)[1].strip().lower()
                    if raw in {"promote", "compress", "discard"}:
                        action = raw
                elif line.startswith("EXISTING_ID:"):
                    raw_id = line.split("EXISTING_ID:", 1)[1].strip()
                    match = re.search(r"([a-fA-F0-9\-]{8,})", raw_id)
                    if match:
                        existing_id = match.group(1)
                    else:
                        existing_id = raw_id.strip("[]`\"'")
                elif line.startswith("REASONING:"):
                    reasoning = line.split("REASONING:", 1)[1].strip()
        except Exception:
            logger.exception("_parse_decision: failed to parse LLM output")
            return Decision(action="promote", reasoning="fallback: parse failed")

        return Decision(action=action, reasoning=reasoning, existing_memory_id=existing_id)

    # ------------------------------------------------------------------
    # Promote / compress
    # ------------------------------------------------------------------

    def _promote(self, entry: EpisodicEntry) -> bool:
        try:
            now = time.time()
            memory = LTMMemory(
                content=entry.content,
                confidence=entry.importance,
                source_episode_ids=[entry.id],
                created_at=now,
                last_accessed=now,
                memory_type=self._infer_memory_type(entry.tags, entry.content),
            )
            if not self.ltm.write(memory):
                return False
            self.buffer.update_status(entry.id, "promoted")
            return True
        except Exception:
            logger.exception("_promote failed for entry %s", entry.id[:8])
            return False

    def _compress(self, entry: EpisodicEntry, existing_memory_id: str) -> bool:
        try:
            existing = self.ltm.read(existing_memory_id)
            if existing is None and existing_memory_id:
                # LLM may have returned only the 8-char prefix shown in the
                # prompt summary — try a prefix lookup before falling back.
                existing = self.ltm.read_by_prefix(existing_memory_id)
            if existing is None:
                logger.warning(
                    "_compress: existing memory %s not found, falling back to promote",
                    existing_memory_id[:8] if existing_memory_id else "<empty>",
                )
                return self._promote(entry)

            existing.content = self._merge_content(existing.content, entry.content)
            existing.confidence = min(1.0, existing.confidence + 0.05)
            if entry.id not in existing.source_episode_ids:
                existing.source_episode_ids.append(entry.id)

            # Update SQLite in-place (no new Qdrant point)
            self.ltm._conn.execute(
                """
                UPDATE ltm_memories
                SET content = ?, confidence = ?, source_episode_ids = ?
                WHERE id = ?
                """,
                (
                    existing.content,
                    existing.confidence,
                    __import__("json").dumps(existing.source_episode_ids),
                    existing.id,
                ),
            )
            self.ltm._conn.commit()

            self.buffer.update_status(entry.id, "promoted")
            return True
        except Exception:
            logger.exception("_compress failed for entry %s", entry.id[:8])
            return False

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def _detect_contradictions(self) -> List[ContradictionFlag]:
        all_memories = self.ltm.get_all()
        if len(all_memories) < 2:
            return []

        # Group by memory_type
        groups: dict[str, list[LTMMemory]] = {}
        for mem in all_memories:
            groups.setdefault(mem.memory_type, []).append(mem)

        flags: List[ContradictionFlag] = []
        for mtype, members in groups.items():
            if len(members) < 2:
                continue
            flags.extend(self._check_group_for_contradictions(members))

        for flag in flags:
            self.ltm.save_contradiction(flag)

        return flags

    def _check_group_for_contradictions(
        self, memories: List[LTMMemory]
    ) -> List[ContradictionFlag]:
        formatted = "\n".join(
            f"- [{m.id[:8]}] {m.content[:150]}" for m in memories
        )
        user_message = f"Check these memories for contradictions:\n{formatted}"

        try:
            response = self._do_chat(_CONTRADICTION_SYSTEM, user_message)
            if response.message.content is None:
                return []
            return self._parse_contradictions(response.message.content, memories)
        except Exception:
            logger.exception("_check_group_for_contradictions: LLM call failed")
            return []

    def _parse_contradictions(
        self, text: str, memories: List[LTMMemory]
    ) -> List[ContradictionFlag]:
        flags: List[ContradictionFlag] = []
        id_prefix_map = {m.id[:8]: m.id for m in memories}

        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            if not line.startswith("CONTRADICTION:"):
                continue
            try:
                # "CONTRADICTION: <id_a> <id_b> | SEVERITY: <s> | REASON: <r>"
                rest = line.split("CONTRADICTION:", 1)[1].strip()
                parts = [p.strip() for p in rest.split("|")]
                ids_part = parts[0].split()
                if len(ids_part) < 2:
                    continue
                id_a_prefix, id_b_prefix = ids_part[0], ids_part[1]

                severity = "low"
                reason = ""
                for part in parts[1:]:
                    if part.startswith("SEVERITY:"):
                        raw_sev = part.split("SEVERITY:", 1)[1].strip().lower()
                        if raw_sev in {"low", "medium", "high"}:
                            severity = raw_sev
                    elif part.startswith("REASON:"):
                        reason = part.split("REASON:", 1)[1].strip()

                full_id_a = id_prefix_map.get(id_a_prefix, id_a_prefix)
                full_id_b = id_prefix_map.get(id_b_prefix, id_b_prefix)
                flags.append(
                    ContradictionFlag(
                        memory_id_a=full_id_a,
                        memory_id_b=full_id_b,
                        reason=reason,
                        severity=severity,
                    )
                )
            except Exception:
                logger.warning("_parse_contradictions: could not parse line: %r", line)

        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_memory_type(self, tags: List[str], content: str = "") -> str:
        tag_set = set(tags)
        if "error_pattern" in tag_set or "debug" in tag_set:
            return "error"
        if "preference" in tag_set or "prefer" in content.lower():
            return "preference"
        if "pattern" in tag_set:
            return "pattern"
        return "fact"

    def _merge_content(self, existing: str, new: str) -> str:
        return f"{existing} | Additional context: {new[:200]}"
