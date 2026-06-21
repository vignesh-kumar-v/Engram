"""Engram system: full HCMA pipeline used as a benchmark contestant."""

from __future__ import annotations

import logging
import tempfile
import time

import ollama

from benchmark.config import OLLAMA_MODEL, TOP_K_RETRIEVAL
from benchmark.systems.base_system import BaseSystem
from hcma.agents.consolidation_agent import ConsolidationAgent
from hcma.agents.task_agent import TaskAgent
from hcma.config import settings
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.memory.ltm_store import LTMStore
from hcma.schemas.memory_types import EpisodicEntry

logger = logging.getLogger(__name__)

_BENCH_SESSION = "benchmark_session"


class EngramSystem(BaseSystem):
    def __init__(self) -> None:
        self._llm = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self._init_components()

    @property
    def name(self) -> str:
        return "engram"

    def _init_components(self) -> None:
        self._buf = EpisodicBuffer(":memory:", capacity=settings.EPISODIC_BUFFER_CAPACITY)
        self._qdrant_tmp = tempfile.mkdtemp(prefix="engram_bench_")
        self._ltm = LTMStore(
            db_path=":memory:",
            qdrant_storage_path=self._qdrant_tmp,
            collection_name="engram_bench",
        )
        self._consolidation_agent = ConsolidationAgent(self._buf, self._ltm)

    def ingest(self, session_id: str, content: str, tags: list[str]) -> None:
        entry = EpisodicEntry(
            content=content,
            source_task=session_id,
            session_id=session_id,
            importance=0.7,
            tags=tags,
        )
        self._buf.write(entry)

    def query(self, question: str) -> str:
        # Consolidate any pending episodic entries before querying
        try:
            self._consolidation_agent.run()
        except Exception:
            logger.exception("EngramSystem.query: consolidation failed")

        try:
            memories = self._ltm.search_semantic(question, top_k=TOP_K_RETRIEVAL)
            context = "\n".join(m.content for m in memories)
            prompt = (
                f"Answer based on this context:\n{context}\n"
                f"Question: {question}"
            ) if context else question

            response = self._llm.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content
        except Exception:
            logger.exception("EngramSystem.query failed")
            return ""

    def after_session(self) -> None:
        try:
            self._consolidation_agent.run()
        except Exception:
            logger.exception("EngramSystem.after_session: consolidation failed")

    def get_session_state(self) -> dict:
        raw_count = len(self._buf.read_all_raw())
        return {
            "buffer_raw": raw_count,
            "ltm_memories": len(self._ltm.get_all()),
            "contradictions": len(self._ltm.get_unresolved_contradictions()),
        }

    def reset(self) -> None:
        self._init_components()
