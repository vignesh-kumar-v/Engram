"""No-memory baseline: every query hits the LLM cold with zero context."""

from __future__ import annotations

import logging

import ollama

from benchmark.config import OLLAMA_MODEL
from benchmark.systems.base_system import BaseSystem

logger = logging.getLogger(__name__)


class NoMemorySystem(BaseSystem):
    def __init__(self) -> None:
        self._client = ollama.Client()

    @property
    def name(self) -> str:
        return "no_memory"

    def ingest(self, session_id: str, content: str, tags: list[str]) -> None:
        pass

    def query(self, question: str) -> str:
        try:
            response = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": question}],
            )
            return response.message.content
        except Exception:
            logger.exception("NoMemorySystem.query failed")
            return ""

    def reset(self) -> None:
        pass
