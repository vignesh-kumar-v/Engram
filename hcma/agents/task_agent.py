"""LangGraph task agent that records episodic memories during coding tasks."""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from hcma.config import settings
from hcma.memory.episodic_buffer import EpisodicBuffer
from hcma.schemas.memory_types import EpisodicEntry

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a coding assistant. Help the user with programming questions, "
    "debugging, and code review. Be concise and precise."
)

_DEBUG_KEYWORDS = {"error", "bug", "fix", "crash", "exception", "traceback", "fail"}
_LLM_FALLBACK = "I encountered an error. Please try again."


class TaskAgent:
    def __init__(self, buffer: EpisodicBuffer, session_id: str) -> None:
        self.buffer = buffer
        self.session_id = session_id
        self._client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self.conversation_history: List[Dict[str, str]] = []

    def run(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self._get_llm_response()
            
        self.conversation_history.append({"role": "assistant", "content": response})
        if response == _LLM_FALLBACK:
            logger.warning("Skipping episodic write due to LLM failure")
        else:
            self._extract_and_store_observations(user_input, response)
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _do_chat(self, messages):
        return self._client.chat(model=settings.OLLAMA_MODEL, messages=messages)

    def _get_llm_response(self) -> str:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + self.conversation_history
        try:
            result = self._do_chat(messages)
            if result.message.content is None:
                return _LLM_FALLBACK
            return result.message.content
        except Exception:
            logger.exception("_get_llm_response failed")
            return "I encountered an error. Please try again."

    def _extract_and_store_observations(self, user_input: str, response: str) -> None:
        now = time.time()

        entries = [
            EpisodicEntry(
                content=f"User asked about: {user_input[:200]}",
                source_task=user_input[:100],
                session_id=self.session_id,
                timestamp=now,
                importance=0.5,
                tags=["user_query"],
            ),
            EpisodicEntry(
                content=f"Assistant response summary: {response[:300]}",
                source_task=user_input[:100],
                session_id=self.session_id,
                timestamp=now,
                importance=0.5,
                tags=["assistant_response"],
            ),
        ]

        input_lower = user_input.lower()
        if any(kw in input_lower for kw in _DEBUG_KEYWORDS):
            entries.append(
                EpisodicEntry(
                    content=f"Debugging interaction: {user_input[:200]}",
                    source_task=user_input[:100],
                    session_id=self.session_id,
                    timestamp=now,
                    importance=0.8,
                    tags=["debug", "error_pattern"],
                )
            )

        for entry in entries:
            self.buffer.write(entry)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return list(self.conversation_history)

    def clear_history(self) -> None:
        self.conversation_history = []
