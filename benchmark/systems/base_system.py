"""Abstract base class for all benchmark memory systems."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSystem(ABC):
    @abstractmethod
    def ingest(self, session_id: str, content: str, tags: list[str]) -> None:
        """Ingest a piece of information into the system's memory."""

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the system; return best answer as a string."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all memory state for a fresh benchmark run."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the system name string."""

    def after_session(self) -> None:
        """Called after a session's facts are ingested. Override to consolidate."""

    def get_session_state(self) -> dict:
        """Return current memory state for trace printing."""
        return {}
