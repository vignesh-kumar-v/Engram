"""HCMA memory type schemas using Python dataclasses."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List

_VALID_STATUSES = {"raw", "promoted", "discarded"}
_VALID_MEMORY_TYPES = {"fact", "preference", "pattern", "error"}
_VALID_ACTIONS = {"promote", "compress", "discard"}
_VALID_SEVERITIES = {"low", "medium", "high"}


@dataclass
class EpisodicEntry:
    content: str
    source_task: str
    session_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    status: str = "raw"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {self.status!r}. Must be one of {_VALID_STATUSES}."
            )
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(
                f"importance must be in range [0.0, 1.0], got {self.importance}."
            )


@dataclass
class LTMMemory:
    content: str
    source_episode_ids: List[str]
    created_at: float
    last_accessed: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 0.7
    access_count: int = 0
    vector_id: str = ""
    memory_type: str = "fact"

    def __post_init__(self) -> None:
        if self.memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type {self.memory_type!r}. "
                f"Must be one of {_VALID_MEMORY_TYPES}."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in range [0.0, 1.0], got {self.confidence}."
            )


@dataclass
class Decision:
    action: str
    reasoning: str
    existing_memory_id: str = ""

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid action {self.action!r}. Must be one of {_VALID_ACTIONS}."
            )


@dataclass
class ConsolidationResult:
    promoted: int = 0
    compressed: int = 0
    discarded: int = 0
    contradictions_found: int = 0
    total_processed: int = 0
    duration_seconds: float = 0.0


@dataclass
class ContradictionFlag:
    memory_id_a: str
    memory_id_b: str
    reason: str
    severity: str = "low"

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity {self.severity!r}. Must be one of {_VALID_SEVERITIES}."
            )
