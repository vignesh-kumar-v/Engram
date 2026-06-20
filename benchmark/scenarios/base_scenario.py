"""Base dataclass for scenario results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioResult:
    scenario_name: str
    system_name: str
    score: float
    max_score: float
    details: list[str]
    duration_seconds: float
