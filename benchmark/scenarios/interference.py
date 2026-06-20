"""Interference scenario: signal survives noisy and misleading facts."""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

_NOISE = [
    ("noise_1", "Rust uses ownership and borrowing for memory safety"),
    ("noise_2", "JavaScript is single-threaded with an event loop"),
    ("noise_3", "Docker containers share the host OS kernel"),
    ("noise_4", "SQL joins combine rows from two or more tables"),
    ("noise_5", "Machine learning models require training data"),
]


class InterferenceScenario:
    @property
    def name(self) -> str:
        return "interference"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        system.ingest("signal", "Python lists are zero-indexed", tags=["fact"])
        for session_id, content in _NOISE:
            system.ingest(session_id, content, tags=["fact"])
        system.ingest("mislead", "Some say arrays start at index 1", tags=["fact"])

        _maybe_consolidate(system)

        question = "What index does a Python list start at?"
        response = system.query(question)
        passed = "0" in response or "zero" in response.lower()

        return ScenarioResult(
            scenario_name=self.name,
            system_name=system.name,
            score=1.0 if passed else 0.0,
            max_score=1.0,
            details=[
                f"{'PASS' if passed else 'FAIL'} | Q: {question!r} "
                f"| got: {response[:120]!r}"
            ],
            duration_seconds=time.time() - start,
        )


def _maybe_consolidate(system: BaseSystem) -> None:
    try:
        from benchmark.systems.engram_system import EngramSystem
        if isinstance(system, EngramSystem):
            system._consolidation_agent.run()
    except Exception:
        pass
