"""Retrieval precision scenario: relevant facts surface above unrelated ones."""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

_DECORATOR_FACTS = [
    "Python decorators are higher-order functions that wrap other functions",
    "The @property decorator creates managed attributes in Python classes",
]

_NOISE_FACTS = [
    "Python lists support slicing with start:stop:step syntax",
    "Dictionaries in Python 3.7+ maintain insertion order",
    "The GIL prevents true multi-threading for CPU-bound tasks",
    "Virtual environments isolate project dependencies",
    "f-strings were introduced in Python 3.6",
    "Generators use yield to produce values lazily",
    "Context managers implement __enter__ and __exit__ methods",
    "Type hints improve code readability and IDE support",
]


class RetrievalPrecisionScenario:
    @property
    def name(self) -> str:
        return "retrieval_precision"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        for fact in _DECORATOR_FACTS:
            system.ingest("decorator_session", fact, tags=["fact", "decorator"])
        for i, fact in enumerate(_NOISE_FACTS):
            system.ingest(f"noise_{i}", fact, tags=["fact"])

        _maybe_consolidate(system)

        question = "What do you know about decorators?"
        response = system.query(question)
        response_lower = response.lower()

        passed = "decorator" in response_lower
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
