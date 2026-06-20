"""Retention scenario: can the system recall facts from earlier sessions?"""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

FACTS = [
    ("session_1", "The user prefers snake_case for variable naming"),
    ("session_1", "The user's main project uses Python 3.11"),
    ("session_2", "The user had a bug with async/await in FastAPI"),
    ("session_3", "The user prefers type hints in all functions"),
    ("session_4", "The user's database is PostgreSQL"),
]

QUERIES = [
    ("What naming convention does the user prefer?", "snake_case"),
    ("What Python version is the user using?", "3.11"),
    ("What framework had async issues?", "FastAPI"),
    ("Does the user use type hints?", "yes"),
    ("What database does the user use?", "PostgreSQL"),
]


class RetentionScenario:
    @property
    def name(self) -> str:
        return "retention"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        for session_id, content in FACTS:
            system.ingest(session_id, content, tags=["fact"])

        # Force consolidation on EngramSystem
        _maybe_consolidate(system)

        correct = 0
        details = []
        for question, expected_keyword in QUERIES:
            response = system.query(question)
            passed = expected_keyword.lower() in response.lower()
            correct += int(passed)
            details.append(
                f"{'PASS' if passed else 'FAIL'} | Q: {question!r} "
                f"| expected {expected_keyword!r} | got: {response[:80]!r}"
            )

        return ScenarioResult(
            scenario_name=self.name,
            system_name=system.name,
            score=correct / len(QUERIES),
            max_score=1.0,
            details=details,
            duration_seconds=time.time() - start,
        )


def _maybe_consolidate(system: BaseSystem) -> None:
    try:
        from benchmark.systems.engram_system import EngramSystem
        if isinstance(system, EngramSystem):
            system._consolidation_agent.run()
    except Exception:
        pass
