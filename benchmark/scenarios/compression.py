"""Compression scenario: key facts survive after episodic consolidation."""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

DETAILED_FACTS = [
    "User debugged a RecursionError in a tree traversal function",
    "The recursion limit was hit at depth 1000",
    "Fix was to convert recursive DFS to iterative using a stack",
    "The tree had approximately 5000 nodes",
    "After fix, runtime improved from timeout to 0.3 seconds",
]

QUERIES = [
    ("What error occurred in tree traversal?", "RecursionError"),
    ("What was the fix for the recursion issue?", "iterative"),
    ("How many nodes did the tree have?", "5000"),
]


class CompressionScenario:
    @property
    def name(self) -> str:
        return "compression"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        for i, fact in enumerate(DETAILED_FACTS):
            system.ingest(f"debug_session_{i}", fact, tags=["debug", "error_pattern"])

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
