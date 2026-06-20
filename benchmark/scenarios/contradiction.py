"""Contradiction scenario: does the system detect conflicting information?"""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

_CONFLICT_WORDS = {"conflict", "contradict", "both", "inconsistent"}


class ContradictionScenario:
    @property
    def name(self) -> str:
        return "contradiction"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        system.ingest("s1", "The user prefers tabs for indentation", tags=["preference"])
        system.ingest("s2", "The user prefers spaces for indentation", tags=["preference"])
        system.ingest("s3", "PEP 8 recommends spaces", tags=["fact"])

        try:
            from benchmark.systems.engram_system import EngramSystem
            if isinstance(system, EngramSystem):
                system._consolidation_agent.run()
                flags = system._ltm.get_unresolved_contradictions()
                passed = len(flags) > 0
                detail = (
                    f"{'PASS' if passed else 'FAIL'} | "
                    f"{len(flags)} contradiction flag(s) detected by HCMA"
                )
                return ScenarioResult(
                    scenario_name=self.name,
                    system_name=system.name,
                    score=1.0 if passed else 0.0,
                    max_score=1.0,
                    details=[detail],
                    duration_seconds=time.time() - start,
                )
        except ImportError:
            pass

        question = "Does the user prefer tabs or spaces?"
        response = system.query(question)
        passed = any(w in response.lower() for w in _CONFLICT_WORDS)
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
