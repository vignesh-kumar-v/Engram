"""Benchmark runner: executes all scenarios against all registered systems."""

from __future__ import annotations

import logging
import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.scenarios.compression import CompressionScenario
from benchmark.scenarios.contradiction import ContradictionScenario
from benchmark.scenarios.interference import InterferenceScenario
from benchmark.scenarios.retention import RetentionScenario
from benchmark.scenarios.noise_degradation import NoiseDegradationScenario
from benchmark.scenarios.retrieval_precision import RetrievalPrecisionScenario
from benchmark.systems.engram_system import EngramSystem
from benchmark.systems.naive_rag_system import NaiveRagSystem
from benchmark.systems.no_memory_system import NoMemorySystem

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self) -> None:
        self.systems = [
            NoMemorySystem(),
            NaiveRagSystem(),
            EngramSystem(),
        ]
        self.scenarios = [
            RetentionScenario(),
            InterferenceScenario(),
            ContradictionScenario(),
            CompressionScenario(),
            RetrievalPrecisionScenario(),
            NoiseDegradationScenario(),
        ]
        self.results: list[ScenarioResult] = []

    def run_all(self) -> list[ScenarioResult]:
        for system in self.systems:
            system.reset()
            logger.info("=== Benchmarking system: %s ===", system.name)
            for scenario in self.scenarios:
                try:
                    result = scenario.run(system)
                    self.results.append(result)
                    print(
                        f"  [{system.name}] {result.scenario_name}: "
                        f"{result.score:.2f}/{result.max_score:.2f}"
                    )
                    logger.info(
                        "  %s/%s — score=%.2f duration=%.2fs",
                        result.scenario_name, system.name,
                        result.score, result.duration_seconds,
                    )
                except Exception:
                    logger.exception(
                        "Scenario %s failed for system %s",
                        scenario.name, system.name,
                    )
        return self.results

    def run_scenario(self, scenario_name: str) -> list[ScenarioResult]:
        scenario = next(
            (s for s in self.scenarios if s.name == scenario_name), None
        )
        if scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_name!r}")

        results: list[ScenarioResult] = []
        for system in self.systems:
            system.reset()
            try:
                result = scenario.run(system)
                self.results.append(result)
                results.append(result)
                print(
                    f"  [{system.name}] {result.scenario_name}: "
                    f"{result.score:.2f}/{result.max_score:.2f}"
                )
            except Exception:
                logger.exception(
                    "Scenario %s failed for system %s", scenario_name, system.name
                )
        return results
