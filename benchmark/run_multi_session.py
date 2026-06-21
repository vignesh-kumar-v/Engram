"""Entry point: run the multi-session lifecycle benchmark."""

from __future__ import annotations

import logging

from benchmark.config import RESULTS_DIR
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.multi_session_runner import MultiSessionRunner
from benchmark.scenarios.multi_session_lifecycle import MultiSessionLifecycleScenario
from benchmark.systems.engram_system import EngramSystem
from benchmark.systems.naive_rag_system import NaiveRagSystem
from benchmark.systems.no_memory_system import NoMemorySystem

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    print("LHMBench — Multi-Session Lifecycle Evaluation")
    print("Systems: engram vs naive_rag vs no_memory")
    print("Scenario: 8-phase coding project (tech stack → bug → mislead → correction → fix)")
    print()

    scenario = MultiSessionLifecycleScenario()
    systems: list = [NoMemorySystem(), NaiveRagSystem(), EngramSystem()]

    runner = MultiSessionRunner(scenario, systems)
    results, _ = runner.run()

    runner.print_trace()

    evaluator = BenchmarkEvaluator(results)
    evaluator.print_report()
    evaluator.save_results(RESULTS_DIR)


if __name__ == "__main__":
    main()
