"""LHMBench entry point — run all scenarios against all systems."""

from __future__ import annotations

import logging

from benchmark.config import RESULTS_DIR
from benchmark.evaluator import BenchmarkEvaluator
from benchmark.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    print("LHMBench — Long-Horizon Memory Benchmark")
    print("Systems: engram vs naive_rag vs no_memory")
    print("Scenarios: retention, interference, contradiction, compression, retrieval_precision")
    print()

    runner = BenchmarkRunner()
    results = runner.run_all()

    evaluator = BenchmarkEvaluator(results)
    evaluator.print_report()
    evaluator.save_results(RESULTS_DIR)


if __name__ == "__main__":
    main()
