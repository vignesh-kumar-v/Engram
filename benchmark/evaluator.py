"""Evaluator: aggregates, reports, and saves benchmark results."""

from __future__ import annotations

import json
import time
from pathlib import Path

from benchmark.config import RESULTS_DIR
from benchmark.scenarios.base_scenario import ScenarioResult

_SCENARIO_ORDER = [
    "retention",
    "interference",
    "contradiction",
    "compression",
    "retrieval_precision",
    "noise_degradation",
]

_SCENARIO_LABELS = {
    "retention": "Retention",
    "interference": "Interference",
    "contradiction": "Contradiction",
    "compression": "Compression",
    "retrieval_precision": "Retrieval",
    "noise_degradation": "Noise Degrad.",
}


class BenchmarkEvaluator:
    def __init__(self, results: list[ScenarioResult]) -> None:
        self.results = results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_summary(self) -> dict:
        systems = self._systems()
        scenarios = self._scenarios()

        systems_data: dict[str, dict] = {}
        for system in systems:
            sys_results = [r for r in self.results if r.system_name == system]
            total_score = sum(r.score for r in sys_results)
            max_score = sum(r.max_score for r in sys_results)
            percentage = (total_score / max_score * 100) if max_score else 0.0
            per_scenario = {
                scenario: next(
                    (r.score for r in sys_results if r.scenario_name == scenario),
                    0.0,
                )
                for scenario in scenarios
            }
            systems_data[system] = {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "per_scenario": per_scenario,
            }

        winner = max(systems_data, key=lambda s: systems_data[s]["percentage"]) if systems_data else ""

        scenario_winners: dict[str, str] = {}
        for scenario in scenarios:
            best_system = ""
            best_score = -1.0
            for system in systems:
                score = systems_data[system]["per_scenario"].get(scenario, 0.0)
                if score > best_score:
                    best_score = score
                    best_system = system
            scenario_winners[scenario] = best_system

        return {
            "systems": systems_data,
            "winner": winner,
            "scenario_winners": scenario_winners,
        }

    def print_report(self) -> None:
        summary = self.compute_summary()
        systems = self._systems()
        scenarios = [s for s in _SCENARIO_ORDER if s in self._scenarios()] + \
                    [s for s in self._scenarios() if s not in _SCENARIO_ORDER]

        # Column widths
        label_w = 14
        col_w = 11

        def hline(left, mid, right, fill="═"):
            cols = left + (fill * label_w) + mid + \
                   (mid.join(fill * col_w for _ in systems)) + right
            return cols

        def row(label, cells, left="║", sep="║", right="║"):
            return (
                left + f" {label:<{label_w - 1}}" + sep +
                sep.join(f"{c:^{col_w}}" for c in cells) + right
            )

        print()
        print(hline("╔", "╦", "╗"))
        title = "LHMBench Results"
        total_w = label_w + (col_w + 1) * len(systems)
        print("║" + title.center(total_w) + "║")
        print(hline("╠", "╦", "╣"))
        print(row("Scenario", [s[:col_w - 2].center(col_w - 2) for s in systems]))
        print(hline("╠", "╬", "╣"))

        for scenario in scenarios:
            label = _SCENARIO_LABELS.get(scenario, scenario).capitalize()
            cells = []
            for system in systems:
                r = next(
                    (x for x in self.results
                     if x.scenario_name == scenario and x.system_name == system),
                    None,
                )
                cells.append(f"{r.score:.1f}/{r.max_score:.1f}" if r else "N/A")
            print(row(label, cells))

        print(hline("╠", "╬", "╣"))

        # Totals row
        total_cells = [
            f"{summary['systems'][s]['total_score']:.1f}/"
            f"{summary['systems'][s]['max_score']:.1f}"
            for s in systems
        ]
        print(row("TOTAL", total_cells))

        # Percentage row
        pct_cells = [
            f"{summary['systems'][s]['percentage']:.1f}%"
            for s in systems
        ]
        print(row("PERCENTAGE", pct_cells))

        print(hline("╚", "╩", "╝"))
        print(f"Winner: {summary['winner']}")
        print()

    def save_results(self, path: str) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        out_file = out_dir / f"lhmbench_{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "summary": self.compute_summary(),
            "results": [
                {
                    "scenario_name": r.scenario_name,
                    "system_name": r.system_name,
                    "score": r.score,
                    "max_score": r.max_score,
                    "details": r.details,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.results
            ],
        }
        out_file.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {out_file}")
        return str(out_file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _systems(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen[r.system_name] = None
        return list(seen)

    def _scenarios(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen[r.scenario_name] = None
        return list(seen)
