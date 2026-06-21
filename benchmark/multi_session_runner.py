"""Multi-session runner: executes a lifecycle scenario session-by-session with
per-session consolidation and a printable memory-state trace."""

from __future__ import annotations

import logging
import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.scenarios.multi_session_lifecycle import MultiSessionLifecycleScenario
from benchmark.systems.base_system import BaseSystem

logger = logging.getLogger(__name__)

_WIDTH = 70


class MultiSessionRunner:
    """
    Runs a MultiSessionLifecycleScenario against a list of systems, calling
    ``system.after_session()`` (consolidation for Engram, no-op for others)
    and ``system.get_session_state()`` after each session so memory evolution
    is visible via ``print_trace()``.
    """

    def __init__(
        self,
        scenario: MultiSessionLifecycleScenario,
        systems: list[BaseSystem],
    ) -> None:
        self.scenario = scenario
        self.systems = systems
        # snapshots[system_name] = [{"session_id", "label", "state"}, ...]
        self.snapshots: dict[str, list[dict]] = {}
        self.results: list[ScenarioResult] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[ScenarioResult], dict[str, list[dict]]]:
        """Run all sessions for all systems; return (results, snapshots)."""
        self.results = []
        self.snapshots = {}

        for system in self.systems:
            system.reset()
            sys_snaps: list[dict] = []

            for session in self.scenario.sessions:
                for content, tags in session.facts:
                    system.ingest(session.session_id, content, tags)

                system.after_session()

                state = system.get_session_state()
                sys_snaps.append({
                    "session_id": session.session_id,
                    "label": session.label,
                    "state": state,
                })
                logger.debug(
                    "session=%s system=%s state=%s",
                    session.session_id, system.name, state,
                )

            self.snapshots[system.name] = sys_snaps
            self.results.append(self._evaluate(system))

        return self.results, self.snapshots

    def print_trace(self) -> None:
        """Print a session-by-session memory-state table for every system."""
        if not self.snapshots:
            print("No trace data — call run() first.")
            return

        print()
        print("━" * _WIDTH)
        print(f"{'Multi-Session Memory Trace':^{_WIDTH}}")
        print(f"{'scenario: ' + self.scenario.name:^{_WIDTH}}")
        print("━" * _WIDTH)

        n_sessions = len(self.scenario.sessions)
        for i, session in enumerate(self.scenario.sessions):
            print(f"\n  Session {i + 1}/{n_sessions}: {session.label}")
            for system in self.systems:
                snap_list = self.snapshots.get(system.name, [])
                state = snap_list[i]["state"] if i < len(snap_list) else {}
                self._print_system_row(system.name, state)

        print()
        print("━" * _WIDTH)
        print(f"{'Query Evaluation (Phase 8)':^{_WIDTH}}")
        print("━" * _WIDTH)
        for result in self.results:
            pct = result.score * 100
            bar_fill = int(pct / 5)
            bar = "█" * bar_fill + "░" * (20 - bar_fill)
            print(f"\n  {result.system_name}")
            print(f"  [{bar}] {pct:.0f}%  ({result.score * len(self.scenario.queries):.0f}/{len(self.scenario.queries)} correct)")
            for detail in result.details:
                print(f"    {detail}")
        print()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        correct = 0
        details: list[str] = []
        for question, keyword in self.scenario.queries:
            response = system.query(question)
            passed = keyword.lower() in response.lower()
            correct += int(passed)
            details.append(
                f"{'PASS' if passed else 'FAIL'} | Q: {question!r} "
                f"| expected {keyword!r} | got: {response[:80]!r}"
            )
        return ScenarioResult(
            scenario_name=self.scenario.name,
            system_name=system.name,
            score=correct / len(self.scenario.queries),
            max_score=1.0,
            details=details,
            duration_seconds=time.time() - start,
        )

    @staticmethod
    def _print_system_row(system_name: str, state: dict) -> None:
        if not state:
            print(f"    {system_name:<14} (stateless)")
            return
        if "buffer_raw" in state:
            print(
                f"    {system_name:<14} "
                f"buffer_raw={state['buffer_raw']:<3} "
                f"ltm={state['ltm_memories']:<4} "
                f"contradictions={state['contradictions']}"
            )
        elif "stored_vectors" in state:
            print(f"    {system_name:<14} stored_vectors={state['stored_vectors']}")
        else:
            fields = "  ".join(f"{k}={v}" for k, v in state.items())
            print(f"    {system_name:<14} {fields}")
