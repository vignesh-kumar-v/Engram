"""Noise degradation scenario: most-recent fact must win over high-volume noise."""

from __future__ import annotations

import time

from benchmark.scenarios.base_scenario import ScenarioResult
from benchmark.systems.base_system import BaseSystem

_NOISE: list[tuple[str, str]] = [
    ("session_2", "User asked about Java syntax and class inheritance"),
    ("session_2", "User asked about JavaScript promises and async/await"),
    ("session_2", "User asked about CSS flexbox layout"),
    ("session_2", "User asked about SQL JOIN types"),
    ("session_2", "User asked about HTML semantic elements"),
    ("session_2", "User asked about Go goroutines"),
    ("session_2", "User asked about TypeScript generics"),
    ("session_2", "User asked about Ruby on Rails conventions"),
    ("session_2", "User asked about Kotlin data classes"),
    ("session_2", "User asked about Swift optionals"),
    ("session_3", "User asked about Docker multi-stage builds"),
    ("session_3", "User asked about Kubernetes pod scheduling"),
    ("session_3", "User asked about Terraform state management"),
    ("session_3", "User asked about GitHub Actions workflow syntax"),
    ("session_3", "User asked about AWS Lambda cold starts"),
    ("session_3", "User asked about Redis pub/sub patterns"),
    ("session_3", "User asked about GraphQL schema design"),
    ("session_3", "User asked about gRPC protobuf definitions"),
    ("session_3", "User asked about Nginx reverse proxy config"),
    ("session_3", "User asked about PostgreSQL indexing strategies"),
    ("session_4", "User asked about React hooks lifecycle"),
    ("session_4", "User asked about Vue 3 Composition API"),
    ("session_4", "User asked about Angular dependency injection"),
    ("session_4", "User asked about Svelte reactive declarations"),
    ("session_4", "User asked about Next.js server components"),
    ("session_4", "User asked about Webpack bundle splitting"),
    ("session_4", "User asked about Vite HMR configuration"),
    ("session_4", "User asked about Jest snapshot testing"),
    ("session_4", "User asked about Cypress end-to-end tests"),
    ("session_4", "User asked about Storybook component stories"),
]


class NoiseDegradationScenario:
    @property
    def name(self) -> str:
        return "noise_degradation"

    def run(self, system: BaseSystem) -> ScenarioResult:
        start = time.time()
        system.reset()

        system.ingest("session_1", "User's primary language is Python", tags=["preference"])

        for session_id, content in _NOISE:
            system.ingest(session_id, content, tags=["fact"])

        system.ingest("session_5", "User switched primary language to Rust", tags=["preference"])

        _maybe_consolidate(system)

        question = "What is the user's current primary programming language?"
        response = system.query(question)
        passed = "rust" in response.lower()

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
