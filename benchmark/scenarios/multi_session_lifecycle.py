"""Multi-session lifecycle scenario: 8-phase coding project with evolving memory."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionSpec:
    session_id: str
    label: str
    facts: list[tuple[str, list[str]]] = field(default_factory=list)


SESSIONS: list[SessionSpec] = [
    SessionSpec(
        session_id="session_1",
        label="Tech stack: Python + FastAPI",
        facts=[
            ("The project uses Python 3.11 as the primary language", ["fact", "tech_stack"]),
            ("FastAPI is the web framework for the REST API", ["fact", "tech_stack"]),
        ],
    ),
    SessionSpec(
        session_id="session_2",
        label="Database: PostgreSQL + asyncpg",
        facts=[
            ("PostgreSQL is the database, accessed via asyncpg for async queries", ["fact", "tech_stack"]),
            ("The asyncpg connection pool is configured with an initial size of 10", ["fact", "configuration"]),
        ],
    ),
    SessionSpec(
        session_id="session_3",
        label="Frontend: React + TypeScript",
        facts=[
            ("React with TypeScript is used for the frontend", ["fact", "tech_stack"]),
            ("The frontend communicates with the backend via REST and WebSockets", ["fact", "tech_stack"]),
        ],
    ),
    SessionSpec(
        session_id="session_4",
        label="Bug: memory leak discovered",
        facts=[
            ("A memory leak was discovered: server RSS grows 50 MB per hour under load", ["fact", "bug"]),
            ("Initial hypothesis: asyncpg connection pool is not releasing connections after queries", ["fact", "bug", "hypothesis"]),
        ],
    ),
    SessionSpec(
        session_id="session_5",
        label="Misleading: ORM layer suspected",
        facts=[
            ("Pool size increased to 20 — memory leak persists, ruling out pool exhaustion", ["fact", "bug"]),
            ("New hypothesis: SQLAlchemy ORM layer may be holding object references and causing the leak", ["fact", "bug", "hypothesis"]),
        ],
    ),
    SessionSpec(
        session_id="session_6",
        label="Correction: ORM ruled out, WebSocket handler is culprit",
        facts=[
            ("ORM hypothesis disproved by profiling — all SQLAlchemy sessions are properly closed", ["fact", "bug"]),
            ("Memory leak root cause confirmed: the WebSocket handler does not call connection.close() on client disconnect", ["fact", "bug", "root_cause"]),
        ],
    ),
    SessionSpec(
        session_id="session_7",
        label="Fix: WebSocket handler patched",
        facts=[
            ("WebSocket handler updated to explicitly call connection.close() on disconnect event", ["fact", "bug", "fix"]),
            ("Memory leak resolved after deployment — RSS stable over 24-hour soak test", ["fact", "bug", "fix"]),
        ],
    ),
]

QUERIES: list[tuple[str, str]] = [
    ("What database does the project use?", "postgresql"),
    ("What web framework is the REST API built with?", "fastapi"),
    ("What frontend framework is used?", "react"),
    ("What was the root cause of the memory leak?", "websocket"),
    ("What memory leak hypothesis was ruled out after profiling?", "orm"),
]


class MultiSessionLifecycleScenario:
    """
    8-phase coding project lifecycle.

    Phases 1–7 are ingestion sessions; phase 8 is the query evaluation.
    Sessions 4–5 introduce a misleading hypothesis; session 6 corrects it.
    The queries test recall of early tech-stack facts plus the correction.
    """

    @property
    def name(self) -> str:
        return "multi_session_lifecycle"

    @property
    def sessions(self) -> list[SessionSpec]:
        return SESSIONS

    @property
    def queries(self) -> list[tuple[str, str]]:
        return QUERIES
