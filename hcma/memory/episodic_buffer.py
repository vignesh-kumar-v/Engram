"""Episodic buffer: short-term SQLite-backed store for raw coding-session events."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from hcma.schemas.memory_types import EpisodicEntry

logger = logging.getLogger(__name__)

_VALID_STATUSES = {"raw", "promoted", "discarded"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodic_entries (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    source_task TEXT NOT NULL,
    importance  REAL NOT NULL,
    status      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    tags        TEXT NOT NULL
)
"""


def _row_to_entry(row: sqlite3.Row) -> EpisodicEntry:
    return EpisodicEntry(
        id=row["id"],
        content=row["content"],
        timestamp=row["timestamp"],
        source_task=row["source_task"],
        importance=row["importance"],
        status=row["status"],
        session_id=row["session_id"],
        tags=json.loads(row["tags"]),
    )


class EpisodicBuffer:
    def __init__(self, db_path: str, capacity: int) -> None:
        self.db_path = db_path
        self.capacity = capacity

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.info("EpisodicBuffer initialised: db=%s capacity=%d", db_path, capacity)

    def write(self, entry: EpisodicEntry) -> bool:
        try:
            self._conn.execute(
                """
                INSERT INTO episodic_entries
                    (id, content, timestamp, source_task, importance, status, session_id, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.content,
                    entry.timestamp,
                    entry.source_task,
                    entry.importance,
                    entry.status,
                    entry.session_id,
                    json.dumps(entry.tags),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            logger.exception("write failed for entry id=%s", entry.id)
            return False

    def read(self, entry_id: str) -> Optional[EpisodicEntry]:
        try:
            row = self._conn.execute(
                "SELECT * FROM episodic_entries WHERE id = ?", (entry_id,)
            ).fetchone()
            return _row_to_entry(row) if row else None
        except Exception:
            logger.exception("read failed for entry_id=%s", entry_id)
            return None

    def read_all_raw(self) -> List[EpisodicEntry]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM episodic_entries WHERE status = 'raw' ORDER BY timestamp ASC"
            ).fetchall()
            return [_row_to_entry(r) for r in rows]
        except Exception:
            logger.exception("read_all_raw failed")
            return []

    def get_count(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM episodic_entries"
            ).fetchone()
            return row[0]
        except Exception:
            logger.exception("get_count failed")
            return 0

    def is_at_capacity(self) -> bool:
        return self.get_count() >= self.capacity

    def update_status(self, entry_id: str, status: str) -> bool:
        if status not in _VALID_STATUSES:
            logger.error("update_status: invalid status %r", status)
            return False
        try:
            cursor = self._conn.execute(
                "UPDATE episodic_entries SET status = ? WHERE id = ?",
                (status, entry_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning("update_status: entry_id=%s not found", entry_id)
                return False
            return True
        except Exception:
            logger.exception("update_status failed for entry_id=%s", entry_id)
            return False

    def delete_entry(self, entry_id: str) -> bool:
        try:
            cursor = self._conn.execute(
                "DELETE FROM episodic_entries WHERE id = ?", (entry_id,)
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning("delete_entry: entry_id=%s not found", entry_id)
                return False
            return True
        except Exception:
            logger.exception("delete_entry failed for entry_id=%s", entry_id)
            return False

    def clear_non_raw(self) -> int:
        try:
            cursor = self._conn.execute(
                "DELETE FROM episodic_entries WHERE status != 'raw'"
            )
            self._conn.commit()
            return cursor.rowcount
        except Exception:
            logger.exception("clear_non_raw failed")
            return 0

    def search_by_tag(self, tag: str) -> List[EpisodicEntry]:
        try:
            # JSON array stored as text; match tag as a JSON string value inside it.
            # Using json_each would require SQLite >= 3.38; a LIKE guard is safe for
            # well-formed tag strings that contain no regex-special JSON characters.
            rows = self._conn.execute(
                """
                SELECT * FROM episodic_entries
                WHERE tags LIKE ?
                ORDER BY timestamp DESC
                """,
                (f'%"{tag}"%',),
            ).fetchall()
            return [_row_to_entry(r) for r in rows]
        except Exception:
            logger.exception("search_by_tag failed for tag=%r", tag)
            return []
