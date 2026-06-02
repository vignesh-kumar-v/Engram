"""Long-term memory store backed by SQLite (metadata) and Qdrant (vectors)."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from hcma.schemas.memory_types import ContradictionFlag, LTMMemory

logger = logging.getLogger(__name__)

_EMBED_MODEL = "nomic-embed-text"
_EMBED_DIM = 768

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ltm_memories (
    id                  TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    confidence          REAL NOT NULL,
    source_episode_ids  TEXT NOT NULL,
    access_count        INTEGER NOT NULL,
    created_at          REAL NOT NULL,
    last_accessed       REAL NOT NULL,
    vector_id           TEXT NOT NULL,
    memory_type         TEXT NOT NULL
)
"""

_CREATE_CONTRADICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS contradictions (
    id          TEXT PRIMARY KEY,
    memory_id_a TEXT NOT NULL,
    memory_id_b TEXT NOT NULL,
    reason      TEXT NOT NULL,
    severity    TEXT NOT NULL,
    detected_at REAL NOT NULL,
    resolved    INTEGER DEFAULT 0
)
"""


def _row_to_memory(row: sqlite3.Row) -> LTMMemory:
    return LTMMemory(
        id=row["id"],
        content=row["content"],
        confidence=row["confidence"],
        source_episode_ids=json.loads(row["source_episode_ids"]),
        access_count=row["access_count"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        vector_id=row["vector_id"],
        memory_type=row["memory_type"],
    )


class LTMStore:
    def __init__(
        self,
        db_path: str,
        qdrant_storage_path: str,
        collection_name: str,
    ) -> None:
        self.collection_name = collection_name

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_CONTRADICTIONS_TABLE)
        self._conn.commit()

        Path(qdrant_storage_path).mkdir(parents=True, exist_ok=True)
        self._qdrant = QdrantClient(path=qdrant_storage_path)

        self._ensure_collection()
        logger.info(
            "LTMStore initialised: db=%s qdrant=%s collection=%s",
            db_path, qdrant_storage_path, collection_name,
        )

    def _ensure_collection(self) -> None:
        try:
            if not self._qdrant.collection_exists(self.collection_name):
                self._qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection %r", self.collection_name)
        except Exception:
            logger.exception("_ensure_collection failed")
            raise

    def _get_embedding(self, content: str) -> List[float]:
        try:
            result = ollama.embeddings(model=_EMBED_MODEL, prompt=content)
            return result.embedding
        except Exception:
            logger.error("_get_embedding failed for content starting with '%s'", content[:60])
            return []

    def write(self, memory: LTMMemory) -> bool:
        try:
            embedding = self._get_embedding(memory.content)

            if not embedding:
                logger.warning(
                    "Empty embedding for memory id=%s — skipping Qdrant write", memory.id
                )
                memory.vector_id = ""
            else:
                vector_id = str(uuid.uuid4())
                self._qdrant.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=vector_id,
                            vector=embedding,
                            payload={"ltm_id": memory.id},
                        )
                    ],
                )
                memory.vector_id = vector_id

            self._conn.execute(
                """
                INSERT OR REPLACE INTO ltm_memories
                    (id, content, confidence, source_episode_ids, access_count,
                     created_at, last_accessed, vector_id, memory_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.confidence,
                    json.dumps(memory.source_episode_ids),
                    memory.access_count,
                    memory.created_at,
                    memory.last_accessed,
                    memory.vector_id,
                    memory.memory_type,
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            logger.exception("write failed for memory id=%s", memory.id)
            return False

    def read(self, memory_id: str) -> Optional[LTMMemory]:
        try:
            row = self._conn.execute(
                "SELECT * FROM ltm_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            return _row_to_memory(row) if row else None
        except Exception:
            logger.exception("read failed for memory_id=%s", memory_id)
            return None

    def search_semantic(self, query: str, top_k: int = 5) -> List[LTMMemory]:
        embedding = self._get_embedding(query)
        if not embedding:
            logger.warning("search_semantic: embedding failed, falling back to search_by_content")
            return self.search_by_content(query, top_k)
        try:
            results = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            memories: List[LTMMemory] = []
            for point in results.points:
                ltm_id = point.payload.get("ltm_id")
                if ltm_id:
                    mem = self.read(ltm_id)
                    if mem:
                        memories.append(mem)
            return memories
        except Exception:
            logger.exception("search_semantic failed, falling back to search_by_content")
            return self.search_by_content(query, top_k)

    def search_by_content(self, query: str, top_k: int = 5) -> List[LTMMemory]:
        try:
            rows = self._conn.execute(
                """
                SELECT * FROM ltm_memories
                WHERE content LIKE ?
                ORDER BY access_count DESC
                LIMIT ?
                """,
                (f"%{query}%", top_k),
            ).fetchall()
            return [_row_to_memory(r) for r in rows]
        except Exception:
            logger.exception("search_by_content failed for query=%r", query)
            return []

    def update_access(self, memory_id: str) -> bool:
        try:
            import time
            cursor = self._conn.execute(
                """
                UPDATE ltm_memories
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
                """,
                (time.time(), memory_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning("update_access: memory_id=%s not found", memory_id)
                return False
            return True
        except Exception:
            logger.exception("update_access failed for memory_id=%s", memory_id)
            return False

    def get_all(self) -> List[LTMMemory]:
        try:
            rows = self._conn.execute(
                "SELECT * FROM ltm_memories ORDER BY created_at DESC"
            ).fetchall()
            return [_row_to_memory(r) for r in rows]
        except Exception:
            logger.exception("get_all failed")
            return []

    def delete(self, memory_id: str) -> bool:
        try:
            row = self._conn.execute(
                "SELECT vector_id FROM ltm_memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                logger.warning("delete: memory_id=%s not found", memory_id)
                return False

            vector_id = row["vector_id"]
            if vector_id:
                try:
                    self._qdrant.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=[vector_id]),
                    )
                except Exception:
                    logger.exception(
                        "delete: Qdrant delete failed for vector_id=%s", vector_id
                    )

            cursor = self._conn.execute(
                "DELETE FROM ltm_memories WHERE id = ?", (memory_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception:
            logger.exception("delete failed for memory_id=%s", memory_id)
            return False

    def save_contradiction(self, flag: ContradictionFlag) -> bool:
        try:
            import time as _time
            self._conn.execute(
                """
                INSERT INTO contradictions
                    (id, memory_id_a, memory_id_b, reason, severity, detected_at, resolved)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    str(uuid.uuid4()),
                    flag.memory_id_a,
                    flag.memory_id_b,
                    flag.reason,
                    flag.severity,
                    _time.time(),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            logger.exception(
                "save_contradiction failed for %s vs %s",
                flag.memory_id_a[:8], flag.memory_id_b[:8],
            )
            return False

    def get_unresolved_contradictions(self) -> List[ContradictionFlag]:
        try:
            rows = self._conn.execute(
                """
                SELECT memory_id_a, memory_id_b, reason, severity
                FROM contradictions
                WHERE resolved = 0
                ORDER BY detected_at DESC
                """
            ).fetchall()
            return [
                ContradictionFlag(
                    memory_id_a=r["memory_id_a"],
                    memory_id_b=r["memory_id_b"],
                    reason=r["reason"],
                    severity=r["severity"],
                )
                for r in rows
            ]
        except Exception:
            logger.exception("get_unresolved_contradictions failed")
            return []

    def resolve_contradiction(self, memory_id_a: str, memory_id_b: str) -> bool:
        try:
            cursor = self._conn.execute(
                """
                UPDATE contradictions SET resolved = 1
                WHERE memory_id_a = ? AND memory_id_b = ? AND resolved = 0
                """,
                (memory_id_a, memory_id_b),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                logger.warning(
                    "resolve_contradiction: no unresolved row for %s vs %s",
                    memory_id_a[:8], memory_id_b[:8],
                )
                return False
            return True
        except Exception:
            logger.exception(
                "resolve_contradiction failed for %s vs %s",
                memory_id_a[:8], memory_id_b[:8],
            )
            return False
