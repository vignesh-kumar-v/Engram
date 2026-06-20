"""Naive RAG baseline: flat vector store, no consolidation or hierarchy."""

from __future__ import annotations

import logging
import tempfile
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from benchmark.config import EMBED_MODEL, OLLAMA_MODEL, TOP_K_RETRIEVAL
from benchmark.systems.base_system import BaseSystem

logger = logging.getLogger(__name__)

_COLLECTION = "naive_rag_bench"
_EMBED_DIM = 768


class NaiveRagSystem(BaseSystem):
    def __init__(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="naive_rag_")
        self._client = ollama.Client()
        self._qdrant = QdrantClient(path=self._tmpdir)
        self._ensure_collection()

    @property
    def name(self) -> str:
        return "naive_rag"

    def _ensure_collection(self) -> None:
        if not self._qdrant.collection_exists(_COLLECTION):
            self._qdrant.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )

    def _embed(self, text: str) -> list[float]:
        try:
            result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return result.embedding
        except Exception:
            logger.exception("NaiveRagSystem._embed failed")
            return []

    def ingest(self, session_id: str, content: str, tags: list[str]) -> None:
        embedding = self._embed(content)
        if not embedding:
            logger.warning("NaiveRagSystem.ingest: empty embedding, skipping")
            return
        self._qdrant.upsert(
            collection_name=_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"content": content, "tags": tags},
                )
            ],
        )

    def query(self, question: str) -> str:
        embedding = self._embed(question)
        if not embedding:
            return ""
        try:
            results = self._qdrant.query_points(
                collection_name=_COLLECTION,
                query=embedding,
                limit=TOP_K_RETRIEVAL,
                with_payload=True,
            )
            context = "\n".join(
                p.payload.get("content", "") for p in results.points
            )
            prompt = (
                f"Answer based on this context:\n{context}\n"
                f"Question: {question}"
            )
            response = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content
        except Exception:
            logger.exception("NaiveRagSystem.query failed")
            return ""

    def reset(self) -> None:
        try:
            if self._qdrant.collection_exists(_COLLECTION):
                self._qdrant.delete_collection(_COLLECTION)
        except Exception:
            logger.exception("NaiveRagSystem.reset: delete_collection failed")
        self._ensure_collection()
