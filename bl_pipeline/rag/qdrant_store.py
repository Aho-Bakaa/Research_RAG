"""qdrant_store.py — Multi-track Qdrant vector store with pre-filtering on payload.

Implements section 5 of RAG_ARCHITECTURE.md:
- Multi-track storage: 'text', 'table', 'figure', 'equation', 'parent', 'memory'.
- Native payload pre-filtering (paper_id, section_path, element_type, case_id).
- Supports local in-memory (":memory:"), on-disk storage, or remote cluster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from bl_pipeline.rag.structured_chunker import StructuredChunk

log = logging.getLogger(__name__)

TRACK_COLLECTIONS = ["text", "table", "figure", "equation", "parent", "memory"]


class QdrantMultiTrackStore:
    """Multi-track vector store powered by Qdrant."""

    def __init__(
        self,
        location: str | Path = ":memory:",
        vector_size: int = 384,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
    ):
        self.vector_size = vector_size
        self.distance = distance
        if isinstance(location, Path):
            location = str(location)
        self.location = location

        if location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif location.startswith("http://") or location.startswith("https://"):
            self.client = QdrantClient(url=location)
        else:
            path = Path(location)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

        self._ensure_collections()

    def _ensure_collections(self) -> None:
        """Create all multi-track collections if they do not exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        for track in TRACK_COLLECTIONS:
            if track not in existing:
                self.client.create_collection(
                    collection_name=track,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )

    def upsert_chunks(
        self,
        track: str,
        chunks: Sequence[StructuredChunk],
        embeddings: Sequence[list[float]],
    ) -> int:
        """Upsert a batch of StructuredChunks and vectors into a specific track."""
        if not chunks or not embeddings:
            return 0
        if track not in TRACK_COLLECTIONS:
            raise ValueError(f"Unknown track {track!r}. Must be one of {TRACK_COLLECTIONS}")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "paper_id": chunk.paper_id,
                "page": chunk.page,
                "section_path": chunk.section_path,
                "heading_level": chunk.heading_level,
                "element_type": chunk.element_type,
                "content": chunk.content,
                **chunk.metadata,
            }
            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=track,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        track: str,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search a single track with optional pre-filtering on payload."""
        if track not in TRACK_COLLECTIONS:
            raise ValueError(f"Unknown track {track!r}")

        q_filter = self._build_qdrant_filter(payload_filter) if payload_filter else None

        results = self.client.query_points(
            collection_name=track,
            query=query_vector,
            query_filter=q_filter,
            limit=limit,
        ).points

        out = []
        for r in results:
            p = r.payload or {}
            out.append({
                "chunk_id": r.id,
                "score": float(r.score) if r.score is not None else 0.0,
                "content": p.get("content", ""),
                "payload": p,
                "track": track,
            })
        return out

    def search_multitrack(
        self,
        tracks: Sequence[str],
        query_vector: list[float],
        limit_per_track: int = 5,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query multiple tracks in parallel and return the aggregated candidate pool."""
        pool: list[dict[str, Any]] = []
        for t in tracks:
            if t in TRACK_COLLECTIONS:
                res = self.search(t, query_vector, limit=limit_per_track, payload_filter=payload_filter)
                pool.extend(res)
        # Sort by similarity score descending
        pool.sort(key=lambda x: x["score"], reverse=True)
        return pool

    @staticmethod
    def _build_qdrant_filter(payload_filter: dict[str, Any]) -> qmodels.Filter:
        """Convert a python dict into Qdrant must condition filter."""
        must_clauses: list[qmodels.FieldCondition] = []
        for key, value in payload_filter.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                must_clauses.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchAny(any=list(value)),
                    )
                )
            else:
                must_clauses.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                )
        return qmodels.Filter(must=must_clauses)
