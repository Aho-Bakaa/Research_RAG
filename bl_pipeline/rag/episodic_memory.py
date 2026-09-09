"""episodic_memory.py — Episodic memory of past failures and Phase-2 experimental diagnostics.

Implements section 7 of RAG_ARCHITECTURE.md:
- Vectorizes past physical discrepancies (e.g. Lambda_x under-prediction at Tu=2.7%).
- Payload-filtered retrieval by case / symptom_type / flow regime.
- Injects relevant historical lessons as few-shot warnings into the reasoner kickoff.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from bl_pipeline.rag.qdrant_store import QdrantMultiTrackStore
from bl_pipeline.rag.structured_chunker import StructuredChunk


@dataclass
class DiagnosticMemory:
    """Structured record of a past experimental failure, discrepancy, or diagnostic insight."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = "general"
    symptom_type: str = "transition_onset_discrepancy"
    velocity_ms: float | None = None
    tu_pct: float | None = None
    root_cause: str = ""
    lesson_learned: str = ""
    suggested_fix: str = ""
    citation: str = ""

    def to_formatted_prompt_text(self) -> str:
        """Format as a few-shot cautionary reminder for the reasoner."""
        lines = [
            f"[HISTORICAL CASE LESSON — {self.symptom_type}]",
            f"Case Context: U={self.velocity_ms} m/s, Tu={self.tu_pct}%, Case={self.case_id}",
            f"Root Cause: {self.root_cause}",
            f"Cautionary Rule: {self.lesson_learned}",
        ]
        if self.suggested_fix:
            lines.append(f"Recommended Action: {self.suggested_fix}")
        return "\n".join(lines)


class EpisodicMemoryManager:
    """Manages recording and retrieving vectorized episodic diagnostics in Qdrant."""

    def __init__(self, qdrant_store: QdrantMultiTrackStore | None = None):
        self.store = qdrant_store or QdrantMultiTrackStore(location=":memory:")

    def record_diagnostic(
        self,
        diagnostic: DiagnosticMemory,
        vector: list[float] | None = None,
    ) -> str:
        """Store a diagnostic failure in the 'memory' track."""
        if vector is None:
            # Simple dummy or fallback vector if embedder offline
            vector = [0.1] * self.store.vector_size

        chunk = StructuredChunk(
            chunk_id=diagnostic.memory_id,
            paper_id=diagnostic.citation or diagnostic.case_id,
            element_type="memory",
            content=diagnostic.to_formatted_prompt_text(),
            metadata={
                "case_id": diagnostic.case_id,
                "symptom_type": diagnostic.symptom_type,
                "velocity_ms": diagnostic.velocity_ms,
                "tu_pct": diagnostic.tu_pct,
                "root_cause": diagnostic.root_cause,
            },
        )
        self.store.upsert_chunks("memory", [chunk], [vector])
        return diagnostic.memory_id

    def retrieve_relevant_memories(
        self,
        query_vector: list[float] | None = None,
        symptom_type: str | None = None,
        limit: int = 2,
    ) -> list[str]:
        """Query memory track for past lessons matching flow conditions."""
        if query_vector is None:
            query_vector = [0.1] * self.store.vector_size

        payload_filter = {}
        if symptom_type:
            payload_filter["symptom_type"] = symptom_type

        hits = self.store.search(
            track="memory",
            query_vector=query_vector,
            limit=limit,
            payload_filter=payload_filter if payload_filter else None,
        )

        memories = []
        for h in hits:
            content = h.get("payload", {}).get("content", "")
            if content:
                memories.append(content)
        return memories
