"""parent_child.py — Hierarchical Small-to-Big Retrieval & Context Assembler.

Bridges fine-grained multi-track child retrieval (text, equation, table) with
rich section-level parent context:
1. Fast in-memory parent cache preloading (O(1) lookups, < 1 MB RAM footprint).
2. Candidate enrichment for MS-MARCO Cross-Encoder scoring.
3. Deduplication of multi-child hits mapping to the same parent section.
4. Structured prompt assembly presenting pinpoint matched passage + full parent section.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence
from qdrant_client import QdrantClient

from bl_pipeline.rag.token_budgeter import RetrievedFocalItem, SemanticTokenBudgeter

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """Manages fast in-memory parent dereferencing, deduplication, and calibrated context packaging."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        parent_collection: str = "parent",
        token_budget: int = 4500,
    ):
        self.client = qdrant_client
        self.parent_collection = parent_collection
        self.parent_cache: dict[str, dict[str, Any]] = {}
        self.budgeter = SemanticTokenBudgeter(total_token_budget=token_budget)
        self._load_cache()

    def _load_cache(self) -> None:
        """Preload all parent sections into memory for zero-latency lookups."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.parent_collection,
                limit=5000,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                payload = pt.payload or {}
                chunk_id = str(pt.id)
                self.parent_cache[chunk_id] = payload
                if "chunk_id" in payload:
                    self.parent_cache[payload["chunk_id"]] = payload
            logger.info(f"Loaded {len(points)} parent sections into memory cache.")
        except Exception as e:
            logger.error(f"Failed to load parent cache: {e}")

    def get_parent_payload(self, parent_id: str | None) -> dict[str, Any] | None:
        """Retrieve cached parent payload by parent UUID."""
        if not parent_id:
            return None
        return self.parent_cache.get(parent_id)

    def enrich_candidate_for_reranking(self, child: dict[str, Any]) -> str:
        """Prepend parent section path and brief context to child text for cross-encoder scoring.
        
        This prevents the cross-encoder from penalizing short (40-80 char) formula fragments
        or parameter statements while maintaining focus on the primary child match.
        """
        p_id = child.get("payload", {}).get("parent_id")
        child_text = child.get("content", "").strip()
        if p_id and p_id in self.parent_cache:
            parent = self.parent_cache[p_id]
            sec = parent.get("section_path", "")
            parent_lead = parent.get("content", "")[:350].replace("\n", " ").strip()
            return f"Section: {sec}\nPassage: {child_text}\nContext: {parent_lead}"
        return child_text

    def expand_and_deduplicate(
        self,
        child_candidates: list[dict[str, Any]],
        max_parents: int = 3,
    ) -> list[dict[str, Any]]:
        """Deduplicate child candidates by parent_id and assemble complete parent contexts.
        
        If multiple child hits stem from the same section (e.g. 3 consecutive equation chunks),
        they are collapsed into a single parent context with the highest candidate rank preserved.
        """
        seen_parents: set[str] = set()
        expanded_contexts: list[dict[str, Any]] = []

        for cand in child_candidates:
            p_id = cand.get("payload", {}).get("parent_id")
            if not p_id or p_id not in self.parent_cache:
                # Fallback: if parent not in cache, preserve child candidate directly
                expanded_contexts.append(cand)
                if len(expanded_contexts) >= max_parents:
                    break
                continue

            if p_id in seen_parents:
                # Aggregate subsequent child matches into the existing parent context
                for ctx in expanded_contexts:
                    if ctx["chunk_id"] == p_id:
                        extra_content = cand.get("content", "").strip()
                        if extra_content and extra_content not in ctx["matched_child_content"]:
                            ctx["matched_child_content"] += "\n\n" + extra_content
                        break
                continue

            seen_parents.add(p_id)
            parent = self.parent_cache[p_id]
            
            child_content = cand.get("content", "").strip()
            parent_content = parent.get("content", "").strip()
            
            merged_context = {
                "chunk_id": p_id,
                "child_chunk_id": cand.get("chunk_id", ""),
                "paper_id": cand.get("paper_id") or parent.get("paper_id", ""),
                "section_path": parent.get("section_path", cand.get("section_path", "")),
                "score": cand.get("rerank_score", cand.get("score", 0.0)),
                "rerank_score": cand.get("rerank_score", cand.get("score", 0.0)),
                "content": parent_content if parent_content else child_content,
                "matched_child_content": child_content,
                "parent_content": parent_content,
                "payload": parent,
                "track": "parent",
            }
            expanded_contexts.append(merged_context)
            if len(expanded_contexts) >= max_parents:
                break

        return expanded_contexts

    def format_llm_context_block(
        self,
        expanded_contexts: list[dict[str, Any]],
        query: str = "",
    ) -> str:
        """Format calibrated context envelope using SemanticTokenBudgeter without arbitrary character cuts."""
        focal_items = []
        for ctx in expanded_contexts:
            payload = ctx.get("payload", {})
            focal_items.append(
                RetrievedFocalItem(
                    item_id=ctx.get("child_chunk_id") or ctx.get("chunk_id", ""),
                    paper_id=ctx.get("paper_id", "unknown_paper"),
                    element_type=payload.get("element_type", "paragraph"),
                    content=ctx.get("matched_child_content") or ctx.get("content", ""),
                    score=float(ctx.get("rerank_score", ctx.get("score", 0.0))),
                    section_path=ctx.get("section_path", ""),
                    page=payload.get("page", 1),
                    equation_ref=payload.get("equation_ref"),
                    table_ref=payload.get("table_no"),
                    parent_section_text=ctx.get("parent_content", ""),
                    cross_reference_texts=ctx.get("cross_reference_texts", []),
                    sibling_texts=ctx.get("sibling_texts", []),
                    metadata=payload,
                )
            )
        return self.budgeter.format_calibrated_context(focal_items, query=query)
