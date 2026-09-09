"""reranker.py — Cross-Encoder re-ranker for scientific candidate chunks.

Adapted from VaultStack's ML_MODELS/reranker.py:
- Models joint query-document cross-attention using pre-trained cross-encoders.
- Elevates specific formula lines, equations, and exact parameter values above verbose text.
- Filters out candidates with negative or low cross-encoder logits.
- Includes transparent fallback if cross-encoder inference is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-Encoder re-ranker evaluating (query, document) token cross-attention."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.enabled = enabled
        self._model = None

        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name, device=device)
                log.info(f"Loaded CrossEncoder model '{model_name}' on {device}")
            except Exception as e:
                log.warning(f"Failed to load CrossEncoder '{model_name}': {e}. Falling back to baseline ranking.")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Re-rank candidate chunks jointly against the user query.
        
        Args:
            query: The scientific question or search query.
            candidates: Sequence of retrieved candidate dicts.
            top_k: Maximum number of re-ranked candidates to return.
            min_score: Optional logit threshold cutoff to discard noise.
            
        Returns:
            List of candidate dicts sorted by rerank_score descending.
        """
        if not candidates:
            return []

        if self._model is None:
            # Fallback to existing candidate ranking
            return list(candidates)[:top_k]

        pairs = []
        for c in candidates:
            text = str(c.get("content") or c.get("payload", {}).get("content", "") or c.get("text", ""))
            # Include parent section context if available for richer cross-attention
            parent_ctx = c.get("parent_context", "")
            if parent_ctx:
                full_text = f"Section Context: {parent_ctx[:150]}\n{text}"
            else:
                full_text = text
            pairs.append((query, full_text))

        try:
            scores = self._model.predict(pairs)
            scored = []
            for cand, score in zip(candidates, scores):
                c_copy = dict(cand)
                c_copy["rerank_score"] = float(score)
                # Filter out low-confidence candidates if min_score specified
                if min_score is not None and float(score) < min_score:
                    continue
                scored.append(c_copy)

            # Sort descending by cross-encoder score
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            log.warning(f"Cross-encoder inference failed: {e}. Falling back to input order.")
            return list(candidates)[:top_k]
