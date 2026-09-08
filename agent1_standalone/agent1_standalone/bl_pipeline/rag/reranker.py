"""reranker.py — cross-encoder reranking for retrieved chunks.

Why a reranker
──────────────
Pure dense (vector) + BM25 hybrid retrieval ranks chunks by query-vs-chunk
similarity in embedding space. For small specialised corpora this is too
coarse — empty section-header chunks ("## 4 Validation for Flat Plate Test
Cases" + a Fig caption) score high on model-name keyword match yet contain
no substantive content. Empirically observed in run f7034604: the top sq3
chunk was Menter 2015's empty §4 header, while the substantive §1 chunk
("Some of the deficiencies of the γ-Reθ model... were removed") was buried.

A cross-encoder reranker fixes this by ACTUALLY READING each candidate vs
the query and scoring relevance directly — not via embedding cosine. The
model sees query+chunk together and outputs a single relevance score per
pair, so empty chunks are penalised because they don't actually answer the
question, and substantive chunks are promoted.

Model choice
────────────
The model + max_length come from the SINGLE config source
(`config.RERANKER_MODEL` / `RERANKER_MAX_LENGTH`, default
`BAAI/bge-reranker-v2-m3` @ 1024) so A1, A2 and A3 all rerank with the
same model — no per-module split.  Override globally with the
`BL_RERANKER_MODEL` env var.  All BGE rerankers are free/local (no API
cost) and English-strong on technical text.

Integration point
─────────────────
`retrieve_for_agent` in `bl_pipeline/shared/agent_subquery.py`. Per
sub-query: ask retrieve() for a wider candidate pool (e.g. 8 instead of 3),
rerank to top-3, then dedupe against the global seen-set. This preserves
facet diversity (each sub-query still contributes its share) while letting
the reranker promote the substantive chunk within each facet.

Failure handling
────────────────
Reranker is OPTIONAL — if the model fails to load (e.g. first-run download
fails, GPU/CPU mismatch), we fall back to embedding-similarity ranking
silently. Reranking is governed by the single canonical switch
`BL_RAG_USE_RERANKER` (default ON, see config.reranker_enabled); set
`BL_RAG_USE_RERANKER=0` to skip entirely.
"""

from __future__ import annotations

import os
from typing import Any

# Module-level singleton (lazy-initialised on first call)
_RERANKER: Any = None
_LOAD_ATTEMPTED: bool = False
_LOAD_ERROR: str | None = None


def _get_reranker() -> Any:
    """Lazy-load the cross-encoder. Returns None on failure (caller must
    fall back to no-rerank path).
    """
    global _RERANKER, _LOAD_ATTEMPTED, _LOAD_ERROR

    if _LOAD_ATTEMPTED:
        return _RERANKER

    _LOAD_ATTEMPTED = True

    from bl_pipeline.shared.config import reranker_enabled
    if not reranker_enabled():
        _LOAD_ERROR = "disabled by BL_RAG_USE_RERANKER=0"
        return None

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        _LOAD_ERROR = f"sentence_transformers not installed: {e!s}"
        return None

    try:
        # Model + max_length come from the SINGLE config source
        # (config.RERANKER_MODEL / RERANKER_MAX_LENGTH) so A1, A2 and A3
        # all rerank with the same model.  First-call download goes to the
        # HF cache; subsequent loads are local.
        from bl_pipeline.shared.config import (
            RERANKER_MODEL, RERANKER_MAX_LENGTH,
        )
        model_name = RERANKER_MODEL
        _RERANKER = CrossEncoder(model_name, max_length=RERANKER_MAX_LENGTH)
    except Exception as e:
        _LOAD_ERROR = f"CrossEncoder load failed: {e!s}"
        _RERANKER = None

    return _RERANKER


def is_available() -> bool:
    """Cheap probe — does NOT trigger model load."""
    from bl_pipeline.shared.config import reranker_enabled
    if not reranker_enabled():
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_load_error() -> str | None:
    """For diagnostic logging."""
    return _LOAD_ERROR


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank `candidates` by cross-encoder relevance to `query`.

    Each candidate is a dict with at least a `text` key (the chunk body).
    Returns the top_k candidates by reranker score, in descending order.

    On reranker-load failure, returns the first top_k candidates unchanged
    (so the call is always safe to wrap around retrieval).
    """
    if not candidates:
        return []
    if top_k <= 0:
        return []
    if len(candidates) <= top_k:
        # Nothing to rerank away — but still score them for `rerank_score`
        # provenance so downstream metadata is consistent.
        pass

    reranker = _get_reranker()
    if reranker is None:
        # Fallback: trust the upstream ranking
        return candidates[:top_k]

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = []
    for c in candidates:
        text = c.get("text", "") or c.get("text_preview", "")
        # Cross-encoders cap at max_length tokens; truncate verbose chunks
        # to ~2000 chars to stay well under that for English text
        if len(text) > 2000:
            text = text[:2000]
        pairs.append((query, text))

    try:
        scores = reranker.predict(pairs)
    except Exception:
        # Inference failure — fall back gracefully
        return candidates[:top_k]

    # Attach reranker score for provenance, then sort descending
    for c, s in zip(candidates, scores):
        c.setdefault("metadata", {})
        c["metadata"]["rerank_score"] = float(s)
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c.get("rerank_score", -1e9), reverse=True)
    return candidates[:top_k]
