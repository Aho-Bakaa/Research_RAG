"""mmr.py — Maximal Marginal Relevance (MMR) diversity selection.

Implements section 6.2 of RAG_ARCHITECTURE.md:
score(c) = lambda * relevance(c) - (1 - lambda) * max_{s in selected} sim(c, s)
with lambda in [0.7, 0.8] to prevent redundant context from filling the window.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def select_diverse_chunks_mmr(
    candidates: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]] | None = None,
    limit: int = 10,
    lambda_param: float = 0.75,
    similarity_fn: Callable[[Any, Any], float] | None = None,
) -> list[dict[str, Any]]:
    """Select a diverse subset of candidate chunks using MMR.

    Parameters
    ----------
    candidates : Sequence[dict[str, Any]]
        Candidate chunks, each having a 'score' (relevance score).
    embeddings : Sequence[Sequence[float]] | None
        Embedding vectors matching candidates.
    limit : int
        Maximum number of diverse chunks to select.
    lambda_param : float
        Trade-off parameter between relevance (1.0) and diversity (0.0). Default 0.75.
    similarity_fn : Callable | None
        Custom similarity function (c1, c2) -> float in [0, 1]. Defaults to cosine similarity.

    Returns
    -------
    list[dict[str, Any]]
        Diverse, ranked list of selected chunks.
    """
    if not candidates:
        return []
    if len(candidates) <= limit:
        return list(candidates)

    n = len(candidates)
    selected_indices: list[int] = []
    unselected_indices: list[int] = list(range(n))

    # Pre-normalize relevance scores only if outside [0, 1]
    scores = [float(c.get("score", 0.0)) for c in candidates]
    if scores and all(0.0 <= s <= 1.0 for s in scores):
        norm_scores = scores
    else:
        max_s = max(scores) if scores else 1.0
        min_s = min(scores) if scores else 0.0
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0
        norm_scores = [(s - min_s) / range_s for s in scores]

    # Pre-calculate pairwise similarities if embeddings available
    sim_matrix: np.ndarray | None = None
    if embeddings is not None and len(embeddings) == n:
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embs = emb_arr / norms
        sim_matrix = np.dot(normalized_embs, normalized_embs.T)

    # 1. Pick the single highest-scoring item first
    best_idx = int(np.argmax(norm_scores))
    selected_indices.append(best_idx)
    unselected_indices.remove(best_idx)

    # 2. Greedily select next items with MMR formula
    while len(selected_indices) < limit and unselected_indices:
        best_mmr_score = -float("inf")
        best_candidate_idx = unselected_indices[0]

        for u_idx in unselected_indices:
            relevance = norm_scores[u_idx]

            # Compute max similarity to any already selected chunk
            if sim_matrix is not None:
                max_sim = max(float(sim_matrix[u_idx, s_idx]) for s_idx in selected_indices)
            elif similarity_fn is not None:
                max_sim = max(similarity_fn(candidates[u_idx], candidates[s_idx]) for s_idx in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate_idx = u_idx

        selected_indices.append(best_candidate_idx)
        unselected_indices.remove(best_candidate_idx)

    return [candidates[i] for i in selected_indices]
