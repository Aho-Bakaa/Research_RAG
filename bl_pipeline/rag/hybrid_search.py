"""hybrid_search.py — Lexical BM25 + Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Adapted and enhanced from VaultStack's hybrid retrieval engine:
- Scientific tokenization preserving equations, symbols (Re_theta, Tu, Lambda_x), and constants (163, 6.91, 400).
- Candidate-level BM25 scoring over multi-track retrieved chunks.
- Reciprocal Rank Fusion (RRF) combining dense semantic search and sparse lexical matching:
    RRF(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


def tokenize_scientific_text(text: str) -> list[str]:
    """Tokenize text preserving scientific terms, equations, underscores, and numbers.
    
    Examples:
    - 'Re_theta_t' -> 're_theta_t', 're_theta', 'theta'
    - '6.91' -> '6.91'
    - 'Tu > 3%' -> 'tu', '3'
    - 'k-kL-omega' -> 'k-kl-omega', 'omega'
    """
    cleaned = text.lower()
    # Find words with optional underscores, hyphens, or decimal points
    raw_tokens = re.findall(r"[a-z0-9]+(?:[_\.-][a-z0-9]+)*", cleaned)
    tokens = []
    for t in raw_tokens:
        tokens.append(t)
        # If token has sub-parts like re_theta, also include component parts
        if "_" in t:
            tokens.extend([part for part in t.split("_") if len(part) > 1])
        if "-" in t:
            tokens.extend([part for part in t.split("-") if len(part) > 1])
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


class SimpleBM25:
    """Lightweight BM25 scorer operating directly over candidate chunk payloads.
    
    Computes exact IDF and length-normalized TF scores across retrieved candidates
    without requiring a heavy separate Lucene/Elasticsearch index.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_candidates(
        self,
        query_tokens: list[str],
        candidates: Sequence[dict[str, Any]],
    ) -> list[float]:
        """Compute BM25 scores for each candidate chunk given query tokens."""
        if not candidates or not query_tokens:
            return [0.0] * len(candidates)

        candidate_texts = [
            str(c.get("content") or c.get("payload", {}).get("content", "") or c.get("text", ""))
            for c in candidates
        ]
        num_docs = len(candidates)
        tokenized_docs = [tokenize_scientific_text(t) for t in candidate_texts]
        avg_doc_len = sum(len(d) for d in tokenized_docs) / max(num_docs, 1)

        # Calculate IDF for each query token across the candidate pool
        idf_scores: dict[str, float] = {}
        for token in set(query_tokens):
            # Document frequency: number of candidates containing token
            df = sum(1 for d in tokenized_docs if token in d)
            # Standard Lucene/Okapi smoothed IDF
            idf_scores[token] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

        scores: list[float] = []
        for doc_tokens in tokenized_docs:
            doc_len = len(doc_tokens)
            token_counts = Counter(doc_tokens)

            score = 0.0
            for token in query_tokens:
                if token not in idf_scores:
                    continue
                tf = token_counts.get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_doc_len))
                    score += idf_scores[token] * (numerator / denominator)
            scores.append(score)

        return scores


def _result_key(result: dict[str, Any]) -> str:
    """Extract unique key for a candidate chunk."""
    cid = result.get("chunk_id") or result.get("id")
    if cid is not None:
        return str(cid)
    paper = result.get("payload", {}).get("paper_id", "")
    content_snippet = str(result.get("content", ""))[:120]
    return f"{paper}|{content_snippet}"


def rrf_fusion(
    vector_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (RRF) combining dense and lexical ranking lists.
    
    Formula:
        RRF(d) = sum(1.0 / (k + rank_i(d)))
    """
    scores: dict[str, float] = {}
    by_key: dict[str, dict[str, Any]] = {}

    # Dense ranks
    for rank, r in enumerate(vector_results, 1):
        key = _result_key(r)
        by_key.setdefault(key, dict(r))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # Sparse (BM25) ranks
    for rank, r in enumerate(sparse_results, 1):
        key = _result_key(r)
        by_key.setdefault(key, dict(r))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = list(by_key.values())
    for r in fused:
        key = _result_key(r)
        r["rrf_score"] = scores.get(key, 0.0)
        # Store fused score as primary ranking score
        r["score"] = r["rrf_score"]

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def apply_hybrid_search(
    query: str,
    dense_candidates: list[dict[str, Any]],
    bm25_scorer: SimpleBM25 | None = None,
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """Execute candidate-level BM25 scoring and fuse with dense rankings via RRF."""
    if not dense_candidates:
        return []

    scorer = bm25_scorer or SimpleBM25()
    query_tokens = tokenize_scientific_text(query)
    bm25_scores = scorer.score_candidates(query_tokens, dense_candidates)

    # Create sparse ranking list (only candidates with positive lexical match)
    sparse_candidates = []
    for cand, b_score in zip(dense_candidates, bm25_scores):
        if b_score > 0.0:
            sc = dict(cand)
            sc["bm25_score"] = b_score
            sparse_candidates.append(sc)

    # Sort sparse candidates by BM25 score descending
    sparse_candidates.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)

    # Fuse dense + sparse via RRF
    return rrf_fusion(dense_candidates, sparse_candidates, k=k_rrf)
