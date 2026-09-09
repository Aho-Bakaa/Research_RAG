"""weighted_retriever.py — consume ExpandedQuery objects, re-rank by metadata weights.

This module is the muscle that executes the expander's plan. Given one
or more ExpandedQuery objects, for each one:

  1. Vector search the target collections (top-K candidates, no filter)
  2. For each candidate, compute an adjusted distance:
         final_distance = raw_distance - DISTANCE_BONUS_SCALE * sum(
             weight for each "field::value" the chunk's metadata matches
         )
     Chunks with matching positive-weight metadata move up; chunks
     with matching negative-weight metadata move down.
  3. Sort by adjusted distance, take top-K.

Merging across sub-queries uses `(paper_id, page_number, chunk_id_tail)`
as the dedup key. The first sub-query's ordering wins ties.

Why additive distance adjustment (not multiplicative score)
──────────────────────────────────────────────────────────
ChromaDB returns cosine distance in roughly [0, 2]. A +3 weight * 0.1
scale subtracts 0.3 from the distance — that's roughly one rank jump
in a tightly-clustered top-10, which is what we want for a meaningful
nudge without overwhelming semantic similarity. Multiplicative scoring
is harder to reason about when logging "why did chunk X rank there?"
and the logs are important for the decision ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bl_pipeline.rag.query_expander import ExpandedQuery


# How much each unit of metadata weight moves the distance.
# 0.1 means +3 weight -> -0.3 distance, -2 weight -> +0.2 distance.
DISTANCE_BONUS_SCALE: float = 0.1

# Hybrid vector + BM25 fusion weights.
#
# Vector gets the lion's share because embeddings capture paraphrase
# and semantic intent that BM25 can't. BM25 is the safety net for
# math-heavy chunks: a chunk like "$$Re_theta_t = 400 Tu^{-5/8}$$"
# has a very thin English embedding (it's mostly LaTeX symbols), so
# it loses to verbose context-rich chunks on vector alone. BM25 loves
# literal tokens like `400`, `Tu`, `5/8` and pulls the naked-formula
# chunk back into the candidate pool.
#
# 0.7 / 0.3 mirrors the weights already used in `RAGEngine.retrieve`
# for consistency.
VECTOR_FUSION_WEIGHT: float = 0.7
BM25_FUSION_WEIGHT: float = 0.3


@dataclass
class ScoredChunk:
    """A retrieved chunk annotated with expander-aware scoring info.

    raw_distance :
        Original ChromaDB cosine distance (smaller = more similar).
    applied_weight :
        Sum of metadata_weight values that matched this chunk.
    adjusted_distance :
        raw_distance - DISTANCE_BONUS_SCALE * applied_weight.
    sub_query_text :
        Which sub-query surfaced this chunk (for audit logs).
    matched_keys :
        List of "field::value" keys from the expander's weights that
        this chunk's metadata matched. Useful for explaining rank moves.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    collection: str
    raw_distance: float
    applied_weight: float
    adjusted_distance: float
    sub_query_text: str
    matched_keys: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "collection": self.collection,
            "raw_distance": self.raw_distance,
            "applied_weight": self.applied_weight,
            "adjusted_distance": self.adjusted_distance,
            "sub_query": self.sub_query_text,
            "matched_keys": self.matched_keys,
        }


def _metadata_matches(md: dict[str, Any], key: str) -> bool:
    """Does this chunk's metadata match the "field::value" key?

    Handles the small type quirks in ChromaDB — booleans may round-trip
    as either Python bool or the strings "True"/"False" depending on
    the client version.
    """
    if "::" not in key:
        return False
    field_name, target = key.split("::", 1)
    actual = md.get(field_name)
    if actual is None:
        return False
    # Exact string compare after lossy-but-safe normalization
    actual_str = str(actual).strip()
    target_str = str(target).strip()
    return actual_str == target_str


def _score_candidate(
    raw_distance: float,
    md: dict[str, Any],
    weights: dict[str, float],
) -> tuple[float, float, list[str]]:
    """Return (applied_weight_sum, adjusted_distance, matched_keys)."""
    matched_keys: list[str] = []
    applied = 0.0
    for key, w in weights.items():
        if _metadata_matches(md, key):
            matched_keys.append(key)
            applied += w
    adjusted = raw_distance - DISTANCE_BONUS_SCALE * applied
    return applied, adjusted, matched_keys


def _fuse_vector_bm25(
    vec_distance: float,
    bm25_score: float,
) -> float:
    """Fuse vector distance (smaller=better) with BM25 score (larger=better)
    into a single 'distance-like' value (smaller = better).

    vec_distance: ChromaDB cosine distance, roughly [0, 2]
    bm25_score:   normalised BM25 in [0, 1] (0 if no token overlap)

    Method: convert both to a common "similarity in [0,1]" space via
      sim_vec  = 1 / (1 + vec_distance)        # cosine -> sim
      sim_bm25 = bm25_score                    # already in [0,1]

    Combine:
      fused_sim = V * sim_vec + B * sim_bm25
      fused_dist = 1 - fused_sim               # back to distance space

    This preserves the "smaller = better" convention the rest of the
    module uses, so the metadata rerank formula
        adjusted = fused_dist - scale * applied_weight
    works unchanged.
    """
    sim_vec = 1.0 / (1.0 + vec_distance)
    fused_sim = VECTOR_FUSION_WEIGHT * sim_vec + BM25_FUSION_WEIGHT * bm25_score
    return 1.0 - fused_sim


def retrieve_for_expanded(
    rag_engine: Any,
    expanded: ExpandedQuery,
    vector_top_k: int = 50,
    final_top_k: int = 5,
) -> list[ScoredChunk]:
    """Execute one sub-query end to end:
       vector search + BM25 fusion + metadata-weighted rerank,
       optionally augmented with HyDE pre-pass and cross-encoder rerank
       post-pass (both feature-flagged).

    Parameters
    ──────────
    rag_engine
        RAGEngine instance. Exposes `get_or_create_collection(name)`
        and `bm25_scores_for_collection(name, query)`.
    expanded
        The planned sub-query.
    vector_top_k
        How many raw candidates to pull per collection before reranking.
        50 is generous enough that metadata boosts can meaningfully
        re-order the list; smaller values cap the boost's leverage.
        BM25's top-N chunks are also considered alongside the vector
        top-K — specifically, any BM25 match with score > 0.3 gets
        pulled into the candidate pool even if vector search missed it.
    final_top_k
        How many chunks to return after reranking, per-collection totals
        merged. Caller typically wants 3-10.

    Feature flags (env-var-driven, default OFF):
      BL_RAG_USE_HYDE=1
        Generate a hypothetical-answer paragraph via the LLM and
        concatenate it onto `expanded.text` BEFORE vector search.  The
        hypothetical bridges the vocabulary gap between question
        phrasing and corpus phrasing — empirically, the bare equation
        `Re_θ,t = 400·Tu^(-5/8)` embeds poorly because its English
        signal is thin; the hypothetical paragraph re-introduces
        anchor words like "transition onset correlation" that the
        embedder can match.  Only the VECTOR side of the search sees
        the augmented text; BM25 keeps the original sub-query so
        literal-token matching (numbers, variable names) isn't
        diluted by hallucinated vocabulary.

      BL_RAG_USE_RERANKER=1
        After the vector+BM25+metadata pass, re-rank the top-N
        candidates with a cross-encoder (BAAI/bge-reranker-base by
        default) against the ORIGINAL sub-query text (not the
        HyDE-augmented version).  Cross-encoders read query and
        chunk together and produce a single relevance score, which
        sharply down-ranks "empty section header" chunks that have
        high keyword density but no substantive content.
    """
    from bl_pipeline.shared.config import hyde_enabled_for, subquery_reranker_enabled

    # HyDE runs only on the diagnostic Phase-2 paths, NOT on this forward
    # A1 retrieval — config.hyde_enabled_for("a1_forward") is False by
    # default (a single BL_RAG_USE_HYDE override can force it globally).
    use_hyde = hyde_enabled_for("a1_forward")
    # DISTINCT experimental PER-SUB-QUERY reranker — SEPARATE from the
    # canonical A1 pool reranker (agent1_fresh/retrieve.py, governed by
    # config.reranker_enabled / BL_RAG_USE_RERANKER).  Kept on its OWN
    # opt-in flag so BL_RAG_USE_RERANKER controls only the canonical
    # reranker and can never double-rerank via this path.
    use_reranker = subquery_reranker_enabled()

    # HyDE pre-pass: generate a hypothetical-answer paragraph and
    # augment the sub-query text used for VECTOR search only.  The
    # original `expanded.text` continues to drive BM25 (literal token
    # matching) and reranker (judging relevance to the actual
    # question, not the hallucinated answer).
    vector_query_text = expanded.text
    hyde_used = False
    if use_hyde:
        try:
            from bl_pipeline.rag.hyde import (
                augment_query, generate_hypothetical_answer,
            )
            hypothetical, _ = generate_hypothetical_answer(expanded.text)
            if hypothetical:
                vector_query_text = augment_query(expanded.text, hypothetical)
                hyde_used = True
        except Exception as e:
            print(f"[WeightedRetriever] HyDE skipped — {e}", flush=True)

    all_candidates: list[ScoredChunk] = []

    for col_name in expanded.target_cols:
        # ── Vector search ─────────────────────────────────────────
        try:
            col = rag_engine.get_or_create_collection(col_name)
            vec_res = col.query(query_texts=[vector_query_text],
                                 n_results=vector_top_k)
        except Exception as e:
            print(f"[WeightedRetriever] vector query failed for {col_name}: {e}")
            continue

        vec_ids   = (vec_res.get("ids")       or [[]])[0]
        vec_docs  = (vec_res.get("documents") or [[]])[0]
        vec_metas = (vec_res.get("metadatas") or [[]])[0]
        vec_dists = (vec_res.get("distances") or [[]])[0]

        # Build lookup of what the vector search returned
        by_id: dict[str, dict[str, Any]] = {}
        for cid, doc, md, dist in zip(vec_ids, vec_docs, vec_metas, vec_dists):
            by_id[cid] = {
                "text": doc or "",
                "metadata": md or {},
                "vec_distance": float(dist),
            }

        # ── BM25 scores across the whole collection ───────────────
        # The engine returns {doc_id: normalised_bm25_score}.
        try:
            bm25_scores = rag_engine.bm25_scores_for_collection(
                col_name, expanded.text
            )
        except Exception as e:
            print(f"[WeightedRetriever] BM25 score fetch failed for {col_name}: {e}")
            bm25_scores = {}

        # Pull chunks BM25 loved even if the vector search missed them.
        # This is the path that rescues naked-equation chunks — their
        # English embeddings are weak, but BM25 lights up on literal
        # tokens like "400", "Tu", "5/8".
        #
        # Threshold 0.15: empirically, BM25 max-normalises against the
        # top-scoring chunk in the collection, which squishes scores
        # for short equation chunks that genuinely match but have
        # fewer tokens than verbose prose. 0.15 still excludes the
        # lukewarm long tail while keeping real equation hits.
        missing_bm25_ids = [
            cid for cid, s in bm25_scores.items()
            if s >= 0.15 and cid not in by_id
        ]
        if missing_bm25_ids:
            try:
                extra = col.get(ids=missing_bm25_ids)
                for cid, doc, md in zip(
                    extra.get("ids") or [],
                    extra.get("documents") or [],
                    extra.get("metadatas") or [],
                ):
                    by_id[cid] = {
                        "text": doc or "",
                        "metadata": md or {},
                        # No vector distance -> treat as "weak": a neutral
                        # distance of 1.0 corresponds to sim_vec = 0.5,
                        # so fusion still takes BM25 into account without
                        # letting vector pretend it was strong.
                        "vec_distance": 1.0,
                    }
            except Exception as e:
                print(f"[WeightedRetriever] BM25-pullback fetch failed: {e}")

        # ── Fuse vector + BM25, then apply metadata weights ───────
        for cid, info in by_id.items():
            bm25_s = bm25_scores.get(cid, 0.0)
            fused_distance = _fuse_vector_bm25(info["vec_distance"], bm25_s)

            applied, adjusted, matched = _score_candidate(
                raw_distance=fused_distance,
                md=info["metadata"],
                weights=expanded.metadata_weights,
            )
            all_candidates.append(ScoredChunk(
                chunk_id=cid,
                text=info["text"],
                metadata=info["metadata"],
                collection=col_name,
                raw_distance=fused_distance,
                applied_weight=applied,
                adjusted_distance=adjusted,
                sub_query_text=expanded.text,
                matched_keys=matched,
            ))

    # Rerank by adjusted distance (smaller = better)
    all_candidates.sort(key=lambda c: c.adjusted_distance)

    # Cross-encoder rerank post-pass (opt-in via BL_RAG_USE_RERANKER).
    # Operates on the top-N hybrid candidates (3x the desired
    # final_top_k so the cross-encoder has room to promote/demote
    # without artificial caps).  Scoring is against the ORIGINAL
    # sub-query text, NOT the HyDE-augmented version — we want the
    # reranker to judge relevance to the actual question.
    if use_reranker and all_candidates:
        try:
            from bl_pipeline.rag.reranker import is_available, rerank
            if is_available():
                rerank_pool_size = max(final_top_k * 3, final_top_k)
                pool = all_candidates[:rerank_pool_size]
                # rerank() takes list[dict] with a `text` key; map
                # ScoredChunk -> dict for the cross-encoder call,
                # then map back preserving the order it returns.
                pool_dicts = [
                    {"text": c.text, "_chunk": c} for c in pool
                ]
                reranked = rerank(
                    query=expanded.text,
                    candidates=pool_dicts,
                    top_k=final_top_k,
                )
                # Pull the ScoredChunk objects back out in reranked
                # order (reranker preserves the dict; we stashed the
                # original chunk under `_chunk`).
                all_candidates = [d["_chunk"] for d in reranked]
                # Reranker already truncated to final_top_k.
                return all_candidates
        except Exception as e:
            print(f"[WeightedRetriever] reranker skipped — {e}", flush=True)

    return all_candidates[:final_top_k]


def retrieve_for_plan(
    rag_engine: Any,
    plan: list[ExpandedQuery],
    vector_top_k: int = 50,
    final_top_k_per_sub: int = 5,
    overall_top_k: int | None = None,
) -> list[ScoredChunk]:
    """Run every sub-query in `plan` and merge results, deduped.

    Dedup key is `(paper_id, page_number, chunk_id_tail)`. First
    occurrence wins. Within a sub-query the ranking comes from
    `adjusted_distance`; across sub-queries we preserve the order in
    which sub-queries appeared in `plan` (the expander's intent).

    If `overall_top_k` is set, the merged list is truncated globally.
    Otherwise every sub-query contributes up to `final_top_k_per_sub`.
    """
    seen: set[tuple] = set()
    merged: list[ScoredChunk] = []

    for eq in plan:
        chunks = retrieve_for_expanded(
            rag_engine=rag_engine,
            expanded=eq,
            vector_top_k=vector_top_k,
            final_top_k=final_top_k_per_sub,
        )
        for ch in chunks:
            md = ch.metadata or {}
            key = (
                md.get("paper_id", "?"),
                str(md.get("page_number", "?")),
                ch.chunk_id[-10:] if ch.chunk_id else "",
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(ch)

    if overall_top_k is not None:
        merged = merged[:overall_top_k]
    return merged


# ── Convenience: plan + retrieve in one call ────────────────────────


def expand_and_retrieve(
    rag_engine: Any,
    user_query: str,
    vector_top_k: int = 50,
    final_top_k_per_sub: int = 5,
    overall_top_k: int | None = None,
) -> tuple[list[ExpandedQuery], list[ScoredChunk]]:
    """End-to-end: Sonnet expands `user_query`, retriever executes.

    Returns the plan and the merged chunks so callers can audit both.
    On expansion failure (empty plan), falls back to using the raw
    user_query as a single sub-query spanning all primary collections.
    """
    from bl_pipeline.rag.query_expander import expand_query
    from bl_pipeline.rag.collections import PRIMARY_COLLECTIONS

    plan = expand_query(user_query)

    if not plan:
        # Defensive fallback: treat the user query as one sub-query
        # against all primary collections with no metadata weights.
        print("[WeightedRetriever] expansion failed, falling back to raw query")
        plan = [ExpandedQuery(
            text=user_query.strip(),
            target_cols=[c.name for c in PRIMARY_COLLECTIONS],
            metadata_weights={},
            rationale="fallback: raw user query, no expansion",
        )]

    chunks = retrieve_for_plan(
        rag_engine=rag_engine,
        plan=plan,
        vector_top_k=vector_top_k,
        final_top_k_per_sub=final_top_k_per_sub,
        overall_top_k=overall_top_k,
    )
    return plan, chunks
