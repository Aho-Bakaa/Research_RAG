"""retrieve.py — adaptive retrieval loop for fresh Agent 1.

Combines:
    • our 3-iteration loop + safety caps          (from agent1_v2/adaptive_retrieval.py)
    • cross-encoder rerank INSIDE the loop        (from Priyanshi/reranker.py)
    • 0-1 numeric scoring on chunks               (from Priyanshi/validator.py)
    • targeted supplementary queries on gaps      (replaces "re-expand whole query")

Per iteration:
    1. expand    — Sonnet generates 3-5 sub-queries via OPTIMIZER_SYSTEM
                   (or supplementary queries from gaps after iter 1).
    2. search    — BM25 + vector + metadata weighting (our weighted_retriever).
    3. merge     — dedup against the running pool.
    4. rerank    — bge-reranker-v2-m3 cross-encoder against ORIGINAL query.
    5. judge     — Sonnet labels each chunk 0-1 + 4-way + sufficient + gaps.
    6. prune     — drop "irrelevant" subject to top-5 immunity, 40% drop cap,
                   12-chunk floor.
    7. loop      — if !sufficient AND iter < max: supplementary queries from gaps,
                   back to step 2 (or skip step 1 since gaps drive queries directly).

Returns a `RetrievalResult` with the validated chunks + full iteration trace.

Cost (typical, 2-iter case):
    Sonnet × 1 (optimizer)
    + Sonnet × 2 (judge per iter)
    + Sonnet × 1 (supplementary, only when needed)
    + cross-encoder inference (local, ~0 USD)
    ≈ $0.04, 15-25 s wall time.
"""
from __future__ import annotations

import time
from typing import Any

from bl_pipeline.agent1_fresh.prompts import (
    JUDGE_SYSTEM,
    OPTIMIZER_SYSTEM,
    SUPPLEMENTARY_SYSTEM,
)
from bl_pipeline.agent1_fresh.state import (
    Chunk,
    IterationTrace,
    RetrievalResult,
    emit_event,
)
from bl_pipeline.rag.collections import ALL_COLLECTIONS
from bl_pipeline.rag.engine import RAGEngine
from bl_pipeline.rag.query_expander import ExpandedQuery
from bl_pipeline.rag.weighted_retriever import ScoredChunk, retrieve_for_plan
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


# Default collections we route every sub-query through.  The full set —
# the fresh Agent 1 doesn't pre-filter at retrieval time, it lets the
# downstream judge prune what isn't useful.  This is intentional:
# pre-filtering by collection is a hardcoded vocabulary in disguise.
_DEFAULT_TARGET_COLS = [c.name for c in ALL_COLLECTIONS]


# ══════════════════════════════════════════════════════════════════════
# Lazy-loaded cross-encoder (BGE reranker)
# ══════════════════════════════════════════════════════════════════════

_CROSS_ENCODER = None


class RerankerLoadError(RuntimeError):
    """Raised when the cross-encoder cannot be loaded while reranking is
    enabled — fail loud instead of silently degrading to a no-rerank,
    non-reproducible pool.  Propagates past the per-iteration handler."""


def _get_cross_encoder():
    """Load the reranker cross-encoder on first use; cache the instance.

    Model + max_length come from the SINGLE config source
    (config.RERANKER_MODEL / RERANKER_MAX_LENGTH), so A1, A2 and A3 all
    rerank with the same model.  First call may take 5-10 s (model
    download or load from the HF cache); later calls return instantly.

    Raises RerankerLoadError if the model cannot be loaded — the operator
    should install sentence-transformers (+ torch) or set
    BL_RAG_USE_RERANKER=0 to run a deterministic no-rerank baseline.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from bl_pipeline.shared.config import (
            RERANKER_MODEL, RERANKER_MAX_LENGTH,
        )
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder(
                RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH,
            )
        except Exception as exc:
            raise RerankerLoadError(
                f"cross-encoder reranker {RERANKER_MODEL!r} failed to load: "
                f"{type(exc).__name__}: {exc}. Install sentence-transformers "
                f"(+ torch), or set BL_RAG_USE_RERANKER=0 to run without "
                f"reranking."
            ) from exc
    return _CROSS_ENCODER


# ══════════════════════════════════════════════════════════════════════
# Stage 1 — sub-query generation
# ══════════════════════════════════════════════════════════════════════

def _expand_query(
    user_query: str,
    flow_context: str = "",
) -> tuple[list[str], float]:
    """Generate 5-7 inventory-aware sub-queries from the user's question.

    The OPTIMIZER now sees the paper inventory (compact summaries of every
    paper in the corpus) and emits paper-targeted queries: each subsequent
    query cites a specific paper_id + equation label when the inventory
    shows that paper has what's needed.  The LLM also emits an
    `inventory_partition` block (anchors / critics / comparisons /
    excluded) which is logged for downstream auditing but not yet
    threaded into PLANNER's user_msg (PLANNER already gets the full
    inventory via plan.py:108).
    """
    # Load the paper inventory.  Identical to what PLANNER/REASONER already
    # see, so the OPTIMIZER's worldview matches the rest of A1.
    from bl_pipeline.agent1_fresh.paper_inventory import (
        load_all_paper_summaries,
        format_paper_inventory_for_prompt,
    )
    summaries = load_all_paper_summaries()
    inventory_text = (
        format_paper_inventory_for_prompt(summaries) if summaries else
        "(no paper inventory available — write generic queries)"
    )

    # FLOW_CONTEXT: tells OPTIMIZER which optional facility inputs the user
    # provided via the sidebar (grid spec, Lambda_x, Tu(x) measurements).
    # Without this, OPTIMIZER only sees the NL query — which may not mention
    # Lambda_x even when the user filled in grid spec → FS20 wrongly excluded.
    flow_block = ""
    if flow_context:
        flow_block = (
            f'FLOW_CONTEXT (structured inputs the user provided via the '
            f'sidebar — use this to decide which papers\' required_inputs '
            f'are satisfiable):\n{flow_context}\n\n'
        )

    user_msg = (
        f'USER_QUERY: "{user_query}"\n\n'
        f'{flow_block}'
        f'INVENTORY (scan this before writing queries — partition the '
        f'corpus into anchors / critics / comparisons / excluded, then '
        f'cite specific paper_ids in your queries):\n\n'
        f'{inventory_text}\n\n'
        f'Generate the inventory-aware search queries per the strategy.'
    )

    response, usage = llm.call(
        task="agent1_fresh_optimize",   # → Sonnet
        system=OPTIMIZER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=2500,                # bumped from 500: queries + inventory_partition
        temperature=0.0,
    )
    parsed = parse_lenient_json(response)
    # Tolerate three shapes from the LLM (same robustness as _judge):
    #   • Canonical:  {"search_queries": [...], "inventory_partition": {...}}
    #   • Bare list:  [...]  — Sonnet sometimes returns just the list
    #   • None / non-dict: defensive default to a single-query plan
    if isinstance(parsed, list):
        queries_str = [q for q in parsed if isinstance(q, str) and q.strip()]
        parsed = {"search_queries": queries_str if queries_str else [user_query]}
    elif not isinstance(parsed, dict):
        parsed = {}
    raw_queries = parsed.get("search_queries") or [user_query]
    if not isinstance(raw_queries, list):
        raw_queries = [user_query]
    queries = [q for q in raw_queries if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [user_query]
    # Ensure the original query is always first.
    if queries[0] != user_query:
        queries = [user_query] + [q for q in queries if q != user_query]

    # Emit inventory_partition as an event for auditing (PLANNER doesn't
    # consume it yet — PLANNER already has the full inventory via plan.py).
    partition = parsed.get("inventory_partition") or {}

    # ── FS20 MANDATORY-ANCHOR ENFORCEMENT ─────────────────────────────
    #
    # When the user's FLOW_CONTEXT exposes a known (or derivable) Λ_x,
    # fransson_shahinfar_2020 MUST be in `anchors` — see OPTIMIZER prompt
    # "MANDATORY ANCHORS" block.  The 2026-06-03 hardcutover_135 run
    # exposed the OLD planner silently dropping FS20 and using only
    # Tu-based correlations (AGS, Mayle) despite a complete grid spec.
    # This code is the safety net: if FS20 is still missing after the
    # prompt-level mandate, we:
    #   1. Force-inject a FS20-targeted search query so its chunks are
    #      retrieved into the pool regardless.
    #   2. Emit `fs20_mandate_violation` so the violation is auditable
    #      (and visible in the frontend trace).
    #
    # Source for the rule: Fransson & Shahinfar 2020 Eq.(3.5) / (3.6) —
    # Re_FST = Tu · Re_Λ as the onset variable; without FS20 the planner
    # discards the user's measured Λ_x.  Jonáš et al (2000) showed
    # empirically that Λ_x advances transition onset.
    _flow_lc = (flow_context or "").lower()
    _lambda_known = (
        "lambda_x" in _flow_lc
        or "grid spec" in _flow_lc
        or "grid_solidity" in _flow_lc
        or "integral length" in _flow_lc
        or "lambda_x_with_provenance" in _flow_lc
        or "fransson decay" in _flow_lc
    )
    anchors_list = partition.get("anchors") or []
    _fs20_in_anchors = any(
        "fransson_shahinfar_2020" in str(a).lower() for a in anchors_list
    )
    if _lambda_known and not _fs20_in_anchors:
        # Safety net 1: inject a FS20 query so its chunks land in the pool
        _fs20_query = (
            "fransson_shahinfar_2020 Eq.3.6 Re_FST Re_Lambda transition "
            "onset Lambda_x-aware"
        )
        if _fs20_query not in queries:
            queries.append(_fs20_query)

        # Safety net 2: audit-trail event (lets reviewers see this fired)
        try:
            emit_event(
                "fs20_mandate_violation",
                reason=(
                    "Λ_x is known via FLOW_CONTEXT but fransson_shahinfar_2020 "
                    "is missing from anchors.  Auto-injected FS20 query into "
                    "the search list as a safety net.  This indicates the "
                    "planner did not follow the OPTIMIZER prompt's "
                    "MANDATORY ANCHORS rule."
                ),
                planner_anchors=anchors_list,
                injected_query=_fs20_query,
            )
        except Exception:
            pass

        # Force FS20 into the anchors list for downstream consumers that
        # read partition["anchors"] (e.g. coverage scoring, frontend).
        partition["anchors"] = list(anchors_list) + ["fransson_shahinfar_2020"]

    # ── A1 LOAD-BEARING ANCHOR SAFETY NETS ─────────────────────────────
    #
    # Run 59c123f7 (2026-06-13) emitted only 2 iter-1 sub-queries — one
    # was the entire multi-paragraph user query verbatim, the other was
    # a single FS20 string.  Iter-1 judge marked INSUFFICIENT, the same
    # anchors got rediscovered in iter 2 and iter 3, costing $0.21 in
    # redundant retrieval before iter 4 would have also fired.
    #
    # The fix is data-driven, NOT a hardcoded paper list (cf. task #135's
    # rule against hardcoded paper choices).  Every paper in the inventory
    # has an `x_t_contribution` role.  A1's load-bearing roles are:
    #   ONSET_CORRELATION  — empirical Re_θt(Tu) formulas
    #   INTERMITTENCY      — transition zone length (Narasimha, etc.)
    #   DECAY_LAW          — Tu(x) decay for grid-derived Tu
    # We inject one query per paper in each role (excluding papers tagged
    # DEFERRED_TO_CFD or in the small manual A1 exclusion list — see
    # _out_of_domain_paper_ids), drawn from the inventory's
    # `how_to_use_for_x_t` field for query phrasing.  If a paper is
    # already covered by an LLM-emitted query (substring match on
    # paper_id), we skip.
    #
    # Blasius θ(x) is added as a non-paper "formula anchor" because
    # there's no Blasius paper in the corpus — it's an algebraic identity.
    _injected: list[str] = []
    _existing_query_blob = " ".join(queries).lower()
    try:
        for _pid, _q in _load_bearing_anchor_queries():
            if _pid.lower() in _existing_query_blob:
                continue
            queries.append(_q)
            _injected.append(_pid)
    except Exception:
        pass   # safety net failure shouldn't break retrieval
    if _injected:
        try:
            emit_event(
                "optimizer_anchor_safety_net_fired",
                injected_anchors=_injected,
                source=("inventory:x_t_contribution in "
                        "{ONSET_CORRELATION,INTERMITTENCY,DECAY_LAW}"),
                reason=("A1 load-bearing anchors not covered by optimizer "
                        "output — injected from inventory so the judge "
                        "doesn't have to gap-trigger them"),
                queries_before=len(queries) - len(_injected),
                queries_after=len(queries),
            )
        except Exception:
            pass

    if partition:
        try:
            emit_event(
                "optimizer_inventory_partition",
                anchors=partition.get("anchors") or [],
                critics=partition.get("critics") or [],
                comparisons=partition.get("comparisons") or [],
                excluded=partition.get("excluded") or [],
                n_queries=len(queries),
            )
        except Exception:
            pass   # logging is best-effort; never break retrieval

    # Cap at 12 queries (was 8) — anchor safety nets above can inject
    # up to 5 load-bearing queries on top of the LLM optimizer's 5-7,
    # plus FS20's safety net = potentially 13 total; cap pulls it back
    # so the hybrid search call has a bounded sub-query count.
    return queries[:12], float(getattr(usage, "cost_usd", 0.0) or 0.0)


def _generate_supplementary(
    user_query: str, gaps: list[str],
) -> tuple[list[str], float]:
    """Convert specific gaps into up to 5 narrow search queries.

    The judge typically flags 3-7 distinct gaps after iteration 1 (e.g.
    missing AGS Eq.18, Fransson L_ref, Suzen-Huang γ-transport, Tu-decay
    benchmark, …).  A previous cap of 2 silently dropped most of them and
    left iter 2/3 doing tiny gap-fills while the actual evidence pool kept
    missing pieces the reasoner needed.  Cap of 5 keeps the supplementary
    pass tightly scoped (each query still targets ONE gap) without
    starving downstream stages of evidence.
    """
    if not gaps:
        return [], 0.0
    user_msg = (
        f'USER_QUERY: "{user_query}"\n\n'
        f'GAPS:\n' + "\n".join(f"  - {g}" for g in gaps[:8])
    )
    response, usage = llm.call(
        task="agent1_fresh_supplementary",   # → Sonnet
        system=SUPPLEMENTARY_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=400,
        temperature=0.0,
    )
    parsed = parse_lenient_json(response)
    if not isinstance(parsed, list):
        return [], float(getattr(usage, "cost_usd", 0.0) or 0.0)
    queries = [str(q) for q in parsed if isinstance(q, str)][:5]
    return queries, float(getattr(usage, "cost_usd", 0.0) or 0.0)


# ══════════════════════════════════════════════════════════════════════
# Stage 2 — hybrid search via our existing weighted retriever
# ══════════════════════════════════════════════════════════════════════

def _hybrid_search(
    rag: RAGEngine,
    sub_queries: list[str],
    *,
    top_k_per_query: int = 12,
) -> list[Chunk]:
    """Run hybrid search across sub-queries; return as Chunk dataclass list.

    Wraps each sub-query string in an ExpandedQuery with default targeting
    (all collections, no metadata weighting).  The judge prunes — not us.

    REVERTED 2026-05-16 to the single-batched-call form that worked in
    run #3.  Earlier attempts to add per-query timing + per-query
    timeout via ThreadPoolExecutor appear to have INTRODUCED the
    iter-3 hang we were trying to diagnose — likely by:
      (a) splitting one batched retrieve_for_plan call into N separate
          calls, multiplying any state-overhead in Chroma/Nemotron,
      (b) creating + tearing down a ThreadPoolExecutor per query,
      (c) emitting many extra events through the event_bus.
    Run #3 used the simple form below and completed cleanly.
    """
    plan = [
        ExpandedQuery(
            text=q,
            target_cols=_DEFAULT_TARGET_COLS,
            metadata_weights={},
            rationale="fresh-agent sub-query",
        )
        for q in sub_queries
    ]
    scored = retrieve_for_plan(
        rag,
        plan=plan,
        final_top_k_per_sub=top_k_per_query,
    )
    return [_scored_to_chunk(sc) for sc in scored]


def _scored_to_chunk(sc: ScoredChunk) -> Chunk:
    md = sc.metadata or {}
    return Chunk(
        chunk_id=sc.chunk_id,
        paper_id=str(md.get("paper_id", "?")),
        page=md.get("page_number", "?"),
        content=sc.text,
        collection=sc.collection or "",
        # rerank_score / score / label / reason get filled later
    )


def _norm_content(s: str | None) -> str:
    """Whitespace-insensitive, case-folded form of a chunk's text — the key
    for content-level dedup."""
    return " ".join((s or "").split()).lower()


def _merge_pool(existing: list[Chunk], new: list[Chunk]) -> tuple[list[Chunk], int]:
    """Dedup new chunks into the pool by chunk_id AND by normalized content.

    Content dedup is required because the SAME source passage is cross-tagged
    into multiple RAG collections at ingest time, producing chunks with
    DIFFERENT chunk_ids (the collection name is baked into the id) but byte-
    identical text — e.g. ``mayle_1991_algebraic_onset_..._p010_75`` and
    ``mayle_1991_algebraic_intermittency_..._p010_75``.  ID-only dedup let
    those repeats through, so the reasoner saw the same paragraph 2-3x and the
    judge paid tokens scoring identical text.  First occurrence wins.
    """
    seen_ids = {c.chunk_id for c in existing}
    seen_content = {_norm_content(c.content) for c in existing}
    out = list(existing)
    n_new = 0
    for c in new:
        ch = _norm_content(c.content)
        if c.chunk_id in seen_ids or (ch and ch in seen_content):
            continue
        out.append(c)
        seen_ids.add(c.chunk_id)
        seen_content.add(ch)
        n_new += 1
    return out, n_new


# ══════════════════════════════════════════════════════════════════════
# Stage 4 — cross-encoder rerank (against the ORIGINAL query)
# ══════════════════════════════════════════════════════════════════════

# A1 is empirical-correlations-only; CFD transport-model papers are
# A2's territory (task #171, #240).  The paper inventory already tags
# every paper with an `x_t_contribution` role — papers tagged
# `DEFERRED_TO_CFD` are exactly the ones A1 must NOT cite.  We use that
# tag as the source of truth (NO hardcoded paper_id list — when a new
# CFD paper enters the corpus, tag it correctly in its summary record
# and this demote takes effect automatically; cf. task #135's
# architectural rule against hardcoding).
_A1_DEMOTE_FACTOR = 0.1
_A1_OUT_OF_DOMAIN_ROLES = frozenset({"DEFERRED_TO_CFD"})

# ── Content-signal patterns (tasks #256 + #257, added 2026-06-13) ─────
# Run 76d5b351 and 5975d774 both showed the same top-10 dominated by
# long descriptive prose chunks AND bibliography pages.  The cross-
# encoder over-weighted them because they're keyword-dense; the actual
# equation chunks (Mayle Eq.9, AGS Eq.3, FS20 Eq.3.6, Narasimha L_tr,
# Blasius θ(x)) were all in the pool but ranked below the top 10.
#
# Cheap signal-based pre-rerank rescoring fixes this without retraining
# the cross-encoder.  Applied AFTER the cross-encoder produces its base
# scores, BEFORE the sort.
import re as _re

# Tokens that indicate equation content (math notation in plain-text
# chunks: Reynolds-number variables, Greek letters, formula operators).
# Tightened 2026-06-13 after first smoke test showed years in
# bibliography pages (2000, 1986, 1998...) and digits in setup prose
# (0.5, 1.5, 2m...) falsely registered as math content.
_EQ_TOKENS = _re.compile(
    r"(?:"
    r"\bRe_\w+|"              # Re_theta_t, Re_x, Re_lambda, Re_FST, R_XE
    r"\\theta\b|\\Theta\b|"
    r"\\Lambda\b|\\nu\b|"
    r"\\sigma\b|\\Sigma\b|"
    r"\\gamma\b|\\delta\b|"
    r"\btheta_t\b|\btheta_S\b|"
    r"\bTu\s*\*\*|Tu\^|"      # `Tu**(-5/8)`, `Tu^(-5/8)`
    r"\bsqrt\s*\(|\\sqrt\b|"
    r"\bexp\s*\(|\\exp\b|"
    r"\\cdot\b|\\times\b|"
    r"\\frac\b|"
    r"\\Eq\b|\bEq\.\s*\("
    r")"
)
# Reference-list markers — patterns that appear ONLY in bibliographies,
# not in main paper body.  Multiple journal abbreviations + et al. +
# DOI patterns close together = a references page.
_REF_TOKENS = _re.compile(
    r"(?:"
    r"\bet al\.?\s*[,\(]|"
    r"\bet al\.\s+\d{4}|"     # "et al. 2018"
    r"\bJ\.\s*Fluid\s*Mech\.|"
    r"\bAIAA\s*J\.|AIAA\s+J\b|"
    r"\bPhys\.\s*Fluids|"
    r"\bAnnu\.\s*Rev\.|"
    r"\bExp\.\s*Fluids|"
    r"\bIntl?\.\s*J\.|"
    r"\bdoi[:\s]*10\."        # DOI line markers
    r")"
)
# Heuristic thresholds (tuned against the chunks that misranked in the
# 2026-06-13 runs — matsubara_alfredsson_2001 p6, kurian_fransson_2009
# p32, comte_bellot_corrsin_1966 p26, etc.).
_EQ_DENSITY_MIN_BOOST   = 8     # ≥8 equation tokens → start boosting
_EQ_DENSITY_FULL_BOOST  = 20    # ≥20 → full 1.4× boost
_REF_DENSITY_MIN_DEMOTE = 6     # ≥6 ref-list markers → start demoting
_REF_DENSITY_FULL_DEMOTE = 15   # ≥15 → full 0.5× demote


def _content_signal_factor(content: str) -> tuple[float, int, int]:
    """Compute a multiplicative rescore factor for chunk content.

    Returns (factor, n_eq_tokens, n_ref_tokens).

    Boost (factor > 1.0) when the chunk is equation-dense; demote
    (factor < 1.0) when it's a reference-list / bibliography page.
    Linearly interpolate between thresholds so a chunk with mixed
    content lands somewhere in the middle.  When both signals fire,
    they multiply (a chunk that has equations AND references gets
    near-neutral rescaling).
    """
    if not content:
        return 1.0, 0, 0
    text = content[:2000]   # bound the scan
    n_eq  = len(_EQ_TOKENS.findall(text))
    n_ref = len(_REF_TOKENS.findall(text))

    # Equation-density boost — linear ramp from 1.0 → 1.4
    if n_eq >= _EQ_DENSITY_FULL_BOOST:
        boost = 1.4
    elif n_eq >= _EQ_DENSITY_MIN_BOOST:
        ratio = (n_eq - _EQ_DENSITY_MIN_BOOST) / (
            _EQ_DENSITY_FULL_BOOST - _EQ_DENSITY_MIN_BOOST
        )
        boost = 1.0 + 0.4 * ratio
    else:
        boost = 1.0

    # Bibliography-page demote — linear ramp from 1.0 → 0.5,
    # ONLY when equation density is also low (the page is mostly
    # references, not a paper that happens to cite many others).
    if n_eq < _EQ_DENSITY_MIN_BOOST:
        if n_ref >= _REF_DENSITY_FULL_DEMOTE:
            demote = 0.5
        elif n_ref >= _REF_DENSITY_MIN_DEMOTE:
            ratio = (n_ref - _REF_DENSITY_MIN_DEMOTE) / (
                _REF_DENSITY_FULL_DEMOTE - _REF_DENSITY_MIN_DEMOTE
            )
            demote = 1.0 - 0.5 * ratio
        else:
            demote = 1.0
    else:
        demote = 1.0

    return boost * demote, n_eq, n_ref


def _out_of_domain_paper_ids() -> frozenset[str]:
    """Return inventory paper_ids whose x_t_contribution role is in the
    A1-out-of-domain set.  Cached after first call.
    """
    global _OUT_OF_DOMAIN_CACHE
    if _OUT_OF_DOMAIN_CACHE is not None:
        return _OUT_OF_DOMAIN_CACHE
    try:
        from bl_pipeline.agent1_fresh.paper_inventory import (
            load_all_paper_summaries,
        )
        summaries = load_all_paper_summaries() or []
    except Exception:
        summaries = []
    pids: set[str] = set()
    for s in summaries:
        pid = (s.get("paper_id") or "").strip()
        rec = s.get("summary_record") or s
        role = (rec.get("x_t_contribution") or "").strip().upper()
        if pid and role in _A1_OUT_OF_DOMAIN_ROLES:
            pids.add(pid)
    _OUT_OF_DOMAIN_CACHE = frozenset(pids)
    return _OUT_OF_DOMAIN_CACHE


_OUT_OF_DOMAIN_CACHE: frozenset[str] | None = None


# Roles A1 reads paper from in iter-1 anchor coverage.
_A1_ANCHOR_ROLES = ("ONSET_CORRELATION", "INTERMITTENCY", "DECAY_LAW")


def _load_bearing_anchor_queries() -> list[tuple[str, str]]:
    """Return [(paper_id, search_query_string), ...] for A1's load-bearing
    anchors — drawn entirely from the paper inventory, filtered by role.

    For each paper whose `x_t_contribution` is in `_A1_ANCHOR_ROLES`
    (and which isn't tagged DEFERRED_TO_CFD), we build a cross-encoder-
    friendly query string from `paper_id` + the head clause of
    `how_to_use_for_x_t` (first sentence / first semicolon split, capped
    at ~200 chars).  Blasius θ(x) is appended as a non-paper "formula
    anchor" because there's no Blasius PDF in the corpus.
    """
    try:
        from bl_pipeline.agent1_fresh.paper_inventory import (
            load_all_paper_summaries,
        )
        summaries = load_all_paper_summaries() or []
    except Exception:
        summaries = []
    out_of_domain = _out_of_domain_paper_ids()
    out: list[tuple[str, str]] = []
    for s in summaries:
        pid = (s.get("paper_id") or "").strip()
        rec = s.get("summary_record") or s
        role = (rec.get("x_t_contribution") or "").strip().upper()
        if not pid or role not in _A1_ANCHOR_ROLES:
            continue
        if pid in out_of_domain:
            continue
        how = (rec.get("how_to_use_for_x_t") or "").strip()
        if not how:
            head = role.replace("_", " ").lower()
        else:
            head = how.split(";")[0].split(".")[0][:200].strip()
        out.append((pid, f"{pid} {head}"))
    # Blasius θ(x) is an algebraic identity used by every onset
    # correlation to convert Re_θt → x_t.  Not in the inventory.
    out.append((
        "blasius",
        "Blasius flat plate laminar boundary layer theta(x) = "
        "0.664 sqrt(nu*x/U_inf) momentum thickness",
    ))
    return out


def _rerank(user_query: str, pool: list[Chunk], top_k: int = 30) -> list[Chunk]:
    """Cross-encoder rerank against the user's original query.

    Mutates each chunk's `rerank_score` and returns the top-k sorted by it.
    Skips if pool is empty (no work to do).

    Post-scoring, chunks from CFD-model papers (out-of-domain for A1) get
    their score multiplied by _A1_DEMOTE_FACTOR — see _A1_OUT_OF_DOMAIN_
    PAPERS.  Observed in run 59c123f7 (2026-06-13): langtry_menter_2009 p6
    top-ranked at 0.958 across every iteration despite A1 having no use
    for transport-model equations; cost-burned three iterations of judge
    calls to keep telling us "highly_relevant" on chunks A1 mustn't cite.
    """
    if not pool:
        return []
    ce = _get_cross_encoder()
    pairs = [(user_query, c.content[:1500]) for c in pool]
    scores = ce.predict(pairs, show_progress_bar=False)
    out_of_domain = _out_of_domain_paper_ids()
    n_demoted = 0
    n_eq_boosted = 0
    n_ref_demoted = 0
    eq_boost_max = 1.0
    ref_demote_min = 1.0
    for c, s in zip(pool, scores):
        raw = float(s)
        # Layer 1: DEFERRED_TO_CFD paper demote (task #243)
        if c.paper_id in out_of_domain:
            raw = raw * _A1_DEMOTE_FACTOR
            n_demoted += 1
        # Layer 2: content-signal rescore (tasks #256 + #257, 2026-06-13)
        # Boost equation-dense chunks (eq tokens >= 8) → ×1.0-1.4.
        # Demote bibliography pages (ref tokens >= 6 with low eq) → ×0.5-1.0.
        # See _content_signal_factor docstring for thresholds.
        sig_factor, n_eq, n_ref = _content_signal_factor(c.content or "")
        if sig_factor > 1.0:
            n_eq_boosted += 1
            if sig_factor > eq_boost_max:
                eq_boost_max = sig_factor
        elif sig_factor < 1.0:
            n_ref_demoted += 1
            if sig_factor < ref_demote_min:
                ref_demote_min = sig_factor
        c.rerank_score = raw * sig_factor
    pool.sort(key=lambda c: c.rerank_score, reverse=True)
    if n_eq_boosted or n_ref_demoted:
        try:
            emit_event(
                "rerank_content_signal_applied",
                n_eq_boosted=n_eq_boosted,
                n_ref_demoted=n_ref_demoted,
                eq_boost_max=round(eq_boost_max, 3),
                ref_demote_min=round(ref_demote_min, 3),
            )
        except Exception:
            pass
    if n_demoted:
        try:
            emit_event(
                "rerank_a1_demote_applied",
                n_chunks_demoted=n_demoted,
                demoted_paper_ids=sorted({
                    c.paper_id for c in pool
                    if c.paper_id in out_of_domain
                }),
                demote_factor=_A1_DEMOTE_FACTOR,
                source="inventory:x_t_contribution=DEFERRED_TO_CFD",
            )
        except Exception:
            pass
    return pool[:top_k]


# ══════════════════════════════════════════════════════════════════════
# Stage 5 — judge (0-1 scores + 4-way labels + sufficient + gaps)
# ══════════════════════════════════════════════════════════════════════

def _build_judge_user_message(user_query: str, pool: list[Chunk]) -> str:
    chunks_lines = []
    for c in pool:
        # Trim long content; keep enough for the judge to see the formula/structure
        body = (c.content or "").strip().replace("\n", " ")
        if len(body) > 600:
            body = body[:600] + "…[truncated]"
        chunks_lines.append(
            f"chunk_id: {c.chunk_id}  (paper={c.paper_id}, page={c.page}, "
            f"rerank={c.rerank_score:.3f})\n  {body}"
        )
    chunks_text = "\n\n".join(chunks_lines) if chunks_lines else "(empty pool)"
    return f'USER_QUERY: "{user_query}"\n\nCHUNK_POOL:\n{chunks_text}'


def _corpus_year_terms() -> set[str]:
    """Cached set of every 4-digit year that appears in any paper_id in
    the inventory.  Used by `_filter_gaps_by_corpus` to drop gap strings
    that demand papers from years no inventory paper covers — e.g. the
    judge in run 59c123f7 kept emitting "Roach 1987 Tu(x) = C·(x/d)^(-5/7)"
    as a gap every iteration even though no Roach 1987 summary exists in
    the inventory (its PDF was never summarised; task #244).  Without
    this check the gap-driven loop chases hallucinated papers forever
    while spending $0.07 per iteration on judge calls.
    """
    global _CORPUS_YEAR_TERMS_CACHE
    if _CORPUS_YEAR_TERMS_CACHE is not None:
        return _CORPUS_YEAR_TERMS_CACHE
    try:
        from bl_pipeline.agent1_fresh.paper_inventory import (
            load_all_paper_summaries,
        )
        summaries = load_all_paper_summaries() or []
    except Exception:
        summaries = []
    years: set[str] = set()
    import re
    for s in summaries:
        pid = (s.get('paper_id') or '').lower()
        for m in re.findall(r'(19\d{2}|20\d{2})', pid):
            years.add(m)
    _CORPUS_YEAR_TERMS_CACHE = years
    return years


_CORPUS_YEAR_TERMS_CACHE: set[str] | None = None


def _filter_gaps_by_corpus(gaps: list[str]) -> tuple[list[str], list[dict]]:
    """Drop gaps that name a year no inventory paper covers.

    Returns (kept, dropped_with_reasons).  A gap is dropped ONLY when it
    explicitly mentions a year (4-digit token like 1987, 2009) and that
    year doesn't appear in any inventory paper_id.  Gaps without a year
    are always kept — they may name a generic concept (e.g. "Blasius
    theta(x) formula") that the corpus has under a different paper.
    This is the lightest-touch corpus-presence check that still stops
    the runaway loop on confidently-hallucinated papers.
    """
    import re
    inventory_years = _corpus_year_terms()
    if not inventory_years:
        return gaps, []   # can't filter without inventory; fail open
    kept: list[str] = []
    dropped: list[dict] = []
    for g in gaps:
        years_in_gap = set(re.findall(r'\b(19\d{2}|20\d{2})\b', g))
        if not years_in_gap:
            kept.append(g)
            continue
        if years_in_gap & inventory_years:
            kept.append(g)
            continue
        dropped.append({
            "gap": g,
            "years_named": sorted(years_in_gap),
            "reason": "no inventory paper covers any of these years",
        })
    return kept, dropped


def _judge(
    user_query: str, pool: list[Chunk],
) -> tuple[dict[str, dict], bool, list[str], float]:
    """Returns (labels_by_chunk_id, sufficient, gaps, cost_usd).

    Tolerates two LLM output shapes:
      • Canonical:  {"chunks": [...], "sufficient": ..., "gaps": [...]}
      • Bare list:  [...]   ← Sonnet sometimes emits this when the
                              "chunks" wrapper feels redundant.  We
                              accept it and conservatively assume
                              sufficient=False + gaps=[] so the loop
                              can decide to iterate or stop on its own.

    Gaps are post-filtered against the corpus inventory via
    `_filter_gaps_by_corpus` — gaps that name a year no inventory paper
    has are dropped (with an audit event), preventing runaway gap loops
    on hallucinated references.
    """
    if not pool:
        return {}, False, ["no chunks retrieved"], 0.0
    user_msg = _build_judge_user_message(user_query, pool)
    # max_tokens=4000 (was 2000): scoring 30 chunks with score+label+reason
    # easily exceeds 2000 output tokens — Sonnet's response gets truncated
    # mid-JSON and parse_lenient_json returns None.  Bumped 2026-05-16
    # after run #7 raw preview showed valid JSON cut off at chunk #4 of 30.
    response, usage = llm.call(
        task="agent1_fresh_judge",       # → Sonnet
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=4000,
        temperature=0.0,
    )
    parsed = parse_lenient_json(response)
    cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)

    # Normalize to the canonical dict shape regardless of what the LLM
    # actually returned.  A bare list is treated as the chunks array.
    if isinstance(parsed, list):
        parsed = {"chunks": parsed, "sufficient": False, "gaps": []}
    elif not isinstance(parsed, dict):
        # None / number / string.  Most common cause we've observed: the
        # response IS valid JSON but was TRUNCATED at max_tokens, leaving
        # the parser to choke on incomplete syntax.  Distinguish the two
        # for clearer diagnostics — a response starting with `{` or `[`
        # was likely truncated; one starting with prose was a refusal.
        raw = (response or "")
        preview = raw[:300].replace("\n", " ")
        likely_truncated = raw.lstrip().startswith(("{", "["))
        cause = (
            "TRUNCATED — Sonnet's JSON was cut off (raise max_tokens)"
            if likely_truncated else
            "PROSE/REFUSAL — Sonnet returned no JSON object/array"
        )
        emit_event(
            "judge_parse_failed",
            raw_preview=preview,
            response_length=len(raw),
            likely_truncated=likely_truncated,
        )
        parsed = {
            "chunks": [], "sufficient": False,
            "gaps": [f"JUDGE_PARSE_FAILED ({cause}). "
                     f"Raw preview: {preview!r}"],
        }

    labels: dict[str, dict] = {}
    for entry in parsed.get("chunks", []) or []:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("chunk_id")
        if not cid:
            continue
        labels[str(cid)] = {
            "score": float(entry.get("score", 0.5)),
            "label": str(entry.get("label", "can_support")),
            "reason": str(entry.get("reason", ""))[:200],
        }
    sufficient = bool(parsed.get("sufficient", False))
    raw_gaps = [str(g) for g in (parsed.get("gaps") or []) if g]
    gaps, dropped = _filter_gaps_by_corpus(raw_gaps)
    if dropped:
        try:
            emit_event(
                "judge_gaps_dropped_corpus_missing",
                n_dropped=len(dropped),
                dropped=dropped[:10],
                kept_count=len(gaps),
            )
        except Exception:
            pass
    return labels, sufficient, gaps, cost


def _apply_judge(pool: list[Chunk], labels: dict[str, dict]) -> None:
    """Stamp judge scores/labels onto each chunk in place."""
    for c in pool:
        lab = labels.get(c.chunk_id)
        if lab:
            c.score = lab["score"]
            c.label = lab["label"]
            c.judge_reason = lab["reason"]


# ══════════════════════════════════════════════════════════════════════
# Stage 6 — prune with safety caps
# ══════════════════════════════════════════════════════════════════════

def _prune_with_safety_caps(
    pool: list[Chunk],
    *,
    chunk_floor: int = 12,
    max_drop_pct: float = 0.4,
    top_n_immune: int = 5,
) -> tuple[list[Chunk], list[str]]:
    """Drop "irrelevant" chunks subject to three caps:

    1. The top-`top_n_immune` chunks by rerank score are immune even if
       the judge labelled them irrelevant (defends against a bad judge).
    2. At most `max_drop_pct` of the pool is dropped per iteration.
    3. The pool never shrinks below `chunk_floor` chunks.
    """
    if not pool:
        return pool, []
    immune_ids = {c.chunk_id for c in pool[:top_n_immune]}
    irrelevant = [c for c in pool if c.label == "irrelevant" and c.chunk_id not in immune_ids]

    # Per-iter drop cap
    max_drop = int(len(pool) * max_drop_pct)
    if len(irrelevant) > max_drop:
        # Drop the lowest-scoring ones first
        irrelevant.sort(key=lambda c: c.rerank_score)
        irrelevant = irrelevant[:max_drop]

    # Floor — never go below the floor
    if len(pool) - len(irrelevant) < chunk_floor:
        # Keep enough to stay at floor
        n_must_keep = chunk_floor - (len(pool) - len(irrelevant))
        # Restore the highest-rerank chunks first
        irrelevant.sort(key=lambda c: c.rerank_score)
        irrelevant = irrelevant[:max(0, len(irrelevant) - n_must_keep)]

    drop_ids = {c.chunk_id for c in irrelevant}
    survivors = [c for c in pool if c.chunk_id not in drop_ids]
    return survivors, sorted(drop_ids)


# ══════════════════════════════════════════════════════════════════════
# Public entry — the adaptive retrieve loop
# ══════════════════════════════════════════════════════════════════════

def _build_flow_context_for_optimizer(flow) -> str:
    """Render the structured FlowConditions as compact text for OPTIMIZER.

    Only includes fields the user EXPLICITLY provided (skips defaults), so
    OPTIMIZER can identify which papers' required_inputs are satisfiable.
    Critically: surfaces lambda_x_provenance so OPTIMIZER knows Λ_x is
    DERIVABLE (via grid spec) even when not given directly.
    """
    if flow is None:
        return ""
    bits: list[str] = []
    if getattr(flow, "velocity_ms", None) is not None:
        bits.append(f"  • U_inf = {flow.velocity_ms} m/s")
    if getattr(flow, "turbulence_intensity_pct", None) is not None:
        bits.append(f"  • Tu_LE = {flow.turbulence_intensity_pct} % at leading edge")
    if getattr(flow, "kinematic_viscosity_m2s", None) is not None:
        bits.append(f"  • ν = {flow.kinematic_viscosity_m2s} m²/s")
    if getattr(flow, "chord_m", None) is not None:
        bits.append(f"  • chord = {flow.chord_m} m")
    if getattr(flow, "geometry_description", None):
        bits.append(f"  • geometry = {flow.geometry_description}")
    # Lambda_x availability — CRUCIAL for FS20 / Gonzalez selection.
    # When the source is the "engineering estimate" branch (grid spec
    # supplied but no direct Λ_x measurement), surface a strong signal
    # so the REASONER knows to re-derive via lookup_equation +
    # compute() rather than trust the pre-estimated number.
    try:
        lambda_info = flow.lambda_x_with_provenance()
        if lambda_info:
            mm = lambda_info["value_m"] * 1000
            src = lambda_info["source"]
            if lambda_info.get("reasoner_should_rederive"):
                bits.append(
                    f"  • Λ_x preview = {mm:.1f} mm  (PRE-ESTIMATE ONLY, "
                    f"source: {src}; REASONER MUST RE-DERIVE via "
                    f"lookup_equation('kurian_fransson_2009', 'Eq.(8)') + "
                    f"compute() with paper_id::Eq.(8) tag so the verifier "
                    f"can judge the math).  FS20/Gonzalez are SATISFIABLE."
                )
            else:
                bits.append(
                    f"  • Λ_x available = YES, {mm:.1f} mm "
                    f"(source: {src})"
                )
        else:
            bits.append("  • Λ_x available = NO (FS20 / Gonzalez NOT satisfiable)")
    except Exception:
        pass
    # Blasius VO inputs — when EITHER is present, PLANNER must dispatch
    # via dubey_2026_thesis::blasius_VO_Tu_effective_method (recipe
    # authored task #220, schema task #221, prompts updated 2026-06-13).
    # WITHOUT exposing these two fields here, the LLMs cannot see the
    # operator's input and the VO branch never fires — i.e. the sidebar
    # input becomes UI decoration, which is exactly the failure mode
    # the operator flagged.  Surface both with strong VO-trigger hints
    # so the OPTIMIZER asks for the right anchors AND the PLANNER's
    # virtual-origin-anchored branch fires.
    if getattr(flow, "x_0_BL_m", None) is not None:
        bits.append(
            f"  • x_0_BL_direct = {flow.x_0_BL_m * 1000:.1f} mm  "
            f"(user-supplied Blasius virtual origin DIRECT input; "
            f"BLASIUS-VO METHOD AVAILABLE — PLANNER MUST DISPATCH "
            f"via dubey_2026_thesis::blasius_VO_Tu_effective_method "
            f"and use Tu_effective = Tu(x_0_BL_m) for ALL three onset "
            f"correlations Mayle / AGS / FS20)"
        )
    _delta_meas = getattr(flow, "delta_x_measurements", None)
    if _delta_meas:
        try:
            _pairs_str = ", ".join(
                f"({x * 1000:.0f},{d * 1000:.2f})"
                for (x, d) in _delta_meas
            )
        except Exception:
            _pairs_str = "<unparseable>"
        bits.append(
            f"  • δ_99 measurements (x_mm, δ_99_mm) = [{_pairs_str}]  "
            f"({len(_delta_meas)} stations; BLASIUS-VO METHOD AVAILABLE "
            f"— PLANNER MUST FIT δ² = K·(x − x_0_BL) linearly to extract "
            f"x_0_BL_FIT, then dispatch via "
            f"dubey_2026_thesis::blasius_VO_Tu_effective_method and use "
            f"Tu_effective = Tu(x_0_BL_FIT) for ALL three onset "
            f"correlations Mayle / AGS / FS20)"
        )
    # Grid spec details
    if getattr(flow, "grid_solidity", None) is not None:
        bits.append(
            f"  • grid spec: σ={flow.grid_solidity}, "
            f"d_bar={flow.grid_bar_diameter_mm} mm, "
            f"x_grid={flow.grid_position_m} m"
        )
    # Decay parameters
    try:
        decay = flow.effective_fransson_decay()
        if decay:
            bits.append(
                f"  • Fransson decay: (C={decay.get('C'):.3g}, "
                f"x_0={decay.get('x_0_m'):.3g}, b={decay.get('b'):.3g}) "
                f"[source: {decay.get('source')}]"
            )
    except Exception:
        pass
    return "\n".join(bits) if bits else ""


def adaptive_retrieve(
    user_query: str,
    rag: RAGEngine | None = None,
    *,
    flow=None,                       # NEW — FlowConditions or None
    max_iterations: int = 3,
    top_k_per_query: int = 12,
    rerank_top_k: int = 30,
    chunk_floor: int = 12,
    max_drop_pct: float = 0.4,
) -> RetrievalResult:
    """Retrieve evidence for `user_query`, iterating until sufficient or capped.

    Returns a `RetrievalResult` with the validated chunks (sorted by judge
    score then rerank score) and the full iteration trace.

    Production-grade defaults:
        max_iterations    3 (caps cost in pathological cases)
        rerank_top_k     30 (judge sees a fixed-size window)
        chunk_floor      12 (downstream reasoner always has minimum context)
    """
    t0 = time.time()
    rag = rag or RAGEngine()

    result = RetrievalResult()
    pool: list[Chunk] = []
    cumulative_gaps: list[str] = []
    emit_event("retrieve_started", user_query=user_query, max_iterations=max_iterations)

    for it in range(1, max_iterations + 1):
        t_iter = time.time()
        trace = IterationTrace(iteration=it)
        emit_event("retrieve_iter_started", iteration=it)

        # Whole iteration wrapped so a per-iter error never destroys the
        # work already done in earlier iters.  We log + emit + break out
        # cleanly with the partial pool.
        #
        # Per-substep instrumentation added 2026-05-16 after 3 silent
        # hangs in iter 2+ (runs #4, #5, #6).  Without per-stage events
        # we couldn't tell WHICH of the 6 stages was hanging.  Every
        # stage now emits start + done with elapsed seconds, so the
        # live printer surfaces exactly where time is going.
        def _stage(name: str):
            """Context manager returning (timer_start, emit_done_fn)."""
            t_stage = time.time()
            emit_event("retrieve_substep_started",
                       iteration=it, stage=name)
            def _done(**extra):
                elapsed = round(time.time() - t_stage, 3)
                emit_event("retrieve_substep_done",
                           iteration=it, stage=name,
                           elapsed_s=elapsed, **extra)
                return elapsed
            return _done
        try:
            # ── Stage 1: queries for this iteration ───────────────────
            done = _stage("queries")
            cost_iter = 0.0
            if it == 1:
                flow_ctx = _build_flow_context_for_optimizer(flow)
                sub_queries, c = _expand_query(user_query, flow_context=flow_ctx)
                cost_iter += c
            else:
                # Use targeted supplementary queries from prior iter's gaps.
                # Stash the triggering gaps on the trace so the UI can render
                # "iter N supplementary — triggered by gaps from iter N-1: ..."
                trace.supplementary_trigger_gaps = list(cumulative_gaps)
                sub_queries, c = _generate_supplementary(user_query, cumulative_gaps)
                cost_iter += c
                if not sub_queries:
                    done(n_queries=0, cost_usd=c, note="no gaps to act on",
                         triggered_by_gaps=trace.supplementary_trigger_gaps)
                    trace.elapsed_s = round(time.time() - t_iter, 3)
                    trace.cost_usd = cost_iter
                    result.iterations.append(trace)
                    break
            done(n_queries=len(sub_queries), cost_usd=c,
                 queries_preview=[q[:80] for q in sub_queries[:3]],
                 triggered_by_gaps=trace.supplementary_trigger_gaps if it > 1 else [])
            trace.sub_queries = sub_queries

            # ── Stage 2: hybrid search (Nemotron embed + BM25 + Chroma) ─
            done = _stage("hybrid_search")
            new_chunks = _hybrid_search(rag, sub_queries, top_k_per_query=top_k_per_query)
            done(n_chunks=len(new_chunks))
            trace.raw_candidates = len(new_chunks)

            # ── Stage 3: merge into running pool ─────────────────────
            done = _stage("merge")
            pool, n_new = _merge_pool(pool, new_chunks)
            done(pool_size=len(pool), n_new=n_new)
            trace.pool_size_after_merge = len(pool)

            # ── Stage 4: cross-encoder rerank ─────────────────────────
            # Gated by the ONE canonical switch (config.reranker_enabled,
            # default ON; BL_RAG_USE_RERANKER=0 disables).  When disabled
            # we load NO model and fall back to hybrid-search order, capped
            # to the same top_k so the judge cost stays bounded.
            done = _stage("rerank")
            from bl_pipeline.shared.config import reranker_enabled
            if reranker_enabled():
                pool = _rerank(user_query, pool, top_k=rerank_top_k)
            else:
                pool.sort(key=lambda c: (c.score or 0.0), reverse=True)
                pool = pool[:rerank_top_k]
                emit_event("rerank_skipped",
                           reason="BL_RAG_USE_RERANKER disabled")
            # Build chunk preview AFTER rerank, BEFORE judge — top 10
            # by rerank_score so the UI shows what actually made the cut.
            top_after_rerank = [
                {
                    "chunk_id":     c.chunk_id,
                    "paper_id":     c.paper_id,
                    "page":         c.page,
                    "rerank_score": round(float(c.rerank_score or 0.0), 4),
                    "snippet":      (c.content or "")[:200],
                }
                for c in pool[:10]
            ]
            trace.chunks_after_rerank = top_after_rerank
            done(pool_size=len(pool), top_chunks_preview=top_after_rerank)
            trace.after_rerank = len(pool)

            # ── Stage 5: judge (Sonnet) ──────────────────────────────
            done = _stage("judge")
            labels, sufficient, gaps, c = _judge(user_query, pool)
            cost_iter += c
            _apply_judge(pool, labels)
            # Build the post-judge preview using the SAME chunk ordering
            # as chunks_after_rerank so the UI can render side-by-side
            # before/after columns naturally.
            chunks_with_verdicts = [
                {
                    "chunk_id":     c.chunk_id,
                    "paper_id":     c.paper_id,
                    "page":         c.page,
                    "rerank_score": round(float(c.rerank_score or 0.0), 4),
                    "snippet":      (c.content or "")[:200],
                    "label":        c.label,
                    "judge_score":  round(float(c.score or 0.0), 3),
                    "judge_reason": c.judge_reason,
                }
                for c in pool[:10]
            ]
            trace.chunks_after_judge = chunks_with_verdicts
            done(n_labels=len(labels), sufficient=sufficient,
                 n_gaps=len(gaps), cost_usd=c,
                 chunks_with_verdicts=chunks_with_verdicts)
            trace.sufficient = sufficient
            trace.gaps = gaps

            # ── Stage 6: prune with safety caps ──────────────────────
            done = _stage("prune")
            pool, dropped = _prune_with_safety_caps(
                pool, chunk_floor=chunk_floor, max_drop_pct=max_drop_pct,
            )
            done(pool_size=len(pool), n_dropped=dropped)
            trace.after_judge_prune = len(pool)

            # ── Bookkeeping ──────────────────────────────────────────
            trace.elapsed_s = round(time.time() - t_iter, 3)
            trace.cost_usd = round(cost_iter, 4)
            result.iterations.append(trace)
            result.total_cost_usd = round(result.total_cost_usd + cost_iter, 4)

            emit_event(
                "retrieve_iter_done",
                iteration=it,
                sub_queries=sub_queries,
                pool_size=len(pool),
                sufficient=sufficient,
                gaps=gaps,
                cost_usd=cost_iter,
                # NEW (2026-06-11) — chunk-level transparency for the UI.
                chunks_after_judge=trace.chunks_after_judge,
                supplementary_trigger_gaps=trace.supplementary_trigger_gaps,
            )

            # ── Loop control ─────────────────────────────────────────
            if sufficient:
                break
            cumulative_gaps = gaps  # next iter uses these
            # Supplementary queries fire at the top of the next iter.

        except RerankerLoadError:
            # Fail loud: a reranker-load failure must NOT be silently
            # swallowed into a truncated, non-reproducible pool.  Re-raise
            # past the generic per-iter handler so the run stops with a
            # clear, actionable message (install sentence-transformers or
            # set BL_RAG_USE_RERANKER=0).
            raise
        except Exception as e:
            # Per-iter crash: keep what we have, mark this iter as failed,
            # and bail out of the loop cleanly with the partial pool.
            trace.elapsed_s = round(time.time() - t_iter, 3)
            trace.cost_usd = round(cost_iter if "cost_iter" in dir() else 0.0, 4)
            trace.gaps = trace.gaps or [f"iter {it} crashed: {type(e).__name__}: {e}"]
            result.iterations.append(trace)
            result.total_cost_usd = round(
                result.total_cost_usd + trace.cost_usd, 4,
            )
            emit_event(
                "retrieve_iter_done",
                iteration=it,
                sub_queries=getattr(trace, "sub_queries", []),
                pool_size=len(pool),
                sufficient=False,
                gaps=trace.gaps,
                cost_usd=trace.cost_usd,
                error=f"{type(e).__name__}: {e}",
            )
            break

    # Sort final pool by judge score then rerank score (best on top)
    pool.sort(key=lambda c: (c.score, c.rerank_score), reverse=True)
    result.validated_chunks = pool
    result.final_pool_size = len(pool)
    result.n_iterations = len(result.iterations)
    result.sufficient = bool(result.iterations and result.iterations[-1].sufficient)
    result.total_elapsed_s = round(time.time() - t0, 3)

    # Emit the FINAL chunk set (post-prune, post-judge-merge across all
    # iterations) as part of retrieve_done so the UI can display the
    # exact pool that gets handed off to PLANNER + REASONER.  Without
    # this the user sees per-iter "Top N chunks" tables but has no view
    # of the consolidated pool the next stage actually receives.
    # Tagged with the same fields the per-iter tables show — paper_id,
    # page, rerank_score, judge score+label+reason, and a 300-char
    # content preview so the user can sanity-check what was kept.
    final_chunks_payload = [
        {
            "chunk_id":     c.chunk_id,
            "paper_id":     c.paper_id,
            "page":         c.page,
            "collection":   c.collection,
            "rerank_score": round(float(c.rerank_score or 0.0), 4),
            "judge_score":  round(float(c.score or 0.0), 4)
                            if c.score is not None else None,
            "judge_label":  c.label,
            "judge_reason": c.judge_reason,
            "preview":      (c.content or "")[:300],
        }
        for c in pool
    ]
    emit_event(
        "retrieve_done",
        final_pool_size=result.final_pool_size,
        n_iterations=result.n_iterations,
        sufficient=result.sufficient,
        total_cost_usd=result.total_cost_usd,
        total_elapsed_s=result.total_elapsed_s,
        final_chunks=final_chunks_payload,
    )
    return result
