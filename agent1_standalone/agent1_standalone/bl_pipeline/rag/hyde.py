"""hyde.py — Hypothetical Document Embeddings for query expansion.

Why HyDE
────────
Pure dense retrieval (vector + BM25) matches the literal query string
against chunks in the corpus. When query phrasing and chunk phrasing
diverge — e.g. query says "fails to predict" but chunks say
"deficiencies", "limitations", "erroneously switches", "tested by" —
embedding similarity scores are weak even when the chunks ARE the right
answer. Empirically observed in run 8f244a7f sq3: max rerank score 0.029
across 25 candidates, none containing the substantive failure-mode
content known to be in the corpus.

HyDE bridges this gap. Steps:
  1. Take each sub-query.
  2. Ask an LLM to write a short hypothetical answer to that sub-query —
     ~80-120 words, in the style of an academic paper paragraph.
  3. Concatenate sub-query + hypothetical answer.
  4. Use the combined string as the retrieval query (both vector and
     BM25 see it).
  5. Pass the ORIGINAL sub-query to the reranker — we want the reranker
     scoring against the actual question, not the hallucination.

The LLM-generated hypothetical answer naturally uses the academic
vocabulary ("deficiencies", "documented limitations", "fail to predict",
"tested by") that lives in real chunks. So the embedding similarity now
matches both the sub-query phrasing AND the corpus phrasing.

Cost
────
~80-120 output tokens per sub-query × 5 sub-queries × Sonnet pricing
≈ $0.008-$0.015 per pipeline run. Negligible relative to the planner LLM
call ($0.05+) and the run-time CFD solver.

Failure handling
────────────────
If the LLM call fails (network, rate-limit, etc.), fall back to the
original sub-query.

Enabling / disabling
────────────────────
HyDE is gated PER PHASE at the call site via
``config.hyde_enabled_for(phase)`` — ON by default only for the A1/A2
Phase-2 diagnostics, OFF for forward retrieval.  Set ``BL_RAG_USE_HYDE``
(1/0) as a GLOBAL override to force it on/off everywhere.  The old
``BL_DISABLE_HYDE`` switch is retired.

Risks
─────
If the hypothetical answer is wrong/hallucinated, retrieval can shift
toward chunks that match the hallucination rather than real evidence.
This is mitigated by:
  - Concatenating with the original sub-query (so original signal is
    preserved alongside the hallucinated vocabulary)
  - Reranking against the original sub-query (so the cross-encoder
    catches HyDE-driven false positives)
"""

from __future__ import annotations

import os
from typing import Any


_HYDE_SYSTEM = (
    "You are writing a short hypothetical answer to a research question, "
    "in the style of an academic paper paragraph. Your answer will NOT be "
    "shown to a user — it is used only as a vector-embedding bridge to "
    "retrieve the most relevant real paper chunks from a corpus.\n\n"
    "Write ONE paragraph, 80-120 words, using academic vocabulary that "
    "would appear in a real paper on the topic. Use canonical scientific "
    "phrasing across THREE registers depending on the question:\n"
    "  - DESCRIPTIVE failure: 'documented limitations', 'has been shown "
    "to underpredict', 'fails to capture', 'reported deficiencies', "
    "'reported sensitivity to'.\n"
    "  - PRESCRIPTIVE exclusion: 'is not recommended for', 'should not "
    "be applied to', 'is inappropriate for', 'is unsuitable for', "
    "'outside the validity range of'.\n"
    "  - APPLICABILITY ENVELOPE: 'validated for Tu in the range', "
    "'calibrated on flat-plate cases with', 'applicable when', "
    "'envelope of validity', 'regime of validity'.\n\n"
    "Cite hypothetical authors if it improves realism (e.g. 'Pacciani "
    "et al.', 'Walters and Cokljat'). Do NOT preface with 'Here is...' "
    "or 'The answer is...'. Just write the paragraph directly. Stay "
    "strictly factually plausible — do NOT invent numerical values, "
    "dataset names, or specific equations.")


def is_enabled() -> bool:
    """DEPRECATED — HyDE is now gated PER PHASE at the call site via
    ``config.hyde_enabled_for(phase)`` (default ON only for the A1/A2
    Phase-2 diagnostics).  The old global ``BL_DISABLE_HYDE`` switch is
    retired; use ``BL_RAG_USE_HYDE`` as a global override instead.  Kept
    as a thin shim for any external importer: reflects the global
    override only (unset → False, i.e. per-phase defaults apply)."""
    return os.environ.get("BL_RAG_USE_HYDE", "").strip().lower() in (
        "1", "true", "yes", "on")


def generate_hypothetical_answer(
    sub_query: str,
    *,
    pipeline_state: Any = None,
    max_tokens: int = 220,
) -> tuple[str, float]:
    """Generate a hypothetical answer paragraph for `sub_query`.

    Whether HyDE runs at all is decided by the CALLER via
    ``config.hyde_enabled_for(phase)`` — this function no longer
    self-gates, so a caller that reaches here has already opted in.
    Returns (hypothetical_answer, cost_usd).  On failure, returns
    ("", 0.0) and the caller is expected to fall back to the original
    sub_query.
    """
    try:
        from bl_pipeline.shared.llm_router import llm
    except ImportError:
        return "", 0.0

    user_msg = (
        f"Question: {sub_query}\n\n"
        "Write the hypothetical answer paragraph now."
    )

    try:
        raw, usage = llm.call(
            task="query_parsing",
            messages=[{"role": "user", "content": user_msg}],
            system=_HYDE_SYSTEM,
            max_tokens=max_tokens,
            pipeline_state=pipeline_state,
        )
    except Exception:
        return "", 0.0

    answer = (raw or "").strip()
    cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)

    # Stream the hypothetical doc to the live right-pane so the
    # operator can audit "what vocabulary did HyDE inject into the
    # retrieval query for this sub-query?".  This is the only point
    # in the pipeline where that paragraph is visible — after
    # augment_query() it's fused into the search string and gone.
    if answer:
        try:
            from bl_pipeline.shared.event_bus import emit as _emit
            _emit("hyde_doc_generated",
                  sub_query=sub_query,
                  hypothetical=answer,
                  n_chars=len(answer),
                  cost_usd=round(cost, 6))
        except Exception:
            pass

    return answer, cost


def augment_query(sub_query: str, hypothetical: str) -> str:
    """Combine the original sub_query with the hypothetical answer
    into a single retrieval query.

    Both BM25 and vector retrieval will see this combined string. The
    sub_query at the front preserves the original keyword signal; the
    hypothetical at the end adds vocabulary bridges.
    """
    if not hypothetical:
        return sub_query
    return f"{sub_query}\n\n{hypothetical}"
