"""verify.py — deterministic plan-vs-execution coverage check.

NO LLM call.  Pure Python comparison between the Plan the planner
produced and the ReasonerTrace the executor produced.  Returns a
`CoverageReport` the orchestrator uses to decide whether to auto-fire
a revision pass.

Coverage rule:
    • Every PlanQuantity.name must appear as a key_findings[*].quantity
      in trace.final, matched case-insensitively after stripping
      whitespace and common annotations (e.g. "_25_75", "_full").

    • A core quantity (is_core=True) missing from the trace ⇒
      coverage.needs_revision = True.  The orchestrator will fire one
      targeted revision pass with the missing names as the brief.

    • A non-core quantity missing is recorded but does NOT trigger
      revision (the executor was allowed to skip it as a cross-check
      it judged redundant — though it must say so in `limitations`).

Decay heuristic:
    decay_used=True iff the trace mentions any of these tokens in
    thinking_segments or tool_call args: "decay", "Fransson", "Tu_local",
    "Tu(x_t)", "iteration with local Tu".  Heuristic, not a proof —
    the critic gets the final say.
"""
from __future__ import annotations

import re
from typing import Iterable

from bl_pipeline.agent1_fresh.state import (
    CoverageReport,
    Plan,
    ReasonerTrace,
    SubQueryCoverage,
    emit_event,
)


# ══════════════════════════════════════════════════════════════════════
# Quantity-name matching
# ══════════════════════════════════════════════════════════════════════

# Variant suffixes the planner / executor sometimes attach (e.g.
# "L_tr_25_75" vs "L_tr").  We strip them for matching.
_VARIANT_SUFFIX_RE = re.compile(
    r"_(25_75|1_99|5_95|10_90|full|inlet|local|decay_corrected|mayle|ags|lm|sh|narasimha)$",
    re.IGNORECASE,
)

# Bug 7 from run #3: the reasoner emits quantity names like
#   "x_t (decay-corrected, PRIMARY)"
#   "L_tr (AGS full zone, γ=0 to ~1, PRIMARY)"
#   "Re_θt (decay-corrected, PRIMARY)"
# and the plain `lower().strip()` couldn't match these to bare "x_t" or
# "L_tr" in the plan.  Coverage said 0/7 when actually 4/4 was computed.
# This regex strips ANY parenthesized qualifier from a name.
_PAREN_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*")
# And strips trailing commas, ellipses, asterisks, and the word PRIMARY
_TRAILING_NOISE_RE = re.compile(
    r"\s*[*,.]+\s*$|\s*\b(primary|secondary)\b\s*$",
    re.IGNORECASE,
)


def _normalise(name: str) -> str:
    """Aggressively normalise a quantity name for matching.

    Handles all observed naming styles:
      "x_t"                                  -> "x_t"
      "X_T"                                  -> "x_t"
      "x_t_inlet"                            -> "x_t"          (suffix strip)
      "x_t_decay_corrected"                  -> "x_t"
      "x_t (decay-corrected, PRIMARY)"       -> "x_t"          (paren strip)
      "L_tr (AGS full zone, γ=0 to ~1)"      -> "l_tr"
      "Re_θt (decay-corrected, PRIMARY)"     -> "re_θt"
      "x_t cross-check (AGS Eq.3, inlet Tu)" -> "x_t cross-check" (hmm)

    Strip order matters: lowercase, then parentheses, then trailing
    noise, then variant suffixes, then re-strip.
    """
    n = (name or "").strip().lower()
    # 1. Strip any parenthesized qualifier(s).  Iterative for nested cases.
    for _ in range(3):
        new = _PAREN_QUALIFIER_RE.sub(" ", n)
        if new == n:
            break
        n = new
    # 2. Strip trailing noise (commas, "PRIMARY", etc.)
    for _ in range(3):
        new = _TRAILING_NOISE_RE.sub("", n)
        if new == n:
            break
        n = new
    # 3. Collapse whitespace and strip again
    n = " ".join(n.split()).strip()
    # 4. Strip variant suffixes iteratively
    for _ in range(3):
        new = _VARIANT_SUFFIX_RE.sub("", n)
        if new == n:
            break
        n = new
    return n


def _final_quantity_names(trace: ReasonerTrace) -> list[str]:
    """Extract `quantity` strings from trace.final.key_findings."""
    if not trace.final:
        return []
    out: list[str] = []
    for entry in (trace.final.get("key_findings") or []):
        if not isinstance(entry, dict):
            continue
        q = entry.get("quantity")
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out


# ══════════════════════════════════════════════════════════════════════
# Decay-handling heuristic
# ══════════════════════════════════════════════════════════════════════

_DECAY_TOKENS = (
    "decay", "Fransson", "Tu(x", "Tu_local", "Tu local",
    "local Tu", "decay-corrected", "iterate Tu", "Tu(x_t)",
    "freestream-turbulence decay",
)


def _decay_used(trace: ReasonerTrace) -> bool:
    """True iff any thinking segment or tool-call arg mentions decay."""
    needles = [t.lower() for t in _DECAY_TOKENS]
    # Check thinking segments
    for seg in trace.thinking_segments or []:
        text = (seg or "").lower()
        if any(n in text for n in needles):
            return True
    # Check tool-call args (code may mention decay variable names)
    for tc in trace.tool_calls or []:
        try:
            text = str(tc.args).lower()
            if any(n in text for n in needles):
                return True
        except Exception:
            continue
    return False


# ══════════════════════════════════════════════════════════════════════
# Per-sub-query coverage — heuristic, NO LLM
# ══════════════════════════════════════════════════════════════════════

# Stopwords we strip when extracting keywords from a sub-query.
# Conservative list — only true noise words.
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "because",
    "in", "on", "at", "by", "for", "with", "of", "to", "from", "as",
    "we", "us", "our", "you", "your", "they", "them", "their",
    "it", "its", "we", "i", "me", "my",
    "would", "should", "could", "must", "can", "may", "might", "will",
    "agent", "user", "answer", "question", "apply", "applies", "applies",
})


def _keywords(text: str) -> set[str]:
    """Extract distinctive lowercase tokens from a string.

    Keeps tokens of length >= 3 that aren't in the stopword list.
    Preserves underscores so technical names like `x_t`, `re_theta_t`,
    `tu`, `l_tr` survive intact.
    """
    if not text:
        return set()
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text.lower())
    return {tok for tok in raw if len(tok) >= 2 and tok not in _STOPWORDS}


def _sub_query_coverage(
    sub_queries: list[str], trace: ReasonerTrace,
) -> list[SubQueryCoverage]:
    """Per-sub-query heuristic coverage.

    A sub-query is considered ADDRESSED if at least ~50% of its
    distinctive keywords appear in trace.thinking_segments OR
    trace.final.  This is intentionally generous — false negatives
    (missing a real answer) are worse than false positives, because
    a false negative would trigger an unnecessary revision pass.
    """
    if not sub_queries or not trace:
        return [SubQueryCoverage(sub_query=q, addressed=False) for q in (sub_queries or [])]

    # Concatenate everything the reasoner produced into one searchable blob.
    blob_parts: list[str] = []
    for seg in trace.thinking_segments or []:
        if seg:
            blob_parts.append(seg)
    if trace.final:
        try:
            import json as _json
            blob_parts.append(_json.dumps(trace.final, default=str))
        except Exception:
            blob_parts.append(str(trace.final))
    blob_tokens = _keywords(" ".join(blob_parts))

    out: list[SubQueryCoverage] = []
    for sq in sub_queries:
        sq_kw = _keywords(sq)
        if not sq_kw:
            out.append(SubQueryCoverage(sub_query=sq, addressed=False))
            continue
        hits = sq_kw & blob_tokens
        # Threshold: at least 50% of sub-query keywords show up, AND at
        # least 2 distinct keywords (so a 2-word sub-query with one
        # generic match doesn't count).
        addressed = (len(hits) / len(sq_kw) >= 0.5) and len(hits) >= 2
        evidence = ""
        if addressed and hits:
            evidence = f"keywords matched: {sorted(hits)[:6]}"
        out.append(SubQueryCoverage(
            sub_query=sq, addressed=addressed, evidence=evidence,
        ))
    return out


# ══════════════════════════════════════════════════════════════════════
# Public entry — verify_plan_coverage
# ══════════════════════════════════════════════════════════════════════

def verify_plan_coverage(plan: Plan, trace: ReasonerTrace) -> CoverageReport:
    """Compare plan vs reasoner trace; return a CoverageReport.

    Pure Python.  Emits a single event for observability.
    """
    if plan is None or not plan.quantities_to_compute:
        # No plan was produced — this is a planner FAILURE.  Surface
        # whatever the reasoner managed to find on its own as `extra`,
        # but DO NOT trigger a coverage-driven revision: there are no
        # planned quantities to revise against, so firing a second
        # 12-turn reasoner pass with a meaningless brief just burns
        # API budget (one such run cost the user $5).  The critic still
        # gets the trace and will flag missing physics independently.
        final_names = _final_quantity_names(trace) if trace else []
        sq_cov = _sub_query_coverage(
            (plan.sub_queries if plan else []), trace,
        )
        report = CoverageReport(
            computed=[],
            missing_core=[],   # ← empty plan ≠ missing core (see comment)
            missing_other=[],
            extra=final_names,
            decay_used=_decay_used(trace) if trace else False,
            sub_query_coverage=sq_cov,
            needs_revision=False,  # ← was True; do NOT auto-revise on empty plan
            summary=(
                "DEGRADED RUN: planner returned an empty plan.  Coverage "
                f"cannot be audited; the reasoner produced {len(final_names)} "
                "finding(s) best-effort.  Skipping coverage-driven revision "
                "to protect API budget — fix the planner instead."
            ),
        )
        emit_event("coverage_done", **{
            "n_planned": 0, "n_computed": 0,
            "missing_core": [],
            "missing_other": [],
            "decay_used": report.decay_used, "needs_revision": False,
            "summary": "planner_empty_no_revision",
        })
        return report

    final_names = _final_quantity_names(trace)
    final_norm = {_normalise(n): n for n in final_names}

    computed: list[str] = []
    missing_core: list[str] = []
    missing_other: list[str] = []

    planned_norm: set[str] = set()
    for q in plan.quantities_to_compute:
        n_norm = _normalise(q.name)
        planned_norm.add(n_norm)
        if n_norm in final_norm:
            computed.append(q.name)
        elif q.is_core:
            missing_core.append(q.name)
        else:
            missing_other.append(q.name)

    # Anything in final that wasn't in plan = extra (executor went beyond)
    extra = [
        final_norm[n] for n in final_norm if n not in planned_norm
    ]

    decay_used = _decay_used(trace)
    needs_revision = bool(missing_core)

    # Per-sub-query coverage — heuristic, separate from quantity coverage
    sq_cov = _sub_query_coverage(plan.sub_queries, trace)
    sq_unaddressed = [s.sub_query for s in sq_cov if not s.addressed]

    if needs_revision:
        summary = (
            f"Coverage gap: {len(missing_core)} core quantity(ies) planned but "
            f"not computed: {missing_core}.  Revision will be triggered."
        )
    elif missing_other:
        summary = (
            f"All {len(computed)} core quantities computed.  "
            f"{len(missing_other)} non-core (cross-check) items were planned "
            f"but skipped — accepted unless critic objects."
        )
    else:
        summary = (
            f"Plan fully executed — {len(computed)}/"
            f"{len(plan.quantities_to_compute)} quantities computed."
        )
    if not decay_used and plan.decay_handling:
        summary += "  Decay was planned but not used in the trace."
    if sq_unaddressed:
        summary += (
            f"  Sub-queries not addressed in reasoner output ({len(sq_unaddressed)} of "
            f"{len(sq_cov)}): {[s[:60] for s in sq_unaddressed[:3]]}"
        )

    report = CoverageReport(
        computed=computed,
        missing_core=missing_core,
        missing_other=missing_other,
        extra=extra,
        decay_used=decay_used,
        sub_query_coverage=sq_cov,
        needs_revision=needs_revision,
        summary=summary,
    )

    emit_event(
        "coverage_done",
        n_planned=len(plan.quantities_to_compute),
        n_computed=len(computed),
        missing_core=missing_core,
        missing_other=missing_other,
        decay_used=decay_used,
        n_sub_queries=len(sq_cov),
        n_sub_queries_addressed=sum(1 for s in sq_cov if s.addressed),
        needs_revision=needs_revision,
        summary=summary,
    )
    return report
