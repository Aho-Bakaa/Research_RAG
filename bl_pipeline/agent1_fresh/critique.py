"""critique.py — peer-review pass on the reasoner's trace.

Sonnet reads the user query + flow + the reasoner's full trace + the
chunk pool the reasoner had access to.  It checks:

  • Physics sanity   — numerical results in plausible ranges
  • Citation honesty — every cited equation traces back to its chunk's
                       paper_id (no misattribution, no invented formulas)
  • Unit discipline  — Tu in percent, lengths in metres, ν consistent
  • Deferral honesty — CFD-required answers labelled as such, not faked
  • Completeness     — every part of the user's query addressed

Verdict ∈ {PASS, NEEDS_REVISION, FAIL}.  NEEDS_REVISION is the signal for
the orchestrator to feed the concerns back into a SECOND reasoner pass
(handled in run.py).
"""
from __future__ import annotations

import json
import time
from typing import Any

from bl_pipeline.agent1_fresh.prompts import CRITIQUE_SYSTEM
from bl_pipeline.agent1_fresh.state import (
    Chunk,
    CritiqueConcern,
    CritiqueResult,
    FlowConditions,
    ReasonerTrace,
    emit_event,
)
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


_VALID_VERDICTS = {"PASS", "NEEDS_REVISION", "FAIL"}
_VALID_SEVERITY = {"minor", "major", "blocker"}
_VALID_CATEGORY = {"physics", "citation", "units", "deferral",
                   "completeness", "other"}


# ══════════════════════════════════════════════════════════════════════
# User-message builder — compact serialisation of the trace for the critic
# ══════════════════════════════════════════════════════════════════════

def _format_trace_for_critic(trace: ReasonerTrace) -> str:
    """Render the reasoner's trace as readable prose for the critic.

    We interleave thinking segments and tool calls in their actual order
    so the critic sees the same reasoning flow the reasoner produced.
    """
    parts: list[str] = []
    # Approximation: thinking_segments[i] precedes tool_calls[i].  When
    # the reasoner ends with a final block (no trailing tool call) the
    # last thinking_segment stands alone.
    n_segs = len(trace.thinking_segments)
    n_calls = len(trace.tool_calls)
    max_i = max(n_segs, n_calls)
    for i in range(max_i):
        if i < n_segs and trace.thinking_segments[i]:
            seg = trace.thinking_segments[i].strip()
            if len(seg) > 1500:
                seg = seg[:1500] + "…[trim]"
            parts.append(f"--- THINKING (turn {i + 1}) ---\n{seg}")
        if i < n_calls:
            tc = trace.tool_calls[i]
            args_str = json.dumps(tc.args, ensure_ascii=False, default=str)[:400]
            res_str = json.dumps(tc.result, ensure_ascii=False, default=str)[:800]
            parts.append(
                f"--- TOOL CALL {tc.step}: {tc.tool_name} ---\n"
                f"args: {args_str}\n"
                f"result: {res_str}"
                + (f"\nerror: {tc.error}" if tc.error else "")
            )
    if trace.final:
        final_str = json.dumps(trace.final, ensure_ascii=False, indent=2, default=str)
        parts.append(f"--- FINAL ---\n{final_str}")
    elif trace.truncated:
        parts.append("--- FINAL ---\n(none — reasoner truncated at max_react_turns)")
    return "\n\n".join(parts)


def _format_chunks_for_critic(chunks: list[Chunk], n: int = 30) -> str:
    """Compact chunk listing for citation-honesty cross-check."""
    out = []
    for c in chunks[:n]:
        body = (c.content or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "…[trim]"
        out.append(
            f"[{c.chunk_id}] paper={c.paper_id} page={c.page} "
            f"score={c.score:.2f}\n  {body}"
        )
    return "\n".join(out) if out else "(empty chunk pool)"


def _build_user_message(
    user_query: str,
    flow: FlowConditions,
    trace: ReasonerTrace,
    chunks: list[Chunk],
) -> str:
    flow_lines = []
    for label, val, unit in [
        ("U_inf", flow.velocity_ms, "m/s"),
        ("Tu", flow.turbulence_intensity_pct, "%"),
        ("nu", flow.kinematic_viscosity_m2s, "m^2/s"),
        ("Lambda", flow.length_scale_m, "m"),
        ("chord", flow.chord_m, "m"),
    ]:
        if val is not None:
            flow_lines.append(f"  {label} = {val} {unit}")
    if flow.geometry_description:
        flow_lines.append(f"  geometry = {flow.geometry_description}")
    if flow.pressure_gradient_description:
        flow_lines.append(f"  pressure_gradient = {flow.pressure_gradient_description}")
    flow_block = "\n".join(flow_lines) or "  (none)"

    return f"""USER_QUERY:
"{user_query}"

FLOW:
{flow_block}

TRACE (reasoner's thinking + tool calls + final):
{_format_trace_for_critic(trace)}

CHUNK_POOL (what the reasoner had access to — for citation cross-check):
{_format_chunks_for_critic(chunks)}

Produce your verdict JSON now.
"""


# ══════════════════════════════════════════════════════════════════════
# Public entry
# ══════════════════════════════════════════════════════════════════════

def critique(
    user_query: str,
    flow: FlowConditions,
    trace: ReasonerTrace,
    chunks: list[Chunk],
    *,
    pipeline_state: Any = None,
    budget_remaining_usd: float | None = None,
) -> CritiqueResult:
    """Run the peer-review pass.  Always returns a `CritiqueResult` —
    never raises (an LLM error becomes a NEEDS_REVISION verdict with the
    error captured in `summary`).

    When `budget_remaining_usd` is provided and <= ~$0.05, skip the LLM
    call entirely (it would just fail with Budget exhausted) and emit a
    graceful fallback verdict.  Added 2026-06-13 after run 76d5b351:
    REASONER blew the $5 cap then critique LLM ran, failed immediately,
    AND writer LLM ran, failed immediately — both burning event-loop
    time for no result.  Pre-call budget check prevents that cascade.
    """
    t0 = time.time()
    emit_event("critique_started", user_query=user_query)

    # Pre-call budget check (2026-06-13 #253)
    if budget_remaining_usd is not None and budget_remaining_usd <= 0.05:
        emit_event("critique_budget_skip",
                   budget_remaining_usd=budget_remaining_usd)
        result = CritiqueResult(
            verdict="BUDGET_EXHAUSTED",
            concerns=[CritiqueConcern(
                severity="major", category="other",
                issue=(f"Critique skipped — pre-call budget check showed "
                       f"${budget_remaining_usd:.3f} remaining, below the "
                       f"$0.05 threshold to safely run the critic LLM."),
                suggested_fix=("Re-run with a higher max_cost_usd cap, OR "
                               "accept the REASONER's FINAL block without "
                               "peer-review polish.  The manuscript writer "
                               "will be told to proceed without critique."),
            )],
            summary="Critique skipped (budget exhausted before this stage).",
            cost_usd=0.0,
            elapsed_s=round(time.time() - t0, 3),
        )
        return result

    user_msg = _build_user_message(user_query, flow, trace, chunks)
    try:
        response, usage = llm.call(
            task="agent1_fresh_critique",   # → Sonnet
            system=CRITIQUE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            # Bumped 1500 → 3000 on 2026-05-16: critique on a full trace
            # with 30 chunks + multiple turns easily exceeds 1500 output
            # tokens and gets truncated mid-JSON.
            max_tokens=3000,
            temperature=0.0,
            pipeline_state=pipeline_state,
        )
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
    except Exception as e:
        result = CritiqueResult(
            verdict="NEEDS_REVISION",
            concerns=[CritiqueConcern(
                severity="major", category="other",
                issue=f"critic LLM call failed: {type(e).__name__}: {e}",
                suggested_fix="retry the critique, or proceed without it"
                              " and flag in the manuscript",
            )],
            summary=f"Critic unavailable: {e}",
            cost_usd=0.0,
            elapsed_s=round(time.time() - t0, 3),
        )
        emit_event("critique_done", verdict=result.verdict, error=str(e))
        return result

    parsed = parse_lenient_json(response)
    # Tolerate two shapes from the LLM:
    #   • Canonical:  {"verdict": "...", "concerns": [...], "summary": "..."}
    #   • Bare list:  [<concern>, ...]  — Sonnet sometimes returns this
    if isinstance(parsed, list):
        parsed = {"verdict": "NEEDS_REVISION", "concerns": parsed, "summary": ""}
    elif not isinstance(parsed, dict):
        # Pure prose / refusal — emit a loud event and surface in the
        # verdict summary so the run report shows the parse failure.
        preview = (response or "(empty)")[:300].replace("\n", " ")
        emit_event(
            "critique_parse_failed",
            raw_preview=preview,
            response_length=len(response or ""),
        )
        parsed = {"verdict": "NEEDS_REVISION", "concerns": [],
                  "summary": f"CRITIQUE_PARSE_FAILED — Sonnet returned no JSON. "
                             f"Raw preview: {preview!r}"}

    verdict = str(parsed.get("verdict", "NEEDS_REVISION")).upper()
    if verdict not in _VALID_VERDICTS:
        verdict = "NEEDS_REVISION"

    raw_concerns = parsed.get("concerns") or []
    concerns: list[CritiqueConcern] = []
    for entry in raw_concerns:
        if not isinstance(entry, dict):
            continue
        sev = str(entry.get("severity", "minor"))
        if sev not in _VALID_SEVERITY:
            sev = "minor"
        cat = str(entry.get("category", "other"))
        if cat not in _VALID_CATEGORY:
            cat = "other"
        concerns.append(CritiqueConcern(
            severity=sev,
            category=cat,
            issue=str(entry.get("issue", ""))[:400],
            suggested_fix=str(entry.get("suggested_fix", ""))[:400],
        ))

    result = CritiqueResult(
        verdict=verdict,
        concerns=concerns,
        summary=str(parsed.get("summary", ""))[:1000],
        cost_usd=round(cost, 4),
        elapsed_s=round(time.time() - t0, 3),
    )
    emit_event(
        "critique_done",
        verdict=result.verdict,
        n_concerns=len(result.concerns),
        cost_usd=result.cost_usd,
    )
    return result
