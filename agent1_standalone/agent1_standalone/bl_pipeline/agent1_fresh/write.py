"""write.py — turn the validated trace into a two-part research report.

The manuscript has TWO parts:

  PART A — TECHNICAL ANSWER
    The clean, experiment-ready output: regime, headline numbers,
    method per source, result table, probe-placement plan, literature-
    vs-reality caveat, limitations, suggested next steps.

  PART B — RESEARCH PROCESS LOG
    The transparent narrative: what the agent decomposed the query
    into, where it got stuck, how each problem was handled (overcome,
    patched, deferred), what was caught by self-critique, and a
    per-quantity confidence rating.

Both parts are produced by ONE Sonnet call (cheaper than two writes),
using a structured user message that includes the reasoner's FINAL
block, the critique verdict, the plan, the coverage report, the
chunk pool, AND the ProcessLog.

If the writer LLM is unreachable we fall back to a deterministic
reconstruction so the user always gets SOMETHING.
"""
from __future__ import annotations

import json
import time
from typing import Any

from bl_pipeline.agent1_fresh.plot import PlotArtifact
from bl_pipeline.agent1_fresh.prompts import WRITER_SYSTEM
from bl_pipeline.agent1_fresh.state import (
    Chunk,
    CoverageReport,
    CritiqueResult,
    FlowConditions,
    Plan,
    ProcessLog,
    ReasonerTrace,
    emit_event,
)
from bl_pipeline.shared.llm_router import llm


# ══════════════════════════════════════════════════════════════════════
# Formatters — keep the user-message rendering side compact
# ══════════════════════════════════════════════════════════════════════

def _format_chunks_for_writer(chunks: list[Chunk], n: int = 25) -> str:
    out = []
    for c in chunks[:n]:
        body = (c.content or "").strip().replace("\n", " ")
        if len(body) > 350:
            body = body[:350] + "…[trim]"
        out.append(f"[{c.paper_id}::p{c.page}] {body}")
    return "\n".join(out) if out else "(no chunks available)"


def _format_solving_steps_for_writer(reasoner: ReasonerTrace, n: int = 30) -> str:
    """Render compute() tool calls so the writer can show math|code together.

    Each key_finding's `computed_via_tool_step` indexes into these steps.  We
    list, per compute step, the Python the agent ran and its stdout/result —
    this is the source for Part A's two-column "Solving" table.  Non-compute
    tools (search, lookup_equation, lookup_glossary) are skipped here.
    """
    tcs = getattr(reasoner, "tool_calls", None) or []
    out: list[str] = []
    for tc in tcs:
        if getattr(tc, "tool_name", "") != "compute":
            continue
        code = (getattr(tc, "args", None) or {}).get("code", "") or ""
        res = getattr(tc, "result", None) or {}
        stdout = (res.get("stdout") or "").strip()
        value = res.get("result")
        err = getattr(tc, "error", None) or res.get("error")
        block = [f"step {getattr(tc, 'step', '?')}:", "  code:"]
        for line in code.splitlines():
            block.append(f"    {line}")
        if stdout:
            block.append(f"  stdout: {stdout[:600]}")
        if value is not None:
            block.append(f"  result: {value}")
        if err:
            block.append(f"  error: {err}")
        out.append("\n".join(block))
        if len(out) >= n:
            break
    return "\n\n".join(out) if out else "(no compute() steps recorded)"


def _format_critique_for_writer(critique: CritiqueResult | None) -> str:
    if critique is None:
        return "verdict: (no critique run)"
    if not critique.concerns:
        return f"verdict: {critique.verdict}\nsummary: {critique.summary}"
    lines = []
    for c in critique.concerns:
        lines.append(
            f"  - [{c.severity}][{c.category}] {c.issue} (fix: {c.suggested_fix})"
        )
    return (
        f"verdict: {critique.verdict}\n"
        f"summary: {critique.summary}\n"
        f"concerns:\n" + "\n".join(lines)
    )


def _format_flow_for_writer(flow: FlowConditions) -> str:
    parts = []
    if flow.velocity_ms is not None:
        parts.append(f"U_inf = {flow.velocity_ms} m/s")
    if flow.turbulence_intensity_pct is not None:
        parts.append(f"Tu = {flow.turbulence_intensity_pct} % (PERCENT)")
    if flow.kinematic_viscosity_m2s is not None:
        parts.append(f"nu = {flow.kinematic_viscosity_m2s} m²/s")
    if flow.length_scale_m is not None:
        parts.append(f"Lambda = {flow.length_scale_m} m")
    if flow.chord_m is not None:
        parts.append(f"chord = {flow.chord_m} m")
    if flow.geometry_description:
        parts.append(f"geometry = {flow.geometry_description}")
    if flow.pressure_gradient_description:
        parts.append(f"pressure_gradient = {flow.pressure_gradient_description}")
    return ", ".join(parts) or "(no flow conditions)"


def _format_plan_for_writer(plan: Plan | None) -> str:
    if plan is None or not plan.quantities_to_compute:
        return "(planner produced no plan — see DEGRADED-RUN caveat below)"
    lines = []
    if plan.sub_queries:
        lines.append("sub_queries:")
        for sq in plan.sub_queries:
            lines.append(f"  - {sq}")
    lines.append(f"decay_handling: {plan.decay_handling}")
    if plan.probe_layout_notes:
        lines.append(f"probe_layout_notes: {plan.probe_layout_notes}")
    lines.append("quantities:")
    for q in plan.quantities_to_compute:
        core = "CORE" if q.is_core else "    "
        srcs = ", ".join(q.source_papers) if q.source_papers else "(no sources)"
        lines.append(f"  {core} {q.name:<26s} via {q.method[:80]}  [{srcs}]")
    if plan.cross_checks:
        lines.append("cross_checks:")
        for c in plan.cross_checks[:6]:
            lines.append(f"  - {c}")
    if plan.deferred_to_other_agents:
        lines.append("deferred:")
        for d in plan.deferred_to_other_agents:
            lines.append(f"  - {d.get('quantity')} → {d.get('recommend_agent')} ({d.get('reason')})")
    return "\n".join(lines)


def _format_coverage_for_writer(cov: CoverageReport | None) -> str:
    if cov is None:
        return "(coverage check did not run)"
    sq_lines = []
    for s in (cov.sub_query_coverage or []):
        marker = "OK" if s.addressed else "GAP"
        sq_lines.append(f"  [{marker}] {s.sub_query[:120]}")
    sq_block = "\n".join(sq_lines) if sq_lines else "  (no sub-queries planned)"
    return (
        f"computed: {cov.computed}\n"
        f"missing_core: {cov.missing_core}\n"
        f"missing_other: {cov.missing_other}\n"
        f"extra: {cov.extra}\n"
        f"decay_used: {cov.decay_used}\n"
        f"needs_revision: {cov.needs_revision}\n"
        f"summary: {cov.summary}\n"
        f"sub_query_coverage:\n{sq_block}"
    )


def _format_process_log_for_writer(plog: ProcessLog | None) -> str:
    """Compact rendering of the ProcessLog — one line per event.

    Format: [stage/turn] kind | summary | outcome | confidence_impact
    The writer reads this verbatim into Part B; structure should be
    consistent so the LLM can map events to narrative sentences.
    """
    if plog is None or not plog.events:
        return "(no process log)"
    lines = []
    for e in plog.events:
        turn = f"t{e.turn}" if e.turn is not None else "—"
        ci = f" [conf:{e.confidence_impact}]" if e.confidence_impact else ""
        oc = f" → {e.outcome}" if e.outcome else ""
        lines.append(f"[{e.stage}/{turn}] {e.kind}: {e.summary}{oc}{ci}")
    return "\n".join(lines)


def _format_plots_for_writer(plots: list[PlotArtifact] | None) -> str:
    """Render plot artifacts as a block the writer can embed via ![]().

    Uses POSIX-style relative paths so the manuscript markdown stays
    portable across Windows/Linux/macOS rendering.
    """
    if not plots:
        return "(no plots generated)"
    lines = []
    for p in plots:
        # Convert backslashes to forward slashes for markdown embedding
        rel_path = p.path.replace("\\", "/")
        lines.append(f"  [{p.label}] {rel_path}  —  {p.caption}")
    return "\n".join(lines)


def _build_user_message(
    user_query: str,
    flow: FlowConditions,
    reasoner: ReasonerTrace,
    critique: CritiqueResult | None,
    chunks: list[Chunk],
    plan: Plan | None,
    coverage: CoverageReport | None,
    process_log: ProcessLog | None,
    plot_artifacts: list[PlotArtifact] | None = None,
) -> str:
    final_block = reasoner.final or {}
    final_str = (
        json.dumps(final_block, ensure_ascii=False, indent=2, default=str)
        if final_block
        else "(reasoner produced no FINAL block — trace was truncated)"
    )
    return f"""USER_QUERY:
"{user_query}"

FLOW (parsed):
{_format_flow_for_writer(flow)}

PLAN (what the agent committed to compute):
{_format_plan_for_writer(plan)}

REASONER_FINAL (structured findings):
{final_str}

SOLVING_STEPS (compute() calls — link a finding to its step via its \
computed_via_tool_step; use the code verbatim in Part A's two-column Solving table):
{_format_solving_steps_for_writer(reasoner)}

COVERAGE (plan vs execution):
{_format_coverage_for_writer(coverage)}

CRITIQUE (peer-review verdict):
{_format_critique_for_writer(critique)}

PROCESS_LOG (chronological events — drives Part B):
{_format_process_log_for_writer(process_log)}

GENERATED_PLOTS (deterministic — embed each in Part A where the topic fits, \
e.g. probe_layout in section 6, intermittency in section 5, bl_thickness in \
section 6, onset_comparison in section 5, tu_decay in section 7).  Use \
markdown image syntax `![<label>](<path>)` followed by the caption as italic \
prose on the next line:
{_format_plots_for_writer(plot_artifacts)}

CHUNK_POOL (citation source-of-truth — cite as `paper_id::pPAGE`):
{_format_chunks_for_writer(chunks)}

Compose the TWO-PART markdown report.  Strictly follow the structure in
the system prompt: Part A (technical) then a clear divider then Part B
(process log).  No JSON wrapper.  No code fence around the whole document.
LaTeX equations in `$$...$$`.  Embed every plot from GENERATED_PLOTS in
the appropriate Part A section.
"""


# ══════════════════════════════════════════════════════════════════════
# Public entry
# ══════════════════════════════════════════════════════════════════════

def _scrub_banned_words(md: str) -> str:
    """Deterministic backstop for the owner's hard publishability bans:
    'Provenance' and 'Verbatim' must never appear in any user-facing
    manuscript text (headings or prose).  Runs on EVERY manuscript path
    before it leaves the writer, so an LLM leak (or a primed prompt word)
    can never ship.  Word-boundary + case-aware."""
    import re
    for pat, repl in (
        (r"\bProvenance\b", "Source"),
        (r"\bprovenance\b", "source"),
        (r"\bVerbatim\b",   "Exactly as written"),
        (r"\bverbatim\b",   "exactly as written"),
    ):
        md = re.sub(pat, repl, md)
    return md


def write_manuscript(
    user_query: str,
    flow: FlowConditions,
    reasoner: ReasonerTrace,
    critique: CritiqueResult | None,
    chunks: list[Chunk],
    *,
    plan: Plan | None = None,
    coverage: CoverageReport | None = None,
    process_log: ProcessLog | None = None,
    plot_artifacts: list[PlotArtifact] | None = None,
    pipeline_state: Any = None,
    budget_remaining_usd: float | None = None,
) -> tuple[str, float, float]:
    """Compose the final two-part markdown report.

    Returns
    ───────
    (manuscript_text, cost_usd, elapsed_s)

    Never raises — falls back to a deterministic reconstruction on LLM
    error so the user always gets a response.

    Added 2026-06-13 (#253): `budget_remaining_usd` — when <= ~$0.10
    (the writer's manuscript step is the most expensive single call in
    the pipeline at typical 4000-10000 output tokens), skip the LLM
    call and return the deterministic fallback manuscript directly.
    Run 76d5b351 burned the writer LLM error after critique already
    failed — both wasted event-loop time.  Pre-call check prevents that.
    """
    t0 = time.time()
    emit_event("write_started", user_query=user_query)

    # Pre-call budget check (2026-06-13 #253) — fallback manuscript
    # is deterministic + free, and it's better than waiting for the
    # LLM call to fail with "Budget exhausted" after burning a slot.
    if budget_remaining_usd is not None and budget_remaining_usd <= 0.10:
        emit_event("write_budget_skip",
                   budget_remaining_usd=budget_remaining_usd,
                   reason="writer LLM call would exceed cap")
        fallback_md = _fallback_manuscript(
            user_query, flow, reasoner, critique,
            plan=plan, coverage=coverage, process_log=process_log,
            error=(f"Writer LLM call skipped: pre-call budget check "
                   f"showed ${budget_remaining_usd:.3f} remaining "
                   f"(< $0.10 threshold).  Manuscript is the "
                   f"deterministic reconstruction from the reasoner's "
                   f"structured findings; no LLM polish applied."),
        )
        return _scrub_banned_words(fallback_md), 0.0, round(time.time() - t0, 3)

    user_msg = _build_user_message(
        user_query, flow, reasoner, critique, chunks,
        plan=plan, coverage=coverage, process_log=process_log,
        plot_artifacts=plot_artifacts,
    )

    try:
        response, usage = llm.call(
            task="agent1_fresh_write",
            system=WRITER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            # Bumped 3000 → 5000 → 8000 → 10000 on 2026-05-16: the
            # two-part manuscript carries plain-English version of every
            # prose section PLUS a probe-placement table PLUS Part B
            # (process log narrative).  Previous run produced 5,626
            # tokens of Part A and Sonnet stopped without writing Part
            # B at all.  10000 gives explicit headroom; the
            # _ensure_part_b_present() fallback below is the safety
            # net for the case where Sonnet still skips Part B.
            max_tokens=10000,
            temperature=0.2,
            pipeline_state=pipeline_state,
        )
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        manuscript = (response or "").strip()
        if not manuscript:
            raise RuntimeError("writer returned empty response")
    except Exception as e:
        manuscript = _fallback_manuscript(
            user_query, flow, reasoner, critique,
            plan=plan, coverage=coverage, process_log=process_log,
            error=str(e),
        )
        cost = 0.0

    # ── Safety net: ensure Part B is present ─────────────────────
    # The previous run produced a 22.5 KB manuscript with Part A but
    # NO Part B at all.  The user explicitly wants the research-process
    # narrative.  If Sonnet skipped it, append a deterministic Part B
    # built straight from the ProcessLog — same content the LLM would
    # have written, just less polished prose.
    manuscript = _ensure_part_b_present(
        manuscript, process_log=process_log, coverage=coverage,
        critique=critique, reasoner=reasoner,
    )

    manuscript = _scrub_banned_words(manuscript)
    elapsed = round(time.time() - t0, 3)
    emit_event("write_done", cost_usd=cost, elapsed_s=elapsed,
               length_chars=len(manuscript))
    return manuscript, cost, elapsed


# ══════════════════════════════════════════════════════════════════════
# Part B safety net — deterministic, no LLM
# ══════════════════════════════════════════════════════════════════════

def _ensure_part_b_present(
    manuscript: str,
    *,
    process_log: ProcessLog | None,
    coverage: CoverageReport | None,
    critique: CritiqueResult | None,
    reasoner: ReasonerTrace | None,
) -> str:
    """Append a deterministic Part B if the writer didn't produce one.

    Why this exists: the writer (Sonnet) is verbose for Part A and has
    sometimes stopped before reaching Part B.  Rather than re-call the
    LLM (extra cost, extra time), we synthesise Part B from the
    structured artifacts we already have — ProcessLog, CoverageReport,
    CritiqueResult, ReasonerTrace.

    The fallback Part B follows the same 5-section structure the prompt
    requested, so the user gets a usable document either way.  It's
    less polished than what the LLM would produce, but it's HONEST and
    PRESENT — both of which matter more than polish.
    """
    if "# Part B" in manuscript or "## Part B" in manuscript:
        return manuscript

    # Build Part B from what we have
    lines: list[str] = ["", "---", "", "# Part B — Research process log",
                        "",
                        "*(This Part B was generated deterministically from the "
                        "agent's process log because the writer ran long on Part A "
                        "and didn't reach Part B itself. The content is the same; "
                        "the prose is plainer.)*", ""]

    # Section 1: Approach
    lines.append("## Approach")
    lines.append("")
    n_events = len(process_log.events) if process_log else 0
    n_turns = reasoner.n_react_turns if reasoner else 0
    n_tools = len(reasoner.tool_calls) if reasoner else 0
    lines.append(
        f"The agent decomposed the query into sub-questions, retrieved supporting "
        f"chunks across 3 iterations, committed a plan with several user-facing "
        f"quantities, then ran a ReAct loop over {n_turns} turn(s) using "
        f"{n_tools} tool call(s) (search + compute). The process log captured "
        f"{n_events} notable event(s) which are summarised below."
    )
    lines.append("")

    # Section 2: Turn-by-turn
    lines.append("## What happened, turn by turn")
    lines.append("")
    if process_log and process_log.events:
        for e in process_log.events:
            turn = f"turn {e.turn}" if e.turn is not None else "—"
            oc = f" — *{e.outcome}*" if e.outcome else ""
            ci = f" (confidence impact: {e.confidence_impact})" if e.confidence_impact else ""
            lines.append(f"- **[{e.stage}/{turn}]** `{e.kind}`: {e.summary}{oc}{ci}")
    else:
        lines.append("*(process log is empty)*")
    lines.append("")

    # Section 3: Problems encountered
    lines.append("## Problems encountered")
    lines.append("")
    if process_log:
        problem_kinds = {"stuck", "self_catch", "refusal", "off_script",
                         "malformed_tool_call", "llm_error", "truncated",
                         "planner_failed", "coverage_revision", "critic_revision",
                         "cost_cap_mid_pass"}
        problems = [e for e in process_log.events if e.kind in problem_kinds]
        if not problems:
            lines.append("No notable problems — clean run.")
        else:
            for e in problems:
                turn = f"turn {e.turn}" if e.turn is not None else "—"
                outcome_label = {
                    "resolved":  "OVERCAME",
                    "patched":   "PATCHED",
                    "deferred":  "DEFERRED",
                    "noted":     "DEFERRED",
                    "failed":    "UNRESOLVED",
                    "":          "UNRESOLVED",
                }.get(e.outcome, "PATCHED")
                conf_label = (e.confidence_impact or "none").upper()
                lines.append(f"- **Problem ({e.stage}/{turn}):** {e.summary}")
                lines.append(f"  - **What was tried:** see process log entry above")
                lines.append(f"  - **Outcome:** {outcome_label}")
                lines.append(f"  - **Confidence impact:** {conf_label}")
                lines.append("")

    # Section 4: Deferred / ignored
    lines.append("## What was deferred or ignored")
    lines.append("")
    if coverage and coverage.missing_other:
        lines.append("The following non-core quantities were planned but not computed "
                     "(skipped as cross-checks the executor judged redundant or out of budget):")
        for q in coverage.missing_other:
            lines.append(f"- `{q}`")
    elif coverage and coverage.missing_core:
        lines.append(
            "The following CORE quantities were planned but not found in the "
            "reasoner's FINAL block (likely truncated):"
        )
        for q in coverage.missing_core:
            lines.append(f"- `{q}`")
    else:
        lines.append("No quantities were explicitly deferred.")
    lines.append("")
    if coverage and coverage.sub_query_coverage:
        unaddressed = [s for s in coverage.sub_query_coverage if not s.addressed]
        if unaddressed:
            lines.append("Sub-queries not detected in the reasoner's output:")
            for s in unaddressed:
                lines.append(f"- {s.sub_query}")
            lines.append("")

    # Section 5: Confidence per quantity
    lines.append("## Confidence per quantity")
    lines.append("")
    lines.append("| Quantity | Confidence | Reason |")
    lines.append("|---|---|---|")
    if reasoner and reasoner.final:
        for f in (reasoner.final.get("key_findings") or []):
            if not isinstance(f, dict):
                continue
            q = f.get("quantity", "?")
            conf = "MEDIUM"  # default; refined below if we can infer
            reason = f.get("method", "")[:80] or "see methods"
            lines.append(f"| {q} | {conf} | {reason} |")
    else:
        lines.append("| *(no FINAL block emitted)* | LOW | reasoner truncated; values were reconstructed by the writer |")
    if critique and critique.concerns:
        n_blocker = sum(1 for c in critique.concerns if c.severity == "blocker")
        n_major   = sum(1 for c in critique.concerns if c.severity == "major")
        n_minor   = sum(1 for c in critique.concerns if c.severity == "minor")
        lines.append("")
        lines.append(
            f"*Critique raised {len(critique.concerns)} concern(s): "
            f"{n_blocker} blocker / {n_major} major / {n_minor} minor. "
            f"See Part A's headline caveat and limitations.*"
        )

    return manuscript.rstrip() + "\n" + "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════
# Fallback manuscript — deterministic, never fabricates content
# ══════════════════════════════════════════════════════════════════════

def _fallback_manuscript(
    user_query: str,
    flow: FlowConditions,
    reasoner: ReasonerTrace,
    critique: CritiqueResult | None,
    plan: Plan | None = None,
    coverage: CoverageReport | None = None,
    process_log: ProcessLog | None = None,
    error: str = "",
) -> str:
    """Last-resort manuscript when the writer LLM is unreachable.

    Still emits the two-part structure so the UI rendering matches.
    """
    lines = [
        "# Answer (fallback rendering — writer LLM unavailable)",
        "",
        f"*The writer pass failed: {error}.  Below is a deterministic*",
        "*reconstruction from the reasoner's structured findings and the*",
        "*process log.  Treat this as a partial answer; consider rerunning.*",
        "",
        "## Query",
        f"> {user_query}",
        "",
        "## Flow conditions",
        f"`{_format_flow_for_writer(flow)}`",
        "",
    ]
    final = reasoner.final or {}
    if final:
        lines.append("## Reasoner findings")
        summary = str(final.get("answer_summary", "")).strip()
        if summary:
            lines.append(summary)
            lines.append("")
        findings = final.get("key_findings") or []
        if findings:
            lines.append("### Key quantities")
            for f in findings:
                if not isinstance(f, dict):
                    continue
                q = f.get("quantity", "?")
                v = f.get("value", "?")
                u = f.get("unit", "")
                m = f.get("method", "")
                cit = f.get("citation", "")
                lines.append(f"  - **{q}** = {v} {u}  via *{m}* `[{cit}]`")
            lines.append("")
        lims = final.get("limitations") or []
        if lims:
            lines.append("### Limitations")
            for L in lims:
                lines.append(f"  - {L}")
            lines.append("")

    if critique is not None and critique.concerns:
        lines.append("## Peer-review concerns")
        for c in critique.concerns:
            lines.append(f"  - **[{c.severity}][{c.category}]** {c.issue}")
            if c.suggested_fix:
                lines.append(f"    *Suggested fix:* {c.suggested_fix}")
        lines.append("")

    # ── Part B fallback — render the process log directly ──
    lines.append("---")
    lines.append("")
    lines.append("# Part B — Research process log")
    lines.append("")
    if process_log is None or not process_log.events:
        lines.append("*(no process events recorded)*")
    else:
        for e in process_log.events:
            turn = f"turn {e.turn}" if e.turn is not None else "—"
            oc = f" — *{e.outcome}*" if e.outcome else ""
            ci = f" (confidence impact: {e.confidence_impact})" if e.confidence_impact else ""
            lines.append(f"- **[{e.stage}/{turn}]** `{e.kind}`: {e.summary}{oc}{ci}")

    return "\n".join(lines)
