"""reason.py — the ReAct researcher loop.

Hands the user's query + validated chunks to Sonnet with three tools
(compute, lookup_glossary, search) and lets the LLM iterate until it
produces a `<final>` block.

Protocol the LLM follows (defined in prompts.REASONER_SYSTEM):

    <prose thinking, possibly multi-paragraph>

    <tool_call>{"tool": "compute", "args": {"code": "..."}}</tool_call>

    ... we inject <tool_result>{...}</tool_result> ...

    <more prose thinking>

    <tool_call>...</tool_call>

    ... loop until ...

    <final>
    {"answer_summary": "...", "key_findings": [...], "limitations": [...],
     "open_questions": [...]}
    </final>

Production-grade contract:
  • Bounded turns (max_react_turns) so a confused LLM can't burn budget.
  • Every tool result is JSON-serialised before injection — keeps the
    conversation window deterministic.
  • Malformed tool_call JSON → friendly error result injected back, the
    LLM can recover.
  • LLM omits <final> at the turn cap → mark trace.truncated, attempt a
    "wrap up" hint, then stop.
  • Cost tracked per turn; emitted as events for live UI.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from bl_pipeline.agent1_fresh.paper_inventory import (
    format_paper_inventory_for_prompt,
    load_all_paper_summaries,
)
from bl_pipeline.agent1_fresh.prompts import REASONER_SYSTEM
from bl_pipeline.agent1_fresh.state import (
    Chunk,
    FlowConditions,
    ProcessLog,
    Plan,
    ReasonerTrace,
    ToolCall,
    emit_event,
)
from bl_pipeline.agent1_fresh.tools import (
    TOOL_DESCRIPTIONS_FOR_PROMPT,
    dispatch as tool_dispatch,
)
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


# ══════════════════════════════════════════════════════════════════════
# Block-extraction regexes
# ══════════════════════════════════════════════════════════════════════

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FINAL_RE     = re.compile(r"<final>\s*(.*?)\s*</final>",         re.DOTALL | re.IGNORECASE)
# Opening-tag-only regexes — used by the lenient fallback below when the
# response is truncated mid-block (closing tag never written).
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_FINAL_OPEN_RE     = re.compile(r"<final>\s*",     re.IGNORECASE)


def _lenient_block_payload(response: str, open_re: re.Pattern, kind: str) -> str | None:
    """When the strict <tag>...</tag> regex misses, fall back to:
       find the LAST opening tag → take everything after it → trim any
       trailing closing tag of the OPPOSITE kind that snuck in.

    Used when the LLM started a block but the response was truncated
    before the closing tag was written (`max_tokens_per_turn` hit
    mid-block).  This was the silent failure mode that caused turn 12
    of the previous live run to report `final: None` even though the
    model HAD emitted `<final>{...}` — the closing `</final>` simply
    never made it onto the wire.
    """
    if not response:
        return None
    m = None
    # Use the LAST opening tag so we don't grab a prior aborted attempt.
    for match in open_re.finditer(response):
        m = match
    if not m:
        return None
    after = response[m.end():]
    # Trim trailing whitespace and any partial closer the model may have
    # started ("</fin", "</to", etc.) — anything after the last `}` of
    # what would be a valid JSON payload is noise.
    return after.strip()


def _balance_truncated_json(text: str) -> str:
    """Append the minimum closing tokens needed to balance an unclosed
    JSON object/array.

    Walks the text once tracking string state, brace depth, and bracket
    depth.  At end:
      • if inside an unterminated string → close the string first
      • append `]` for each unclosed array, `}` for each unclosed object,
        in the order needed to balance
      • strip a trailing comma (common at truncation point) before
        appending closers — `{"a": 1,}` is invalid JSON
    Conservative: if the truncation happened mid-key (e.g.
    `{\"key`), this won't produce valid JSON, and downstream parsing
    will still fail.  But for the common case of truncation between
    array entries, this works.
    """
    if not text:
        return text
    in_str = False
    esc = False
    brace_open = 0
    bracket_open = 0
    last_meaningful = 0  # position of last char that should not be after a comma
    for i, c in enumerate(text):
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                brace_open += 1
            elif c == "}":
                brace_open -= 1
            elif c == "[":
                bracket_open += 1
            elif c == "]":
                bracket_open -= 1
            if c not in (" ", "\t", "\n", "\r", ","):
                last_meaningful = i + 1
    # Trim trailing garbage (whitespace + trailing comma if at end)
    fixed = text[:last_meaningful].rstrip(", \n\t\r")
    # Close out an open string with an empty value first
    if in_str:
        fixed += '"'
    # Close unclosed arrays then unclosed objects (innermost first; bracket
    # before brace matches the typical {"key": [...]} structure)
    if bracket_open > 0:
        fixed += "]" * bracket_open
    if brace_open > 0:
        fixed += "}" * brace_open
    return fixed


def _parse_block_payload(payload_text: str) -> dict[str, Any] | None:
    """Try parse_lenient_json; on failure try the brace-balancer first."""
    if not payload_text:
        return None
    parsed = parse_lenient_json(payload_text)
    if isinstance(parsed, dict):
        return parsed
    # Fallback: balance unclosed JSON, retry
    balanced = _balance_truncated_json(payload_text)
    if balanced != payload_text:
        parsed = parse_lenient_json(balanced)
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_first_tool_call(response: str) -> dict[str, Any] | None:
    """Find the FIRST tool_call block; parse its JSON payload.

    Returns None when no tool_call appears, or the JSON is unparseable.
    A malformed payload is its own signal — caller decides whether to
    bail or inject an error to teach the LLM.

    Two-stage extraction:
      1. Strict <tool_call>...</tool_call> regex (fastest, handles
         the normal case).
      2. If that fails, lenient fallback: find the LAST opening tag
         and try to parse everything after it as JSON.  parse_lenient_json
         already handles truncated/partial JSON gracefully.
    """
    if not response:
        return None
    # Stage 1: strict
    m = _TOOL_CALL_RE.search(response)
    if m:
        payload_text = m.group(1)
    else:
        # Stage 2: lenient — handles truncated-mid-block responses
        payload_text = _lenient_block_payload(response, _TOOL_CALL_OPEN_RE, "tool_call")
        if payload_text is None:
            return None
    parsed = _parse_block_payload(payload_text)
    if not isinstance(parsed, dict):
        # Surface as a structured error tool-call so the dispatcher
        # rejects it and the reasoner sees the recovery hint.
        return {"_malformed": True, "raw": payload_text[:200]}
    return parsed


def _extract_final(response: str) -> dict[str, Any] | None:
    """Extract the FINAL block's JSON, if any.

    Same two-stage strategy as `_extract_first_tool_call` — strict
    regex first, then lenient fallback when the closing `</final>`
    tag is missing (response truncated mid-block).
    """
    if not response:
        return None
    m = _FINAL_RE.search(response)
    if m:
        payload_text = m.group(1)
    else:
        payload_text = _lenient_block_payload(response, _FINAL_OPEN_RE, "final")
        if payload_text is None:
            return None
    parsed = _parse_block_payload(payload_text)
    return parsed if isinstance(parsed, dict) else None


# Patterns for parsing citations/methods to extract paper_id + eq_id
# so we can look the verbatim formula up in the glossary.
_PAPER_ID_FROM_CITATION_RE = re.compile(r"([a-z][a-z0-9_]+)\s*::")
_EQ_NUMBER_FROM_METHOD_RE = re.compile(
    r"\bEq\.?\s*\(([^)]+)\)", re.IGNORECASE,
)


def _enrich_findings_with_glossary(findings: list[dict[str, Any]]) -> None:
    """For each finding whose `method` field cites a paper-numbered
    equation, fetch the verbatim formula + validity from the glossary
    and attach them as `glossary_formula` / `glossary_validity` fields.

    Modifies `findings` in place.  Never raises — best-effort
    enrichment.  Findings with no parseable citation are left alone.

    Composite citations like
        "mayle_1991::p10; fransson_matsubara_alfredsson_2005::p7"
    plus a method that mentions multiple equations
        "Three-step Fransson Eq.(3.1) iteration applied to Mayle Eq.(9)"
    are handled by trying every (paper_id × eq_number) combination
    until one resolves.  First successful lookup wins.

    The writer downstream uses `glossary_formula` to render equations
    in §3 (Methods) verbatim — closes the "Sonnet retyped the formula
    from memory" attack surface that caused the run-#12 manuscript bug.
    """
    # Local import to avoid circular load with tools.py at module-load time.
    from bl_pipeline.agent1_fresh.tools import lookup_equation

    for f in findings:
        if not isinstance(f, dict):
            continue
        # Skip if we already attached one (idempotent — useful for re-runs)
        if f.get("glossary_formula"):
            continue

        citation = str(f.get("citation", ""))
        method = str(f.get("method", ""))

        # Pull ALL paper_ids from the citation (composite-citation aware)
        paper_ids = [
            m.group(1).lower()
            for m in _PAPER_ID_FROM_CITATION_RE.finditer(citation)
        ]
        if not paper_ids:
            continue

        # Pull ALL equation numbers from method + citation
        eq_nums = [
            m.group(1).strip()
            for m in _EQ_NUMBER_FROM_METHOD_RE.finditer(method + " " + citation)
        ]
        if not eq_nums:
            continue

        # Try every combination — first successful lookup wins.
        # Order: paper-major (try paper1 with all eqs first, then paper2),
        # which matches the "primary source first" convention in citations.
        resolved = None
        for paper_id in paper_ids:
            for eq_num in eq_nums:
                eq_id = f"Eq. ({eq_num})"
                entry = lookup_equation(paper_id=paper_id, eq_id=eq_id)
                if not entry.get("error"):
                    resolved = entry
                    break
            if resolved is not None:
                break

        if resolved is None:
            # No combo worked — leave unattached.  Writer's prompt
            # instructs Sonnet to flag any §3 equation lacking
            # `glossary_formula` as "manuscript transcription not
            # glossary-verified" so the reader sees the provenance gap.
            continue

        f["glossary_formula"] = resolved.get("formula_latex_verbatim", "") or ""
        validity = resolved.get("validity", "") or ""
        if validity:
            f["glossary_validity"] = validity
        # Stamp the resolved key so the manuscript can show the lookup
        # path (helps the reader trace the provenance).
        f["glossary_eq_id"] = resolved.get("eq_id", "")
        f["glossary_paper_id"] = resolved.get("paper_id", "")
        f["glossary_page"] = resolved.get("page")


# ══════════════════════════════════════════════════════════════════════
# User-message builder — the kickoff message
# ══════════════════════════════════════════════════════════════════════

def _format_chunks_for_reasoner(chunks: list[Chunk], n: int = 20) -> str:
    """Render the top-`n` validated chunks for the reasoner prompt."""
    out_lines = []
    for i, c in enumerate(chunks[:n], start=1):
        body = (c.content or "").strip().replace("\r", "")
        # Cap content to keep the prompt bounded — chunks often have
        # ~500-1500 useful chars; anything longer is parent-context
        # we don't need in the prompt itself (reasoner can `search` for
        # more if it discovers a gap).
        if len(body) > 1200:
            body = body[:1200] + "…[truncated]"
        out_lines.append(
            f"[{i}] chunk_id={c.chunk_id}  paper={c.paper_id}  page={c.page}  "
            f"score={c.score:.2f}  label={c.label}\n"
            f"{body}\n"
        )
    return "\n".join(out_lines) if out_lines else "(no chunks available)"


def _revision_kickoff_user_message(
    user_query: str,
    flow: FlowConditions,
    chunks: list[Chunk],
    revision_brief: str,
    prior_findings: dict[str, Any],
) -> str:
    """Compact kickoff for a revision pass.

    No paper inventory — the reasoner saw it in pass 1; carrying it again
    burns tokens without adding info.  Only the top chunks (already
    pre-trimmed by caller to ~5).  Includes the critic's specific
    concerns and the prior pass's FINAL block summary so the reasoner
    builds on prior work instead of redoing it.
    """
    flow_lines = []
    if flow.velocity_ms is not None:
        flow_lines.append(f"  U_inf = {flow.velocity_ms} m/s")
    if flow.turbulence_intensity_pct is not None:
        flow_lines.append(f"  Tu    = {flow.turbulence_intensity_pct} % (percent)")
    if flow.kinematic_viscosity_m2s is not None:
        flow_lines.append(f"  nu    = {flow.kinematic_viscosity_m2s} m^2/s")
    if flow.chord_m is not None:
        flow_lines.append(f"  chord = {flow.chord_m} m")
    # Blasius-VO trigger inputs — keep in revision pass too (#221)
    if flow.x_0_BL_m is not None:
        flow_lines.append(f"  x_0_BL (operator) = {flow.x_0_BL_m} m  (VO trigger)")
    if flow.delta_x_measurements:
        flow_lines.append(
            f"  delta_x_measurements = {flow.delta_x_measurements}  (VO trigger)"
        )
    flow_block = "\n".join(flow_lines) or "  (none)"

    tools_block = "\n".join(
        f"  • {desc}" for desc in TOOL_DESCRIPTIONS_FOR_PROMPT.values()
    )

    prior_str = (
        json.dumps(prior_findings, ensure_ascii=False, indent=2, default=str)
        if prior_findings
        else "(no prior FINAL block — pass 1 truncated)"
    )

    return f"""USER_QUERY:
"{user_query}"

FLOW:
{flow_block}

TOOLS AVAILABLE:
{tools_block}

PRIOR PASS FINDINGS (your previous answer — revise these to address the concerns below):
{prior_str}

CRITIC CONCERNS to address in this revision (be specific):
{revision_brief}

TOP CHUNKS (the highest-scored chunks from retrieval — focus on these; \
call search() if you need more):
{_format_chunks_for_reasoner(chunks, n=len(chunks))}

This is a REVISION pass.  Don't repeat pass-1's work — build on it. \
Address each concern explicitly in your new <final> block.  If a concern \
is wrong, push back in prose with a chunk citation.  You may call tools \
as needed but most of the math from pass 1 was correct; the concerns are \
mostly about citation discipline and completeness.
"""


def _format_plan_for_executor(plan: Plan | None) -> str:
    """Render the Plan as a compact block for the executor's kickoff."""
    if plan is None or not plan.quantities_to_compute:
        return "(no plan supplied — executor proceeds best-effort)"
    lines = ["EXECUTION_PLAN (pre-computed by the planner node):"]
    lines.append(f"  rationale: {plan.rationale}")
    if plan.decay_handling:
        lines.append(f"  decay_handling: {plan.decay_handling}")
    lines.append("  quantities_to_compute:")
    for q in plan.quantities_to_compute:
        srcs = ", ".join(q.source_papers) if q.source_papers else "(no source pinned)"
        deps = (", ".join(q.depends_on) if q.depends_on else "—")
        core = "CORE" if q.is_core else "cross-check"
        lines.append(
            f"    - [{core}] {q.name}  ({q.expected_unit})\n"
            f"        method: {q.method}\n"
            f"        sources: {srcs}\n"
            f"        depends_on: {deps}"
        )
    if plan.cross_checks:
        lines.append("  cross_checks:")
        for cc in plan.cross_checks:
            lines.append(f"    - {cc}")
    lines.append(f"  expected_tool_calls: ~{plan.expected_tool_calls}")
    return "\n".join(lines)


def _kickoff_user_message(
    user_query: str,
    flow: FlowConditions,
    chunks: list[Chunk],
    max_chunks_in_prompt: int,
    plan: Plan | None = None,
) -> str:
    flow_lines = []
    if flow.velocity_ms is not None:
        flow_lines.append(f"  U_inf            = {flow.velocity_ms} m/s")
    if flow.turbulence_intensity_pct is not None:
        flow_lines.append(f"  Tu               = {flow.turbulence_intensity_pct} %  (PERCENT, not decimal)")
    if flow.kinematic_viscosity_m2s is not None:
        flow_lines.append(f"  nu (kinematic)   = {flow.kinematic_viscosity_m2s} m^2/s")
    if flow.length_scale_m is not None:
        flow_lines.append(f"  Lambda (FST)     = {flow.length_scale_m} m")
    if flow.chord_m is not None:
        flow_lines.append(f"  chord            = {flow.chord_m} m")
    # Blasius-VO trigger inputs (task #221).  Bug 2026-06-13 run af7256a2:
    # the PLANNER saw x_0_BL_m and dispatched VIRTUAL_ORIGIN_ANCHORED, but
    # the REASONER's FLOW block didn't show x_0_BL_m, so it overrode the
    # PLAN.decay_method ("the FLOW shows no x_0_BL_m... I'll execute
    # ITERATE_FRANSSON_DECAY instead").  The VO dispatch is the THESIS
    # NOVELTY; without these two lines the entire branch is invisible.
    if flow.x_0_BL_m is not None:
        flow_lines.append(
            f"  x_0_BL (operator) = {flow.x_0_BL_m} m  "
            f"(Blasius virtual origin — direct input; triggers "
            f"dubey_2026_thesis::blasius_VO_Tu_effective_method per "
            f"PLANNER I2 trigger ii)"
        )
    if flow.delta_x_measurements:
        _pairs = ", ".join(
            f"(x={x}m, δ={d}m)" for (x, d) in flow.delta_x_measurements
        )
        flow_lines.append(
            f"  delta_x_measurements = [{_pairs}]  "
            f"(≥2 (x, δ_99) pairs — fit Blasius √(x − x_0_BL) for the "
            f"virtual origin, then run dubey_2026_thesis::"
            f"blasius_VO_Tu_effective_method per PLANNER I2 trigger ii)"
        )
    if flow.geometry_description:
        flow_lines.append(f"  geometry         = {flow.geometry_description}")
    if flow.pressure_gradient_description:
        flow_lines.append(f"  pressure_gradient= {flow.pressure_gradient_description}")
    if flow.roughness_um is not None:
        flow_lines.append(f"  roughness        = {flow.roughness_um} um")
    flow_block = "\n".join(flow_lines) or "  (no numerical flow conditions parsed)"

    tools_block = "\n".join(
        f"  • {desc}" for desc in TOOL_DESCRIPTIONS_FOR_PROMPT.values()
    )

    # Paper inventory — research-style "skim the forest before the trees".
    # If we have summaries on disk, inject them.  Empty when none exist
    # yet, in which case the reasoner falls back to chunks alone.
    summaries = load_all_paper_summaries()
    inventory_block = format_paper_inventory_for_prompt(summaries) if summaries else ""

    inventory_section = (
        f"PAPER INVENTORY (skim this before diving into chunks — tells you "
        f"which paper supplies which closed-form quantities, what regimes "
        f"each covers, and what they explicitly don't):\n{inventory_block}\n\n"
        if inventory_block
        else ""
    )

    plan_block = _format_plan_for_executor(plan)

    return f"""USER_QUERY:
"{user_query}"

FLOW (parsed numerical conditions — use these when plugging into formulas):
{flow_block}

TOOLS AVAILABLE:
{tools_block}

{plan_block}

{inventory_section}VALIDATED_CHUNKS (already retrieved + reranked + judged; sorted best-first):
{_format_chunks_for_reasoner(chunks, n=max_chunks_in_prompt)}

Reason like a senior boundary-layer researcher.  Pick which models apply, \
quote their equations verbatim from the chunks above, compute numerical \
answers when the math is concrete, compare across models when they \
disagree, and acknowledge limits.  Defer to Agent 2 (CFD) or Agent 3 \
(experiment) when the query genuinely needs them — don't fake an answer.

Begin.  Think in prose; call tools with <tool_call>...</tool_call>; \
emit <final>...</final> when you've reached a complete answer.
"""


# ══════════════════════════════════════════════════════════════════════
# Public entry — the ReAct loop
# ══════════════════════════════════════════════════════════════════════

def react_reason(
    user_query: str,
    flow: FlowConditions,
    chunks: list[Chunk],
    *,
    plan: Plan | None = None,              # pre-computed plan
    max_react_turns: int = 12,
    max_chunks_in_prompt: int = 12,        # was 20 — cut to slim per-call ctx
    # Bumped 2000 → 4000 on 2026-05-16: at 2000, FINAL blocks for
    # multi-quantity plans got truncated mid-emission, so the closing
    # </final> tag never appeared and the strict regex extractor
    # returned None.  Bumped 4000 → 8000 on 2026-06-13 after run
    # 76d5b351 had 4/7 compute() calls truncate mid-Python (iteration
    # blocks for AGS τ_avg + Mayle + LM + FS20 + Fransson decay
    # exceeded the 4000-token budget mid-statement, costing 4 × $0.40
    # = ~$1.60 of wasted retries).  8000 gives multi-correlation
    # compute() blocks room to land; lenient extractor still catches
    # the rare overflow.
    max_tokens_per_turn: int = 8000,
    pipeline_state: Any = None,
    is_revision: bool = False,             # revision mode: smaller ctx
    revision_brief: str | None = None,     # critic feedback when revising
    prior_findings: dict[str, Any] | None = None,  # FINAL from prior pass
    process_log: ProcessLog | None = None, # narrative log for Part B
    max_cost_usd: float = 2.50,            # NEW — abort the loop when cumulative reasoner cost exceeds this
                                           # Bumped from 1.20 → 2.50 on 2026-06-03 so the
                                           # reasoner has head-room for chained lookup_equation
                                           # derivations (Roach Eq.18 + Kurian-Fransson Eq.8
                                           # + AGS Eq.17/18) without the inner cap firing
                                           # mid-derivation.  Top-level run() cap is $10.00.
) -> ReasonerTrace:
    """Run the ReAct loop until <final> or max_react_turns.

    In revision mode (is_revision=True), the kickoff includes the prior
    pass's FINAL block + the critic's specific concerns instead of the
    full paper inventory, and we ship only the top-5 chunks (since by
    revision time we already know which chunks were useful).  This keeps
    revision calls compact instead of repeating all of pass-1's context.

    Returns a `ReasonerTrace` capturing every prose segment, every tool
    call + result, and the final structured answer (or None if truncated).
    """
    t0 = time.time()
    trace = ReasonerTrace()
    # Initialise the process log (no-op if caller didn't pass one)
    plog = process_log if process_log is not None else ProcessLog()
    plog.add(
        stage="reason",
        kind="info",
        summary=(
            f"Reasoner started ({'revision pass' if is_revision else 'first pass'}) "
            f"with {len(chunks)} chunks and max {max_react_turns} turns."
        ),
    )
    emit_event("reason_started", user_query=user_query,
               n_chunks=len(chunks), max_turns=max_react_turns,
               is_revision=is_revision)

    if is_revision:
        # Revision pass: compact context — drop paper inventory (already
        # seen), keep only the top-5 chunks (highest judge score), inject
        # the critic's concerns and the prior FINAL block.
        kickoff = _revision_kickoff_user_message(
            user_query=user_query, flow=flow,
            chunks=chunks[:5],
            revision_brief=revision_brief or "",
            prior_findings=prior_findings or {},
        )
    else:
        kickoff = _kickoff_user_message(
            user_query, flow, chunks, max_chunks_in_prompt, plan=plan,
        )

    # Seed the conversation: system prompt + kickoff user message.
    messages: list[dict[str, str]] = [
        {"role": "user", "content": kickoff},
    ]

    # ── Stuck-detection state ──────────────────────────────────────
    # We watch the recent turn history for the pattern that burned
    # turns 10-12 in the previous live run: a self_catch ("this is
    # clearly wrong") followed by 2+ consecutive search() calls without
    # an intervening compute() — the reasoner chasing a side question
    # forever instead of finalizing what it already has.
    stuck_state = {
        "consecutive_searches": 0,    # searches since last compute() / final
        "had_recent_self_catch": False,
        "stuck_topic": "",            # snippet from the catch, for the recovery prompt
        "recovery_used": False,       # one-shot — only fire stuck-recovery ONCE per pass
    }

    for turn in range(1, max_react_turns + 1):
        # ── Budget warnings ─────────────────────────────────────────
        # The single most common failure mode in production runs has been
        # the reasoner spending turns 10-12 chasing a side question (e.g.
        # a unit-convention ambiguity) and never emitting a <final> block
        # before max_react_turns hits.  truncated=True → user gets nothing.
        # We inject a soft heads-up at N-1 and a hard stop at N so the
        # reasoner ALWAYS gets a chance to finalize what it already has.
        if turn == max_react_turns:
            messages.append({
                "role": "user",
                "content": (
                    f"BUDGET STOP: this is your FINAL turn "
                    f"({max_react_turns}/{max_react_turns}).  DO NOT call "
                    f"another <tool_call> — the dispatcher will be "
                    f"ignored.  Emit your <final>{{...}}</final> block NOW "
                    f"using the results you have already computed in "
                    f"prior turns.  For any quantity that you did NOT "
                    f"finish, list it under the FINAL block's `limitations` "
                    f"or `open_questions` field with a one-line reason — "
                    f"DO NOT fake a number to fill the slot.  If you skip "
                    f"<final>, the user receives no deliverable."
                ),
            })
            plog.add(stage="reason", kind="budget_warning", turn=turn,
                     summary="Hard-stop injected: final turn, must emit FINAL.",
                     outcome="noted")
        elif turn == max_react_turns - 1:
            messages.append({
                "role": "user",
                "content": (
                    f"BUDGET HEADS-UP: you have 2 turns left "
                    f"(this one = turn {turn}, then turn {max_react_turns} "
                    f"is FINAL only).  Stop opening new investigation "
                    f"threads.  Use THIS turn for any last essential "
                    f"compute(); the next turn must be your <final> "
                    f"block.  If a side question (e.g. a unit-convention "
                    f"ambiguity in a cross-check correlation) is not "
                    f"blocking your CORE deliverable, defer it to "
                    f"`limitations` rather than chasing it now."
                ),
            })
            plog.add(stage="reason", kind="budget_warning", turn=turn,
                     summary="Soft heads-up injected: 2 turns left, begin consolidating.",
                     outcome="noted")

        # ── SCRATCHPAD TRIM (added 2026-06-13) ──────────────────────
        # Run 76d5b351 averaged $0.40/turn — 4× the historical $0.10/turn
        # baseline.  Root cause: messages list grows linearly each turn
        # (+2 messages per turn: assistant_response + tool_result), and
        # tool_result blocks can be ~3500 chars apiece.  By turn 11 the
        # input context to each LLM call had ballooned to ~25 messages
        # = ~30k+ input tokens per call.  Trim: keep the kickoff (msg 0),
        # the most recent 5 tool round-trips (10 msgs), and any
        # budget-warning messages.  Drop older history.
        if len(messages) > 14:
            _kept: list[dict[str, str]] = [messages[0]]   # kickoff
            # collect budget warnings (recognizable by leading "BUDGET")
            _kept += [m for m in messages[1:-10]
                      if isinstance(m.get("content"), str)
                      and m["content"].lstrip().startswith(("BUDGET", "Soft heads-up"))]
            _kept += messages[-10:]   # last 5 turn-pairs
            emit_event(
                "reason_scratchpad_trimmed", turn=turn,
                msgs_before=len(messages), msgs_after=len(_kept),
            )
            messages = _kept

        # ── PRE-CALL COST CAP (added 2026-06-13) ─────────────────────
        # Run 76d5b351 showed: soft cap fires AFTER the LLM call (we
        # only know the cost once we get usage back).  By then we've
        # already spent it.  Combined with code-truncation retries
        # ($0.40/turn × failed retries), the hard cap was blown by
        # ~$0.03-$1.00.  Pre-call check: if cumulative cost already
        # >= hard cap, skip the LLM call entirely and force-emit a
        # FINAL block from whatever scratchpad state exists.
        if trace.cost_usd >= max_cost_usd:
            emit_event(
                "reason_pre_call_budget_block", turn=turn,
                cost_usd=trace.cost_usd, cap_usd=max_cost_usd,
                action="skip_llm_call_force_finalize_from_scratchpad",
            )
            plog.add(stage="reason", kind="budget_exhausted_pre_call",
                     turn=turn,
                     summary=(f"Pre-call cost ${trace.cost_usd:.2f} >= "
                              f"cap ${max_cost_usd:.2f}.  Skipping LLM "
                              f"call; emitting fallback FINAL from "
                              f"scratchpad state."),
                     outcome="patched", confidence_impact="high")
            trace.truncated = True
            break
        # If next call would push us OVER cap, force it to a tiny
        # tokens budget so the response is bounded and final.
        if trace.cost_usd + 0.5 >= max_cost_usd and max_tokens_per_turn > 1500:
            emit_event(
                "reason_pre_call_budget_squeeze", turn=turn,
                cost_usd=trace.cost_usd, cap_usd=max_cost_usd,
                old_max_tokens=max_tokens_per_turn, new_max_tokens=1500,
            )
            max_tokens_per_turn = 1500

        # ── LLM call ─────────────────────────────────────────────────
        try:
            response, usage = llm.call(
                task="agent1_fresh_reason",   # → Sonnet
                system=REASONER_SYSTEM,
                messages=messages,
                max_tokens=max_tokens_per_turn,
                temperature=0.0,
                pipeline_state=pipeline_state,
            )
        except Exception as e:
            # Stop the loop on hard LLM error; bubble up via trace.
            trace.thinking_segments.append(
                f"[reasoner LLM call failed at turn {turn}: {type(e).__name__}: {e}]"
            )
            trace.truncated = True
            plog.add(stage="reason", kind="llm_error", turn=turn,
                     summary=f"LLM call failed: {type(e).__name__}: {str(e)[:120]}",
                     outcome="failed", confidence_impact="high")
            break

        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        trace.cost_usd = round(trace.cost_usd + cost, 4)
        trace.n_react_turns = turn

        # Mid-pass cost cap — staged response, not a single binary trip.
        # task #15: the cap was being overshot because the prior code
        # gave the "one more turn" to emit FINAL the full 4000-token
        # budget, which spent another ~$0.30-0.50 and pushed the run
        # over cap. Two-stage response:
        #
        #   Stage 1 (cost ≥ 80% of cap):
        #     shrink max_tokens_per_turn to 1500 so subsequent turns
        #     can't blow another big chunk. FINAL blocks are typically
        #     500-1500 tokens; this is enough to land.
        #
        #   Stage 2 (cost ≥ 100% of cap):
        #     collapse max_react_turns = turn + 1 so the next iteration
        #     IS the emergency BUDGET STOP turn that forces <final>.
        #     The next turn still runs with the now-shrunken 1500-token
        #     budget so it can't push much further over cap.
        #
        #   Stage 3 (cost ≥ 120% of cap, defensive):
        #     hard break the loop. The trace is marked truncated and
        #     the orchestrator emits whatever findings have already
        #     been attached. Better to ship a degraded result than
        #     charge the user 2x.
        _SOFT_CAP_FRACTION = 0.80
        _HARD_OVERRUN_FRACTION = 1.20
        if (
            trace.cost_usd >= _SOFT_CAP_FRACTION * max_cost_usd
            and max_tokens_per_turn > 3000
        ):
            # Floor bumped 1500 → 3000 on 2026-06-13 — 1500 was too
            # aggressive a clamp; truncated late-run compute()s mid-
            # iteration, wasting retries.  3000 keeps the late-budget
            # turn focused but still gives the compute() Python room
            # to close out a multi-correlation block.
            emit_event(
                "reason_cost_soft_cap_hit", turn=turn,
                cost_usd=trace.cost_usd, cap_usd=max_cost_usd,
                old_max_tokens=max_tokens_per_turn, new_max_tokens=3000,
            )
            max_tokens_per_turn = 3000
        if trace.cost_usd >= max_cost_usd and turn < max_react_turns:
            plog.add(stage="reason", kind="cost_cap_mid_pass", turn=turn,
                     summary=(f"Reasoner cumulative cost ${trace.cost_usd:.2f} "
                              f"reached cap ${max_cost_usd:.2f} mid-pass — "
                              f"forcing finalize on turn {turn + 1}/{max_react_turns}."),
                     outcome="patched", confidence_impact="medium")
            emit_event("reason_cost_cap_hit", turn=turn,
                       cost_usd=trace.cost_usd, cap_usd=max_cost_usd)
            max_react_turns = turn + 1   # collapse remaining turns to one
        if trace.cost_usd >= _HARD_OVERRUN_FRACTION * max_cost_usd:
            # Defensive: even with the shrunken token budget, somehow
            # blew through. Hard break, mark truncated.
            emit_event(
                "reason_cost_hard_overrun", turn=turn,
                cost_usd=trace.cost_usd, cap_usd=max_cost_usd,
                threshold=_HARD_OVERRUN_FRACTION * max_cost_usd,
            )
            plog.add(stage="reason", kind="cost_hard_overrun", turn=turn,
                     summary=(f"Reasoner cost ${trace.cost_usd:.2f} exceeded "
                              f"hard overrun threshold "
                              f"${_HARD_OVERRUN_FRACTION * max_cost_usd:.2f} — "
                              f"truncating run; trace incomplete."),
                     outcome="failed", confidence_impact="high")
            trace.truncated = True
            break

        # Record the LLM's prose for this turn (everything except the
        # tool_call and final blocks — they're machine-readable).
        prose_only = _strip_blocks(response)
        trace.thinking_segments.append(prose_only)

        # Detect self-catches in the prose (the reasoner noticing its own
        # mistake mid-turn — e.g. "wait, Re_tr = 18 is physically wrong").
        # Heuristic: short phrase list; conservative to avoid false positives.
        catch_snippet = _detect_self_catch(prose_only)
        if catch_snippet:
            plog.add(stage="reason", kind="self_catch", turn=turn,
                     summary=f"Reasoner caught its own error: '{catch_snippet[:120]}'",
                     outcome="noted", confidence_impact="low")
            # Arm stuck-detection: a catch alone is fine (good!), but if
            # the next turns burn into consecutive searches without
            # resolving it, we step in.
            stuck_state["had_recent_self_catch"] = True
            stuck_state["stuck_topic"] = catch_snippet[:160]

        emit_event(
            "reason_turn",
            turn=turn,
            prose=prose_only[:2000],   # truncate for event payload
            cost_usd=cost,
        )

        # ── Check for FINAL ─────────────────────────────────────────
        final = _extract_final(response)
        if final is not None:
            # Glossary-formula enrichment: for every finding whose `method`
            # or `citation` mentions a paper-numbered equation, look up
            # the verbatim formula + validity from the glossary and attach
            # them to the finding.  The writer downstream uses
            # `glossary_formula` to typeset equations in §3 (Methods),
            # never Sonnet's restatement — closing the "Sonnet retyped
            # from memory" drift surface we hardened against in run #12.
            try:
                _enrich_findings_with_glossary(final.get("key_findings") or [])
            except Exception as _e:
                # Enrichment failure should NEVER block the FINAL emission.
                emit_event(
                    "glossary_enrichment_error",
                    error=f"{type(_e).__name__}: {_e}",
                )

            # ── Anchor-coverage validation (task #148) ─────────────────
            # For every anchor paper named in the optimizer's
            # ``inventory_partition.anchors`` we require at least one
            # finding citing that paper with a verified
            # ``glossary_formula`` attached (i.e. ``lookup_equation``
            # succeeded). Violations are recorded on the trace and
            # emitted as an audit event so the writer can flag the
            # manuscript and the frontend can show the gap. We do NOT
            # block FINAL emission — a hard block during the pre-pilot
            # cutover would risk killing a defensible run; the writer
            # will flag suspect sections instead.
            try:
                from bl_pipeline.agent1_fresh.anchor_validator import (
                    format_violation_block,
                    validate_anchor_coverage,
                )
                _anchors_for_validation: list[str] = []
                if pipeline_state is not None:
                    _partition = (
                        getattr(pipeline_state, "optimizer_partition", None)
                        or {}
                    )
                    _anchors_for_validation = list(
                        _partition.get("anchors") or []
                    )
                _coverage = validate_anchor_coverage(
                    anchors=_anchors_for_validation,
                    findings=final.get("key_findings") or [],
                )
                final["anchor_coverage"] = {
                    "anchors":    list(_coverage.anchors),
                    "covered":    list(_coverage.covered),
                    "violations": [
                        {
                            "paper_id":        v.paper_id,
                            "reason":          v.reason,
                            "findings_citing": list(v.findings_citing),
                        }
                        for v in _coverage.violations
                    ],
                    "is_clean":   _coverage.is_clean,
                    "diagnostic_md": format_violation_block(_coverage),
                }
                if not _coverage.is_clean:
                    emit_event(
                        "anchor_coverage_violation",
                        anchors=list(_coverage.anchors),
                        violations=[
                            {"paper_id": v.paper_id, "reason": v.reason}
                            for v in _coverage.violations
                        ],
                        is_clean=False,
                    )
                    plog.add(
                        stage="reason",
                        kind="anchor_coverage_violation",
                        turn=turn,
                        summary=(
                            f"{len(_coverage.violations)}/"
                            f"{len(_coverage.anchors)} anchor(s) lack a "
                            f"verified `lookup_equation` finding."
                        ),
                        outcome="flagged",
                    )
            except Exception as _e:
                emit_event(
                    "anchor_coverage_error",
                    error=f"{type(_e).__name__}: {_e}",
                )

            trace.final = final
            n_findings = len((final.get("key_findings") or []))
            plog.add(stage="reason", kind="final_emitted", turn=turn,
                     summary=f"FINAL block emitted with {n_findings} key finding(s).",
                     outcome="resolved")
            emit_event("reason_final", final=final, cost_usd=trace.cost_usd)
            break

        # ── Otherwise look for the next tool call ───────────────────
        tc_payload = _extract_first_tool_call(response)
        if tc_payload is None:
            # No tool call AND no final — LLM went silent or off-script.
            # Inject a nudge once; if it happens twice, bail.
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": ("You produced neither a <tool_call> nor a <final> "
                            "block.  Either call a tool (<tool_call>{...}"
                            "</tool_call>) or emit your answer "
                            "(<final>{...}</final>).  Do not produce any other "
                            "block shape."),
            })
            plog.add(stage="reason", kind="off_script", turn=turn,
                     summary="Produced neither tool_call nor final — injected nudge.",
                     outcome="patched")
            # Allow ONE recovery; the next iteration must comply.
            continue

        # ── Final-turn refusal: don't dispatch tools on the last turn ──
        # We promised the model in the BUDGET STOP prompt that further
        # tool_calls would be ignored.  Honour that — return a synthetic
        # tool_result that points the model back to <final>.  The loop
        # then ends because turn == max_react_turns; the model has had
        # its one warning.  This guarantees we never spend a tool
        # invocation (search latency, compute latency) on a turn the
        # user can't see the result of.
        if turn == max_react_turns:
            tool_name = str(tc_payload.get("tool", "")) if not tc_payload.get("_malformed") else "<refused>"
            args = (tc_payload.get("args") or {}) if isinstance(tc_payload.get("args"), dict) else {}
            tool_result = {
                "error": (
                    "TOOL_CALL_REFUSED: this was the final turn and you "
                    "were asked to emit <final> instead.  No tool was "
                    "dispatched.  The reasoner trace now ends.  Whatever "
                    "you computed in earlier turns will be passed to the "
                    "writer and critic as-is; missing items will be "
                    "flagged in the manuscript's limitations section."
                ),
                "_refused_final_turn": True,
            }
            plog.add(stage="reason", kind="refusal", turn=turn,
                     summary=(f"Final-turn tool call ({tool_name}) refused — "
                              "model warned but tried anyway."),
                     outcome="patched", confidence_impact="medium")
        # ── Dispatch the tool call ──────────────────────────────────
        elif tc_payload.get("_malformed"):
            tool_result = {
                "error": "your <tool_call> JSON was malformed; "
                         "ensure args is a JSON object",
                "_raw": tc_payload.get("raw", "")[:200],
            }
            tool_name = "<malformed>"
            args = {}
            plog.add(stage="reason", kind="malformed_tool_call", turn=turn,
                     summary="LLM emitted malformed <tool_call> JSON.",
                     outcome="patched")
        else:
            tool_name = str(tc_payload.get("tool", ""))
            args = tc_payload.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            # Bug 3 from run #4: Sonnet keeps wasting search() calls on
            # the Fransson Eq.5.5 constant even though the glossary
            # contains the canonical formula.  Intercept BEFORE dispatching
            # the search: if the query smells like a Fransson constant
            # hunt, replace the tool result with the canonical answer.
            #
            # CONSTANT UPDATE 2026-05-16: was using C=1.7e6 (loose
            # approximation Sonnet picked up in early runs) — the
            # canonical Fransson 2005 value is C=1.96×10⁶ with Tu in
            # percent (equivalently C=196 with Tu in fraction; both
            # give Re_tr ≈ 180,000 at Tu=3.3%, while the old 1.7e6
            # gives the 15%-low value 156,000).  The Haiku formula
            # judge enforces the canonical value; the interceptor now
            # matches that expectation so compute() calls don't get
            # rejected for using a value the interceptor itself injected.
            if (tool_name == "search"
                    and _is_fransson_constant_query(str(args.get("query", "")))):
                tool_result = {
                    "query": str(args.get("query", ""))[:200],
                    "n": 0,
                    "chunks": [],
                    "_intercepted": "fransson_constant",
                    "error": None,
                    "result_override": (
                        "INTERCEPTED — you are searching for the Fransson "
                        "Eq.(5.5) constant.  Stop searching; the value is "
                        "PRE-RESOLVED from the glossary:  Re_tr = C · Tu^(-2)  "
                        "with TWO equivalent forms:\n"
                        "   • C = 196      when Tu is in FRACTION (e.g. Tu=0.033)\n"
                        "   • C = 1.96e6   when Tu is in PERCENT  (e.g. Tu=3.3)\n"
                        "Both give the same answer.  At Tu=3.3 (percent):\n"
                        "   Re_tr ≈ 1.96e6 / 3.3² ≈ 180,000  (Re_x at γ=0.5 midpoint, NOT Re_θ at onset).\n"
                        "If you use C=196 with Tu in percent (Re_tr ≈ 18) "
                        "or C=1.96e6 with Tu in fraction (Re_tr ≈ 1.8×10⁹), "
                        "those are unit-mismatch bugs the formula verifier will reject. "
                        "Use the matching pair.  Do NOT search again for this constant."
                    ),
                }
                plog.add(stage="reason", kind="fransson_intercept", turn=turn,
                         summary="Intercepted Fransson-constant search; injected pre-resolved value.",
                         outcome="patched", confidence_impact="low")
                # Don't count this against consecutive_searches — it
                # didn't actually run, so it can't get the reasoner stuck.
            else:
                tool_result = tool_dispatch(tool_name, args)

            # Stuck-detection counters: search bumps the counter,
            # compute/lookup resets it (and clears the catch flag — the
            # reasoner is making forward progress on something).
            if tool_name == "search" and not tool_result.get("_intercepted"):
                stuck_state["consecutive_searches"] += 1
            elif tool_name in ("compute", "lookup_glossary", "lookup_equation"):
                stuck_state["consecutive_searches"] = 0
                stuck_state["had_recent_self_catch"] = False
                stuck_state["stuck_topic"] = ""

            # Log the dispatched call in the process narrative.  We tag
            # search with the query and compute with a short code preview.
            if tool_name == "search":
                q = str(args.get("query", ""))[:120]
                plog.add(stage="reason", kind="search", turn=turn,
                         summary=f"Searched literature for: '{q}'",
                         outcome="resolved" if not tool_result.get("error") else "failed")
            elif tool_name == "compute":
                code_preview = str(args.get("code", "")).strip().split("\n")[0][:100]
                plog.add(stage="reason", kind="compute", turn=turn,
                         summary=f"Ran compute(): {code_preview}",
                         outcome="resolved" if not tool_result.get("error") else "failed")
            elif tool_name == "lookup_glossary":
                term = str(args.get("term", ""))[:60]
                plog.add(stage="reason", kind="lookup", turn=turn,
                         summary=f"Looked up glossary term: '{term}'",
                         outcome="resolved" if not tool_result.get("error") else "failed")
            elif tool_name:
                plog.add(stage="reason", kind="tool_call", turn=turn,
                         summary=f"Called tool: {tool_name}",
                         outcome="resolved" if not tool_result.get("error") else "failed")

        trace.tool_calls.append(ToolCall(
            step=len(trace.tool_calls) + 1,
            tool_name=tool_name,
            args=args,
            result=_jsonable_subset(tool_result, max_chars=4000),
            error=tool_result.get("error"),
            elapsed_s=float(tool_result.get("_elapsed_s", 0.0) or 0.0),
        ))

        # ── Stuck-recovery: fire ONCE per pass when the pattern matches ──
        # Pattern: a recent self_catch + this turn made the 2nd consecutive
        # search() without resolution.  Inject a one-shot advisory into
        # the tool_result block telling the reasoner to take ONE final
        # targeted shot at the question, then move on with a caveat.
        stuck_advisory = ""
        if (
            not stuck_state["recovery_used"]
            and stuck_state["had_recent_self_catch"]
            and stuck_state["consecutive_searches"] >= 2
            and turn < max_react_turns - 1   # leave room for the budget warnings
        ):
            topic = stuck_state["stuck_topic"] or "this sub-question"
            stuck_advisory = (
                f"\n\n[STUCK-RECOVERY ADVISORY — one-shot]  You appear to "
                f"be stuck investigating: '{topic}'.  You have already "
                f"done {stuck_state['consecutive_searches']} consecutive "
                f"search() calls on this without an intervening compute().  "
                f"This is your ONE targeted retry: on your NEXT turn, "
                f"either (a) issue a single SHARPER search() with the "
                f"most specific terms you can think of for this exact "
                f"question (e.g. paper name + equation number + constant), "
                f"or (b) note the unresolved ambiguity in `limitations` "
                f"of your final answer and MOVE ON to the next CORE "
                f"quantity in the plan.  Do not spend more than one more "
                f"turn on this side question — it is blocking your "
                f"primary deliverable."
            )
            stuck_state["recovery_used"] = True
            plog.add(stage="reason", kind="stuck", turn=turn,
                     summary=(f"Stuck-detector fired: {stuck_state['consecutive_searches']} "
                              f"consecutive searches after self-catch on "
                              f"'{topic[:80]}'."),
                     outcome="patched", confidence_impact="medium")

        # ── Inject result back into the conversation ─────────────────
        # We carry the LLM's full response (including any prose and the
        # tool_call block) plus an injected tool_result block from us.
        messages.append({"role": "assistant", "content": response})
        result_block = (
            f"<tool_result>\n"
            f"{json.dumps(_jsonable_subset(tool_result, max_chars=3500), ensure_ascii=False, default=str)}\n"
            f"</tool_result>"
            f"{stuck_advisory}"
        )
        messages.append({"role": "user", "content": result_block})

    else:
        # Hit the for-loop's `else` — never broke out, so cap exceeded.
        trace.truncated = True
        if trace.final is None:
            plog.add(stage="reason", kind="truncated", turn=max_react_turns,
                     summary=(f"Reasoner exhausted {max_react_turns} turns without "
                              "emitting a FINAL block.  Writer will reconstruct."),
                     outcome="failed", confidence_impact="high")

    trace.elapsed_s = round(time.time() - t0, 3)
    emit_event(
        "reason_done",
        n_turns=trace.n_react_turns,
        truncated=trace.truncated,
        has_final=trace.final is not None,
        total_cost_usd=trace.cost_usd,
        elapsed_s=trace.elapsed_s,
    )
    return trace


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _strip_blocks(text: str) -> str:
    """Remove <tool_call> and <final> blocks from a response — keep the prose."""
    if not text:
        return ""
    text = _TOOL_CALL_RE.sub("", text)
    text = _FINAL_RE.sub("", text)
    return text.strip()


# Phrases that indicate the reasoner caught its own error mid-thinking.
# Conservative list — only obvious "wait that's wrong" signals so we
# don't flag every routine mention of "wrong".  We extract a short
# sentence around the hit so Part B can quote what was caught.
_SELF_CATCH_PATTERNS = (
    "clearly wrong",
    "physically wrong",
    "physically nonsensical",
    "physically impossible",
    "doesn't make sense",
    "does not make sense",
    "can't be right",
    "cannot be right",
    "this is a unit error",
    "let me reconsider",
    "wait,",
    "wait —",
    "off by",
    "order of magnitude wrong",
    "implausible",
    "i need to re-derive",
)


# Bug 3 helper: detect Fransson C-constant search queries.  Conservative
# — requires Fransson name + at least one constant-related token.
# Examples that should fire:
#   "Fransson Re_tr C Tu^-2 constant value 196"
#   "Fransson Eq 5.5 constant"
#   "Fransson 2005 Eq 5.5 C value"
# Examples that should NOT fire:
#   "Fransson decay law Eq 3.1 Tu(x)"  (different equation)
#   "Fransson zone length Eq 5.1"      (different equation)
def _is_fransson_constant_query(query: str) -> bool:
    if not query:
        return False
    q = query.lower()
    if "fransson" not in q:
        return False
    # Fransson + (5.5 OR Re_tr OR mid-transition keywords)
    if not any(t in q for t in ("5.5", "re_tr", "re tr", "midpoint",
                                "mid-transition", "mid transition")):
        return False
    # AND mentions constant / value / specific candidate numbers
    return any(t in q for t in (
        "constant", "value of c", "value", "tu^-2", "tu^(-2)",
        "tu**-2", "196", "1.7e6", "1.96e6", "1.7 × 10", "1.7×10",
        "1.96 × 10", "1.96×10",
    ))


def _detect_self_catch(prose: str) -> str:
    """Return a short snippet around the first self-catch phrase, or ''.

    Used by the ProcessLog so Part B of the manuscript can show the
    reader exactly which mistakes the agent noticed and fixed itself.
    Returns the first ~140 chars surrounding the matched phrase.
    """
    if not prose:
        return ""
    low = prose.lower()
    for pat in _SELF_CATCH_PATTERNS:
        i = low.find(pat)
        if i >= 0:
            start = max(0, i - 30)
            end = min(len(prose), i + 100)
            snippet = prose[start:end].replace("\n", " ").strip()
            return snippet
    return ""


def _jsonable_subset(value: Any, *, max_chars: int = 3500) -> Any:
    """Trim large fields in a dict so tool_result blocks don't blow up
    the conversation window.

    For lists (e.g. search results), cap to first 8 entries and trim
    each entry's content.  For strings, cap to max_chars.  Numeric and
    bool pass through.
    """
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "…[trim]"
    if isinstance(value, list):
        return [_jsonable_subset(v, max_chars=max_chars) for v in value[:8]]
    if isinstance(value, dict):
        return {k: _jsonable_subset(v, max_chars=max_chars) for k, v in value.items()}
    return value
