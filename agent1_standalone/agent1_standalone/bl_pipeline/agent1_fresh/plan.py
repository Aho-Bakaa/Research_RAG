"""plan.py — the planner node.

ONE Sonnet call.  Reads user_query + flow + validated_chunks + paper
inventory, produces a structured `Plan` dataclass committing to:

    • which quantities to compute (with sources + dependencies)
    • which cross-checks to run
    • how decay will be USED (not just mentioned)
    • how many tool calls the reasoner should expect

Why a separate node (not just a phase inside the reasoner):

    • The plan is an AUDITABLE ARTIFACT — sits at AgentResult.plan,
      visible in the UI, comparable across runs.
    • A downstream deterministic verifier checks plan-vs-execution
      coverage and triggers a targeted revision if items were dropped.
    • The reasoner stays focused on EXECUTION, not on planning —
      saves tokens and turns.

Production-grade rules:
  • Never raises — on LLM error, returns an empty Plan with the error
    captured in `rationale`.  The orchestrator can still run a
    "best-effort" reasoner pass; verifier will flag missing coverage.
  • Validates the planner's source citations against the actual
    paper_ids in `chunks` and the inventory — hallucinated sources are
    dropped with a warning.
"""
from __future__ import annotations

import time
from typing import Any

from bl_pipeline.agent1_fresh.paper_inventory import (
    format_paper_inventory_for_prompt,
    load_all_paper_summaries,
)
from bl_pipeline.agent1_fresh.prompts import PLANNER_SYSTEM
from bl_pipeline.agent1_fresh.state import Chunk, FlowConditions, Plan, PlanQuantity, emit_event
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


# ══════════════════════════════════════════════════════════════════════
# User-message builder
# ══════════════════════════════════════════════════════════════════════

def _format_chunks_for_planner(chunks: list[Chunk], n: int = 15) -> str:
    """Compact chunk listing for the planner.

    Smaller window than the executor (15 vs 12) — planner needs to see
    WHICH papers/equations are available, not the full content.  We trim
    each chunk to ~600 chars so the planner prompt stays under 12k tokens.
    """
    out = []
    for i, c in enumerate(chunks[:n], 1):
        body = (c.content or "").strip().replace("\n", " ")
        if len(body) > 600:
            body = body[:600] + "…[trim]"
        out.append(
            f"[{i}] {c.chunk_id}  paper={c.paper_id}  page={c.page}  "
            f"score={c.score:.2f}\n  {body}"
        )
    return "\n".join(out) if out else "(no chunks)"


def _build_user_message(
    user_query: str, flow: FlowConditions, chunks: list[Chunk],
) -> str:
    """Build the planner's USER message.

    Important architectural choice (validated by `diag_planner_sonnet.py`,
    run 2026-05-16): the planner does NOT receive raw chunks.  It plans
    against the PAPER INVENTORY (which paper supplies which equations) +
    flow + query.  Feeding the full chunk pool to the planner pushed the
    prompt to ~25k tokens of chunk text and Sonnet stopped emitting clean
    JSON — see run #6, every planner call returned prose.  In the
    diagnostic call WITHOUT chunks, Sonnet returned a clean 9-quantity
    plan with full dependency graph in $0.035.

    Separation of concerns:
      • Planner: strategy.  Reads inventory.  Decides quantities + sources.
      • Reasoner: execution.  Reads chunks.  Computes via tools.
      • Both share the user query + flow.

    The `chunks` parameter is retained on the signature for the
    citation-validation step (paper_ids that appear in retrieved chunks
    are also accepted as valid sources) but not rendered into the prompt.
    """
    flow_lines = []
    if flow.velocity_ms is not None:
        flow_lines.append(f"  U_inf = {flow.velocity_ms} m/s")
    if flow.turbulence_intensity_pct is not None:
        flow_lines.append(f"  Tu    = {flow.turbulence_intensity_pct} % (percent)")
    if flow.kinematic_viscosity_m2s is not None:
        flow_lines.append(f"  nu    = {flow.kinematic_viscosity_m2s} m^2/s")
    if flow.length_scale_m is not None:
        flow_lines.append(f"  Lambda(FST) = {flow.length_scale_m} m")
    if flow.chord_m is not None:
        flow_lines.append(f"  chord = {flow.chord_m} m")
    if flow.geometry_description:
        flow_lines.append(f"  geometry = {flow.geometry_description}")
    if flow.pressure_gradient_description:
        flow_lines.append(f"  pressure_gradient = {flow.pressure_gradient_description}")
    # Grid spec (Roach 1987 Eq.18 → grid-derived Lambda_x; FS20 satisfiability)
    if getattr(flow, "grid_solidity", None) is not None:
        flow_lines.append(
            f"  grid: sigma={flow.grid_solidity}, "
            f"d_bar={flow.grid_bar_diameter_mm} mm, "
            f"x_grid={flow.grid_position_m} m"
        )
    # Blasius-VO inputs (task #221) — SAME exposure as the OPTIMIZER sees
    # in retrieve.py:_build_flow_context_for_optimizer.  Without these
    # lines, PLANNER cannot see the user's sidebar inputs and the VO
    # dispatch branch added today (2026-06-13) never fires.
    if getattr(flow, "x_0_BL_m", None) is not None:
        flow_lines.append(
            f"  x_0_BL_direct = {flow.x_0_BL_m * 1000:.1f} mm  "
            f"(BLASIUS-VO INPUT → dispatch via "
            f"dubey_2026_thesis::blasius_VO_Tu_effective_method; use "
            f"Tu_effective = Tu(x_0_BL) for ALL three onset correlations)"
        )
    _delta_meas = getattr(flow, "delta_x_measurements", None)
    if _delta_meas:
        try:
            _pairs = ", ".join(
                f"({x * 1000:.0f},{d * 1000:.2f})"
                for (x, d) in _delta_meas
            )
        except Exception:
            _pairs = "<unparseable>"
        flow_lines.append(
            f"  delta_x_measurements (x_mm, delta99_mm) = [{_pairs}]  "
            f"({len(_delta_meas)} stations; BLASIUS-VO INPUT → fit "
            f"delta^2 = K*(x - x_0_BL) linearly, then dispatch via "
            f"dubey_2026_thesis::blasius_VO_Tu_effective_method)"
        )
    flow_block = "\n".join(flow_lines) or "  (none parsed)"

    summaries = load_all_paper_summaries()
    inventory = (
        format_paper_inventory_for_prompt(summaries) if summaries else "(empty)"
    )

    return f"""USER_QUERY:
"{user_query}"

FLOW:
{flow_block}

PAPER INVENTORY (which paper supplies which closed-form quantities — \
this is your STRATEGIC view of what's in the corpus; the downstream \
reasoner has the full retrieved chunks for execution):

{inventory}

Produce the JSON plan now.  Commit to specific quantities, sources, and \
dependencies.  Spell out how decay will be USED (not just mentioned).
"""


# ══════════════════════════════════════════════════════════════════════
# Plan validation — drop hallucinated source citations
# ══════════════════════════════════════════════════════════════════════

def _valid_paper_ids(chunks: list[Chunk]) -> set[str]:
    """Set of paper_ids that legitimately appear in the chunk pool +
    inventory.  Used to filter hallucinated citations.
    """
    pids = {c.paper_id for c in chunks if c.paper_id and c.paper_id != "?"}
    # Also accept summarised papers — those are valid sources even if
    # they didn't surface in this query's chunk pool.
    for entry in load_all_paper_summaries():
        if entry.get("paper_id"):
            pids.add(entry["paper_id"])
    return pids


def _filter_source_papers(
    sources: list[str], valid_pids: set[str],
) -> list[str]:
    """Drop hallucinated `paper_id::pPAGE` citations."""
    out: list[str] = []
    for s in sources or []:
        s = str(s).strip()
        if not s:
            continue
        # paper_id is everything before "::"
        pid = s.split("::", 1)[0].strip()
        if pid in valid_pids:
            out.append(s)
    return out


# ══════════════════════════════════════════════════════════════════════
# Salvage: extract complete {...} entries from a TRUNCATED JSON array.
# ══════════════════════════════════════════════════════════════════════

def _salvage_quantities(raw: str) -> list[dict[str, Any]]:
    """Best-effort recovery when Sonnet's planner response was truncated
    mid-array.  Walks the raw text from after `"quantities_to_compute": [`
    and returns every complete `{...}` object up to the first incomplete
    one.

    Used only as a last-resort fallback — if max_tokens is set correctly
    we never get here.  But when we DO, salvaging 6 complete quantities
    from a truncated response of 12 is far better than emitting 0.
    """
    if not raw:
        return []
    marker = '"quantities_to_compute"'
    i = raw.find(marker)
    if i < 0:
        return []
    # Find the opening `[` after the marker
    bracket = raw.find("[", i)
    if bracket < 0:
        return []

    pos = bracket + 1
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    entries: list[str] = []

    while pos < len(raw):
        c = raw[pos]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                if depth == 0:
                    start = pos
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    entries.append(raw[start:pos + 1])
                    start = None
            elif c == "]" and depth == 0:
                break
        pos += 1

    out: list[dict[str, Any]] = []
    import json
    for entry in entries:
        try:
            obj = json.loads(entry)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


# ══════════════════════════════════════════════════════════════════════
# Public entry — the planner node
# ══════════════════════════════════════════════════════════════════════

def make_plan(
    user_query: str,
    flow: FlowConditions,
    chunks: list[Chunk],
    *,
    pipeline_state: Any = None,
) -> Plan:
    """Build a structured Plan via one Sonnet call.

    Returns an empty-ish Plan on LLM error; never raises.  The orchestrator
    can still proceed to a best-effort reasoner pass, and the verifier
    will flag coverage gaps loudly when execution returns.
    """
    t0 = time.time()
    emit_event("plan_started", user_query=user_query, n_chunks=len(chunks))

    user_msg = _build_user_message(user_query, flow, chunks)

    # max_tokens=8000 (was 4000): real measurements on the live corpus
    # show a 21-quantity plan can hit ~9k-10k output tokens with full
    # method + dependency + decay-handling text.  4000 truncated the
    # array mid-entry → parse_lenient_json could not recover → 0
    # quantities → executor wandered.  8000 gives 2x headroom; we ALSO
    # salvage partial arrays in `_salvage_quantities()` below as a
    # belt-and-suspenders for the rare case where 8000 isn't enough.
    response: str = ""
    cost_total = 0.0
    tokens_out_total = 0
    parsed: Any = None
    for attempt in (1, 2):
        try:
            response, usage = llm.call(
                task="agent1_fresh_plan",        # → Sonnet
                system=PLANNER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                max_tokens=8000,
                temperature=0.0,
                pipeline_state=pipeline_state,
            )
            cost_total += float(getattr(usage, "cost_usd", 0.0) or 0.0)
            tokens_out_total += int(getattr(usage, "tokens_out", 0) or 0)
        except Exception as e:
            plan = Plan(
                rationale=f"planner LLM call failed: {type(e).__name__}: {e}",
                cost_usd=round(cost_total, 4),
                elapsed_s=round(time.time() - t0, 3),
            )
            emit_event("plan_done", error=str(e), n_quantities=0,
                       cost_usd=cost_total, elapsed_s=plan.elapsed_s)
            return plan

        parsed = parse_lenient_json(response)
        # Bare-list shape: treat as the quantities array directly
        if isinstance(parsed, list):
            parsed = {"quantities_to_compute": parsed}
        elif not isinstance(parsed, dict):
            # Most common cause: response was VALID JSON but TRUNCATED
            # at max_tokens.  Try to salvage complete quantity entries
            # from the partial array before giving up.
            salvaged = _salvage_quantities(response or "")
            raw = response or ""
            preview = raw[:300].replace("\n", " ")
            likely_truncated = raw.lstrip().startswith(("{", "["))
            emit_event(
                "plan_parse_failed",
                attempt=attempt,
                raw_preview=preview,
                response_length=len(raw),
                tokens_out=int(getattr(usage, "tokens_out", 0) or 0),
                likely_truncated=likely_truncated,
                salvaged_n=len(salvaged),
            )
            if salvaged:
                parsed = {"quantities_to_compute": salvaged}
            else:
                parsed = {}

        if parsed.get("quantities_to_compute"):
            break  # got a real plan, stop retrying
        if attempt == 1:
            emit_event("plan_retry",
                       reason="empty quantities_to_compute on attempt 1",
                       raw_preview=(response or "")[:300])

    cost = cost_total

    valid_pids = _valid_paper_ids(chunks)

    # Build PlanQuantity list
    quantities: list[PlanQuantity] = []
    for entry in parsed.get("quantities_to_compute", []) or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        sources = _filter_source_papers(
            entry.get("source_papers") or [], valid_pids,
        )
        quantities.append(PlanQuantity(
            name=name,
            method=str(entry.get("method", ""))[:300],
            source_papers=sources,
            depends_on=[str(d) for d in (entry.get("depends_on") or []) if d],
            expected_unit=str(entry.get("expected_unit", ""))[:30],
            is_core=bool(entry.get("is_core", True)),
        ))

    # Parse the new structured fields the upgraded PLANNER_SYSTEM emits.
    sub_queries = [
        str(s).strip()[:200] for s in (parsed.get("sub_queries") or []) if s
    ][:10]  # cap at 10 sub-queries

    raw_deferred = parsed.get("deferred_to_other_agents") or []
    deferred: list[dict[str, Any]] = []
    for entry in raw_deferred:
        if not isinstance(entry, dict):
            continue
        deferred.append({
            "quantity": str(entry.get("quantity", ""))[:120],
            "reason": str(entry.get("reason", ""))[:200],
            "recommend_agent": str(entry.get("recommend_agent", ""))[:80],
        })

    # ENFORCE the ≤8 quantities rule (Invariant I1 in PLANNER_SYSTEM).
    # If the planner still emitted more, keep CORE first, then highest-
    # priority non-core up to 8 total.  Better to truncate gracefully
    # than to ship a bloated plan that confuses the verifier.
    if len(quantities) > 8:
        core_q = [q for q in quantities if q.is_core]
        non_core_q = [q for q in quantities if not q.is_core]
        kept = (core_q + non_core_q)[:8]
        dropped = (core_q + non_core_q)[8:]
        emit_event(
            "plan_quantities_truncated",
            kept=len(kept),
            dropped=[q.name for q in dropped],
            cap=8,
        )
        quantities = kept

    plan = Plan(
        sub_queries=sub_queries,
        quantities_to_compute=quantities,
        cross_checks=[str(x) for x in (parsed.get("cross_checks") or []) if x],
        decay_handling=str(parsed.get("decay_handling", ""))[:600],
        probe_layout_notes=str(parsed.get("probe_layout_notes", ""))[:400],
        deferred_to_other_agents=deferred,
        # NEW: planner's honest list of missing-chunk descriptions; capped
        # to 10 gaps × 200 chars so a runaway emit doesn't bloat the trace.
        retrieval_gaps=[
            str(g).strip()[:200]
            for g in (parsed.get("retrieval_gaps") or [])
            if g and isinstance(g, str)
        ][:10],
        expected_tool_calls=int(parsed.get("expected_tool_calls", 0) or 0),
        rationale=str(parsed.get("rationale", ""))[:1200],
        cost_usd=round(cost, 4),
        elapsed_s=round(time.time() - t0, 3),
    )

    # If we STILL have no quantities after retry, log the raw response
    # in the rationale so the next debug session can see what Sonnet
    # actually emitted (without bloating every plan dump).
    if not plan.quantities_to_compute and response:
        preview = response[:800].replace("\n", " ")
        plan.rationale = (plan.rationale or "") + (
            f"  [DEBUG: planner returned no usable quantities after retry. "
            f"Raw response preview: {preview!r}]"
        )

    # Serialize the full plan content into the event payload so the
    # frontend can render the actual quantities / cross-checks / decay
    # method, not just counts.  Run 76d5b351 (2026-06-13) shipped only
    # the metadata counts (n_quantities=8, etc.) — the frontend's
    # PlanPane reads plan.quantities_to_compute (a LIST), saw undefined,
    # rendered "0 Quantities" even though the actual plan had 8.  The
    # fix: include the structured lists in the payload alongside counts.
    _q_dicts: list[dict] = []
    for q in plan.quantities_to_compute:
        try:
            _q_dicts.append({
                "quantity":    q.name,
                "unit":        q.expected_unit,
                "method":      (q.method or "")[:600],
                "sources":     list(q.source_papers or []),
                "is_core":     bool(q.is_core),
                "depends_on":  list(q.depends_on or []),
            })
        except Exception as _e:
            emit_event("plan_payload_quantity_serialization_failed",
                       error_type=type(_e).__name__,
                       error=str(_e)[:200])
            continue
    emit_event(
        "plan_done",
        n_sub_queries=len(plan.sub_queries),
        n_quantities=len(plan.quantities_to_compute),
        n_core_quantities=sum(1 for q in plan.quantities_to_compute if q.is_core),
        n_cross_checks=len(plan.cross_checks),
        decay_handled=bool(plan.decay_handling),
        probe_layout_planned=bool(plan.probe_layout_notes),
        n_deferred=len(plan.deferred_to_other_agents),
        n_retrieval_gaps=len(plan.retrieval_gaps),
        expected_tool_calls=plan.expected_tool_calls,
        tokens_out=tokens_out_total,
        cost_usd=plan.cost_usd,
        elapsed_s=plan.elapsed_s,
        # NEW (2026-06-13) — actual plan content so the UI can render
        # the quantities + cross-checks + decay method.  Keep field
        # names matching the frontend's PlanPane reads.
        quantities_to_compute=_q_dicts,
        cross_checks=list(plan.cross_checks or []),
        decay_handling=str(plan.decay_handling or "")[:600],
        probe_layout_notes=str(plan.probe_layout_notes or "")[:400],
        deferred_to_other_agents=list(plan.deferred_to_other_agents or []),
        retrieval_gaps=list(plan.retrieval_gaps or []),
        rationale=str(plan.rationale or "")[:1200],
    )
    return plan
