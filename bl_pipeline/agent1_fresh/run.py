"""run.py — public entry point for fresh Agent 1.

End-to-end:

    user query
       │
       ▼
    [1] parse        Haiku — extract flow numbers + verbatim query
       │
       ▼
    [2] retrieve     adaptive 3-iter loop (rerank inside, supplementary on gaps)
       │
       ▼
    [3] reason       Sonnet ReAct (tools: compute, lookup_glossary, search)
       │
       ▼
    [4] critique     Sonnet peer review → PASS / NEEDS_REVISION / FAIL
       │
       ├── if NEEDS_REVISION: rerun [3] with concerns in context, recritique once
       ▼
    [5] write        Sonnet — markdown answer with LaTeX + inline citations
       │
       ▼
    AgentResult — JSON-serialisable, manuscript inside

Production-grade rules:
  • Top-level try/except: pipeline never raises; on catastrophe we still
    return an AgentResult with status="error" and a useful trace.
  • Every node's cost + latency aggregated into the result.
  • Events emitted at every transition; UI can render live.
  • Run ID = short UUID + timestamp (deterministic ordering when needed).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from bl_pipeline.agent1_fresh.critique import critique as run_critique
from bl_pipeline.agent1_fresh.plan import make_plan
from bl_pipeline.agent1_fresh.plot import generate_plots
from bl_pipeline.agent1_fresh.prompts import PARSE_SYSTEM
from bl_pipeline.agent1_fresh.provenance import enforce_provenance
from bl_pipeline.agent1_fresh.reason import react_reason
from bl_pipeline.agent1_fresh.retrieve import adaptive_retrieve
from bl_pipeline.agent1_fresh.state import (
    AgentResult,
    Chunk,
    CritiqueResult,
    FlowConditions,
    ProcessLog,
    ReasonerTrace,
    emit_event,
)
from bl_pipeline.agent1_fresh.verify import verify_plan_coverage
from bl_pipeline.agent1_fresh.write import write_manuscript
from bl_pipeline.rag.engine import RAGEngine
from bl_pipeline.shared.config import RUNS_DIR
from bl_pipeline.shared.event_bus import (
    set_current_run, clear_current_run,
    install_default_listeners, close_default_listeners,
)
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


# ══════════════════════════════════════════════════════════════════════
# Node [1] — Parse
# ══════════════════════════════════════════════════════════════════════

def _parse_query(
    user_query: str,
    pipeline_state: Any = None,
    *,
    facility_overrides: dict | None = None,
) -> tuple[FlowConditions, float, list[str]]:
    """Run the entry parser (Haiku).  Returns (flow, cost_usd, override_log_lines).

    Never raises — on LLM error returns a default FlowConditions with
    every field None and `defaulted_fields=["llm_unavailable"]`.

    CRITICAL — applies facility_overrides BEFORE emitting parse_done.
    Sidebar values (U / Tu / ν / chord / Λ_x / x_0_BL_m / delta_x_measurements)
    are user-supplied and have priority over NL-parsed values, but if we
    emit parse_done with the pre-merge flow, the frontend's Parse panel
    renders the null pre-override values and the user concludes the
    sidebar inputs were ignored (run 2026-06-13).  Merging inside this
    function and emitting AFTER ensures parse_done carries the final
    flow exactly as downstream stages will see it.
    """
    emit_event("parse_started", user_query=user_query)
    t0 = time.time()
    try:
        response, usage = llm.call(
            task="agent1_fresh_parse",      # → Haiku
            system=PARSE_SYSTEM,
            messages=[{"role": "user", "content": user_query}],
            max_tokens=500,
            temperature=0.0,
            pipeline_state=pipeline_state,
        )
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
    except Exception as e:
        flow = FlowConditions(defaulted_fields=["llm_unavailable"])
        # Even on LLM failure, give facility_overrides a chance to fill
        # the flow so the rest of the pipeline can still run with the
        # operator-provided values.
        override_log: list[str] = []
        if facility_overrides:
            for key, value in facility_overrides.items():
                if value is None:
                    continue
                if hasattr(flow, key):
                    setattr(flow, key, value)
                    override_log.append(f"{key}={value}")
        emit_event("parse_done", flow=flow,
                   error=f"{type(e).__name__}: {e}",
                   elapsed_s=round(time.time() - t0, 3))
        return flow, 0.0, override_log

    parsed = parse_lenient_json(response)
    # Defensive against list-shape / non-dict responses (same pattern
    # we hardened in _judge, critique, plan, _expand_query — every JSON
    # parse site must check shape before calling .get()).
    if not isinstance(parsed, dict):
        parsed = {}
    f = parsed.get("flow") or {}
    if not isinstance(f, dict):
        f = {}
    flow = FlowConditions(
        velocity_ms=_safe_float(f.get("velocity_ms")),
        turbulence_intensity_pct=_safe_float(f.get("turbulence_intensity_pct")),
        kinematic_viscosity_m2s=_safe_float(f.get("kinematic_viscosity_m2s")),
        length_scale_m=_safe_float(f.get("length_scale_m")),
        chord_m=_safe_float(f.get("chord_m")),
        grid_solidity=_safe_float(f.get("grid_solidity")),
        grid_bar_diameter_mm=_safe_float(f.get("grid_bar_diameter_mm")),
        grid_position_m=_safe_float(f.get("grid_position_m")),
        geometry_description=_safe_str(f.get("geometry_description")),
        pressure_gradient_description=_safe_str(f.get("pressure_gradient_description")),
        roughness_um=_safe_float(f.get("roughness_um")),
        defaulted_fields=[str(x) for x in (f.get("defaulted_fields") or []) if x],
    )
    # Merge facility_overrides BEFORE emitting parse_done so the frontend's
    # Parse panel renders the final flow the downstream stages will use.
    override_log = []
    if facility_overrides:
        for key, value in facility_overrides.items():
            if value is None:
                continue
            if hasattr(flow, key):
                setattr(flow, key, value)
                override_log.append(f"{key}={value}")
    emit_event("parse_done", flow=flow, cost_usd=cost,
               elapsed_s=round(time.time() - t0, 3),
               facility_overrides_applied=override_log)
    return flow, cost, override_log


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ══════════════════════════════════════════════════════════════════════
# Revision loop helper
# ══════════════════════════════════════════════════════════════════════

def _revision_kickoff_message(
    user_query: str, prior_trace: ReasonerTrace,
    concerns: list[Any], summary: str,
) -> str:
    """Build a 'rerun reasoner with concerns' addendum.

    Used when critique returns NEEDS_REVISION.  We don't re-feed the
    full trace — that would explode tokens; instead we summarise what
    the prior pass produced and pin the concerns the critic raised.
    """
    concerns_lines = [
        f"  - [{c.severity}][{c.category}] {c.issue}  (fix: {c.suggested_fix})"
        for c in concerns
    ] or ["  (no specific concerns listed)"]

    prior_final = prior_trace.final or {}
    prior_summary = (
        json.dumps(prior_final, ensure_ascii=False, indent=2, default=str)
        if prior_final else "(no prior FINAL block — trace was truncated)"
    )

    return f"""USER_QUERY:
"{user_query}"

PRIOR FINDINGS (your previous pass — revise these to address the concerns below):
{prior_summary}

CRITIC CONCERNS to address in this revision:
{chr(10).join(concerns_lines)}

CRITIC OVERALL: {summary}

You may re-call tools as needed.  Address each concern explicitly in your \
new <final> block.  If a concern is wrong, push back in the prose and explain \
why — but cite a chunk for the pushback."""


# ══════════════════════════════════════════════════════════════════════
# Public entry
# ══════════════════════════════════════════════════════════════════════

def run(
    user_query: str,
    *,
    max_retrieval_iterations: int = 3,
    max_react_turns: int = 12,
    max_revisions: int = 1,        # 0 = no revision loop; 1 = one retry on NEEDS_REVISION
    max_cost_usd: float = 10.00,   # HARD per-run cap; skip further passes when exceeded
                                   # Bumped from 5.00 → 10.00 on 2026-06-15 for live demo
                                   # head-room (Bumped from 2.50 → 5.00 on 2026-06-03 to
                                   # give the reasoner head-room for lookup_equation chains
                                   # e.g., Roach Eq.18 → Kurian-Fransson Eq.8 derivation of Λ_x).
    rag: RAGEngine | None = None,
    run_id: str | None = None,
    facility_overrides: dict | None = None,
) -> AgentResult:
    """Run fresh Agent 1 end-to-end.

    Returns a fully populated `AgentResult`.  Never raises.  Caller can
    inspect `result.status`; on "error", `result.error` holds the cause.

    Cost protection:
        `max_cost_usd` is a HARD ceiling checked after each major node.
        If cumulative spend exceeds it, the orchestrator skips remaining
        revision passes and the critic, and produces a write-up from
        whatever the reasoner has so far.  Default $10.00 keeps a runaway
        from costing more than a small coffee.  One earlier run with the
        planner silently failing cost $5+ before this guard existed.
    """
    t_start = time.time()
    run_id = run_id or f"a1f_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    result = AgentResult(run_id=run_id, user_query=user_query)
    # Process log is created up front and threaded through every node
    # so the writer can produce an honest "how the agent solved it"
    # narrative (Part B of the manuscript).
    plog: ProcessLog = result.process_log
    plog.add(stage="run", kind="info",
             summary=f"Run started: '{user_query[:140]}'")

    # The shared `emit()` drops events when no run is in context.
    # Pin this run as current so every downstream emit lands on the bus.
    set_current_run(run_id)
    # Persist every event to events.jsonl (the same dir the API /events
    # endpoint reads) so the frontend LiveLog can REPLAY the full history on
    # connect/refresh — not just stream live from the moment it subscribes.
    # Without this, a tab opened mid-run shows only the tail (the "6 events"
    # problem).  Best-effort; never blocks the run.
    try:
        install_default_listeners(
            run_id,
            events_file=RUNS_DIR / run_id / "agent1_output" / "events.jsonl",
        )
    except Exception:
        pass
    emit_event("agent_started", run_id=run_id, user_query=user_query)

    try:
        # ── [1] Parse ──────────────────────────────────────────────
        # facility_overrides are merged INSIDE _parse_query before the
        # parse_done event is emitted (2026-06-13 fix — without this the
        # frontend Parse panel rendered the pre-override null values
        # even though the merged flow had the operator's sidebar inputs).
        flow, parse_cost, override_log = _parse_query(
            user_query, facility_overrides=facility_overrides,
        )
        result.flow = flow
        result.total_cost_usd = round(result.total_cost_usd + parse_cost, 4)

        # Audit-trail entries for each override applied by _parse_query.
        for entry in override_log:
            plog.add(
                stage="parse", kind="info",
                summary=f"Facility override applied: {entry} (from UI input)",
            )

        plog.add(stage="parse", kind="info",
                 summary=(f"Parsed flow conditions: U={flow.velocity_ms} m/s, "
                          f"Tu={flow.turbulence_intensity_pct}%, "
                          f"nu={flow.kinematic_viscosity_m2s} m²/s, "
                          f"chord={flow.chord_m} m"))

        # ── [2] Retrieve ──────────────────────────────────────────
        rag = rag or RAGEngine()
        retrieval = adaptive_retrieve(
            user_query, rag=rag, max_iterations=max_retrieval_iterations,
            flow=flow,    # NEW — gives OPTIMIZER structured grid/Lambda_x context
        )
        result.retrieval = retrieval
        result.total_cost_usd = round(
            result.total_cost_usd + retrieval.total_cost_usd, 4,
        )
        plog.add(stage="retrieve", kind="info",
                 summary=(f"Retrieved {retrieval.final_pool_size} chunks across "
                          f"{retrieval.n_iterations} iteration(s); "
                          f"judge_sufficient={retrieval.sufficient}"),
                 outcome="resolved" if retrieval.sufficient else "patched",
                 confidence_impact="" if retrieval.sufficient else "medium")

        if not retrieval.validated_chunks:
            # No evidence — write a deferred manuscript.
            result.manuscript = (
                f"# Cannot answer — no relevant evidence found\n\n"
                f"The user query *{user_query!r}* did not surface any "
                f"chunks the retrieval system judged relevant after "
                f"{retrieval.n_iterations} iteration(s).  Please refine "
                f"the query or check the corpus coverage."
            )
            result.status = "complete"
            result.total_elapsed_s = round(time.time() - t_start, 3)
            emit_event("agent_done", status=result.status,
                       cost_usd=result.total_cost_usd,
                       elapsed_s=result.total_elapsed_s, no_evidence=True)
            return result

        # ── [3] Plan ──────────────────────────────────────────────
        plan = make_plan(
            user_query=user_query, flow=flow,
            chunks=retrieval.validated_chunks,
        )
        result.plan = plan
        result.total_cost_usd = round(result.total_cost_usd + plan.cost_usd, 4)
        n_q = len(plan.quantities_to_compute)
        n_core = sum(1 for q in plan.quantities_to_compute if q.is_core)
        if n_q == 0:
            plog.add(stage="plan", kind="planner_failed",
                     summary="Planner produced no quantities (parse failure).",
                     outcome="failed", confidence_impact="high")
        else:
            plog.add(stage="plan", kind="info",
                     summary=(f"Plan committed: {n_q} quantities ({n_core} core), "
                              f"{len(plan.cross_checks)} cross-checks, "
                              f"decay_handled={bool(plan.decay_handling)}"),
                     outcome="resolved")

        # ── [4] ReAct executor (now plan-aware) ───────────────────
        reasoner = react_reason(
            user_query=user_query,
            flow=flow,
            chunks=retrieval.validated_chunks,
            plan=plan,
            max_react_turns=max_react_turns,
            process_log=plog,
            # Pass remaining budget down so the reasoner can self-abort
            # if a single pass alone tries to blow the cap.  This is the
            # mid-pass guard that the earlier between-nodes guard couldn't
            # catch — one reasoner pass once spent $1.58 in this codebase.
            max_cost_usd=max(0.10, max_cost_usd - result.total_cost_usd),
        )
        result.reasoner = reasoner
        result.total_cost_usd = round(result.total_cost_usd + reasoner.cost_usd, 4)

        # ── [5] Verify plan-vs-execution coverage (deterministic) ─
        coverage = verify_plan_coverage(plan, reasoner)
        result.coverage = coverage

        # ── [5b] Auto-revision when CORE coverage gaps exist ─────
        # Cost-cap guard: if we've already burned the budget on
        # retrieval + planner + first reasoner pass, skip revision.
        # Better to ship a degraded result than charge the user $5.
        over_budget = result.total_cost_usd > max_cost_usd
        if over_budget:
            emit_event("cost_cap_hit",
                       stage="before_coverage_revision",
                       cost_usd=result.total_cost_usd,
                       cap_usd=max_cost_usd)
        if coverage.needs_revision and max_revisions > 0 and not over_budget:
            emit_event("coverage_revision_started",
                       missing_core=coverage.missing_core)
            plog.add(stage="verify", kind="coverage_revision",
                     summary=(f"Coverage gap detected — firing revision pass for "
                              f"{len(coverage.missing_core)} missing CORE quantity(ies): "
                              f"{coverage.missing_core[:5]}"),
                     outcome="patched", confidence_impact="medium")
            coverage_brief = (
                "Coverage check found the following CORE quantities planned "
                "but not computed in your previous pass:\n"
                + "\n".join(f"  - {q}" for q in coverage.missing_core)
                + "\n\nCompute each of them now via compute() and include in "
                  "your new FINAL block."
            )
            reasoner2 = react_reason(
                user_query=user_query,
                flow=flow,
                chunks=retrieval.validated_chunks,
                plan=plan,
                max_react_turns=max_react_turns,
                is_revision=True,
                revision_brief=coverage_brief,
                prior_findings=reasoner.final or {},
                process_log=plog,
                max_cost_usd=max(0.10, max_cost_usd - result.total_cost_usd),
            )
            # Merge revision into the existing reasoner trace
            reasoner.thinking_segments.extend(
                ["\n--- COVERAGE-DRIVEN REVISION ---\n"] + reasoner2.thinking_segments,
            )
            reasoner.tool_calls.extend(reasoner2.tool_calls)
            reasoner.final = reasoner2.final or reasoner.final
            reasoner.cost_usd = round(reasoner.cost_usd + reasoner2.cost_usd, 4)
            reasoner.elapsed_s = round(reasoner.elapsed_s + reasoner2.elapsed_s, 3)
            reasoner.n_react_turns += reasoner2.n_react_turns
            reasoner.truncated = reasoner.truncated or reasoner2.truncated
            result.total_cost_usd = round(
                result.total_cost_usd + reasoner2.cost_usd, 4,
            )
            # Re-verify with the merged trace
            coverage = verify_plan_coverage(plan, reasoner)
            result.coverage = coverage
            emit_event("coverage_revision_done",
                       missing_core_now=coverage.missing_core)
            # NB: max_revisions is decremented implicitly by skipping the
            # critic-driven revision below if coverage already burned the
            # budget.  Honest accounting.
            max_revisions = max(0, max_revisions - 1)

        # ── [6] Critique ──────────────────────────────────────────
        # Thread the remaining budget so critique() can skip its LLM
        # call when the budget is already exhausted (added 2026-06-13
        # task #253 — prevents the cascade failure where critique +
        # writer both fired LLM calls knowing they'd error out).
        crit = run_critique(
            user_query=user_query,
            flow=flow,
            trace=reasoner,
            chunks=retrieval.validated_chunks,
            budget_remaining_usd=max(0.0, max_cost_usd - result.total_cost_usd),
        )
        result.critique = crit
        result.total_cost_usd = round(result.total_cost_usd + crit.cost_usd, 4)

        # ── Optional critic-driven revision ──────────────────────
        # Cost-cap guard again: even if the critic wants a revision,
        # if we're past budget we'd rather ship the degraded result.
        # Also skip if the planner failed (no plan to revise against
        # meaningfully — let the manuscript flag the degraded run).
        over_budget = result.total_cost_usd > max_cost_usd
        plan_empty = plan is None or not plan.quantities_to_compute
        if over_budget:
            emit_event("cost_cap_hit",
                       stage="before_critic_revision",
                       cost_usd=result.total_cost_usd,
                       cap_usd=max_cost_usd)
        if plan_empty:
            emit_event("critic_revision_skipped",
                       reason="planner_produced_no_plan",
                       cost_usd=result.total_cost_usd)
        if (crit.verdict == "NEEDS_REVISION" and max_revisions > 0
                and not over_budget and not plan_empty):
            emit_event("revision_started", concerns=len(crit.concerns))
            plog.add(stage="critique", kind="critic_revision",
                     summary=(f"Critic verdict NEEDS_REVISION with "
                              f"{len(crit.concerns)} concern(s) — firing revision."),
                     outcome="patched", confidence_impact="medium")
            # Build a compact revision brief from the critic's concerns.
            # The reasoner gets this + the prior FINAL block + top-5 chunks
            # only — no paper inventory, no 30-chunk pool re-shipped.
            revision_brief = "\n".join(
                f"  - [{c.severity}][{c.category}] {c.issue}  (fix: {c.suggested_fix})"
                for c in crit.concerns
            ) or "  (no specific concerns listed)"

            reasoner2 = react_reason(
                user_query=user_query,             # ORIGINAL query, not nested
                flow=flow,
                chunks=retrieval.validated_chunks,
                plan=plan,
                max_react_turns=max_react_turns,
                is_revision=True,
                revision_brief=revision_brief,
                prior_findings=reasoner.final or {},
                process_log=plog,
                max_cost_usd=max(0.10, max_cost_usd - result.total_cost_usd),
            )
            # Merge reasoner trace: keep both passes' segments + tool calls
            # for full transparency.
            reasoner.thinking_segments.extend(
                ["\n--- REVISION PASS ---\n"] + reasoner2.thinking_segments
            )
            reasoner.tool_calls.extend(reasoner2.tool_calls)
            reasoner.final = reasoner2.final or reasoner.final
            reasoner.cost_usd = round(reasoner.cost_usd + reasoner2.cost_usd, 4)
            reasoner.elapsed_s = round(reasoner.elapsed_s + reasoner2.elapsed_s, 3)
            reasoner.truncated = reasoner.truncated or reasoner2.truncated
            reasoner.n_react_turns += reasoner2.n_react_turns

            # Re-critique once
            crit2 = run_critique(
                user_query=user_query,
                flow=flow,
                trace=reasoner,
                chunks=retrieval.validated_chunks,
            )
            # Keep the second verdict as the authoritative one but merge concerns
            crit2.concerns = list(crit.concerns) + list(crit2.concerns)
            result.critique = crit2
            result.total_cost_usd = round(
                result.total_cost_usd + reasoner2.cost_usd + crit2.cost_usd, 4,
            )
            emit_event("revision_done", verdict=crit2.verdict,
                       n_concerns=len(crit2.concerns))

        # Log critique outcome to the process log
        if result.critique is not None:
            plog.add(stage="critique", kind="info",
                     summary=(f"Critique verdict: {result.critique.verdict} "
                              f"({len(result.critique.concerns)} concern(s))"),
                     outcome=("resolved" if result.critique.verdict == "PASS"
                              else "patched"))

        # ── [6a-pre] Value reconciliation (CORRECTNESS) ───────────
        # The reasoner often COMPUTES the right value but retypes a WRONG
        # one in the FINAL block.  Pull the computed value back in BEFORE
        # the gate, so reconciled values then pass; only truly un-traceable
        # ones remain flagged.  See provenance.reconcile_findings.
        try:
            from bl_pipeline.agent1_fresh.provenance import reconcile_findings
            recon = reconcile_findings(reasoner)
            if recon.n_reconciled:
                emit_event("values_reconciled", n=recon.n_reconciled)
                plog.add(stage="verify", kind="value_reconciled",
                         summary=(f"Reconciled {recon.n_reconciled} FINAL value(s) "
                                  f"from compute() output (reasoner had retyped "
                                  f"them): {[r[0] for r in recon.reconciled][:5]}"),
                         outcome="resolved", confidence_impact="high")
        except Exception as _rec_e:
            emit_event("value_reconcile_error",
                       error=f"{type(_rec_e).__name__}: {_rec_e}")

        # ── [6a] Provenance gate — ENFORCEMENT, not instruction ───
        # Every numeric value in the FINAL key_findings must trace to a
        # number a compute() actually PRINTED this run, else it is flagged
        # UNVERIFIED.  Structural backstop for prompts.py §3 ("NO MENTAL
        # ARITHMETIC"): the instruction asks; this enforces.
        try:
            prov = enforce_provenance(reasoner)
            result.provenance = prov.to_dict()
            emit_event("provenance_gate_done",
                       n_findings=prov.n_findings,
                       n_verified=prov.n_verified,
                       n_unverified=prov.n_unverified)
            if prov.n_unverified:
                plog.add(stage="verify", kind="provenance_unverified",
                         summary=(f"{prov.n_unverified}/{prov.n_findings} FINAL "
                                  f"value(s) not traced to a compute() result: "
                                  f"{[q for q, _ in prov.unverified][:6]}"),
                         outcome="flagged", confidence_impact="high")
        except Exception as _pe:
            emit_event("provenance_gate_error",
                       error=f"{type(_pe).__name__}: {_pe}")

        # ── [6b] Generate deterministic plots (offline, no LLM) ──
        # Standard plot suite the experimentalist needs: probe layout,
        # intermittency profile, BL thickness, onset comparison, Tu decay.
        # Saved to runs/_logs/<run_id>/plots/.  Writer embeds them in
        # Part A.  If matplotlib isn't installed or any specific plot
        # can't be drawn, this returns whatever it could produce — never
        # raises.
        # Invariant 4: write under the SAME canonical RUNS_DIR the API
        # serves from and Agent 3's output_loader reads from (was the
        # hardcoded runs/_logs, which the API/downstream never looked in).
        plots_dir = RUNS_DIR / run_id / "plots"
        plot_artifacts = generate_plots(reasoner, flow, plan, plots_dir)
        if plot_artifacts:
            plog.add(stage="write", kind="info",
                     summary=(f"Generated {len(plot_artifacts)} plot(s): "
                              f"{[a.label for a in plot_artifacts]}"),
                     outcome="resolved")
        else:
            plog.add(stage="write", kind="info",
                     summary="No plots generated (matplotlib unavailable or insufficient data).",
                     outcome="noted")

        # ── [5] Write ─────────────────────────────────────────────
        # Thread the remaining budget so write_manuscript() can fall
        # back to the deterministic reconstruction when the budget is
        # already exhausted (added 2026-06-13 task #253 — prevents
        # the wasteful "LLM call, fail, deterministic fallback anyway"
        # path that fired in run 76d5b351 at $5.03 cap overrun).
        manuscript, write_cost, _ = write_manuscript(
            user_query=user_query,
            flow=flow,
            reasoner=reasoner,
            critique=result.critique,
            chunks=retrieval.validated_chunks,
            plan=plan,
            coverage=result.coverage,
            process_log=plog,
            plot_artifacts=plot_artifacts,
            budget_remaining_usd=max(0.0, max_cost_usd - result.total_cost_usd),
        )
        result.manuscript = manuscript
        # Hard provenance guarantee on the OUTPUT: if any FINAL value could
        # not be traced to a computed result, prepend a banner the writer
        # cannot suppress (added by code, after the writer ran).
        if result.provenance and result.provenance.get("n_unverified"):
            _unv = result.provenance.get("unverified") or []
            _lines = "\n".join(
                f">   - {u.get('quantity', '?')}: {u.get('value', '?')}"
                for u in _unv[:12]
            )
            result.manuscript = (
                "> ⚠️ **PROVENANCE WARNING — "
                f"{result.provenance['n_unverified']} of "
                f"{result.provenance['n_findings']} reported values could NOT "
                "be traced to a computed result this run** (possible LLM "
                "mental-arithmetic — do not trust these numbers):\n>\n"
                f"{_lines}\n>\n"
                "> Inserted by the provenance gate (code, not the writer).\n\n"
            ) + result.manuscript
        result.total_cost_usd = round(result.total_cost_usd + write_cost, 4)

        result.status = "complete"
    except Exception as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        plog.add(stage="run", kind="agent_error",
                 summary=f"Pipeline raised: {type(e).__name__}: {str(e)[:160]}",
                 outcome="failed", confidence_impact="high")
        emit_event("agent_error", run_id=run_id,
                   error=result.error)
    finally:
        result.total_elapsed_s = round(time.time() - t_start, 3)

        # ── Persist the AgentResult to disk for downstream agents ──
        # Agent 3's input_resolver reads Agent 1's output from
        # runs/_logs/<run_id>/agent1_output/agent_result.json.
        # Persist on every run (even error/truncated runs) so debugging
        # is possible from disk alone.  Never raises — best-effort.
        try:
            from dataclasses import asdict
            import json as _json
            out_dir = RUNS_DIR / run_id / "agent1_output"  # Invariant 4: canonical RUNS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            # Serialise via the api_runner's tree-walker so dataclasses
            # become JSON-safe dicts.  Local import to avoid a cycle
            # at module load.
            from bl_pipeline.agent1_fresh.api_runner import agent_result_to_dict
            result_dict = agent_result_to_dict(result)
            (out_dir / "agent_result.json").write_text(
                _json.dumps(result_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Also write the manuscript as a standalone .md for easy
            # browsing/PDF conversion (scripts/manuscript_to_pdf.py
            # accepts either the .json dump or this .md file).
            if result.manuscript:
                (out_dir / "manuscript.md").write_text(
                    result.manuscript, encoding="utf-8",
                )
                # Sidecar metadata so Agent 6 can discover this
                # section by globbing
                # data/runs/<pipeline_id>/agent*_output/manuscript_meta.json
                try:
                    from bl_pipeline.shared.manuscript_meta import (
                        write_manuscript_meta,
                    )
                    _plots: list[str] = []
                    _plots_dir = out_dir / "plots"
                    if _plots_dir.is_dir():
                        _plots = sorted(
                            p.name for p in _plots_dir.glob("*.png")
                        )
                    write_manuscript_meta(
                        agent_id="agent1",
                        pipeline_id=str(run_id),
                        output_dir=out_dir,
                        plot_files=[f"plots/{n}" for n in _plots],
                        upstream_run_ids=None,
                    )
                except Exception as _meta_e:
                    print(f"  [warn] agent1 manuscript_meta write "
                          f"failed: {_meta_e}")
            # (Standalone A1: the downstream Agent-1 -> Agent-3 handoff writer
            # was removed with the other downstream modules.  This project ends
            # at the Agent-1 result.)
            emit_event("agent1_fresh_persisted",
                       path=str(out_dir / "agent_result.json"),
                       run_id=run_id)
        except Exception as _persist_err:
            emit_event("agent1_fresh_persist_error",
                       error=f"{type(_persist_err).__name__}: {_persist_err}",
                       run_id=run_id)

        emit_event(
            "agent_done",
            run_id=run_id,
            status=result.status,
            error=result.error,
            cost_usd=result.total_cost_usd,
            elapsed_s=result.total_elapsed_s,
            verdict=(result.critique.verdict if result.critique else None),
        )
        try:
            clear_current_run()
        except Exception:
            pass
        try:
            close_default_listeners(run_id)
        except Exception:
            pass

    return result
