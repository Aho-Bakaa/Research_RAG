"""state.py — what flows through the fresh Agent 1.

Pure Python dataclasses.  No UI imports.  Every field is JSON-serializable
so the Streamlit app (today) and the React/FastAPI port (later) read the
same shapes.

Two layers:

1.  **State containers** (dataclasses): a typed snapshot of what each node
    produced.  `AgentResult` is the final artifact.

2.  **Events** (functions in this module that emit through the existing
    `bl_pipeline.shared.event_bus`): real-time progress signals.  The UI
    subscribes to these to render the live trace; the agent core does not
    know what the UI is.

Serialization rule (production-grade):
    Every event payload MUST be JSON-serializable as-is (no numpy types,
    no datetime objects without `.isoformat()`, no dataclass instances
    that haven't been `asdict()`-ed).  Helper `_jsonable` enforces this.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# Flow conditions — parsed from the user's query
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FlowConditions:
    """Numerical inputs the reasoner can plug into formulas.

    `geometry_description` and `pressure_gradient_description` stay as
    verbatim user strings — they are NOT classified into a fixed set.
    Downstream nodes read them with their own judgment.

    All NEW grid- and decay-related fields below are OPTIONAL.  They unlock
    progressively smarter A1 behaviour when supplied:

      - lambda_x_mm   → enables FS20 Eq.3.6 (length-scale-aware Re_tr).
      - grid_*        → A1 can derive lambda_x_mm via Comte-Bellot scaling
                        if it isn't measured directly.
      - tu_x_measurements / fransson_* → facility-specific decay calibration
                        (overrides the Fransson 2005 KTH defaults).

    When none of these are provided, A1 falls back to AGS/Mayle with an
    explicit "Λ_x absence" flag in the FINAL block (user is told which
    uncertainty drivers are dominant).
    """
    # ── Core flow inputs (existing) ────────────────────────────────────
    velocity_ms: float | None = None
    turbulence_intensity_pct: float | None = None
    kinematic_viscosity_m2s: float | None = None
    length_scale_m: float | None = None             # legacy generic length scale
    chord_m: float | None = None
    geometry_description: str | None = None
    pressure_gradient_description: str | None = None
    roughness_um: float | None = None

    # ── NEW: integral length scale at LE (preferred over legacy
    #         length_scale_m once UI exposes this explicitly) ─────────
    lambda_x_mm: float | None = None

    # ── NEW: grid geometry (lets A1 derive lambda_x if not provided) ──
    grid_solidity: float | None = None
    grid_bar_diameter_mm: float | None = None
    grid_position_m: float | None = None             # signed, x = 0 at LE; negative ⇒ upstream

    # ── NEW: facility decay calibration ────────────────────────────────
    #   Option A — user provides multi-station Tu(x) measurements; A1
    #              fits Fransson (C, x_0, b) via fit_fransson_decay().
    tu_x_measurements: list[tuple[float, float]] | None = None
    #   Option B — user provides Fransson constants directly (e.g. from a
    #              prior A6 calibration run).  Overrides any fit.
    fransson_C: float | None = None
    fransson_x_0_m: float | None = None
    fransson_b: float | None = None

    # ── NEW (2026-06-12, task #221) — Blasius-VO Tu_effective method ──
    # Two ways to provide the boundary-layer virtual origin x_0_BL:
    #
    #   Option A — user already fitted (or computed by hand) the Blasius
    #              virtual origin and gives the value directly in metres.
    #              When supplied, A1 uses this value as x_0_BL.
    x_0_BL_m: float | None = None
    #
    #   Option B — user supplies ≥2 (x, δ_99) measurements; A1 fits
    #              the Blasius √(x − x_0_BL) virtual origin via the
    #              compute tool (closed-form for 2 points, nonlinear
    #              least-squares for ≥3 points with R² report).
    #              The list values are (x_m, delta_99_m) — both in metres.
    delta_x_measurements: list[tuple[float, float]] | None = None
    #
    # When BOTH are provided, the direct x_0_BL_m wins (user-override
    # semantics), but A1's reasoner is instructed (via the standing
    # validity rule and the blasius_VO_Tu_effective_method recipe entry)
    # to ALSO fit from delta_x_measurements and report the agreement as
    # a cross-check in the manuscript.  When only delta_x_measurements
    # is supplied, A1 performs the fit and uses the fitted x_0_BL.
    # When neither is supplied, the Blasius-VO recipe is not applicable
    # and A1 falls back to Tu_LE (with the Dick-Kubacki 2017 caveat).

    # ── Provenance / defaulting trail ──────────────────────────────────
    defaulted_fields: list[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────
    # Helper methods — graceful fallback logic
    # ──────────────────────────────────────────────────────────────────

    def has_required(self) -> bool:
        """Minimum inputs the reasoner needs to compute anything quantitative."""
        return self.velocity_ms is not None and self.turbulence_intensity_pct is not None

    def effective_lambda_x_m(self) -> float | None:
        """Best Λ_x estimate (m), or None if no information available.

        Returns the VALUE only — for source/provenance metadata use
        `lambda_x_with_provenance()`.  Both methods follow the same
        fallback chain: user_input → legacy_length_scale → grid_derived
        → None.
        """
        info = self.lambda_x_with_provenance()
        return info["value_m"] if info else None

    def lambda_x_with_provenance(self) -> dict | None:
        """Best Λ_x estimate WITH provenance metadata.

        Returns a dict:
            {
              "value_m":         float,      # Λ_x in metres
              "source":          str,        # "user_input" | "legacy_length_scale"
                                             # | "grid_derived_comte_bellot"
                                             # | "a6_measured_autocorrelation"
              "formula_used":    str | None, # textual formula reference
              "uncertainty_pct": float,      # honest uncertainty estimate
            }
        or None if no information is available.

        Phase 2 reads `source` to identify which assumption broke; e.g.
        when `source == "grid_derived_comte_bellot"` and A6 returns a
        very different measured value, Phase 2 can flag the Comte-Bellot
        default α as the dominant uncertainty driver.

        Priority order (most-trustworthy first):
          1. lambda_x_mm explicitly given by user.
          2. derived from grid spec via Comte-Bellot 1966 scaling.
          3. legacy length_scale_m (LAST RESORT — the LLM query parser
             routinely puts chord_m into this field, which is then
             indistinguishable from a real Λ_x.  Only trust it when no
             grid spec was supplied and no value looks chord-like
             (>0.5 m), since real wind-tunnel Λ_x is typically a few
             mm to a few cm).
          4. (future) facility-measured via A6 autocorrelation.
        """
        if self.lambda_x_mm is not None and self.lambda_x_mm > 0:
            return {
                "value_m": self.lambda_x_mm * 1e-3,
                "source": "user_input",
                "formula_used": None,
                "uncertainty_pct": 5.0,    # user knows their measurement
            }
        if (
            self.grid_solidity is not None
            and self.grid_bar_diameter_mm is not None
            and self.grid_position_m is not None
        ):
            # ── 2026-06-03 hard cutover (task #135) ─────────────────
            # The hardcoded engineering blend (α=0.10, n=0.40, k=3) is
            # no longer applied in production.  When grid spec is
            # available, we emit a DEFERRED sentinel that carries:
            #   * `value_m` = None              — no number for the
            #     OPTIMIZER to grab as a shortcut
            #   * `reasoner_must_call` =        — explicit tool hint so
            #     the reasoner uses `lookup_equation` + `compute`
            #     with a paper_id::Eq.(N) tag the formula_verifier
            #     can judge
            #   * `inputs`                       — the (σ, d, x_grid)
            #     to plug into the looked-up formula
            # The reasoner is REQUIRED to call lookup_equation and
            # compute Λ_x itself.  No silent engineering-blend value
            # downstream.  See correlations.estimate_lambda_x_from_grid
            # for the legacy function (preserved for tests but no
            # longer on the production path).
            return {
                "value_m":              None,
                "source":               "grid_spec_DEFERRED_TO_REASONER",
                "formula_used":         None,
                "uncertainty_pct":      None,
                "reasoner_should_rederive": True,
                "reasoner_must_call":   (
                    "lookup_equation('roach_1987', 'Eq.(18)') "
                    "OR lookup_equation('kurian_fransson_2009', 'Eq.(8)') "
                    "→ compute Λ_x with a paper_id::Eq.(N) tag"
                ),
                "inputs": {
                    "grid_solidity":           self.grid_solidity,
                    "grid_bar_diameter_mm":    self.grid_bar_diameter_mm,
                    "grid_position_m":         self.grid_position_m,
                },
            }
        # Legacy length_scale_m — only trustworthy when value looks like
        # a length scale (not a chord).  Real wind-tunnel Λ_x is
        # O(1 mm to 10 cm); the parser puts chord_m here when it
        # can't tell.  Cap at 0.5 m to reject chord-like values.
        if (
            self.length_scale_m is not None
            and self.length_scale_m > 0
            and self.length_scale_m < 0.5
        ):
            return {
                "value_m": self.length_scale_m,
                "source": "legacy_length_scale",
                "formula_used": None,
                "uncertainty_pct": 10.0,
            }
        return None

    def effective_fransson_decay(self) -> dict | None:
        """Best Fransson decay parameters (C, x_0_m, b), or None.

        Priority order:
          1. All three of (fransson_C, fransson_x_0_m, fransson_b) given directly.
          2. tu_x_measurements provided → fit via fit_fransson_decay().
          3. Grid geometry given → infer x_0 from grid + Fransson defaults
             (b=0.6, C derived from Tu_LE boundary condition).
          4. None of the above → returns a "facility-default" dict that the
             REASONER must flag in the FINAL block.

        Always returns a dict with at minimum keys (C, x_0_m, b, source).
        `source` ∈ {"user_direct", "facility_fit", "grid_derived",
                    "facility_default"}.
        """
        if all(
            v is not None for v in (self.fransson_C, self.fransson_x_0_m, self.fransson_b)
        ):
            return {
                "C": self.fransson_C,
                "x_0_m": self.fransson_x_0_m,
                "b": self.fransson_b,
                "source": "user_direct",
            }

        if self.tu_x_measurements is not None and len(self.tu_x_measurements) >= 3:
            try:
                from bl_pipeline.agent1_fresh.correlations import fit_fransson_decay
                fit = fit_fransson_decay(self.tu_x_measurements)
                if fit.get("converged"):
                    return {
                        "C": fit["C"], "x_0_m": fit["x_0"], "b": fit["b"],
                        "rms_residual_pct": fit.get("rms_residual_pct"),
                        "source": "facility_fit",
                    }
            except Exception:
                pass

        # Grid-derived: infer x_0 from grid, use Fransson b default, derive C
        # from boundary condition Tu(0) = Tu_LE if Tu_LE known.
        if (
            self.grid_solidity is not None
            and self.grid_bar_diameter_mm is not None
            and self.grid_position_m is not None
            and self.turbulence_intensity_pct is not None
        ):
            try:
                from bl_pipeline.agent1_fresh.correlations import (
                    mesh_size_from_solidity,
                    virtual_origin_from_grid,
                )
                M_mm = mesh_size_from_solidity(self.grid_solidity, self.grid_bar_diameter_mm)
                x_0_m = virtual_origin_from_grid(self.grid_position_m, M_mm * 1e-3)
                b = 0.6
                C = self.turbulence_intensity_pct * (-x_0_m) ** b
                return {
                    "C": C, "x_0_m": x_0_m, "b": b,
                    "source": "grid_derived",
                }
            except Exception:
                pass

        # All-defaults fallback: only Tu_LE known.  x_0 ≈ -1 m, b = 0.6.
        # The REASONER MUST surface this as the dominant uncertainty in FINAL.
        return {
            "C": (self.turbulence_intensity_pct or 1.0),
            "x_0_m": -1.0,
            "b": 0.6,
            "source": "facility_default",
            "warning": "Fransson decay using KTH-facility defaults (Tu_0, x_0=-1m, b=0.6). "
                       "Real rig values may differ — supply tu_x_measurements OR "
                       "fransson_C/x_0/b OR grid spec to remove this uncertainty.",
        }

    def has_length_scale_info(self) -> bool:
        """True if A1 can use FS20 (any path: explicit lambda, grid, or legacy)."""
        return self.effective_lambda_x_m() is not None

    def missing_for_smart_a1(self) -> list[str]:
        """List of fields whose absence forces A1 to fall back from FS20 → AGS/Mayle.

        Use in FINAL block to flag "to improve prediction, provide ..."
        """
        missing = []
        if self.effective_lambda_x_m() is None:
            missing.append(
                "Λ_x (integral length scale at LE) — needed for FS20; "
                "either measure directly OR provide grid geometry "
                "(solidity, bar diameter, position) to derive it"
            )
        if self.effective_fransson_decay() is None or \
                self.effective_fransson_decay().get("source") == "facility_default":
            missing.append(
                "Fransson decay parameters (C, x_0, b) — currently using "
                "KTH facility defaults; provide tu_x_measurements or "
                "fransson_C/x_0/b to calibrate to your rig"
            )
        return missing


# ══════════════════════════════════════════════════════════════════════
# Chunks and retrieval trace
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """One retrieved chunk after judging.  Score is the judge's 0-1 number."""
    chunk_id: str
    paper_id: str
    page: int | str           # ints when known; str fallback for legacy
    content: str
    score: float = 0.0        # judge 0-1
    label: str = "relevant"   # highly_relevant|relevant|can_support|irrelevant
    judge_reason: str = ""
    rerank_score: float = 0.0  # cross-encoder
    collection: str = ""


@dataclass
class IterationTrace:
    """Audit record for one pass through the retrieval loop."""
    iteration: int
    sub_queries: list[str] = field(default_factory=list)
    raw_candidates: int = 0
    after_rerank: int = 0
    after_judge_prune: int = 0
    pool_size_after_merge: int = 0
    sufficient: bool = False
    gaps: list[str] = field(default_factory=list)
    supplementary_queries: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    # ── NEW (2026-06-11) — transparency fields for the UI ─────────────
    # Top-K chunks AFTER rerank, BEFORE the judge runs.  Each entry is
    # {chunk_id, paper_id, page, rerank_score, snippet}.  Capped at 10
    # entries so the event payload stays bounded.
    chunks_after_rerank: list[dict] = field(default_factory=list)
    # Same chunks AFTER the judge stamps label + score + reason.  Each
    # entry adds {label, judge_score, judge_reason} to the rerank preview.
    chunks_after_judge: list[dict] = field(default_factory=list)
    # Gaps from the PREVIOUS iteration's judge that drove this iter's
    # supplementary queries.  Empty for iter 1.  Lets the UI render a
    # "this re-retrieval was triggered by..." banner.
    supplementary_trigger_gaps: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    iterations: list[IterationTrace] = field(default_factory=list)
    validated_chunks: list[Chunk] = field(default_factory=list)
    final_pool_size: int = 0
    n_iterations: int = 0
    sufficient: bool = False
    total_cost_usd: float = 0.0
    total_elapsed_s: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Reasoner trace (ReAct-style)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """One tool invocation in the ReAct loop."""
    step: int
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class ReasonerTrace:
    """Full ReAct trace — interleaved prose thinking + tool calls."""
    thinking_segments: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final: dict[str, Any] | None = None        # the FINAL block
    n_react_turns: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0
    truncated: bool = False                    # hit max_react_turns


# ══════════════════════════════════════════════════════════════════════
# Critique
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CritiqueConcern:
    severity: str  # minor|major|blocker
    category: str  # physics|citation|units|deferral|completeness|other
    issue: str
    suggested_fix: str


@dataclass
class CritiqueResult:
    verdict: str   # PASS | NEEDS_REVISION | FAIL
    concerns: list[CritiqueConcern] = field(default_factory=list)
    summary: str = ""
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Plan — what the planner produces, what the executor executes,
# what the verifier checks
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PlanQuantity:
    """One quantity the planner commits to computing.

    The verifier later checks that `name` appears in
    `reasoner.final.key_findings[*].quantity`.  If not, that's a
    coverage gap and triggers a targeted revision.
    """
    name: str                                 # canonical short label, e.g. "x_t"
    method: str = ""                          # one-line description of approach
    source_papers: list[str] = field(default_factory=list)  # paper_id::pPAGE refs
    depends_on: list[str] = field(default_factory=list)     # other quantity names
    expected_unit: str = ""                   # "m", "-", "1/m", etc.
    is_core: bool = True                      # user explicitly asked for this?
                                              # (drives verifier escalation)


@dataclass
class Plan:
    """Structured plan produced by the planner node.

    The reasoner executes this.  The verifier checks coverage against
    `reasoner.final.key_findings`.  Both planner and reasoner produce
    artifacts that downstream nodes can audit independently.
    """
    sub_queries: list[str] = field(default_factory=list)  # NEW — decomposition of user query
    quantities_to_compute: list[PlanQuantity] = field(default_factory=list)
    cross_checks: list[str] = field(default_factory=list)   # free-form
    decay_handling: str = ""                  # how decay will be USED, not just mentioned
    probe_layout_notes: str = ""              # NEW — guidance for the writer's probe table
    deferred_to_other_agents: list[dict[str, Any]] = field(default_factory=list)  # NEW
    # NEW (2026-06-11) — research-student persona addition: when the planner
    # would like to take a step but doesn't see a supporting chunk, it lists
    # the missing-chunk description here rather than inventing an equation
    # from memory.  Downstream retrieval can act on these gaps (Phase 1 —
    # gaps are logged only; Phase 2 — gaps auto-trigger SUPPLEMENTARY).
    retrieval_gaps: list[str] = field(default_factory=list)
    expected_tool_calls: int = 0              # planner's self-estimate
    rationale: str = ""                       # one-paragraph explanation
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Coverage report — deterministic plan-vs-execution check
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SubQueryCoverage:
    """Per-sub-query coverage record.

    `addressed` is a heuristic: the reasoner's trace + findings mention
    enough of the sub-query's distinctive keywords that we believe it
    was actually answered.  `evidence` is a short snippet supporting
    the decision (or empty if not addressed).
    """
    sub_query: str
    addressed: bool = False
    evidence: str = ""


@dataclass
class CoverageReport:
    """Output of `verify.verify_plan_coverage(plan, trace)`.

    Pure Python comparison — no LLM call.  Tells the orchestrator
    whether to fire a targeted revision pass for missing items.
    """
    computed:           list[str] = field(default_factory=list)  # names in BOTH plan + final
    missing_core:       list[str] = field(default_factory=list)  # is_core, not in final
    missing_other:      list[str] = field(default_factory=list)  # !is_core, not in final
    extra:              list[str] = field(default_factory=list)  # in final, not in plan
    decay_used:         bool = False                              # heuristic
    sub_query_coverage: list[SubQueryCoverage] = field(default_factory=list)  # NEW
    needs_revision:     bool = False                              # = bool(missing_core)
    summary:            str = ""


# ══════════════════════════════════════════════════════════════════════
# Process log — the research narrative that drives Part B of the
# two-part manuscript.  Every notable event during a run (search,
# compute, self-catch, stuck-recovery, patch, defer) gets appended
# here so the writer can produce an honest "how the agent solved it"
# section alongside the technical answer.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ProcessEvent:
    """One notable moment during the agent's run.

    Categories:
      stage   — which pipeline node produced this (parse|retrieve|plan|
                reason|verify|critique|write)
      kind    — what kind of event (search|compute|self_catch|stuck|
                patch|defer|info|off_script|budget_warning|refusal)
      summary — one-line plain-language description for the reader
      outcome — how it ended (resolved|patched|deferred|noted|failed|"")
      confidence_impact — "" (none) | "high" | "medium" | "low" — how
                this event affects the confidence the reader should
                place in the final numbers
      turn    — reasoner turn number if applicable
    """
    stage: str
    kind: str
    summary: str
    outcome: str = ""
    confidence_impact: str = ""
    turn: int | None = None
    ts: float = 0.0


@dataclass
class ProcessLog:
    """The honest research narrative.

    Append-only during the run.  The writer reads it to produce
    Part B of the manuscript.
    """
    events: list[ProcessEvent] = field(default_factory=list)

    def add(
        self,
        stage: str,
        kind: str,
        summary: str,
        *,
        outcome: str = "",
        confidence_impact: str = "",
        turn: int | None = None,
    ) -> None:
        """Append an event.  Never raises — observability never breaks runs."""
        try:
            self.events.append(ProcessEvent(
                stage=stage, kind=kind, summary=summary,
                outcome=outcome, confidence_impact=confidence_impact,
                turn=turn, ts=time.time(),
            ))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# Top-level result
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """The complete artifact returned by `agent1_fresh.run.run()`.

    Everything in here is JSON-serializable via dataclasses.asdict —
    use `to_dict()` for transport.
    """
    run_id: str
    user_query: str
    flow: FlowConditions = field(default_factory=FlowConditions)
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)
    plan: Plan | None = None                    # planner output
    reasoner: ReasonerTrace = field(default_factory=ReasonerTrace)
    coverage: CoverageReport | None = None      # verifier output
    critique: CritiqueResult | None = None
    # Final-number provenance report (provenance.py): how many FINAL
    # key_findings traced to an executed compute() vs were flagged
    # UNVERIFIED (LLM mental-math).  None until the provenance gate runs.
    provenance: dict | None = None
    process_log: ProcessLog = field(default_factory=ProcessLog)  # NEW — feeds Part B
    manuscript: str = ""
    total_cost_usd: float = 0.0
    total_elapsed_s: float = 0.0
    status: str = "running"                     # running|complete|error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
# Event emission — wraps the existing event_bus with JSON-safe payloads
# ══════════════════════════════════════════════════════════════════════

def _jsonable(value: Any) -> Any:
    """Coerce a value into something JSON serialization can handle.

    Conservative: any unknown type becomes its repr() so we never raise
    inside an emit() call (an exception in the event path would crash
    the whole agent, which is the opposite of what we want).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # Dataclasses (any of ours)
    try:
        return _jsonable(asdict(value))
    except TypeError:
        pass
    # Numpy scalars / arrays
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    # Last resort
    return repr(value)


def emit_event(event_type: str, **payload: Any) -> None:
    """Emit a typed event through the shared event_bus.

    Adds `ts` (wall-clock timestamp) and `agent="agent1_fresh"` so the
    UI can demux events from different agents on the same bus.

    NEVER raises — a failed emit must not break the run.
    """
    try:
        from bl_pipeline.shared.event_bus import emit as _bus_emit
    except ImportError:
        return  # No bus available — silently degrade.

    safe = {k: _jsonable(v) for k, v in payload.items()}
    safe["ts"] = time.time()
    safe["agent"] = "agent1_fresh"
    try:
        _bus_emit(event_type, **safe)
    except Exception:
        # Production rule: never let observability break execution.
        pass
