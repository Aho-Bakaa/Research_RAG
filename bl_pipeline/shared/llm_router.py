"""Multi-provider LLM client factory.

Routes tasks to the appropriate model (Claude / OpenAI / Grok) based on
task type.  Tracks token usage and cost for the decision logger.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic import APIConnectionError as _AnthropicConnErr
from anthropic import RateLimitError as _AnthropicRateLimit
from openai import OpenAI
from openai import APIConnectionError as _OpenAIConnErr
from openai import RateLimitError as _OpenAIRateLimit


def _retry_backoff(fn, *, tries: int = 5, base_s: float = 8.0, cap_s: float = 60.0):
    """Retry `fn()` on rate-limit / connection errors with exponential backoff.

    The OpenAI TPM window is 60 s.  Start at 8 s, double each retry
    (8, 16, 32, 60, 60), so a wedged Methodology-Discussion-Summary
    chain has ~2 min of headroom before it gives up.  Any other
    exception is re-raised on the first attempt.
    """
    retryable = (
        _AnthropicRateLimit, _OpenAIRateLimit,
        _AnthropicConnErr,   _OpenAIConnErr,
    )
    last_err: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except retryable as exc:
            last_err = exc
            if i == tries - 1:
                break
            delay = min(cap_s, base_s * (2 ** i))
            # If the SDK surfaces a Retry-After hint, honour it.
            try:
                headers = getattr(getattr(exc, "response", None),
                                  "headers", {}) or {}
                hinted = headers.get("Retry-After") or headers.get("retry-after")
                if hinted:
                    delay = max(delay, float(hinted))
            except Exception:
                pass
            time.sleep(delay)
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable: _retry_backoff exhausted with no error")

from bl_pipeline.shared.config import (
    ANTHROPIC_API_KEY,
    BUDGET_LIMIT_USD,
    GROQ_API_KEY,
    MODELS,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    XAI_API_KEY,
    persist_llm_trace,
)


# ── LLM trace persistence ────────────────────────────────────────────
# When enabled (default), every successful LLM call appends a record
# to data/runs/{run_id}/agent1_output/llm_calls.jsonl with the full
# system prompt, messages, and response text.  This is what powers the
# "Trace" tab on the Agent 1 page — without it the frontend can only
# show task names and token counts (which is what the existing
# `llm_call_completed` event carries).
#
# Persistence is opt-out via BLP_PERSIST_LLM_TRACE=false for ops modes
# that need to minimise disk usage.  No truncation here: per the
# product decision the on-disk file is full-fidelity, and the frontend
# truncates to ~4 KB per pane in its default view.
_TRACE_LOCK = threading.Lock()


def _trace_enabled() -> bool:
    return persist_llm_trace()


def _persist_llm_call(
    run_id: str,
    *,
    task: str,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    response: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    duration_s: float,
    agent_dir: str = "agent1_output",
    extra: dict | None = None,
) -> None:
    """Append one LLM exchange to the run's ``<agent_dir>/llm_calls.jsonl``.

    ``agent_dir`` selects which agent's output folder the trace lands in
    (default ``agent1_output`` for back-compat — that is the only folder
    the ``/llm_calls`` reader defaulted to historically).  Agents that
    route their own trace elsewhere (e.g. A6 → ``agent6_output``) pass
    their folder here so the router no longer pollutes A1's trace with
    another agent's calls.  ``extra`` merges caller-specific fields
    (e.g. A6's ``stage`` / ``section``) into the record.

    Best-effort — swallow all errors so a disk hiccup never breaks a
    pipeline run.  The lock guards against interleaved writes when
    Agent 1's per-model ThreadPoolExecutor fires three Sonnet calls in
    parallel.
    """
    if not run_id or not _trace_enabled():
        return
    try:
        from bl_pipeline.shared.config import RUNS_DIR
        out_dir = RUNS_DIR / run_id / agent_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "task":       task,
            "model":      model,
            "system":     system or "",
            "messages":   messages,
            "response":   response,
            "tokens_in":  tokens_in,
            "tokens_out": tokens_out,
            "cost_usd":   round(cost_usd, 6),
            "duration_s": round(duration_s, 3),
            "timestamp":  time.time(),
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _TRACE_LOCK:
            with (out_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
    except Exception:
        # Trace persistence is observability infra, not pipeline state.
        # Never raise from here.
        pass


# ── Cost per 1M tokens (USD) ─────────────────────────────────────
_COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    MODELS["haiku"]:           (1.00, 5.00),
    MODELS["sonnet"]:          (3.00, 15.00),
    MODELS["opus"]:             (15.00, 75.00),
    MODELS["gpt4o"]:           (2.50, 10.00),
    MODELS["gpt4o_mini"]:      (0.15, 0.60),
    MODELS["grok2"]:           (2.00, 10.00),
    # Groq's free tier — Llama 3.3 70B and 3.1 8B both cost $0 today.
    # Track at zero so the budget gauge doesn't double-count them.
    MODELS["groq_llama_70b"]:  (0.00, 0.00),
    MODELS["groq_llama_8b"]:   (0.00, 0.00),
    # Local Ollama-hosted models — zero monetary cost; the cost is
    # GPU/CPU time and electricity.  We track at $0 so the budget
    # ledger reflects what you're actually being billed for.
    MODELS["qwen_coder_32b"]:  (0.00, 0.00),
}


@dataclass
class UsageRecord:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""


@dataclass
class CumulativeUsage:
    """Cost / token ledger.  Thread-safe — `add()` is guarded by a
    single Lock so concurrent LLM calls (e.g. ThreadPoolExecutor in
    Agent 1's per-model parallel block) don't lose updates on the
    `+=` operations.

    Used in two roles:
      • PROCESS GLOBAL: `usage_tracker` (below) — cumulative totals
        across every run the FastAPI process has ever served.  Useful
        for the /api/cost endpoint's "total spend so far" display.
      • PER-RUN: created by `_get_or_create_run_usage(run_id)` and
        reaped by `reset_run_usage(run_id)` at run end so the run-
        summary reports accurate per-run cost (was contaminated by
        every previous run's totals before this fix).
    """
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0
    per_model: dict[str, float] = field(default_factory=dict)
    # Mutex: protects all four counters + per_model dict from
    # concurrent `add()` races.  Excluded from repr/eq so the dataclass
    # stays comparable in tests and printable in logs.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False,
    )

    def add(self, rec: UsageRecord) -> None:
        with self._lock:
            self.total_tokens_in += rec.tokens_in
            self.total_tokens_out += rec.tokens_out
            self.total_cost_usd += rec.cost_usd
            self.calls += 1
            self.per_model[rec.model] = (
                self.per_model.get(rec.model, 0.0) + rec.cost_usd
            )

    @property
    def budget_remaining(self) -> float:
        return BUDGET_LIMIT_USD - self.total_cost_usd

    @property
    def over_budget(self) -> bool:
        return self.total_cost_usd >= BUDGET_LIMIT_USD


# ── Global process-cumulative tracker ─────────────────────────────
# Kept for /api/cost ("total spend ever") and the budget guard.  This
# is separate from the per-run bucket: every LLM call updates BOTH so
# the global stays accurate even when run-scoped readers grab the
# per-run bucket.
usage_tracker = CumulativeUsage()


# ── Per-run usage buckets ────────────────────────────────────────
# Keyed by run_id.  Each entry is a fresh CumulativeUsage that
# accumulates ONLY this run's calls.  Populated by `call()` /
# `_call_anthropic` / `_call_openai` etc. when the current ContextVar
# `current_run_id` is set; reaped by `reset_run_usage()` at run end.
#
# The dict itself is guarded by a Lock for concurrent dict-mutation;
# each CumulativeUsage's own Lock guards its counters.  No nesting
# issues: lookup-then-add releases the dict lock before holding the
# bucket lock.
_run_usage_buckets: dict[str, CumulativeUsage] = {}
_run_usage_lock:    threading.Lock              = threading.Lock()


def _get_or_create_run_usage(run_id: str) -> CumulativeUsage:
    with _run_usage_lock:
        bucket = _run_usage_buckets.get(run_id)
        if bucket is None:
            bucket = CumulativeUsage()
            _run_usage_buckets[run_id] = bucket
        return bucket


def get_run_usage(run_id: str) -> CumulativeUsage | None:
    """Read-only accessor — returns the bucket WITHOUT creating one.

    Useful for the run summary / cost-report endpoint.  Returns None
    when the run hasn't issued any LLM calls (or has already been
    reaped via reset_run_usage).
    """
    with _run_usage_lock:
        return _run_usage_buckets.get(run_id)


def reset_run_usage(run_id: str) -> CumulativeUsage:
    """Pop and return the bucket; subsequent get_run_usage(run_id)
    will return None.  Call this at the end of a pipeline run after
    the run-summary records the totals so memory doesn't grow without
    bound.  Always returns a CumulativeUsage (empty if no calls)."""
    with _run_usage_lock:
        return _run_usage_buckets.pop(run_id, None) or CumulativeUsage()


def _record_usage(rec: UsageRecord) -> None:
    """Fan out a usage record to BOTH the global tracker AND the
    current-run bucket (if a run context is set).  Replaces the bare
    `usage_tracker.add(rec)` calls — those reported run totals
    contaminated by every previous run's cost.  Now:

      • Process-global tracker: still gets every call (backward-compat
        for /api/cost and the budget guard).
      • Per-run bucket: also gets the call IFF current_run_id ContextVar
        is set.  Orchestrators read this bucket to report accurate
        per-run cost in run_summary.json.

    The current-run id comes from `bl_pipeline.shared.event_bus.
    current_run_id` (an existing ContextVar), so this works for any
    code path that already calls `set_current_run(...)` at run start.
    """
    usage_tracker.add(rec)
    # Late import: event_bus -> llm_router would be a circular import.
    try:
        from bl_pipeline.shared.event_bus import current_run_id
        rid = current_run_id.get()
    except Exception:
        rid = None
    if rid:
        _get_or_create_run_usage(rid).add(rec)


# ── Task → model mapping ─────────────────────────────────────────
#
# Routing policy:
#   Sonnet — physics-aware reasoning + writing (almost everything).
#            Strong enough for model identification, ranking, critique,
#            code generation, and manuscript writing.
#   Haiku  — pure text→JSON extraction (no physics reasoning needed).
#   Opus   — reserved for one task: physics validation (Node 7, not
#            yet implemented). Small inputs, small outputs, critical
#            yes/no gate — the one place where the premium reasoning
#            actually pays for itself.
#   Grok-2 — one niche task (bulk data processing code).
#
# History note: model_selection, model_ranking, and _identify_math used
# to route to Opus. Empirically Sonnet handles them just as well at
# ~5x lower cost, and Opus kept truncating mid-JSON on the shortlist
# call. Switched 2026-04-17.
#
# Routing tiers used by TASK_MODEL_MAP below.
#
# Tiers were originally:
#   FREE     — Groq Llama 3.3 70B Versatile (zero cost, 100k token/day cap)
#   CHEAP    — gpt-4o-mini  (~$0.15/$0.60 per 1M)
#   PREMIUM  — gpt-4o       (~$2.50/$10 per 1M)
#
# Current routing (mini-default, gpt-4o on heavy tasks):
#   FREE     → gpt-4o-mini   (was Groq llama-3.3-70b; daily quota hit)
#   CHEAP    → gpt-4o-mini   (unchanged)
#   PREMIUM  → gpt-4o        (heavy tasks: manuscript, composition
#                             code gen, physics gate, vision OCR —
#                             where hallucinations have outsized cost)
_TIER_FREE     = MODELS["gpt4o_mini"]       # was Groq llama-3.3-70b
_TIER_CHEAP    = MODELS["gpt4o_mini"]
_TIER_PREMIUM  = MODELS["gpt4o"]            # heavy tasks pinned here

# ── Claude (Anthropic) tiers for Agent 1 v2 ────────────────────────
# v2 architecture uses Claude across the board because (a) the user
# has an Anthropic API key, (b) JSON-grounding + instruction-following
# is materially better than gpt-4o-mini for the structured tasks we
# care about, and (c) Haiku is cheap enough to run on every query.
#
# Routing intent:
#   Haiku  — cheap filters: query parsing (flow + scope), pool judge.
#   Sonnet — most reasoning: expansion, symbol/convention, conflict,
#            planning, comparison, recommendation, manuscript.
#   Opus   — physics safety gate (critic_v2) only — small input, small
#            output, but premium reasoning where wrong answers cost the
#            most downstream.
_TIER_CLAUDE_HAIKU   = MODELS["haiku"]
_TIER_CLAUDE_SONNET  = MODELS["sonnet"]
_TIER_CLAUDE_OPUS    = MODELS["opus"]


TASK_MODEL_MAP: dict[str, str] = {
    # ══ Agent 1 v2 — Claude routing tiers ═══════════════════════════
    # [0] query parser: flow extraction + flat-plate ZPG scope check
    #     in one Haiku call. ~$0.0005 per run. Replaces the legacy
    #     flow_extraction (gpt-4o-mini) + scope_guard (Sonnet) split.
    "agent1_query_parser":        _TIER_CLAUDE_HAIKU,

    # [1.3] adaptive-retrieval judge — 4-way chunk labeling + overall
    #       sufficiency + gap identification.  Sonnet is the right tier:
    #       Haiku can't reliably distinguish "relevant" from "can_support"
    #       on subtle physics text, and gpt-4o-mini hallucinated chunk
    #       ids in early tests.  ~$0.05 per call × up to 3 calls = $0.15
    #       worst case per retrieval.
    "agent1_pool_judge":          _TIER_CLAUDE_SONNET,

    # [4] conflict_detect + summary_analysis in ONE call.  Promoted to
    #     Opus on 2026-05: this call cascades into [5] planner +
    #     [10] manuscript, so a missed subtle convention disagreement
    #     here (e.g. Tu percent vs decimal between Mayle and AGS) ends
    #     up as wrong unit_convention strings on the executed plans
    #     → wrong numbers reaching the user.  Opus pays for itself
    #     here.  ~$0.10-0.15 per call, once per run.
    "agent1_conflict_v2":         _TIER_CLAUDE_OPUS,

    # [5] solution_planner — merges legacy model_shortlist + composition
    #     into one Opus call producing a ranked plans[] list with
    #     execution strategies + chunk-grounded analysis per model.
    #     This is THE most cascading decision in the pipeline: the
    #     planner's formulas drive the executor, its unit_convention
    #     strings drive code-gen, its expected_range strings drive the
    #     comparator's band checks, its source_chunks drive the
    #     manuscript citations.  Wrong call here = self-consistent
    #     garbage downstream.  Promoted to Opus on 2026-05 per the
    #     upstream-first routing principle.  ~$0.15-0.20 per call.
    "agent1_solution_planner":    _TIER_CLAUDE_OPUS,

    # [7] results_comparator inference paragraph.  Reads a pre-computed
    #     numerical table + agreement bins + band checks + red flags and
    #     synthesises a 4-6 sentence verdict for the operator.  Sonnet
    #     is plenty here — the heavy lifting is deterministic upstream.
    #     ~$0.02 per call.
    "agent1_results_inference":   _TIER_CLAUDE_SONNET,

    # [8] plan_validator — physics safety gate.  RETIRED on 2026-05:
    #     under the upstream-first routing principle, Opus now sits at
    #     [4] conflict_v2 and [5] solution_planner so the executor sees
    #     stronger inputs.  By the time we reach [8], the deterministic
    #     red_flags + band checks + self_validation flags already catch
    #     ~95% of what an Opus validator would.  Per-plan validation
    #     now runs in PlanValidator's deterministic mode (no LLM call).
    #     Kept in the map for traceability; if you ever re-enable the
    #     LLM path, set BL_USE_LLM_PLAN_VALIDATOR=1 and this routing
    #     will be honoured.
    "agent1_plan_validator":      _TIER_CLAUDE_SONNET,  # only used if explicitly re-enabled

    # [9] plan_recommender — picks the recommended plan from the
    #     validator's approved set.  Pure judgement over already-vetted
    #     candidates → Sonnet is the right tier.  ~$0.02 per call.
    "agent1_plan_recommender":    _TIER_CLAUDE_SONNET,

    # ── Free tier ──────────────────────────────────────────────────
    # Simple JSON extraction / routing
    "flow_extraction":            _TIER_FREE,  # "Tu=3%" → {Tu: 3.0}
    "routing":                    _TIER_FREE,  # which agent to invoke
    # Reasoning where the validator / critic catches model drift
    "query_parsing":              _TIER_FREE,  # expand query, rerank chunks
    # theory_analysis writes the per-model Python solver code and
    # extracts the algorithm card from the retrieved chunks.  This is
    # the single LLM call that picked the WRONG Mayle equation
    # (σ̂_n = 1.5e-11·Tu^(7/4) instead of Re_θ,t = 400·Tu^(-5/8)) in
    # run 0966292d.  Pinned to gpt-4o — mini's pattern-matching grabs
    # whichever equation happens to be in the retrieved chunk
    # without reasoning about which one answers the user's question.
    # Cost impact: ~$0.10-0.20 extra per A1 run (theory_analysis
    # fires ~20 times per run).  Worth it.
    "theory_analysis":            _TIER_PREMIUM,  # per-model physics + code gen

    # ══ Agent 1 FRESH — explicit Claude routing per node ════════════
    # The fresh research-agent rewrite (bl_pipeline/agent1_fresh/) uses
    # these task names so every node lands on the right Claude tier
    # rather than silently falling into the gpt-4o premium pool.  Added
    # 2026-05 after discovering all fresh-agent nodes were running on
    # gpt-4o because they used the generic "theory_analysis" task name.
    "agent1_fresh_parse":         _TIER_CLAUDE_SONNET,  # query parser — Sonnet for publication-grade extraction (every flow/grid quantity); was Haiku
    "agent1_fresh_optimize":      _TIER_CLAUDE_SONNET,  # sub-query gen
    "agent1_fresh_judge":         _TIER_CLAUDE_SONNET,  # in-loop chunk judge
    "agent1_fresh_supplementary": _TIER_CLAUDE_SONNET,  # targeted gap queries
    "agent1_fresh_plan":          _TIER_CLAUDE_SONNET,  # planner node
    "agent1_fresh_reason":        _TIER_CLAUDE_SONNET,  # ReAct executor
    "agent1_fresh_critique":      _TIER_CLAUDE_SONNET,  # peer review
    "agent1_fresh_write":         _TIER_CLAUDE_SONNET,  # manuscript writer
    "agent1_fresh_summarize":     _TIER_CLAUDE_SONNET,  # paper-summary script
    # Formula-equivalence judge: compares Sonnet's compute() Python
    # against the glossary's LaTeX formula and decides whether the
    # algebra matches.  Pure pattern-matching on math text; no physics
    # reasoning.  Haiku at ~$0.001-0.003 per compute() call → ~$0.03
    # per full run with ~8 compute calls.  Added after run #12 shipped
    # a manuscript with `Re_theta_t = 172.80` hardcoded under a fake
    # "Mayle Eq.(9)" print label — see bl_pipeline/agent1_fresh/
    # formula_verifier.py for the full layer.
    "agent1_fresh_formula_judge": _TIER_CLAUDE_HAIKU,
    # NOTE on critic-class tasks: gpt-4o-mini was caught flipping the
    # sign of an exponent in a Mayle-style correlation (Tu^(-5/8) →
    # Tu^(+5/8)) during run 6606b08f, corrupting downstream code.
    # Critics MUST be at least as capable as the code generator they
    # review — pinned to gpt-4o (PREMIUM tier).
    "critique":                   _TIER_PREMIUM,  # peer-review pass
    "conflict_resolution":        _TIER_PREMIUM,  # detect literature conflicts
    "model_selection":            _TIER_FREE,  # Node 3b Pass 1 shortlist
    "model_ranking":              _TIER_FREE,  # pick best of shortlist
    "agent2_concern_review":      _TIER_PREMIUM,  # Stage 3.5 — case review

    # Mechanical OCR correction of Marker-produced markdown.  Pure
    # find-and-replace of systematic OCR errors (Greek letters → Latin,
    # = → 5, etc.).  Haiku is perfect for this — fast, cheap, accurate
    # on well-specified text transformations.  ~$0.015-0.02 per paper.
    "marker_ocr_correct":         _TIER_CLAUDE_HAIKU,

    # Per-agent manuscript writeup.  Each agent (2/3/4/5/6) calls the
    # shared writeup helper at the end of its run, which uses this task
    # to compose a 200-400 word researcher-voice paragraph from the
    # agent's structured output.  These six paragraphs become the
    # primary narrative source for Agent 7's integration pass.  Haiku
    # is right-sized — the agent already produced the facts; the LLM
    # just turns them into prose.  ~$0.001-0.003 per call.
    "agent_writeup":              _TIER_CLAUDE_HAIKU,
    "report_writing":             _TIER_FREE,  # shorter reports
    "chat":                       _TIER_FREE,  # conversational
    "experiment_planning":        _TIER_FREE,  # legacy

    # ── Cheap-paid tier (gpt-4o-mini) ─────────────────────────────
    # Anywhere code or rig dictionaries are generated — silent
    # formula errors are too easy on a free model.
    "openfoam_generation":        _TIER_CHEAP,
    "openfoam_error_diagnosis":   _TIER_CHEAP,
    "agent2_strategic_planning":  _TIER_CHEAP,  # Stage 0.5 — model + mesh
    "agent2_solver_repair":       _TIER_CHEAP,  # Stage 5 — diagnose crash
    "code_generation":            _TIER_CHEAP,  # scientific Python / SymPy
    "sympy_code":                 _TIER_CHEAP,
    "data_processing_code":       _TIER_CHEAP,  # was Grok-2

    # ── Premium-paid tier (gpt-4o) ─────────────────────────────────
    # Manuscript writing: previously Opus.  v2 manuscript is 60-100k
    # chars; gpt-4o handles self-consistency at that length well
    # enough.  Watch the critic for attribution drift on the first
    # run — escalate back to Opus if it shows up.
    "manuscript_writing":          _TIER_PREMIUM,
    "agent2_comparison_diagnosis": _TIER_PREMIUM,  # theory↔CFD verdict prose

    # Physics validation — small JSON-in / small JSON-out gate where
    # wrong answers have outsized downstream cost.
    "physics_validation":          _TIER_PREMIUM,

    # Composition discovery + prioritisation.  Decides which cross-
    # model compositions are worth solving for the current query.
    # One call per run, ~15-25k tokens in.
    "composition_discovery":       _TIER_PREMIUM,

    # Composition code generation.  The composition_solver writes raw
    # Python from chunks; this is exactly where formula hallucinations
    # happen (e.g. inventing a "widely used" R_Λ formula that produced
    # a 120 km transition zone in a prior run).  Premium tier here is
    # bought, not optional.  K calls per run, typically 3-5.
    "composition_code_gen":        _TIER_PREMIUM,

    # Solo-model code rewrite when physics_validator flags a number
    # as non-physical.  Fires only on validator-flagged cases
    # (typically 0-1 models per run), so cost is bounded.  Trigger is
    # a real physics violation, not the model's own doubt — must be
    # more conservative on the retry, not less.
    "code_rewrite_validator_flagged": _TIER_PREMIUM,

    # VLM OCR for paper ingestion (vision → markdown).  Groq Llama
    # 70B has no vision input; gpt-4o is the affordable vision model
    # in scope and OCR fidelity is image-bound anyway.
    "vlm_ocr":                     _TIER_PREMIUM,
}


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    in_rate, out_rate = _COST_TABLE.get(model, (3.0, 15.0))
    return (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000


def _prepend_pipeline_state(system: str, pipeline_state: Any) -> str:
    """If `pipeline_state` has a `.render()` method, prepend its text
    to the system prompt so the model sees cumulative pipeline context.

    Keeps the rendered graph visually separated from the caller's
    system prompt by a blank line. Tolerates None (no-op) and any
    render() exception (skipped with a visible warning — a broken
    graph must never break an LLM call).
    """
    if pipeline_state is None:
        return system
    render = getattr(pipeline_state, "render", None)
    if not callable(render):
        return system
    try:
        block = render()
    except Exception as e:
        print(f"[llm_router] pipeline_state.render() failed: {e}", flush=True)
        return system
    if not block:
        return system
    if system:
        return f"{block}\n\n{system}"
    return block


class LLMRouter:
    """Unified interface for calling Claude, OpenAI, Grok, or Groq."""

    def __init__(self) -> None:
        self._anthropic: Anthropic | None = None
        self._openai: OpenAI | None = None
        self._grok: OpenAI | None = None  # Grok uses OpenAI-compatible API
        self._groq: OpenAI | None = None  # Groq uses OpenAI-compatible API
        self._ollama: OpenAI | None = None  # Local Ollama, OpenAI-compatible API

    # ── Lazy client init ──────────────────────────────────────────

    @property
    def anthropic(self) -> Anthropic:
        if self._anthropic is None:
            # Bounded timeout + single retry so a wedged connection fails
            # fast.  The SDK default (600 s x retries) lets a hung call
            # stall a run for ~20-30 min (observed 2026-05-24).  300 s
            # safely covers the long writer call (~10k tokens); on timeout
            # the caller's never-raise wrapper degrades the node gracefully.
            self._anthropic = Anthropic(
                api_key=ANTHROPIC_API_KEY, timeout=300.0, max_retries=1,
            )
        return self._anthropic

    @property
    def openai(self) -> OpenAI:
        if self._openai is None:
            self._openai = OpenAI(
                api_key=OPENAI_API_KEY, timeout=300.0, max_retries=1,
            )
        return self._openai

    @property
    def grok(self) -> OpenAI:
        if self._grok is None:
            self._grok = OpenAI(
                api_key=XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
        return self._grok

    @property
    def groq(self) -> OpenAI:
        """Groq's OpenAI-compatible endpoint (free tier).

        Used for schema-validated extraction / planning / critique
        tasks where the downstream validator can catch model drift.
        See TASK_MODEL_MAP for the per-task routing decisions.
        """
        if self._groq is None:
            self._groq = OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
        return self._groq

    @property
    def ollama(self) -> OpenAI:
        """Local Ollama service via its OpenAI-compatible API.

        Default endpoint: http://localhost:11434/v1 (overridable via
        OLLAMA_BASE_URL env var).  Ollama doesn't authenticate by
        default — we pass a dummy api_key="ollama" because the OpenAI
        client requires a non-empty key.

        Models routed here include qwen2.5-coder:32b (primary code-gen
        + reasoning model for Agent 1 — chosen for thesis
        reproducibility: weights pinned, no API drift, no rate limits,
        no per-call cost).  Any model name starting with "qwen" is
        dispatched here automatically (see call() below).

        Failure mode: if the Ollama service isn't running, the first
        call raises a ConnectionError with a clear message.  The
        pipeline does NOT silently fall back to a cloud model — that
        would defeat the reproducibility guarantee.
        """
        if self._ollama is None:
            self._ollama = OpenAI(
                api_key="ollama",            # ignored by Ollama
                base_url=OLLAMA_BASE_URL,
            )
        return self._ollama

    # ── State-bound wrapper ──────────────────────────────────────
    #
    # Many per-node LLM calls happen deep inside helper methods (e.g.
    # PDESolver._run_algebraic → self-validation → retry). Threading
    # `pipeline_state` through every signature is invasive and easy to
    # miss. Instead, each node's public run() takes `pipeline_state`
    # once and binds it here:
    #
    #     self.llm = llm.bind(pipeline_state) if pipeline_state else llm
    #
    # All downstream `self.llm.call(...)` / `self.llm.call_json(...)`
    # calls then transparently forward the bound state. Binding
    # returns a NEW lightweight wrapper — no mutation of the shared
    # router singleton, so calls running in parallel ThreadPoolExecutor
    # threads don't leak state into each other.

    def bind(self, pipeline_state: Any) -> "_BoundLLM":
        """Return a lightweight wrapper whose call()/call_json() forward
        `pipeline_state` automatically. Use when a node has a graph and
        doesn't want to thread `pipeline_state=` through every helper.
        """
        return _BoundLLM(self, pipeline_state)

    # ── Core call methods ─────────────────────────────────────────

    def call(
        self,
        task: str,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
        model_override: str | None = None,
        pipeline_state: Any = None,
        agent_dir: str = "agent1_output",
        trace_extra: dict | None = None,
    ) -> tuple[str, UsageRecord]:
        """Route a call to the right provider. Returns (text, usage).

        If `pipeline_state` is a `PipelineGraph` (or any object with a
        `.render()` method returning a string), its rendered text is
        prepended to the system prompt so the model sees the cumulative
        pipeline context. The param is opt-in — callers that don't pass
        it see identical behaviour to before.

        `agent_dir` selects which run subfolder the LLM trace persists to
        (default ``agent1_output``); `trace_extra` merges caller-specific
        fields (e.g. A6's stage/section) into the persisted record.  Both
        are opt-in — omitting them reproduces the historical behaviour.
        """
        if usage_tracker.over_budget:
            raise RuntimeError(
                f"Budget exhausted: ${usage_tracker.total_cost_usd:.2f} "
                f">= ${BUDGET_LIMIT_USD:.2f}"
            )

        system = _prepend_pipeline_state(system, pipeline_state)
        model = model_override or TASK_MODEL_MAP.get(task, MODELS["sonnet"])

        # Emit a "call starting" event so the UI can show a spinner.
        # Importing locally so the router module has no hard dependency
        # on event_bus — if event_bus is absent (e.g. in a stripped
        # test harness) the emit is a no-op.
        # Resolve the active run_id so trace persistence can find the
        # right run directory. ContextVar is set by the API in
        # _run_pipeline_task; CLI runs and tests may leave it unset.
        try:
            from bl_pipeline.shared.event_bus import (
                emit as _emit, current_run_id,
            )
            _emit("llm_call_started", task=task, model=model,
                  max_tokens=max_tokens)
            _run_id = current_run_id.get() or ""
        except Exception:
            _emit = None  # type: ignore[assignment]
            _run_id = ""

        _t0 = time.time()
        try:
            if model.startswith("claude"):
                text, rec = self._call_anthropic(
                    model, messages, system, temperature, max_tokens)
            elif model.startswith("grok"):
                text, rec = self._call_openai_compat(
                    self.grok, model, messages, system,
                    temperature, max_tokens, json_mode,
                )
            elif model.startswith("llama"):
                # Groq's free Llama models — OpenAI-compatible API at
                # api.groq.com/openai/v1.  Same client class as Grok.
                text, rec = self._call_openai_compat(
                    self.groq, model, messages, system,
                    temperature, max_tokens, json_mode,
                )
            else:
                text, rec = self._call_openai_compat(
                    self.openai, model, messages, system,
                    temperature, max_tokens, json_mode,
                )
        except Exception as e:
            if _emit is not None:
                try:
                    _emit("llm_call_failed", task=task, model=model,
                          error=str(e)[:300])
                except Exception:
                    pass
            raise
        _duration_s = time.time() - _t0

        if _emit is not None:
            try:
                _emit("llm_call_completed", task=task, model=model,
                      tokens_in=rec.tokens_in, tokens_out=rec.tokens_out,
                      cost_usd=round(rec.cost_usd, 6))
            except Exception:
                pass

            # NOTE: llm_call_payload emit DISABLED — sending the full
            # system + messages + response over WS for every call was
            # causing the WS broadcast pipeline to hang on large
            # conflict_resolution / theory_analysis prompts (~10kB+
            # JSON frames).  The full exchange is still persisted to
            # data/runs/<run>/agent1_output/llm_calls.jsonl via
            # _persist_llm_call below, and the post-run "Trace" tab
            # reads from there — so the data is NOT lost, it just
            # isn't streamed live.  Reinstate this emit only after
            # the WS bridge has a non-blocking send with size cap.

        # Persist the full exchange (system + messages + response) so
        # the Agent 1 "Trace" tab can show what was asked and what
        # came back. This is the only place the raw text survives —
        # event_bus events only carry metadata.
        _persist_llm_call(
            _run_id,
            task=task, model=model,
            system=system, messages=messages, response=text,
            tokens_in=rec.tokens_in, tokens_out=rec.tokens_out,
            cost_usd=rec.cost_usd, duration_s=_duration_s,
            agent_dir=agent_dir, extra=trace_extra,
        )
        return text, rec

    def _call_anthropic(
        self,
        model: str,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, UsageRecord]:
        resp = _retry_backoff(lambda: self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a scientific research assistant.",
            messages=messages,
        ))
        text = resp.content[0].text
        rec = UsageRecord(
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            cost_usd=_compute_cost(model, resp.usage.input_tokens, resp.usage.output_tokens),
            model=model,
        )
        _record_usage(rec)
        return text, rec

    def _call_openai_compat(
        self,
        client: OpenAI,
        model: str,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> tuple[str, UsageRecord]:
        oai_messages: list[dict[str, str]] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = _retry_backoff(lambda: client.chat.completions.create(**kwargs))
        text = resp.choices[0].message.content or ""
        t_in = resp.usage.prompt_tokens if resp.usage else 0
        t_out = resp.usage.completion_tokens if resp.usage else 0
        rec = UsageRecord(
            tokens_in=t_in,
            tokens_out=t_out,
            cost_usd=_compute_cost(model, t_in, t_out),
            model=model,
        )
        _record_usage(rec)
        return text, rec

    # ── Convenience wrappers ──────────────────────────────────────

    def call_json(
        self,
        task: str,
        messages: list[dict[str, str]],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        pipeline_state: Any = None,
        agent_dir: str = "agent1_output",
        trace_extra: dict | None = None,
    ) -> tuple[dict[str, Any], UsageRecord]:
        """Call and parse the result as JSON.

        `agent_dir` / `trace_extra` thread straight through to
        :meth:`call` so structured-output callers can route their trace
        (e.g. A6 → ``agent6_output`` with stage/section tags).
        """
        model = TASK_MODEL_MAP.get(task, MODELS["sonnet"])

        if model.startswith("claude"):
            # Anthropic: instruct JSON in prompt
            if system:
                system += "\n\nRespond with valid JSON only. No markdown fences."
            else:
                system = "Respond with valid JSON only. No markdown fences."
            text, rec = self.call(
                task, messages, system, temperature, max_tokens,
                pipeline_state=pipeline_state,
                agent_dir=agent_dir, trace_extra=trace_extra,
            )
        else:
            text, rec = self.call(
                task, messages, system, temperature, max_tokens,
                json_mode=True, pipeline_state=pipeline_state,
                agent_dir=agent_dir, trace_extra=trace_extra,
            )

        # Use the lenient parser — Sonnet sometimes emits perfectly
        # valid JSON followed by a trailing prose sentence, or prepends
        # a "Here is the JSON:" preamble, or includes unescaped
        # backslashes inside LaTeX strings. Strict json.loads fails on
        # any of those; parse_lenient_json recovers via a cascade of
        # cleanup strategies (strip fences, extract outermost
        # {...} block, escape lone backslashes, fix trailing commas,
        # escape embedded newlines in strings).
        from bl_pipeline.shared.json_utils import parse_lenient_json
        parsed = parse_lenient_json(text)
        if parsed is None:
            # Fall back to strict parse so the caller still gets the
            # original JSONDecodeError message for debugging. The
            # lenient parser returns None when nothing at all worked,
            # which is strictly rarer than strict failure.
            cleaned = text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
            return json.loads(cleaned), rec
        if not isinstance(parsed, dict):
            # call_json's contract is "returns a dict". If Sonnet sent
            # a list, raise explicitly so the caller can decide how to
            # coerce (see algorithm_extractor for the list-unwrap
            # precedent).
            raise ValueError(
                f"call_json expected dict but got {type(parsed).__name__} "
                f"(content: {str(parsed)[:200]})"
            )
        return parsed, rec


# ──────────────────────────────────────────────────────────────────
# _BoundLLM — lightweight state-bound router wrapper
# ──────────────────────────────────────────────────────────────────

class _BoundLLM:
    """Wraps an LLMRouter and forwards a fixed `pipeline_state` to
    every call() / call_json(). Constructed via `llm.bind(state)`.

    Attributes are all set once in __init__ and never mutated, so
    this object is safe to share across threads — each parallel
    worker simply creates its own bound wrapper (or reuses one since
    it's read-only).
    """

    __slots__ = ("_router", "_state")

    def __init__(self, router: "LLMRouter", pipeline_state: Any) -> None:
        self._router = router
        self._state = pipeline_state

    def call(self, *args: Any, **kwargs: Any):  # noqa: ANN401
        kwargs.setdefault("pipeline_state", self._state)
        return self._router.call(*args, **kwargs)

    def call_json(self, *args: Any, **kwargs: Any):  # noqa: ANN401
        kwargs.setdefault("pipeline_state", self._state)
        return self._router.call_json(*args, **kwargs)

    # Passthrough for any other router attribute (e.g. callers poking
    # at usage tracker through the router). Read-only — no __setattr__
    # override needed because __slots__ already locks attribute set.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


# Module-level singleton
llm = LLMRouter()
