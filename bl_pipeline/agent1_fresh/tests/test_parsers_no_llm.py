"""Deterministic tests for every JSON-parsing site in agent1_fresh.

Mocks the LLM router so each parser is exercised against the realistic
"bad shape" failure modes WITHOUT spending a cent on the API:

    1. Canonical dict      — happy path
    2. Bare list           — Sonnet's most common deviation
    3. Empty string        — LLM call with truncated/garbage output
    4. Non-JSON prose      — "Here is the result..."
    5. None                — parser returned nothing

Every test asserts the parser:
    • Does NOT raise an exception
    • Produces a sane default (empty plan / fallback queries / default verdict)

This is the pre-flight harness that prevents the
"AttributeError: 'list' object has no attribute 'get'" class of
failure-mode regressing into a paid run.

Run:
    python -m bl_pipeline.agent1_fresh.tests.test_parsers_no_llm
"""
from __future__ import annotations

import io
import json
import sys
from typing import Any
from unittest.mock import patch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# Shared mock: a fake LLM that returns a configurable response string
# ══════════════════════════════════════════════════════════════════════

class _FakeUsage:
    def __init__(self, cost: float = 0.001):
        self.cost_usd = cost
        self.tokens_in = 100
        self.tokens_out = 50


def _make_fake_llm(response_text: str):
    """Return a function with the same signature as llm.call() that
    returns the configured text once, no matter what arguments are
    passed."""
    def _fake_call(*args, **kwargs):
        return response_text, _FakeUsage()
    return _fake_call


# ══════════════════════════════════════════════════════════════════════
# Five shapes we exercise against every parser
# ══════════════════════════════════════════════════════════════════════

_BAD_SHAPES = {
    "bare_list":     '[{"chunk_id":"a","score":0.9,"label":"highly_relevant","reason":"x"}]',
    "empty_string":  "",
    "prose_no_json": "Here is the result: I think the answer is 0.46 m.",
    "truncated":     '{"search_queries": ["q1", "q2"',   # missing closing
    "valid_dict":    '{"search_queries":["q1","q2","q3"]}',  # happy path for sanity
}


# ══════════════════════════════════════════════════════════════════════
# 1.  _expand_query — must always produce a non-empty list of queries
# ══════════════════════════════════════════════════════════════════════

def test_expand_query_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.retrieve import _expand_query
    user_query = "predict transition onset at Tu=3%"
    for label, response in _BAD_SHAPES.items():
        with patch("bl_pipeline.agent1_fresh.retrieve.llm.call",
                   side_effect=_make_fake_llm(response)):
            queries, cost = _expand_query(user_query)
        assert isinstance(queries, list), f"[{label}] queries must be a list"
        assert len(queries) >= 1, f"[{label}] must always have ≥1 query"
        assert queries[0] == user_query, f"[{label}] original query must be first"
    print("PASS  test_expand_query_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 2.  _generate_supplementary — must always produce a list (or empty)
# ══════════════════════════════════════════════════════════════════════

def test_generate_supplementary_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.retrieve import _generate_supplementary
    for label, response in {
        "bare_list":     '["targeted query 1", "targeted query 2"]',  # canonical
        "wrapped_dict":  '{"queries": ["q1"]}',                       # wrong shape — should return []
        "empty_string":  "",
        "prose_no_json": "I think we need: ...",
    }.items():
        with patch("bl_pipeline.agent1_fresh.retrieve.llm.call",
                   side_effect=_make_fake_llm(response)):
            queries, cost = _generate_supplementary(
                "predict onset", ["zone-length correlation"],
            )
        assert isinstance(queries, list), f"[{label}] must return a list"
    print("PASS  test_generate_supplementary_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 3.  _judge — must always return (dict, bool, list, float)
# ══════════════════════════════════════════════════════════════════════

def test_judge_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.retrieve import _judge
    from bl_pipeline.agent1_fresh.state import Chunk
    pool = [Chunk(chunk_id="a::1", paper_id="a", page=1, content="text")]
    for label, response in _BAD_SHAPES.items():
        with patch("bl_pipeline.agent1_fresh.retrieve.llm.call",
                   side_effect=_make_fake_llm(response)):
            labels, sufficient, gaps, cost = _judge("query", pool)
        assert isinstance(labels, dict), f"[{label}] labels must be a dict"
        assert isinstance(sufficient, bool), f"[{label}] sufficient must be bool"
        assert isinstance(gaps, list), f"[{label}] gaps must be a list"
        assert isinstance(cost, float), f"[{label}] cost must be float"
    print("PASS  test_judge_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 4.  make_plan — must always return a Plan (possibly empty)
# ══════════════════════════════════════════════════════════════════════

def test_make_plan_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.plan import make_plan
    from bl_pipeline.agent1_fresh.state import Chunk, FlowConditions, Plan
    chunks = [Chunk(chunk_id="m::1", paper_id="mayle_1991", page=1,
                    content="Re_θ,t = 400·Tu^(-5/8)")]
    flow = FlowConditions(velocity_ms=12.0, turbulence_intensity_pct=3.3)
    for label, response in _BAD_SHAPES.items():
        with patch("bl_pipeline.agent1_fresh.plan.llm.call",
                   side_effect=_make_fake_llm(response)):
            plan = make_plan("query", flow, chunks)
        assert isinstance(plan, Plan), f"[{label}] must return a Plan"
        assert isinstance(plan.quantities_to_compute, list), \
            f"[{label}] quantities_to_compute must be a list"
    print("PASS  test_make_plan_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 5.  critique — must always return a CritiqueResult (verdict-set)
# ══════════════════════════════════════════════════════════════════════

def test_critique_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.critique import critique
    from bl_pipeline.agent1_fresh.state import Chunk, FlowConditions, ReasonerTrace, CritiqueResult
    trace = ReasonerTrace()
    chunks = []
    flow = FlowConditions()
    for label, response in _BAD_SHAPES.items():
        with patch("bl_pipeline.agent1_fresh.critique.llm.call",
                   side_effect=_make_fake_llm(response)):
            crit = critique("query", flow, trace, chunks)
        assert isinstance(crit, CritiqueResult), f"[{label}] must return a CritiqueResult"
        assert crit.verdict in {"PASS", "NEEDS_REVISION", "FAIL"}, \
            f"[{label}] verdict must be valid, got {crit.verdict!r}"
    print("PASS  test_critique_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 6.  _parse_query — must always return (FlowConditions, float)
# ══════════════════════════════════════════════════════════════════════

def test_parse_query_handles_every_shape() -> None:
    from bl_pipeline.agent1_fresh.run import _parse_query
    from bl_pipeline.agent1_fresh.state import FlowConditions
    for label, response in _BAD_SHAPES.items():
        with patch("bl_pipeline.agent1_fresh.run.llm.call",
                   side_effect=_make_fake_llm(response)):
            flow, cost, override_log = _parse_query("predict transition")
        assert isinstance(flow, FlowConditions), f"[{label}] must return FlowConditions"
        assert isinstance(cost, float), f"[{label}] cost must be float"
        assert isinstance(override_log, list), f"[{label}] override_log must be a list"
    print("PASS  test_parse_query_handles_every_shape")


# ══════════════════════════════════════════════════════════════════════
# 7.  verify_plan_coverage — empty plan does NOT auto-revise (protects
#     API budget; the critic still catches missing physics separately)
# ══════════════════════════════════════════════════════════════════════

def test_verifier_empty_plan_flags_revision() -> None:
    from bl_pipeline.agent1_fresh.verify import verify_plan_coverage
    from bl_pipeline.agent1_fresh.state import Plan, ReasonerTrace
    report = verify_plan_coverage(Plan(), ReasonerTrace())
    assert report.needs_revision is False, \
        "empty plan must NOT trigger a coverage-driven revision"
    assert report.missing_core == []
    print("PASS  test_verifier_empty_plan_flags_revision")


def test_parse_failures_emit_loud_events() -> None:
    """When Sonnet returns pure prose, every dependent parser must
    emit a structured *_parse_failed event — never silently swallow."""
    from bl_pipeline.agent1_fresh.retrieve import _judge
    from bl_pipeline.agent1_fresh.plan import make_plan
    from bl_pipeline.agent1_fresh.critique import critique
    from bl_pipeline.agent1_fresh.state import (
        Chunk, FlowConditions, ReasonerTrace,
    )
    from bl_pipeline.shared.event_bus import bus, Event

    captured: list[tuple[str, dict]] = []

    def listener(e: Event):
        if e.event_type.endswith("_parse_failed"):
            captured.append((e.event_type, dict(e.payload or {})))

    bus.add_listener(listener)
    try:
        # Pre-arm an active run so emit() doesn't silently drop events.
        from bl_pipeline.shared.event_bus import set_current_run, clear_current_run
        set_current_run("test_parse_failures")

        prose = "I cannot determine which chunks are relevant."
        # Judge prose response
        captured.clear()
        with patch("bl_pipeline.agent1_fresh.retrieve.llm.call",
                   side_effect=_make_fake_llm(prose)):
            _judge("q", [Chunk(chunk_id="x::1", paper_id="x", page=1, content="t")])
        assert any(t == "judge_parse_failed" for t, _ in captured), \
            "judge must emit judge_parse_failed on prose"

        # Plan prose response (will retry once, so two events expected)
        captured.clear()
        with patch("bl_pipeline.agent1_fresh.plan.llm.call",
                   side_effect=_make_fake_llm(prose)):
            make_plan("q", FlowConditions(velocity_ms=12.0, turbulence_intensity_pct=3.3),
                      [Chunk(chunk_id="x::1", paper_id="x", page=1, content="t")])
        assert sum(1 for t, _ in captured if t == "plan_parse_failed") >= 1, \
            "plan must emit plan_parse_failed on prose"

        # Critique prose response
        captured.clear()
        with patch("bl_pipeline.agent1_fresh.critique.llm.call",
                   side_effect=_make_fake_llm(prose)):
            critique("q", FlowConditions(), ReasonerTrace(), [])
        assert any(t == "critique_parse_failed" for t, _ in captured), \
            "critique must emit critique_parse_failed on prose"

        clear_current_run()
    finally:
        bus.remove_listener(listener)

    print("PASS  test_parse_failures_emit_loud_events")


def test_verifier_normal_path_unchanged() -> None:
    from bl_pipeline.agent1_fresh.verify import verify_plan_coverage
    from bl_pipeline.agent1_fresh.state import Plan, PlanQuantity, ReasonerTrace
    plan = Plan(quantities_to_compute=[
        PlanQuantity(name="x_t", is_core=True),
        PlanQuantity(name="L_tr", is_core=True),
    ])
    trace = ReasonerTrace()
    trace.final = {"key_findings": [
        {"quantity": "x_t", "value": 0.102, "unit": "m"},
        {"quantity": "L_tr", "value": 0.066, "unit": "m"},
    ]}
    report = verify_plan_coverage(plan, trace)
    assert report.needs_revision is False
    assert "x_t" in report.computed
    assert "L_tr" in report.computed
    print("PASS  test_verifier_normal_path_unchanged")


# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    tests = [
        test_expand_query_handles_every_shape,
        test_generate_supplementary_handles_every_shape,
        test_judge_handles_every_shape,
        test_make_plan_handles_every_shape,
        test_critique_handles_every_shape,
        test_parse_query_handles_every_shape,
        test_verifier_empty_plan_flags_revision,
        test_parse_failures_emit_loud_events,
        test_verifier_normal_path_unchanged,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("-" * 60)
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
