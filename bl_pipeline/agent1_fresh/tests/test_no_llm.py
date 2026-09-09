"""Deterministic tests for fresh Agent 1 — NO LLM calls, NO chroma access.

Exercises:
  • state.py round-trip + JSON safety
  • reason.py block parsers (<tool_call>, <final>, prose stripping)
  • tools.py dispatcher error paths (unknown tool, bad args, broken code)
  • tools.compute on a clean physics formula (Mayle Re_θ,t at Tu=3.3%)
  • Tu-convention guard: Mayle correlation gives ~190 at Tu=3.3% (percent)
                        and ~3373 at Tu=0.033 (decimal — the historical bug)

Run from repo root:
    python -m bl_pipeline.agent1_fresh.tests.test_no_llm
"""
from __future__ import annotations

import io
import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                       errors="replace")
    except Exception:
        pass

from bl_pipeline.agent1_fresh.reason import (
    _extract_final, _extract_first_tool_call, _strip_blocks,
)
from bl_pipeline.agent1_fresh.state import (
    AgentResult, Chunk, FlowConditions, IterationTrace, ToolCall, _jsonable,
)
from bl_pipeline.agent1_fresh.tools import TOOLS, dispatch


# ══════════════════════════════════════════════════════════════════════
# state.py — JSON serialisation + dataclass round-trip
# ══════════════════════════════════════════════════════════════════════

def test_agent_result_serialises_to_json() -> None:
    r = AgentResult(run_id="t1", user_query="onset for U=12 Tu=3.3")
    r.flow = FlowConditions(velocity_ms=12.0, turbulence_intensity_pct=3.3)
    r.retrieval.iterations.append(
        IterationTrace(iteration=1, sub_queries=["q1", "q2"], pool_size_after_merge=15),
    )
    r.reasoner.tool_calls.append(ToolCall(step=1, tool_name="compute",
                                          args={"code": "result={'x':1}"}))
    blob = json.dumps(r.to_dict())          # must NOT raise
    parsed = json.loads(blob)
    assert parsed["user_query"] == r.user_query
    assert parsed["flow"]["velocity_ms"] == 12.0
    print("PASS  test_agent_result_serialises_to_json")


def test_jsonable_handles_numpy_and_dataclasses() -> None:
    """_jsonable must not blow up on numpy scalars or dataclass instances."""
    try:
        import numpy as np
        v = _jsonable({"arr": np.array([1, 2, 3]), "scalar": np.float64(3.14)})
        assert v == {"arr": [1, 2, 3], "scalar": 3.14}
    except ImportError:
        pass  # numpy not available in test env
    fc = FlowConditions(velocity_ms=10.0)
    v = _jsonable(fc)
    assert isinstance(v, dict) and v["velocity_ms"] == 10.0
    print("PASS  test_jsonable_handles_numpy_and_dataclasses")


# ══════════════════════════════════════════════════════════════════════
# reason.py — block parsers
# ══════════════════════════════════════════════════════════════════════

def test_extract_tool_call_clean() -> None:
    text = (
        "Thinking...\n"
        "<tool_call>\n"
        '{"tool": "lookup_glossary", "args": {"symbol": "Re_theta_t"}}\n'
        "</tool_call>"
    )
    tc = _extract_first_tool_call(text)
    assert tc is not None
    assert tc["tool"] == "lookup_glossary"
    assert tc["args"] == {"symbol": "Re_theta_t"}
    print("PASS  test_extract_tool_call_clean")


def test_extract_tool_call_malformed_flagged() -> None:
    text = "<tool_call>not json at all</tool_call>"
    tc = _extract_first_tool_call(text)
    assert tc is not None
    assert tc.get("_malformed") is True
    print("PASS  test_extract_tool_call_malformed_flagged")


def test_extract_final_clean() -> None:
    text = (
        "All done.\n"
        '<final>{"answer_summary": "x_t ≈ 0.46 m", "key_findings": []}</final>'
    )
    final = _extract_final(text)
    assert final and final["answer_summary"].startswith("x_t")
    print("PASS  test_extract_final_clean")


def test_strip_blocks_keeps_prose_only() -> None:
    text = (
        "Some prose.\n"
        "<tool_call>{}</tool_call>\n"
        "More prose.\n"
        "<final>{}</final>"
    )
    stripped = _strip_blocks(text)
    assert "tool_call" not in stripped
    assert "final" not in stripped
    assert "Some prose" in stripped and "More prose" in stripped
    print("PASS  test_strip_blocks_keeps_prose_only")


# ══════════════════════════════════════════════════════════════════════
# tools.py — dispatcher error paths + compute happy path
# ══════════════════════════════════════════════════════════════════════

def test_dispatch_unknown_tool() -> None:
    r = dispatch("not_a_tool", {})
    assert r.get("error", "").startswith("unknown tool")
    print("PASS  test_dispatch_unknown_tool")


def test_dispatch_bad_args_type() -> None:
    r = dispatch("compute", "bad string instead of dict")  # type: ignore[arg-type]
    assert "error" in r and "dict" in r["error"]
    print("PASS  test_dispatch_bad_args_type")


def test_compute_mayle_at_3pct_gives_canonical_value() -> None:
    """Mayle Re_theta_t = 400 * Tu^(-5/8) at Tu = 3.3 % → ~190."""
    r = dispatch("compute", {"code":
        "Tu = 3.3\nresult = {'Re_theta_t_Mayle': 400 * Tu**(-5/8)}"})
    assert r.get("error") is None, f"unexpected error: {r.get('error')}"
    val = r["result"]["Re_theta_t_Mayle"]
    assert 180 < val < 200, f"expected ~190, got {val}"
    print(f"PASS  test_compute_mayle_at_3pct_gives_canonical_value  (got {val:.2f})")


def test_compute_mayle_at_decimal_gives_buggy_value() -> None:
    """Same formula at Tu=0.033 (decimal — the historical bug) gives ~3373.
    Documenting the unit-convention pitfall in a test."""
    r = dispatch("compute", {"code":
        "Tu = 0.033\nresult = {'Re_theta_t_Mayle_BUG': 400 * Tu**(-5/8)}"})
    val = r["result"]["Re_theta_t_Mayle_BUG"]
    assert 3000 < val < 4000, f"expected ~3373, got {val}"
    print(f"PASS  test_compute_mayle_at_decimal_gives_buggy_value  (got {val:.2f} — the bug)")


def test_compute_broken_syntax_captured() -> None:
    r = dispatch("compute", {"code": "this is not python !!"})
    assert r.get("error") is not None
    print("PASS  test_compute_broken_syntax_captured")


def test_tool_registry_complete() -> None:
    """The three contractual tools must be in the registry."""
    for name in ("compute", "lookup_glossary", "search"):
        assert name in TOOLS, f"missing tool: {name}"
    print("PASS  test_tool_registry_complete")


# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    tests = [
        test_agent_result_serialises_to_json,
        test_jsonable_handles_numpy_and_dataclasses,
        test_extract_tool_call_clean,
        test_extract_tool_call_malformed_flagged,
        test_extract_final_clean,
        test_strip_blocks_keeps_prose_only,
        test_dispatch_unknown_tool,
        test_dispatch_bad_args_type,
        test_compute_mayle_at_3pct_gives_canonical_value,
        test_compute_mayle_at_decimal_gives_buggy_value,
        test_compute_broken_syntax_captured,
        test_tool_registry_complete,
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
