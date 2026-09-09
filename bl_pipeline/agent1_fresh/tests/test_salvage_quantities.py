"""Test the truncated-JSON salvage path in plan.py.

NO API spend — synthetic truncated responses, salvage extracts what
it can.

Run:
    python -m bl_pipeline.agent1_fresh.tests.test_salvage_quantities
"""
from __future__ import annotations
import io, sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

from bl_pipeline.agent1_fresh.plan import _salvage_quantities


def test_complete_json_salvaged() -> None:
    raw = '{"quantities_to_compute":[{"name":"x_t","method":"Mayle"},{"name":"L_tr","method":"AGS"}]}'
    out = _salvage_quantities(raw)
    assert len(out) == 2
    assert out[0]["name"] == "x_t"
    assert out[1]["name"] == "L_tr"
    print("PASS  test_complete_json_salvaged")


def test_truncated_mid_entry_salvages_completed_ones() -> None:
    """Sonnet truncated mid 3rd entry — should still get first 2."""
    raw = (
        '{"quantities_to_compute":[\n'
        '  {"name":"Re_theta_t","method":"Mayle Eq.9","is_core":false},\n'
        '  {"name":"x_t","method":"Blasius back-out","is_core":true},\n'
        '  {"name":"L_tr","met'   # ← cut here, no closing
    )
    out = _salvage_quantities(raw)
    assert len(out) == 2, f"expected 2 salvaged entries, got {len(out)}"
    assert out[0]["name"] == "Re_theta_t"
    assert out[1]["name"] == "x_t"
    print("PASS  test_truncated_mid_entry_salvages_completed_ones")


def test_truncated_after_first_entry() -> None:
    raw = (
        '{"quantities_to_compute":[\n'
        '  {"name":"x_t","method":"Blasius","source_papers":["m::1"]},\n'
        '  {"nam'   # ← cut almost immediately
    )
    out = _salvage_quantities(raw)
    assert len(out) == 1
    assert out[0]["name"] == "x_t"
    print("PASS  test_truncated_after_first_entry")


def test_no_marker_returns_empty() -> None:
    raw = 'This is just prose, no JSON marker at all.'
    out = _salvage_quantities(raw)
    assert out == []
    print("PASS  test_no_marker_returns_empty")


def test_empty_array_returns_empty() -> None:
    raw = '{"quantities_to_compute":[]}'
    out = _salvage_quantities(raw)
    assert out == []
    print("PASS  test_empty_array_returns_empty")


def test_handles_nested_braces_in_strings() -> None:
    """The method string can contain `{` and `}` — depth counter must
    only count braces outside string literals."""
    raw = (
        '{"quantities_to_compute":[\n'
        '  {"name":"x_t","method":"Re_x = Re_{theta}^2 / 0.664^2 — note {nested}"}\n'
        ']}'
    )
    out = _salvage_quantities(raw)
    assert len(out) == 1
    assert out[0]["name"] == "x_t"
    assert "{nested}" in out[0]["method"]
    print("PASS  test_handles_nested_braces_in_strings")


def test_handles_escaped_quotes_in_strings() -> None:
    raw = (
        '{"quantities_to_compute":[\n'
        '  {"name":"foo","method":"contains \\"quoted\\" word"}\n'
        ']}'
    )
    out = _salvage_quantities(raw)
    assert len(out) == 1
    assert out[0]["method"] == 'contains "quoted" word'
    print("PASS  test_handles_escaped_quotes_in_strings")


# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_complete_json_salvaged,
        test_truncated_mid_entry_salvages_completed_ones,
        test_truncated_after_first_entry,
        test_no_marker_returns_empty,
        test_empty_array_returns_empty,
        test_handles_nested_braces_in_strings,
        test_handles_escaped_quotes_in_strings,
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
