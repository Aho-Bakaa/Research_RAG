"""Tests for strip_trailing_prose — the compile-walk-back helper that
recovers LLM codegen with trailing prose paragraphs.

Run from repo root:
    python -m bl_pipeline.shared.tests.test_strip_trailing_prose
"""
from __future__ import annotations

import io
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )
    except Exception:
        pass

from bl_pipeline.shared.json_utils import strip_trailing_prose


def test_valid_code_unchanged() -> None:
    code = "x = 1\nprint(x)\n"
    assert strip_trailing_prose(code) == code
    print("PASS  test_valid_code_unchanged")


def test_a609c5ee_mayle_real_failure() -> None:
    """The exact failure mode from run a609c5ee: prose with apostrophe."""
    code = (
        "import numpy as np\n"
        "Tu = 3.3\n"
        "Re_theta_t = 400 * Tu**(-5/8)\n"
        "result = {'Re_theta_t_Mayle': Re_theta_t}\n"
        "print('Final result:', result)\n"
        "\n"
        "This code computes the transition onset Reynolds number using "
        "Mayle's correlation, adhering strictly to the conventions...\n"
    )
    cleaned = strip_trailing_prose(code)
    # The trailing prose paragraph must be gone; the print line must remain.
    assert "This code computes" not in cleaned
    assert "print('Final result:'" in cleaned
    assert "Re_theta_t = 400 * Tu" in cleaned
    # And the result MUST be compilable.
    compile(cleaned, "<test>", "exec")
    print("PASS  test_a609c5ee_mayle_real_failure")


def test_multiline_prose_paragraph_stripped() -> None:
    code = (
        "x = 1\n"
        "y = 2\n"
        "z = x + y\n"
        "\n"
        "This script demonstrates basic arithmetic.\n"
        "It uses two variables x and y.\n"
        "The result is stored in z.\n"
    )
    cleaned = strip_trailing_prose(code)
    assert "This script" not in cleaned
    assert "z = x + y" in cleaned
    compile(cleaned, "<test>", "exec")
    print("PASS  test_multiline_prose_paragraph_stripped")


def test_prose_with_apostrophe_stripped() -> None:
    """Prose containing an apostrophe is the exact trigger we hit."""
    code = "x = 1\nMayle's correlation result is shown above.\n"
    cleaned = strip_trailing_prose(code)
    assert "Mayle's" not in cleaned
    assert "x = 1" in cleaned
    print("PASS  test_prose_with_apostrophe_stripped")


def test_empty_input() -> None:
    assert strip_trailing_prose("") == ""
    assert strip_trailing_prose("   \n  \n") == "   \n  \n"
    print("PASS  test_empty_input")


def test_only_prose_returns_unchanged() -> None:
    """If NOTHING compiles, return as-is so the caller's error surfaces."""
    text = "This is just a sentence with an apostrophe.\n"
    cleaned = strip_trailing_prose(text)
    # We don't truncate to empty — caller will see the SyntaxError.
    assert cleaned == text
    print("PASS  test_only_prose_returns_unchanged")


def test_preserves_blank_lines_within_code() -> None:
    """Blank lines mid-code must not confuse the walk-back."""
    code = (
        "def f():\n"
        "    return 1\n"
        "\n"
        "x = f()\n"
        "\n"
        "y = 2\n"
    )
    cleaned = strip_trailing_prose(code)
    assert cleaned == code
    print("PASS  test_preserves_blank_lines_within_code")


def test_handles_decorator_at_end() -> None:
    """A trailing decorator alone is invalid Python; walk-back drops it."""
    code = "x = 1\n@my_decorator\n"
    cleaned = strip_trailing_prose(code)
    # The decorator without a function is a SyntaxError; cleaned ends at x = 1.
    assert "x = 1" in cleaned
    compile(cleaned, "<test>", "exec")
    print("PASS  test_handles_decorator_at_end")


def test_handles_unclosed_string_at_end() -> None:
    """An unclosed string literal at end → walk back drops it."""
    code = 'x = 1\nbroken = "unterminated\n'
    cleaned = strip_trailing_prose(code)
    assert "x = 1" in cleaned
    assert "unterminated" not in cleaned
    compile(cleaned, "<test>", "exec")
    print("PASS  test_handles_unclosed_string_at_end")


# ───────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_valid_code_unchanged,
        test_a609c5ee_mayle_real_failure,
        test_multiline_prose_paragraph_stripped,
        test_prose_with_apostrophe_stripped,
        test_empty_input,
        test_only_prose_returns_unchanged,
        test_preserves_blank_lines_within_code,
        test_handles_decorator_at_end,
        test_handles_unclosed_string_at_end,
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
