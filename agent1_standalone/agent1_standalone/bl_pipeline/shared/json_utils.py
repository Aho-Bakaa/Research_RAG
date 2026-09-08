"""json_utils.py — small helpers for tolerantly parsing LLM JSON output.

Language models occasionally return JSON that is syntactically slightly
off: unescaped backslashes inside LaTeX strings, trailing commas,
prose-then-JSON preambles, markdown code fences, or raw newlines inside
string values. `parse_lenient_json` applies a sequence of salvage
strategies so nodes don't fail hard on these common quirks.

This is a shared helper because the same pattern comes up in several
nodes (pde_solver Layer 3 self-validation, physics_gap_detector,
paper_gap_detector, etc.). Having one canonical implementation keeps
the behaviour consistent and makes future improvements land in one place.
"""

from __future__ import annotations

import json
import re
from typing import Any


def strip_fences(text: str) -> str:
    """Extract the FIRST fenced code/JSON block; drop any prose around it.

    Why this matters
    ────────────────
    LLMs (gpt-4o in particular) routinely emit responses shaped like:

        ```python
        import numpy as np
        ...
        print(result)
        ```

        This script calculates the transition Reynolds number using …

    The old implementation removed every line starting with ``` but
    kept the prose appended after the closing fence.  Downstream
    consumers then either:
      • tried to `exec()` it → SyntaxError on the prose lines (observed
        in run 5558c9f3: all 3 models failed with `SyntaxError at line 30`
        where line 30 was the first prose word "This");
      • tried to `json.loads()` it → JSON decode failure on the prose.

    New behaviour
    ─────────────
    If the text contains a ```…``` block, return ONLY the content
    between the first pair of fences. Any leading preamble ("Here is the
    code:") and any trailing explanation are dropped.  If only an
    opening fence exists with no closer, return everything after it.
    If there are no fences at all, return the text unchanged.

    This is the right call for BOTH callers:
      • code: exec gets pure Python, no trailing prose to choke on.
      • JSON: json.loads gets pure JSON, no preamble to skip over.
    """
    text = (text or "").strip()
    if not text:
        return ""

    open_idx = text.find("```")
    if open_idx == -1:
        return text  # no fences, leave as-is

    # Skip past the opening-fence line.  The opener may carry a
    # language tag (```python, ```json) before the newline; everything
    # from start-of-opener to the first newline is the fence header.
    open_nl = text.find("\n", open_idx)
    if open_nl == -1:
        # Whole text is the opener line (no body) — nothing to extract.
        return ""

    # Find the closing fence.  Search after the opener; we look first
    # for "\n```" (closer at start of a line) which is the canonical
    # form, then fall back to bare "```" so a sloppy LLM that put the
    # closer mid-line still gets handled.
    close_idx = text.find("\n```", open_nl)
    if close_idx == -1:
        close_idx = text.find("```", open_nl + 1)
        if close_idx == -1:
            # No closer at all — take everything after the opener
            # (and trust the caller's parser to deal with truncation).
            return text[open_nl + 1:].strip()

    return text[open_nl + 1:close_idx].strip()


def strip_trailing_prose(code: str) -> str:
    """Trim trailing prose from LLM-generated Python source.

    Why this matters
    ────────────────
    Despite explicit prompt instructions to emit code-only, LLMs (Sonnet
    in particular when reasoning through a derivation) occasionally
    append an explanatory paragraph AFTER the Python — without code
    fences.  `strip_fences()` returns such text unchanged (correct
    behaviour given the no-fence case), but then `exec()` chokes on the
    prose with a SyntaxError.  Symptom in run a609c5ee:

        # ... real Mayle code ...
        print('Final result:', result)

        This code computes the transition onset Reynolds number using
        Mayle's correlation, adhering strictly to the conventions ...

    The trailing prose contained an apostrophe ("Mayle's") that the
    Python tokenizer parsed as an unterminated string literal at line
    26.  All algebraic models in that run produced executed=False with
    SyntaxError; downstream A3 then step-blocked silently.

    How this helper works
    ─────────────────────
    1. Try compiling the whole text as Python.  If it works, return it.
    2. Otherwise walk back from the end one line at a time and return
       the longest line-prefix that does compile.
    3. If NO prefix compiles, return the original text so the caller's
       error reporting still surfaces the underlying problem (don't
       silently swallow a genuinely-broken codegen).

    The walk-back is O(N²) in line-count in the worst case but real-
    world codegen is well under 100 lines so compile-cost dominates;
    this stays under 5 ms per call in practice.

    Limitations
    ───────────
    A mid-file syntax error (rare — the LLM would have to inject prose
    inside the code body) is NOT recovered by this; we accept that
    trade-off.  The typical failure mode this helper fixes is trailing
    prose, which is the one we observed in production.
    """
    if not code or not code.strip():
        return code
    try:
        compile(code, "<llm_codegen>", "exec")
        return code  # already valid
    except SyntaxError:
        pass
    lines = code.splitlines()
    # Walk back. Keep at least one line so we always return *some* code.
    for cut in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:cut])
        if not candidate.strip():
            continue
        try:
            compile(candidate, "<llm_codegen>", "exec")
            return candidate
        except SyntaxError:
            continue
    return code  # nothing compiled — surface the error to the caller


def parse_lenient_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Try to parse a string as JSON, with a series of fallbacks.

    Strategies, in order:
      1. Strip markdown fences, try strict json.loads.
      2. Extract the outermost {...} or [...] block and try again.
      3. Clean common LLM errors and retry:
           - collapse actual newlines inside strings into \\n
           - fix trailing commas inside arrays/objects
           - escape lone backslashes that aren't valid JSON escapes

    Returns the parsed object on success, or None if every strategy
    fails. Callers decide how to handle None (usually fall back to a
    safe default so the pipeline doesn't die).
    """
    if not text:
        return None

    # ── Strategy 1: strict parse of fence-stripped text ─────────────
    cleaned = strip_fences(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: extract outermost {...} or [...] block ──────────
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        first = cleaned.find(open_ch)
        last  = cleaned.rfind(close_ch)
        if first == -1 or last <= first:
            continue
        candidate = cleaned[first:last + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except json.JSONDecodeError:
            # Try the cleaning pass on this subsection
            patched = _apply_cleanup_patches(candidate)
            try:
                obj = json.loads(patched)
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                continue

    # ── Strategy 3: apply cleanup on the full text ──────────────────
    patched = _apply_cleanup_patches(cleaned)
    try:
        obj = json.loads(patched)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        return None

    return None


# ── Cleanup patches ──────────────────────────────────────────────────

def _apply_cleanup_patches(text: str) -> str:
    """Apply the common 'LLM JSON' repairs.

    These are intentionally conservative — we'd rather fail to parse
    than produce a parsed-but-wrong structure. Each patch targets a
    specific known pattern.
    """
    # 1. Remove trailing commas before closing brackets (common LLM slip):
    #    {"a": 1,} -> {"a": 1}       ["x",] -> ["x"]
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # 2. Escape lone backslashes that aren't already part of a valid
    #    JSON escape sequence. Valid escapes start with \", \\, \/, \b,
    #    \f, \n, \r, \t, \u followed by 4 hex digits. Anything else
    #    (like \lambda, \theta) needs the backslash doubled.
    def _fix_backslash(m: re.Match[str]) -> str:
        ch = m.group(1)
        if ch in ('"', "\\", "/", "b", "f", "n", "r", "t"):
            return m.group(0)
        if ch == "u":
            return m.group(0)   # leave unicode escapes alone
        return "\\\\" + ch
    text = re.sub(r"\\(.)", _fix_backslash, text)

    # 3. Replace raw newlines inside string values with \n.
    #    We do this by walking the text and tracking whether we're
    #    inside a double-quoted string, flipping state on unescaped
    #    quotes. Heavier than a regex but safe.
    text = _escape_newlines_in_strings(text)

    return text


def _escape_newlines_in_strings(text: str) -> str:
    """Replace real newlines / carriage returns inside double-quoted
    strings with their escape sequences. Leaves newlines outside
    strings (JSON whitespace) alone.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            out.append("\\r")
            continue
        out.append(ch)
    return "".join(out)
