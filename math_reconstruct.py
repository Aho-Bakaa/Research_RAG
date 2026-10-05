"""math_reconstruct.py — Rule-based mathematical reconstructor with equation classification.

Pipeline:
1. Inspects equation regions in PDF using PyMuPDF rawdict and drawing vectors.
2. Classifies each equation into 'simple' (rule-based) vs 'hard' (image-based formula model):
   - 'hard' if contains: AdvP415979 glyphs, radicals, sum/integral, matrix/cases, deep nesting, or unmapped codes.
   - 'simple' routed to geometric rule reconstructor.
3. Decodes character spans overriding PDF /ToUnicode with glyph_map.json.
4. Detects horizontal vector bars for fraction numerator/denominator extraction.
5. Emits clean LaTeX for simple equations and flags/routes hard equations.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Any
import fitz

GLYPH_MAP_PATH = Path(__file__).parent / "glyph_map.json"

with open(GLYPH_MAP_PATH, "r", encoding="utf-8") as f:
    GLYPH_MAP: dict[str, dict[str, dict[str, str]]] = json.load(f)

MATH_FONTS = {"AdvP49C2A1", "AdvP3E483E", "AdvP4A4712", "AdvP415979", "AdvP3DD108"}

GREEK_AND_SYMBOLS = [
    r"\\theta", r"\\rho", r"\\mu", r"\\omega", r"\\gamma",
    r"\\delta", r"\\lambda", r"\\nu", r"\\sigma", r"\\Omega",
    r"\\cdot", r"\\le", r"\\ge", r"\\partial"
]
ALL_SYM_PATTERN = "(?:" + "|".join(GREEK_AND_SYMBOLS) + ")"


def clean_font_name(fn: str) -> str:
    """Strip 6-character random subset prefix (e.g. 'BMGPDA+AdvP49C2A1' -> 'AdvP49C2A1')."""
    return fn.split("+")[-1]


def decode_char(font: str, code: int) -> str:
    """Decode a character code using glyph_map.json, strictly overriding PDF /ToUnicode."""
    clean_f = clean_font_name(font)
    if clean_f in MATH_FONTS:
        entry = GLYPH_MAP.get(clean_f, {}).get(str(code))
        if entry:
            return entry["latex"]
        print(f"WARNING: Unknown glyph in {clean_f}: code {code} (0x{code:02x})")
        return chr(code) if 32 <= code <= 126 else f"\\x{code:02x}"
    # Standard text font (e.g. AdvOT563941f4)
    return chr(code) if 32 <= code <= 126 else chr(code)


def decode_span_text(span: dict[str, Any]) -> str:
    fn = span["font"]
    text = ""
    for ch in span.get("chars", []):
        c = ch["c"]
        code = ord(c) if isinstance(c, str) else int(c)
        text += decode_char(fn, code)
    return text


def format_script_string(s: str) -> str:
    if not s:
        return ""
    for sym in GREEK_AND_SYMBOLS:
        s = re.sub(f"({sym})([a-zA-Z0-9])", r"\1 \2", s)
    return s


def post_process_latex(text: str) -> str:
    """Apply contextual cleanups: decimal colons, argument semicolons, macro spacing."""
    # 1. Decimal point: digit:digit -> digit.digit (AIAA math colon hijacking)
    text = re.sub(r"(\d+):(\d+)", r"\1.\2", text)
    # 2. Semicolons in argument lists -> comma: e.g. min(max(F_{onset1}; F_{onset1}^4); 2.0)
    text = text.replace(";", ", ")
    # 3. Standard functions
    text = re.sub(r"\bmin\b", r"\\min", text)
    text = re.sub(r"\bmax\b", r"\\max", text)
    # 4. Spacing between LaTeX math symbol macros and letters
    text = re.sub(f"({ALL_SYM_PATTERN})([a-zA-Z0-9])", r"\1 \2", text)
    text = re.sub(f"({ALL_SYM_PATTERN})({ALL_SYM_PATTERN})", r"\1 \2", text)
    # 5. Clean up duplicate whitespace and format equals sign
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*=\s*", " = ", text)
    # Optional single letter subscript style R_{T} -> R_T if single ASCII letter
    text = re.sub(r"_\{([A-Z])\}", r"_\1", text)
    return text


def build_line_text(spans: list[dict[str, Any]]) -> str:
    """Assemble horizontally sorted spans into LaTeX with baseline-derived sub/superscripts."""
    if not spans:
        return ""

    sizes = [s["size"] for s in spans]
    base_size = max(set(sizes), key=sizes.count)
    base_ys = [s["origin"][1] for s in spans if abs(s["size"] - base_size) < 0.5]
    base_y = sum(base_ys) / len(base_ys) if base_ys else spans[0]["origin"][1]

    tokens: list[dict[str, Any]] = []
    for s in spans:
        decoded = decode_span_text(s)
        if not decoded.strip():
            continue
        sy = s["origin"][1]
        sz = s["size"]
        is_script = (sz <= base_size * 0.75) or (sz <= 6.5 and base_size >= 8.5)
        is_sub = is_script and (sy > base_y + 0.5)
        is_sup = is_script and (sy < base_y - 0.5)

        tokens.append({
            "text": decoded,
            "x": s["origin"][0],
            "y": sy,
            "size": sz,
            "is_sub": is_sub,
            "is_sup": is_sup,
            "is_base": not (is_sub or is_sup)
        })

    res = ""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t["is_base"]:
            txt = t["text"]
            if txt == "=":
                res = res.rstrip() + " = "
            else:
                res += txt

            # Collect scripts attached to this base
            sub_parts = []
            sup_parts = []
            j = i + 1
            while j < len(tokens) and not tokens[j]["is_base"] and (tokens[j]["x"] - t["x"] < 25.0):
                if tokens[j]["is_sub"]:
                    sub_parts.append(tokens[j]["text"])
                elif tokens[j]["is_sup"]:
                    sup_parts.append(tokens[j]["text"])
                j += 1
            
            sub_str = format_script_string("".join(sub_parts))
            sup_str = format_script_string("".join(sup_parts))

            if sub_str and sup_str:
                res += f"_{{{sub_str}}}^{{{sup_str}}}"
                i = j
                continue
            elif sub_str:
                res += f"_{{{sub_str}}}"
                i = j
                continue
            elif sup_str:
                res += f"^{{{sup_str}}}"
                i = j
                continue
            else:
                i += 1
        else:
            txt = format_script_string(t["text"])
            if t["is_sub"]:
                res += f"_{{{txt}}}"
            elif t["is_sup"]:
                res += f"^{{{txt}}}"
            else:
                res += txt
            i += 1

    return res


def reconstruct_simple_equation(doc: fitz.Document, page_num: int, eq_bbox: tuple[float, float, float, float]) -> str:
    """Reconstruct a simple equation into valid LaTeX using geometric and font rules."""
    page = doc[page_num]
    raw_dict = page.get_text("rawdict", clip=eq_bbox)
    
    # 1. Search for horizontal vector drawing bars (fraction line)
    all_drawings = page.get_drawings()
    fraction_bars = []
    for dw in all_drawings:
        r = dw.get("rect")
        if r and r.height < 2.0 and r.width > 5.0:
            if (eq_bbox[0] - 5 <= r.x0 and r.x1 <= eq_bbox[2] + 5 and
                eq_bbox[1] - 5 <= r.y0 and r.y1 <= eq_bbox[3] + 5):
                fraction_bars.append(r)

    # 2. Extract spans, filtering out equation tags "(n)"
    spans = []
    for b in raw_dict.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                text_raw = "".join(ch["c"] if isinstance(ch["c"], str) else chr(ch["c"]) for ch in s.get("chars", []))
                if re.match(r"^\(\d+\)$", text_raw.strip()) and s["origin"][0] > 500:
                    continue
                if not text_raw.strip():
                    continue
                spans.append(s)

    if not spans:
        return ""

    spans.sort(key=lambda s: s["origin"][0])

    # 3. Fraction decomposition
    if fraction_bars:
        bar = fraction_bars[0]
        bar_x0, bar_x1 = bar.x0, bar.x1
        bar_y = (bar.y0 + bar.y1) / 2.0

        lhs_spans = []
        num_spans = []
        den_spans = []
        rhs_spans = []

        for s in spans:
            sx_center = (s["bbox"][0] + s["bbox"][2]) / 2.0
            sy = s["origin"][1]
            if sx_center < bar_x0 - 3.0:
                lhs_spans.append(s)
            elif sx_center > bar_x1 + 3.0:
                rhs_spans.append(s)
            else:
                if sy < bar_y:
                    num_spans.append(s)
                else:
                    den_spans.append(s)

        lhs_text = build_line_text(lhs_spans)
        num_text = build_line_text(num_spans)
        den_text = build_line_text(den_spans)
        rhs_text = build_line_text(rhs_spans)

        frac_part = f"\\frac{{{num_text}}}{{{den_text}}}"
        parts = [p for p in [lhs_text, frac_part, rhs_text] if p]
        full = " ".join(parts)
        return post_process_latex(full)
    else:
        full = build_line_text(spans)
        return post_process_latex(full)


def classify_equation(doc: fitz.Document, page_num: int, eq_bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Classify an equation as 'simple' or 'hard' based on structural criteria."""
    page = doc[page_num]
    d = page.get_text("rawdict", clip=eq_bbox)

    reasons = []
    has_advp415979 = False
    has_radical = False
    has_sum_integral = False
    has_unmapped_glyph = False

    spans = []
    lines = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            lines.append(l)
            for s in l.get("spans", []):
                text_raw = "".join(ch["c"] if isinstance(ch["c"], str) else chr(ch["c"]) for ch in s.get("chars", []))
                if re.match(r"^\(\d+\)$", text_raw.strip()) and s["origin"][0] > 500:
                    continue
                if not text_raw.strip():
                    continue
                spans.append(s)

    for s in spans:
        fn = clean_font_name(s["font"])
        if fn == "AdvP415979":
            has_advp415979 = True
            reasons.append("Contains AdvP415979 (multi-piece delimiter / radical)")

        for ch in s.get("chars", []):
            c = ch["c"]
            code = ord(c) if isinstance(c, str) else int(c)
            if fn in MATH_FONTS:
                if str(code) not in GLYPH_MAP.get(fn, {}):
                    has_unmapped_glyph = True
                    reasons.append(f"Unmapped glyph in {fn}: code {code}")
                entry = GLYPH_MAP.get(fn, {}).get(str(code), {})
                latex_sym = entry.get("latex", "")
                if "\\surd" in latex_sym or "\\sqrt" in latex_sym or "\\overline" in latex_sym:
                    has_radical = True
                    reasons.append("Contains radical / square root")
                if "\\int" in latex_sym or "\\sum" in latex_sym:
                    has_sum_integral = True
                    reasons.append("Contains sum/integral")

    # Multi-line cases / piecewise layout
    has_matrix_cases = len(lines) >= 3 and has_advp415979
    if has_matrix_cases:
        reasons.append("Multi-line cases / piecewise layout")

    is_hard = has_advp415979 or has_radical or has_sum_integral or has_matrix_cases or has_unmapped_glyph

    return {
        "classification": "hard" if is_hard else "simple",
        "reasons": list(set(reasons))
    }
