"""chunk_classifier.py — Per-chunk metadata enrichment.

Adds two fields to every chunk at ingestion time:

  evidence_type: one of
      'experimental'         — reports measured data, hot-wire/PIV/oil-flow etc.
      'DNS'                  — reports direct numerical simulation results
      'LES'                  — reports large-eddy simulation results
      'CFD_validated'        — reports RANS/steady CFD with stated validation
      'theoretical_bound'    — states a mathematical bound or limit
      'empirical_correlation' — states a fit or formula calibrated to data
      'review_statement'     — paraphrases or summarises other work
      'derivation'           — pure algebra / manipulation of equations
      'other'                — anything that doesn't fit above

  contains_numeric_data: bool
      True if the chunk contains tabulated or inline numeric values that
      could serve as a validation reference point (e.g. "Re_θ,t = 210",
      "180 ≤ Re_θ ≤ 240", "measured x_t = 0.23 m").

Design choices:

  • Numeric-data detection is DETERMINISTIC (regex). Cheap, consistent
    across runs, no LLM cost.
  • Evidence-type classification uses a small LLM call (Haiku). One call
    per chunk at ingestion time, then cached forever in the chunk's
    metadata. Downstream stages never re-classify.

Both fields are additive — existing chunks without them are harmless;
they just get the default 'other' / False until the backfill runs.
"""

from __future__ import annotations

import re
from typing import Any


# ══════════════════════════════════════════════════════════════════
# Numeric-data detection (deterministic, regex-based)
# ══════════════════════════════════════════════════════════════════

# Patterns that strongly suggest reportable numeric data:
#   "Re_θ,t = 210"  |  "Re_{θ,t} = 210"
#   "x_t = 0.15 m"  |  "x_t ≈ 0.15 m"
#   "Tu = 3 %"      |  "Tu = 3%"
#   "C_f = 0.0025"  |  "δ = 2.3 mm"
#   "180–240"       (range notation)
#   "1.5 × 10⁻⁵"    (scientific notation)
#
# We require either a Greek letter or a recognisable fluids-variable
# token next to an equals/approx/range/number so we avoid counting
# section numbers and page numbers as "data".
#
# Kept intentionally broad — the validator can filter further.

_SYMBOL_TOKEN = (
    r"(?:"
    # Plain ASCII aliases for common transition / BL quantities
    r"Re_?[a-zθtθ,\\_\{\}xLTw ]*"
    r"|x_?[tLe01a-z]"
    r"|theta_?t|θ_?t"
    r"|Tu|T_?u"
    r"|lambda_?θ|λ_?θ"
    r"|C_?f|C_?p"
    r"|delta[*\*]?_?\d*|δ[*\*]?_?\d*"
    r"|gamma|γ"
    r"|n[·\.\\s]?sigma|n[·\.\\s]?σ|n[·\.\\s]?hat[·\.\\s]?sigma"
    r"|Re_?x|Re_?L|Re_?LT"
    r"|F\(lambda\)|F\(λ\)"
    r")"
)

_NUMBER_LITERAL = (
    r"(?:"
    r"\d+\.?\d*"                              # 3, 3.0, 0.1149
    r"(?:[eE][\-\+]?\d+)?"                     # scientific 1.5e-5
    r"|\d+\.?\d*\s*[×xX*]\s*10"               # 1.5 × 10 (prefix of sci notation)
    r"[\^\s]*[\-−⁻]?\s*"                       # optional caret / ASCII minus / Unicode minus / superscript minus
    r"[\d⁰¹²³⁴⁵⁶⁷⁸⁹\-\+]+"                    # digits (ASCII or Unicode superscripts)
    r")"
)

_EQUAL_SIGN = r"(?:=|≈|≃|~|:)"

# Pattern 1: "<symbol> = <number>"
_PATTERN_ASSIGNMENT = re.compile(
    _SYMBOL_TOKEN + r"\s*" + _EQUAL_SIGN + r"\s*" + _NUMBER_LITERAL,
    re.IGNORECASE,
)

# Pattern 2: numeric range, e.g. "180–240", "180 to 240", "1.0e5-1.5e5"
_PATTERN_RANGE = re.compile(
    _NUMBER_LITERAL + r"\s*(?:[-–—]|to)\s*" + _NUMBER_LITERAL,
    re.IGNORECASE,
)

# Pattern 3: scientific notation floating around. Covers BOTH forms:
#   (a) "1.5 × 10⁻⁵"   Unicode multiplier + Unicode/ASCII superscripts
#   (b) "1.5e-5"        ASCII 'e' / 'E' form
_PATTERN_SCIENTIFIC = re.compile(
    r"\d+\.?\d*\s*[×xX*]\s*10[\^\s]*[\-−⁻]?\s*[\d⁰¹²³⁴⁵⁶⁷⁸⁹\-\+]+"
    r"|"
    r"\d+\.?\d*[eE][\-\+]?\d+",
)


def contains_numeric_data(text: str) -> bool:
    """True if the chunk likely contains reportable validation data.

    We require at least ONE of:
      • a symbol=value assignment (strongest signal)
      • a numeric range (e.g. "180–240")
      • multiple scientific-notation numbers (table-like)

    Single numbers in prose ("Figure 3") are NOT counted.  Very short
    inputs (< 6 chars) are rejected outright — prevents noise on empty
    or ultra-short text.  A short but dense string like "Tu=3%" still
    passes because the assignment regex matches.
    """
    if not text or len(text) < 6:
        return False

    if _PATTERN_ASSIGNMENT.search(text):
        return True
    if _PATTERN_RANGE.search(text):
        return True
    # Multiple scientific-notation numbers in one chunk → probably a table
    if len(_PATTERN_SCIENTIFIC.findall(text)) >= 2:
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# Evidence-type classification (LLM, one call per chunk at ingest)
# ══════════════════════════════════════════════════════════════════

_EVIDENCE_TYPES = [
    "experimental",
    "DNS",
    "LES",
    "CFD_validated",
    "theoretical_bound",
    "empirical_correlation",
    "review_statement",
    "derivation",
    "other",
]

_EVIDENCE_PROMPT = """Classify this chunk from a fluid-mechanics paper by the type of scientific evidence it reports. Return ONE label from this exact list:

  experimental          — measured data (hot-wire, PIV, LDA, oil-flow, etc.)
  DNS                   — direct numerical simulation results
  LES                   — large-eddy simulation results
  CFD_validated         — RANS/steady CFD with explicit validation vs data
  theoretical_bound     — mathematical bound or stability limit
  empirical_correlation — a fit or formula calibrated to data (e.g. Mayle's 400·Tu^(-5/8))
  review_statement      — summary or paraphrase of other work, no new results
  derivation            — algebraic manipulation, no data, no measurement
  other                 — anything that doesn't cleanly fit the above

Rules:
  • Return the label ONLY, no prose, no punctuation, no quotes.
  • If the chunk discusses multiple types, pick the DOMINANT one.
  • 'empirical_correlation' ≠ 'experimental': the correlation itself is not data,
    but a chunk that lists measured Re_θ,t values IS experimental.

CHUNK:
\"\"\"
{text}
\"\"\"

LABEL:"""


def classify_evidence_type(text: str, llm_call) -> str:
    """Classify a chunk's evidence type via a small LLM call.

    `llm_call` is the router's callable (we pass it in instead of
    importing at module top so tests can inject a fake).

    Returns one of _EVIDENCE_TYPES. On any failure returns 'other' —
    the validator treats 'other' as unreliable and down-weights it.
    """
    if not text or len(text) < 30:
        return "other"
    try:
        raw, _usage = llm_call(
            task="chunk_classification",
            messages=[{"role": "user",
                        "content": _EVIDENCE_PROMPT.format(text=text[:1500])}],
            system="You are a strict classifier. Return one label only.",
            max_tokens=20,
            temperature=0.0,
        )
    except Exception:
        return "other"

    # Pull the first recognised label word from the response.
    lowered = (raw or "").strip().lower()
    for label in _EVIDENCE_TYPES:
        if label.lower() in lowered:
            return label
    return "other"


# ══════════════════════════════════════════════════════════════════
# Public entry point — used by ingestion AND backfill
# ══════════════════════════════════════════════════════════════════

def enrich_chunk_metadata(
    text: str,
    existing_metadata: dict[str, Any],
    llm_call,
    skip_if_present: bool = True,
) -> dict[str, Any]:
    """Add `evidence_type` and `contains_numeric_data` to chunk metadata.

    Parameters
    ----------
    text : str
        The chunk's text content.
    existing_metadata : dict
        Whatever metadata the chunk already has. Returned as a NEW dict
        with the enrichment fields added.
    llm_call : callable
        Router callable. Injected so tests can use a fake.
    skip_if_present : bool
        If True (default), don't re-classify chunks that already have
        the enrichment fields — important for the incremental backfill
        so we don't re-pay API cost on already-enriched chunks.

    Returns
    -------
    dict
        A shallow copy of existing_metadata with added fields.
    """
    out = dict(existing_metadata or {})

    if skip_if_present and "evidence_type" in out and "contains_numeric_data" in out:
        return out

    out["contains_numeric_data"] = bool(contains_numeric_data(text))
    # Only call the LLM if we don't already have an evidence_type.
    if "evidence_type" not in out:
        out["evidence_type"] = classify_evidence_type(text, llm_call)
    return out
