"""query_expander.py — Sonnet-driven query expansion with metadata steering.

Problem this module solves
──────────────────────────
Raw vector similarity retrieves semantically-close chunks, but "close"
isn't always "right." When a user asks "what's Mayle's Re_theta,t
correlation?" the equation chunk

    $$Re_{theta,t} = 400 * Tu^{-5/8}$$   (Mayle Eq. 9)

loses on cosine similarity to a VERBOSE figure-caption chunk

    "### Figure 11 — Comparison of present correlation with
    Abu-Ghannam and Shaw's... Y-axis: Re_theta... X-axis: Tu%..."

both describe the same idea, but only one is the formula. Plain
similarity can't tell them apart.

Solution
────────
Before hitting ChromaDB, a Sonnet "expander" reads the user query and
emits one or more SUB-QUERIES, each annotated with:
  - which collection(s) to search
  - metadata weights that tell the retriever which SHAPE of chunk
    to prefer (e.g. equation_block +3, figure -1, within_figure
    _annotation -2)

The retriever then:
  1. Runs each sub-query as a pure vector search (top-50, no filter)
  2. Re-ranks each candidate by `distance - 0.1 * sum(applicable_boosts)`
  3. Keeps the top-K per sub-query
  4. Merges + dedupes across sub-queries

The metadata schema is a CLOSED vocabulary — only the keys this module
knows about are legal. Sonnet is given the schema explicitly in the
system prompt and warned not to invent new keys.

Design constraints
──────────────────
- No `paper_id` or `page_number` in metadata_weights — semantics
  should pick the paper, not a hard-coded guess from the expander.
- Negative weights are allowed and important (e.g. -2 on
  within_figure_annotation to exclude plot labels from equation
  retrieval — a real bug we hit earlier where Mayle Fig 13's
  end-of-transition annotation was misapplied as the onset formula).
- Auto-generated schema: the collection summaries and metadata key
  list are built from `collections.py` and the known chunk_metadata
  fields, so the prompt stays in sync automatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from bl_pipeline.rag.collections import ALL_COLLECTIONS, COLLECTION_MAP
from bl_pipeline.shared.json_utils import parse_lenient_json
from bl_pipeline.shared.llm_router import llm


# ═══════════════════════════════════════════════════════════════════
# Metadata schema — the CLOSED vocabulary Sonnet may use
# ═══════════════════════════════════════════════════════════════════
#
# Each entry is (key, description, typical_use). This is the single
# source of truth both for the Sonnet prompt AND for the retriever's
# validation step. If you add a new metadata field to
# `chunk_metadata.py`, add it here too (and update the retriever
# validator).


ALLOWED_WEIGHT_KEYS: dict[str, str] = {
    "content_type::equation_block":
        "Chunks dominated by display math ($$...$$). NOTE: With the "
        "orphan-merge chunker, formulas usually live inside PROSE chunks "
        "that have has_equations=True — this tag is becoming rarer. For "
        "formula retrieval, prefer has_equations::True +2 with "
        "content_type::prose +1 instead of relying on equation_block alone.",
    "content_type::table":
        "Chunks containing a markdown table. Boost when the user wants "
        "tabulated data (benchmark Tu/Re values, calibration constants, "
        "test matrix entries).",
    "content_type::figure":
        "Figure blocks with captions + axis descriptions + data "
        "observations. Boost when the user asks about a specific figure "
        "or trend. PENALIZE (-1 or less) when the user wants a formula, "
        "since figure chunks describe where formulas appear, not the "
        "formulas themselves.",
    "content_type::prose":
        "Regular prose paragraphs. Boost when the user wants explanation, "
        "derivation, physical reasoning, or methodology.",
    "has_equations::True":
        "Chunk contains at least one math expression ($...$ or $$...$$). "
        "Useful soft-boost for any formula-adjacent query.",
    "has_tables::True":
        "Chunk contains a markdown table OR a '**Table N.**' caption. "
        "Soft-boost for tabulated data queries.",
    "within_figure_annotation::True":
        "Chunk contains formulas that the VLM marked as 'curve-fit "
        "label' inside a figure (not free-standing paper equations). "
        "USE NEGATIVE WEIGHTS when the user wants paper correlations. "
        "This prevents treating plot annotations as governing formulas "
        "(a real bug we hit with Mayle Fig 13's Re_x = 62*Re_theta^(5/4) "
        "annotation being misapplied as the onset correlation).",
}


# ═══════════════════════════════════════════════════════════════════
# Data classes — the expander's output shape
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ExpandedQuery:
    """One sub-query produced by the expander.

    text
        Keyword-dense string to hand to the vector retriever. 8-15
        scientific terms; not a natural-language sentence.
    target_cols
        Which 1-3 ChromaDB collections this sub-query should search.
    metadata_weights
        Dict from "field::value" key to signed float weight. Only keys
        listed in ALLOWED_WEIGHT_KEYS are legal. Typical magnitudes
        are in [-3.0, +3.0]; the retriever clamps to that range.
    rationale
        One-sentence explanation of what this sub-query is hunting
        for. Kept so the decision ledger can explain "we pulled
        Mayle Eq. 9 because the expander was looking for an equation
        block in primary_correlations."
    """

    text: str
    target_cols: list[str]
    metadata_weights: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# System prompt — auto-built from schema so it never drifts
# ═══════════════════════════════════════════════════════════════════


def _build_collection_block() -> str:
    """Auto-generate the collection-summary block from collections.py."""
    lines = []
    for c in ALL_COLLECTIONS:
        lines.append(f"  {c.name:<28} — {c.description}")
    return "\n".join(lines)


def _build_metadata_schema_block() -> str:
    """Auto-generate the allowed-metadata-keys block."""
    lines = []
    for key, desc in ALLOWED_WEIGHT_KEYS.items():
        # Wrap the description to ~100 chars for readable prompt
        lines.append(f"  {key}")
        for seg in _wrap(desc, width=96, indent="      "):
            lines.append(seg)
    return "\n".join(lines)


def _wrap(text: str, width: int = 96, indent: str = "") -> list[str]:
    """Simple word-wrap for long descriptions in the prompt."""
    out: list[str] = []
    words = text.split()
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            out.append(indent + cur)
            cur = w
    if cur:
        out.append(indent + cur)
    return out


_FEW_SHOT_EXAMPLES = """\
FEW-SHOT EXAMPLES:

────────────────────────────────────────────────────────────────────
User: "find the algebraic correlation for transition onset Reynolds number as a function of free-stream turbulence on a flat plate"
Output (illustrative shape; substitute author/formula names from the
ACTUAL retrieved corpus rather than copying labels from this example):
{
  "expanded_queries": [
    {
      "text": "<descriptive prose sub-query naming the quantity sought, e.g. transition onset Reynolds number Tu flat plate correlation>",
      "target_cols": ["primary_correlations"],
      "metadata_weights": {
        "has_equations::True":             2.5,
        "content_type::prose":             1.0,
        "content_type::equation_block":    2.0,
        "content_type::figure":           -1.0,
        "within_figure_annotation::True": -2.5
      },
      "rationale": "descriptive query for the formula in prose+equation context; figure-annotation chunks down-weighted to avoid catching plot labels as the governing formula"
    },
    {
      "text": "<math-token sub-query — literal numbers, variable names, equation labels, e.g. '<coefficient> <variable> <exponent> Eq <number> <quantity name> <author>'>",
      "target_cols": ["primary_correlations"],
      "metadata_weights": {
        "has_equations::True":             2.5,
        "content_type::equation_block":    2.0,
        "within_figure_annotation::True": -2.5
      },
      "rationale": "math-token sub-query targeting BM25 keyword match on the literal formula; useful when the descriptive sub-query buries the canonical equation chunk under more verbose introductory prose"
    }
  ]
}

The math-token sub-query is the highest-leverage pattern in this
domain — when a user query implies a specific equation, emit one
sub-query whose text is the LITERAL tokens of that equation
(coefficient, variable names, exponent, equation number). Do not copy
the placeholder above; derive the actual tokens from the user query
and from common conventions for the regime.

────────────────────────────────────────────────────────────────────
User: "ERCOFTAC T3A experiment inlet conditions"
Output:
{
  "expanded_queries": [
    {
      "text": "ERCOFTAC T3 series flat plate inlet turbulence intensity Reynolds number tables",
      "target_cols": ["primary_benchmarks"],
      "metadata_weights": {
        "has_tables::True":       3.0,
        "content_type::table":    3.0,
        "content_type::prose":    0.5
      },
      "rationale": "tabulated benchmark inlet conditions live in table chunks; prose down-weighted"
    }
  ]
}

────────────────────────────────────────────────────────────────────
User: "qualitative mechanism of bypass transition at high free-stream turbulence"
Output:
{
  "expanded_queries": [
    {
      "text": "bypass transition mechanism streak shear sheltering free-stream turbulence boundary layer",
      "target_cols": ["secondary_bypass", "secondary_stability_physics"],
      "metadata_weights": {
        "content_type::prose":  1.0,
        "has_equations::True":  0.3
      },
      "rationale": "qualitative physics explanation in prose; equations down-weighted"
    },
    {
      "text": "DNS bypass transition streak breakdown secondary instability",
      "target_cols": ["secondary_bypass"],
      "metadata_weights": {
        "content_type::prose":  0.8,
        "content_type::figure": 0.5
      },
      "rationale": "DNS literature — figure chunks often carry the headline findings"
    }
  ]
}
"""


def build_system_prompt() -> str:
    """Assemble the full system prompt. Called once at module load."""
    return (
        "You are a query-expansion engine for a scientific RAG pipeline on "
        "boundary-layer transition research. Given one user query, produce "
        "1-5 sub-queries that together cover the information needed to "
        "answer the user. For each sub-query, declare which collection(s) "
        "to search AND which metadata weights to apply.\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "AVAILABLE COLLECTIONS (pick 1-3 per sub-query):\n"
        "═══════════════════════════════════════════════════════════════\n"
        f"{_build_collection_block()}\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "ALLOWED metadata_weights KEYS (format: \"field::value\"):\n"
        "Only these keys are legal. Using any other key is an ERROR.\n"
        "═══════════════════════════════════════════════════════════════\n"
        f"{_build_metadata_schema_block()}\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "RULES:\n"
        "═══════════════════════════════════════════════════════════════\n"
        "1. 1-5 sub-queries total. Favor multiple focused sub-queries over "
        "   one giant catch-all.\n"
        "2. Each sub_query text is 8-15 scientific keywords, not a "
        "   natural-language sentence. Keep author-year if useful.\n"
        "3. Pick 1-3 target_cols per sub-query. Do not spray into all 11.\n"
        "4. Metadata weights are floats in [-3.0, +3.0]. USE NEGATIVE "
        "   WEIGHTS to exclude wrong-shape chunks (especially "
        "   within_figure_annotation when the user wants a real equation, "
        "   and content_type::figure when the user wants a formula).\n"
        "5. DO NOT emit paper_id or page_number weights. Semantic retrieval "
        "   picks the paper; your job is to shape WHAT KIND of chunk is "
        "   preferred, not WHICH paper.\n"
        "6. DO NOT invent metadata keys — only the ones listed above.\n"
        "7. MATH-TOKEN SUB-QUERY RULE: If the user query demands a solution "
        "that will need a specific correlation, formula, or equation, "
        "include ONE or more additional sub-queries made of literal math "
        "tokens (numbers, variable names, equation numbers) that you'd "
        "expect to appear in that formula's chunk. This sub-query has NO "
        "English grammar — just whitespace-separated tokens like:\n"
        "   \"400 Tu 5/8 -5/8 Eq 9 Re_theta_t Mayle\"\n"
        "   \"62 Re_x 5/4 Re_theta Mayle transition length\"\n"
        "   \"F lambda_theta 6.91 12.75 63.64 AGS Eq 12\"\n"
        "The naked-formula chunk is short and mostly LaTeX, so its "
        "English embedding is weak — but BM25 keyword search will light "
        "up on the literal numbers. This sub-query rescues it from the "
        "long tail. Use content_type::equation_block +3 and "
        "within_figure_annotation::True -2.5 on these math-token queries "
        "so plot-label annotations (NOT free-standing paper equations) "
        "are excluded.\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        "═══════════════════════════════════════════════════════════════\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY this JSON (no markdown fences, no prose preamble):\n"
        "═══════════════════════════════════════════════════════════════\n"
        '{\n'
        '  "expanded_queries": [\n'
        '    {\n'
        '      "text":             "<keyword-dense search string>",\n'
        '      "target_cols":      ["<col>", ...],\n'
        '      "metadata_weights": {"<field::value>": <float>, ...},\n'
        '      "rationale":        "<one sentence>"\n'
        '    },\n'
        '    ... (1-5 entries)\n'
        '  ]\n'
        '}\n'
    )


# Build the system prompt once — schema changes require a module reload.
_SYSTEM_PROMPT = build_system_prompt()


# ═══════════════════════════════════════════════════════════════════
# Validation — defends against Sonnet hallucinations
# ═══════════════════════════════════════════════════════════════════


_VALID_COLLECTIONS = set(COLLECTION_MAP.keys())
_VALID_METADATA_KEYS = set(ALLOWED_WEIGHT_KEYS.keys())


def _validate_and_clean(raw: dict[str, Any]) -> list[ExpandedQuery]:
    """Coerce raw Sonnet JSON into ExpandedQuery objects.

    Silently drops:
      - entries missing required fields
      - unknown target_cols
      - unknown metadata_weights keys
      - metadata weights outside [-3.0, 3.0] (clamped, not dropped)

    Warnings go to stdout so the caller can see what was filtered.
    """
    out: list[ExpandedQuery] = []

    items = raw.get("expanded_queries") or []
    if not isinstance(items, list):
        print(f"[QueryExpander] expanded_queries is not a list: {type(items)}")
        return out

    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            print(f"[QueryExpander] entry {i} is not a dict, skipping")
            continue

        text = str(entry.get("text") or "").strip()
        if not text:
            print(f"[QueryExpander] entry {i} has empty text, skipping")
            continue

        raw_cols = entry.get("target_cols") or []
        if not isinstance(raw_cols, list):
            raw_cols = [raw_cols]
        target_cols = [c for c in raw_cols if c in _VALID_COLLECTIONS]
        dropped_cols = set(raw_cols) - set(target_cols)
        if dropped_cols:
            print(f"[QueryExpander] entry {i}: dropped unknown collections {dropped_cols}")
        if not target_cols:
            print(f"[QueryExpander] entry {i} has no valid target_cols, skipping")
            continue

        raw_weights = entry.get("metadata_weights") or {}
        if not isinstance(raw_weights, dict):
            raw_weights = {}
        clean_weights: dict[str, float] = {}
        for key, val in raw_weights.items():
            if key not in _VALID_METADATA_KEYS:
                print(f"[QueryExpander] entry {i}: dropped unknown weight key {key!r}")
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                print(f"[QueryExpander] entry {i}: non-numeric weight for {key}, skipping")
                continue
            # Clamp to [-3, 3] — anything larger is probably a hallucination
            fv = max(-3.0, min(3.0, fv))
            clean_weights[key] = fv

        rationale = str(entry.get("rationale") or "").strip()

        out.append(ExpandedQuery(
            text=text,
            target_cols=target_cols,
            metadata_weights=clean_weights,
            rationale=rationale,
        ))

    return out


# ═══════════════════════════════════════════════════════════════════
# Public API — expand_query()
# ═══════════════════════════════════════════════════════════════════


def expand_query(user_query: str, max_tokens: int = 2000) -> list[ExpandedQuery]:
    """Ask Sonnet to fan a single user query into planned sub-queries.

    Parameters
    ──────────
    user_query
        The original user input. Free-form natural language is fine.
    max_tokens
        Output cap for Sonnet. 2000 is plenty for 5 sub-queries; raise
        if you enlarge few-shot examples or add more rationale fields.

    Returns
    ───────
    A list of ExpandedQuery. Empty list on total failure — callers
    should treat that as "no expansion, fall back to raw user_query".
    """
    if not user_query or not user_query.strip():
        return []

    try:
        raw_text, _usage = llm.call(
            task="query_parsing",  # → Sonnet per llm_router config
            messages=[{"role": "user", "content": user_query.strip()}],
            system=_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[QueryExpander] Sonnet call failed: {e}")
        return []

    parsed = parse_lenient_json(raw_text or "")
    if not isinstance(parsed, dict):
        print(f"[QueryExpander] JSON parse failed. Raw output (first 500 chars):\n"
              f"  {(raw_text or '')[:500]}")
        return []

    expanded = _validate_and_clean(parsed)

    # Stream each accepted sub-query to the live right-pane so the
    # operator can SEE exactly what Sonnet planned to search for —
    # the literal text it'll hand to BM25/vector, which collections,
    # which metadata weights, and the rationale for each.  Without
    # this, the right-pane shows only "Literature retrieval" with no
    # way to audit whether the expander asked the right questions.
    # Failures are swallowed so any test harness without an event bus
    # still runs.
    try:
        from bl_pipeline.shared.event_bus import emit as _emit
        _emit("query_expansion_done",
              user_query=user_query.strip(),
              count=len(expanded))
        for _i, _eq in enumerate(expanded, start=1):
            _emit("subquery_generated",
                  idx=_i,
                  text=_eq.text,
                  target_cols=list(_eq.target_cols),
                  metadata_weights=dict(_eq.metadata_weights),
                  rationale=_eq.rationale)
    except Exception:
        pass

    return expanded
