"""chunk_metadata.py — structure-derived metadata for VLM chunks.

After the VLM page parser emits structured markdown and the markdown
chunker splits it into chunks, we run this module to derive extra
metadata fields per chunk: content type, section/subsection heading,
presence of equations/tables, figure references.

Zero LLM calls. Pure regex on the markdown itself. This metadata lives
alongside `paper_id` and `page_number` in ChromaDB and enables:

  - Targeted retrieval: "fetch only FIGURE chunks from Mayle" or
    "fetch only TABLE chunks containing calibration constants".
  - Better prompts downstream: Node 4 AlgorithmExtractor can weight
    equation-rich chunks higher when looking for governing equations.
  - Richer manuscript citations: Node 10 can report
    "retrieved 4 prose chunks + Figure 13 + Table 2 from Mayle 1991"
    instead of a flat chunk count.

Why this is free: Sonnet already structured the page as markdown with
explicit `## / ### / **Table N.** / ### Figure N / $$…$$` markers.
We just read those markers.
"""

from __future__ import annotations

import re
from typing import Any


# ── Compiled patterns (module-level for speed) ──────────────────────

# A ## or ### heading. Captures the title text after the # marks.
_H2_RX = re.compile(r"^##\s+(?!#)(.+)$", re.MULTILINE)
_H3_RX = re.compile(r"^###\s+(.+)$",   re.MULTILINE)

# A "### Figure N" block marker (N can be multi-digit, may have letters).
_FIGURE_HEADING_RX = re.compile(r"###\s+Figure\s+([0-9A-Za-z]+)", re.IGNORECASE)

# A **Table N.** caption block (our VLM prompt emits this format).
_TABLE_CAPTION_RX = re.compile(r"\*\*Table\s+([0-9A-Za-z]+)\.?\*\*", re.IGNORECASE)

# A markdown table separator row: `|---|---|` or with alignment `|:---:|`.
_TABLE_SEPARATOR_RX = re.compile(r"^\s*\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
                                  re.MULTILINE)

# Inline math $…$  (lookbehind negates double-dollar)
_INLINE_MATH_RX = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+?(?<!\$)\$(?!\$)")

# Display math $$…$$ (may span multiple lines)
_DISPLAY_MATH_RX = re.compile(r"\$\$.+?\$\$", re.DOTALL)

# In-text references like "Fig. 13", "Figure 7", "Table 2"
_FIG_REF_RX   = re.compile(r"(?:Fig(?:ure|\.)?\s*)([0-9]{1,3})", re.IGNORECASE)
_TABLE_REF_RX = re.compile(r"Table\s+([0-9]{1,3})", re.IGNORECASE)

# "Curve-fit labels within figure" / "Annotations within figure" marker.
# Our VLM prompt uses these — when present, the chunk contains formula
# ANNOTATIONS from a plot, not free-standing paper equations.
_FIG_ANNOTATION_RX = re.compile(
    r"(?:curve[- ]fit\s+label|annotation)s?\s+(?:within\s+figure|in\s+figure)",
    re.IGNORECASE,
)


# ── Public API ───────────────────────────────────────────────────────


def derive_chunk_metadata(
    chunk_text: str,
    page_markdown: str | None = None,
) -> dict[str, Any]:
    """Return a metadata dict derived from the markdown structure of a
    chunk. All fields are safe to use as ChromaDB metadata values
    (scalar types only).

    Args:
        chunk_text:
            The chunk as it will be indexed. This is what we mostly read.
        page_markdown:
            The full page's markdown, optional. If supplied, we use it
            to resolve the chunk's enclosing section / subsection even
            when the chunk itself starts mid-section (no local heading).

    Returns:
        dict with keys:
            content_type:      "figure" | "table" | "equation_block" | "prose"
            section:           nearest ## heading text (str | empty)
            subsection:        nearest ### heading text (str | empty)
            has_equations:     bool
            has_tables:        bool
            n_display_math:    int     # count of $$…$$ blocks
            n_inline_math:     int     # count of inline $…$ occurrences
            figure_refs:       str     # comma-joined figure numbers referenced
            table_refs:        str     # comma-joined table numbers referenced
            within_figure_annotation: bool
                True if the chunk contains formula annotations that the
                VLM explicitly marked as "within figure" (curve-fit
                labels). Downstream this tells Node 4 NOT to treat
                those equations as free-standing paper correlations.
    """
    text = chunk_text or ""

    # Section / subsection — first try the chunk itself, then fall back
    # to the page-level markdown if supplied.
    section = _last_heading_match(_H2_RX, text)
    if not section and page_markdown:
        section = _section_containing_chunk(page_markdown, text, level=2)

    subsection = _last_heading_match(_H3_RX, text)
    if not subsection and page_markdown:
        subsection = _section_containing_chunk(page_markdown, text, level=3)

    # Classify content type — check most-specific markers first.
    has_display_math = bool(_DISPLAY_MATH_RX.search(text))
    has_inline_math = bool(_INLINE_MATH_RX.search(text))
    has_equations = has_display_math or has_inline_math
    has_tables = bool(_TABLE_SEPARATOR_RX.search(text)) or bool(_TABLE_CAPTION_RX.search(text))

    if _FIGURE_HEADING_RX.search(text):
        content_type = "figure"
    elif has_tables:
        content_type = "table"
    elif has_display_math and not _is_mostly_prose(text):
        content_type = "equation_block"
    else:
        content_type = "prose"

    # Reference lists — sorted, deduped, joined into a comma-string
    # (ChromaDB metadata doesn't accept lists — strings only).
    fig_refs   = sorted({m.group(1) for m in _FIG_REF_RX.finditer(text)} |
                        {m.group(1) for m in _FIGURE_HEADING_RX.finditer(text)})
    table_refs = sorted({m.group(1) for m in _TABLE_REF_RX.finditer(text)} |
                        {m.group(1) for m in _TABLE_CAPTION_RX.finditer(text)})

    return {
        "content_type":             content_type,
        "section":                  section or "",
        "subsection":               subsection or "",
        "has_equations":            has_equations,
        "has_tables":               has_tables,
        "n_display_math":           len(_DISPLAY_MATH_RX.findall(text)),
        "n_inline_math":            len(_INLINE_MATH_RX.findall(text)),
        "figure_refs":              ",".join(fig_refs),
        "table_refs":               ",".join(table_refs),
        "within_figure_annotation": bool(_FIG_ANNOTATION_RX.search(text)),
    }


# ── Internals ────────────────────────────────────────────────────────


def _last_heading_match(rx: re.Pattern[str], text: str) -> str:
    """Return the last heading title matched by rx, or empty string."""
    matches = rx.findall(text)
    return matches[-1].strip() if matches else ""


def _section_containing_chunk(
    page_md: str,
    chunk_text: str,
    level: int,
) -> str:
    """Find the nearest `level`-heading (## or ###) that comes BEFORE
    where the chunk starts in the page markdown.

    Fallback used when the chunk itself was split mid-section and
    carries no local heading.
    """
    # Try to locate the chunk in the page. Use a fingerprint (first
    # 80 non-space chars) to sidestep whitespace differences introduced
    # by the chunker.
    rx = _H2_RX if level == 2 else _H3_RX

    fingerprint = re.sub(r"\s+", "", chunk_text)[:80]
    if not fingerprint:
        return ""
    page_collapsed = re.sub(r"\s+", "", page_md)
    pos_collapsed = page_collapsed.find(fingerprint)
    if pos_collapsed < 0:
        # Couldn't locate — return the page's last heading of this level.
        return _last_heading_match(rx, page_md)

    # Map collapsed-index back to an approximate original-index.
    # Since we collapsed whitespace, every char in the collapsed form
    # came from at least that many chars in the original. We walk the
    # original string counting non-whitespace chars to find the
    # original position.
    target = pos_collapsed
    count = 0
    orig_pos = 0
    for i, ch in enumerate(page_md):
        if not ch.isspace():
            if count == target:
                orig_pos = i
                break
            count += 1
    else:
        orig_pos = len(page_md)

    # Last heading of this level at or before orig_pos
    last_match = ""
    for m in rx.finditer(page_md):
        if m.start() <= orig_pos:
            last_match = m.group(1).strip()
        else:
            break
    return last_match


def _is_mostly_prose(text: str) -> bool:
    """True if the chunk looks like prose with some embedded math,
    rather than an equation block with surrounding sentence fragments.

    Heuristic: if the non-math text is more than 3x the length of the
    math content, it's prose. Used to distinguish a paragraph that
    contains inline math from a chunk that IS primarily equations.
    """
    math_chars = sum(len(m) for m in _DISPLAY_MATH_RX.findall(text))
    math_chars += sum(len(m) for m in _INLINE_MATH_RX.findall(text))
    non_math = max(0, len(text) - math_chars)
    return non_math > 3 * max(math_chars, 1)
