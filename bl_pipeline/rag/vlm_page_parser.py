"""vlm_page_parser.py — Vision-language-model-based PDF page parser.

Problem
───────
The default `PyMuPDF.get_text("text")` path loses critical structure:

  - Math formulas extracted with pieces floating (exponents, subscripts
    detached from their base symbols).
  - Greek letters mis-OCR'd to digits (δ → 6, θ → 0).
  - Figure captions separated from their figures in the output stream.
  - Tables flattened to ungrouped text with row/column layout gone.

NVIDIA's NeMo Retriever solves this with Nemotron-Parse-v1.1, a
purpose-built VLM. We don't have that on hand, but Claude Sonnet's
vision capability is a close functional equivalent — it reads the page
as an image and returns structured markdown with LaTeX math.

This module is the local analogue. For each page of a PDF:
  1. Render the page to PNG at a resolution VLMs handle cleanly.
  2. Send the image to Sonnet with a strict extraction prompt.
  3. Get back structured markdown with inline $LaTeX$ for math,
     explicit section headings, tables, and figure captions that stay
     paired with their figures.

Output is a list of per-page markdown strings. The ingestion pipeline
chunks this markdown with an equation-aware chunker so formulas stay
intact.

Cost profile
────────────
  Sonnet vision roughly: ~$0.015 per page (at 1M-pixel renders).
  For the three papers under immediate test (~90 pages total) this
  costs ~$1.35 one-time.  Full corpus (~1000 pages) would be ~$15.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from bl_pipeline.shared.llm_router import llm


# Prompt kept deliberately firm and specific. VLMs drift into prose
# summaries otherwise.
_PARSE_SYSTEM = (
    "You are a scientific-document extraction engine. Given an image of "
    "one PDF page, return a STRUCTURED MARKDOWN REPRESENTATION of the "
    "page content. You are not summarising — you are transcribing with "
    "structure preserved."
)

_PARSE_PROMPT = (
    "Transcribe this PDF page as structured markdown. Follow these "
    "rules STRICTLY:\n\n"
    "1. TEXT: preserve paragraphs and reading order. Do NOT rewrite "
    "   or summarise the prose.\n\n"
    "2. MATH: every equation goes in LaTeX. Inline math uses $…$, "
    "   display math uses $$…$$. NEVER let exponents, subscripts, or "
    "   Greek letters float away from their base symbol. If you see "
    "   ‘Re_x = 62 · Re_θ^(5/4)’ typeset, output exactly "
    "   `$Re_x = 62 \\cdot Re_\\theta^{5/4}$` with the 5/4 as the "
    "   superscript and θ as \\theta.\n\n"
    "3. EQUATION NUMBERING: if the paper assigns equation numbers "
    "   (e.g. '(7)'), keep them.\n\n"
    "4. TABLES — follow every sub-rule:\n"
    "   a. Put the table's caption/title on the line IMMEDIATELY "
    "      above the table, formatted as `**Table N.** caption text`. "
    "      If the paper labels it differently (e.g. 'Table 2'), mirror "
    "      that label.\n"
    "   b. Render as a proper markdown table with pipes and a header "
    "      separator row. Multi-column / multi-row headers collapse "
    "      into one header row with units in parentheses: "
    "      `| Tu (%) | λ_θ | Re_θt |`.\n"
    "   c. Preserve column UNITS in the header — `Tu (%)`, not `Tu`; "
    "      `x_t (m)`, not `x_t`.\n"
    "   d. Preserve numerical precision as shown in the paper. Do not "
    "      round. If a cell is blank, use an em-dash `—`.\n"
    "   e. If a cell contains an equation, keep it as inline LaTeX.\n"
    "   f. Footnote markers (*, †, a, b, …) stay in the cell. Write "
    "      each footnote on its own line immediately BELOW the table, "
    "      prefixed with the same marker: `* Re based on chord length`.\n\n"
    "5. FIGURES — follow every sub-rule:\n"
    "   a. Start the block with `### Figure N` (matching the paper's "
    "      numbering).\n"
    "   b. Line 1 of the block: the figure's caption verbatim, in bold.\n"
    "   c. Then a bulleted METADATA list:\n"
    "      - `Y-axis:` label, units, and approximate range shown\n"
    "      - `X-axis:` label, units, and approximate range shown\n"
    "      - `Legend entries:` comma-separated list of datasets / "
    "        authors represented on the plot\n"
    "      - `Annotations within figure:` any TEXT that appears on "
    "        the plot itself (arrow labels, curve-fit equations, "
    "        regime boundaries). For curve-fit equations render in "
    "        LaTeX AND note they are 'fitted line label', NOT free-"
    "        standing paper equations.\n"
    "   d. Then, if the figure is a DATA PLOT (scatter / line chart / "
    "      bar chart — NOT a schematic), add a `**Data observations:**` "
    "      subsection with 3-5 short bullets covering:\n"
    "      - Overall trend shape: monotonic up/down, peak, plateau, "
    "        bimodal, etc.\n"
    "      - Approximate extent of the data cloud on each axis "
    "        (e.g. 'Tu spans 0.1%–10%, Re_θt spans 100–800').\n"
    "      - Relative ordering of the datasets if multiple are shown "
    "        (e.g. 'Blair's data sits above Gostelow's at fixed Tu').\n"
    "      - Presence of error bars, scatter width, or fit-line "
    "        residuals if visible.\n"
    "      - Any regime boundary or transition point the figure "
    "        highlights visually.\n"
    "   e. DO NOT attempt to read exact (x, y) values from scatter "
    "      markers — that is a digitisation task outside this extraction.\n"
    "   f. If the figure is a SCHEMATIC (experimental setup, flow "
    "      diagram, mechanism sketch) rather than a data plot, skip "
    "      the `Data observations:` block and instead add a "
    "      `**Schematic description:**` subsection with a 2-3 "
    "      sentence description of what the schematic shows.\n\n"
    "6. SECTION HEADINGS: `##` for main sections, `###` for "
    "   sub-sections and figure blocks.\n\n"
    "7. NEVER invent content. If something is illegible, mark it "
    "   `[UNREADABLE]` rather than guessing. If a figure is too "
    "   small / low-resolution to read its legend, write "
    "   `[LEGEND ILLEGIBLE]` in place of the legend entries.\n\n"
    "8. DO NOT wrap your output in a fence. Return raw markdown "
    "   starting with the first readable element on the page.\n\n"
    "Begin transcription now."
)


def render_page_to_png(
    pdf_path: str | Path,
    page_num: int,
    dpi: int = 180,
) -> bytes:
    """Render a single PDF page to PNG bytes.

    page_num is 1-indexed to match the ingestion pipeline's conventions.
    180 dpi gives a ~1600×2100 image for a US-letter page — within the
    sweet spot for Sonnet vision (under the 1.15 M-pixel soft cap) and
    with enough resolution to read subscripts and Greek letters clearly.
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_num - 1]
        # fitz.Matrix scales from 72 dpi base to requested dpi
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def parse_page(
    pdf_path: str | Path,
    page_num: int,
    dpi: int = 180,
    max_tokens: int = 8000,
) -> dict[str, Any]:
    """Extract one page via VLM.

    Returns:
        {"page": int, "markdown": str, "success": bool, "error": str | None}

    The caller decides what to do with failed pages (usually: fall back
    to the legacy text-only extraction just for that page).
    """
    try:
        png = render_page_to_png(pdf_path, page_num, dpi=dpi)
    except Exception as e:
        return {
            "page": page_num,
            "markdown": "",
            "success": False,
            "error": f"page render failed: {e}",
        }

    b64 = base64.standard_b64encode(png).decode("ascii")

    # Anthropic's vision input format — image block + text block
    user_content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "media_type",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {"type": "text", "text": _PARSE_PROMPT},
    ]
    # The anthropic SDK expects "source" with "type": "base64" for b64 payloads
    # (not "media_type"). Fix the shape here.
    user_content[0]["source"]["type"] = "base64"

    try:
        markdown, _usage = llm.call(
            task="vlm_ocr",   # Sonnet — dedicated OCR slot
            messages=[{"role": "user", "content": user_content}],
            system=_PARSE_SYSTEM,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        return {
            "page": page_num,
            "markdown": "",
            "success": False,
            "error": f"VLM call failed: {e}",
        }

    return {
        "page": page_num,
        "markdown": (markdown or "").strip(),
        "success": True,
        "error": None,
    }


def parse_page_with_retry(
    pdf_path: str | Path,
    page_num: int,
    dpi: int = 180,
    max_tokens: int = 8000,
    retries: int = 2,
) -> dict[str, Any]:
    """parse_page with up to `retries` additional attempts on failure.

    Single-page failures are usually transient (rate-limit, timeout,
    one-off malformed response). Retrying sequentially gives us much
    better coverage without amplifying API pressure.
    """
    import time as _t
    last: dict[str, Any] = {}
    for attempt in range(retries + 1):
        last = parse_page(pdf_path, page_num, dpi=dpi, max_tokens=max_tokens)
        if last["success"] and last["markdown"]:
            if attempt > 0:
                print(f"    [retry ok] p{page_num} on attempt {attempt + 1}")
            return last
        if attempt < retries:
            _t.sleep(2.0)   # small backoff before retry
    return last


def parse_paper(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    dpi: int = 180,
    retries: int = 2,
) -> list[dict[str, Any]]:
    """Extract every page (or a subset) of a PDF SEQUENTIALLY.

    Sequential, not parallel, by deliberate choice: the user's priority
    is quality over speed. One page at a time means:
      - No rate-limit contention between concurrent calls
      - Each failed page gets its full retry budget without competing
        for API capacity
      - Deterministic, reproducible progress output
      - Slower — ~10-20 sec per page — but measurably cleaner extraction

    Each failed page is retried up to `retries` times before we accept
    the failure.
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    if pages is None:
        pages = list(range(1, total_pages + 1))
    pages = [p for p in pages if 1 <= p <= total_pages]

    results: list[dict[str, Any]] = []
    for p in pages:
        results.append(
            parse_page_with_retry(pdf_path, p, dpi=dpi, retries=retries)
        )
    return results
