"""page_parser_cascade.py — layered PDF page parser with automatic fallback.

For every page, tries three extraction methods in order:

  1. **Claude Sonnet vision** (meta-skip prompt, copyright-safe)
     Highest-quality structured markdown. Our Phase 1 diagnostic
     proved the meta-skip prompt avoids most of Anthropic's content
     filter triggers, so we use it as the primary path.

  2. **OpenAI GPT-4o vision**
     Different content-filter infrastructure from Anthropic. In our
     empirical test, GPT-4o recovered 67/67 pages that Anthropic
     blocked — so it's the go-to fallback for scientific pages that
     Sonnet refuses. Slightly less consistent with our exact prompt
     formatting than Sonnet, but the scientific content is intact.

  3. **PyMuPDF raw text**
     Pure local, no LLM. Guaranteed to return something as long as
     the PDF has extractable text. Output is unstructured: math
     floats apart, tables collapse to whitespace-separated tokens.
     Used only when both vision LLMs fail (network outage, both
     APIs down, or page has no text content at all).

Each success path tags the returned dict with `extracted_via` so
the ingestion pipeline can propagate provenance into ChromaDB
metadata. Downstream retrieval can use this to weight chunks by
quality (e.g. prefer sonnet > gpt4o > pymupdf when scoring).
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from openai import OpenAI

from bl_pipeline.rag.vlm_page_parser import render_page_to_png
from bl_pipeline.shared.config import MODELS, OPENAI_API_KEY
from bl_pipeline.shared.llm_router import (
    UsageRecord,
    _compute_cost,
    _record_usage,
    llm,
)


# ── Prompts ─────────────────────────────────────────────────────────
# The SYSTEM prompt is shared across both VLM layers — keeps tone
# consistent regardless of which model we land on.

_VLM_SYSTEM = "You are a scientific-document extraction engine."


# The Sonnet prompt is deliberately kept to the short meta-skip
# variant that cleared every test page in our Phase 1 diagnostic.
# Longer prompts with extra TABLE/FIGURE rules re-triggered the
# content filter on some pages, so we stay lean here.
_SONNET_PROMPT = (
    "Transcribe the scientific content of this PDF page as structured markdown.\n\n"
    "SCOPE — transcribe ONLY:\n"
    "- Section headings and body paragraphs (math, tables, figure captions)\n"
    "- Equations as LaTeX\n\n"
    "EXPLICITLY SKIP and replace with [omitted]:\n"
    "- Author names, affiliations, institutional addresses\n"
    "- Correspondence, email, ORCID, DOI lines\n"
    "- Acknowledgments and funding statements\n"
    "- Journal masthead / logo / copyright notices\n"
    "- Page numbers and running headers/footers\n\n"
    "If the entire page is author info or acknowledgments, output just: "
    "`[metadata page]`.\n\n"
    "Return raw markdown, no code fences."
)


# GPT-4o has a more permissive content policy for scientific
# transcription, so we include the fuller table/figure instructions
# here that we had to drop from Sonnet.
_GPT4O_PROMPT = (
    "Transcribe the scientific content of this PDF page as structured "
    "markdown. Rules:\n\n"
    "1. TEXT: preserve paragraphs verbatim.\n"
    "2. MATH: inline $...$ and display $$...$$. Never split exponents "
    "   or subscripts from their base symbols.\n"
    "3. TABLES: full markdown tables, caption above as `**Table N.** "
    "   caption`. Preserve units in headers.\n"
    "4. FIGURES: `### Figure N`, bold caption, then bulleted metadata "
    "   (Y-axis / X-axis / Legend / Annotations within figure).\n"
    "5. SECTION HEADINGS: `##` main, `###` sub.\n"
    "6. Omit author names, affiliations, acknowledgments, and copyright "
    "   text (replace with [omitted] if they interrupt scientific "
    "   content).\n\n"
    "Return raw markdown, no code fences."
)


# ── Layer 1: Claude Sonnet ──────────────────────────────────────────


def _try_sonnet(
    pdf_path: str | Path,
    page_num: int,
    dpi: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    """Attempt Sonnet vision extraction with retry-on-transient. Returns
    {"success": bool, "markdown": str, "error": str|None}.
    """
    try:
        png = render_page_to_png(pdf_path, page_num, dpi=dpi)
    except Exception as e:
        return {"success": False, "markdown": "", "error": f"render failed: {e}"}

    b64 = base64.standard_b64encode(png).decode("ascii")
    img_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }
    user_content = [img_block, {"type": "text", "text": _SONNET_PROMPT}]

    last_err = ""
    for attempt in range(retries + 1):
        try:
            md, _usage = llm.call(
                task="vlm_ocr",   # dedicated slot → Sonnet (cheap OCR)
                messages=[{"role": "user", "content": user_content}],
                system=_VLM_SYSTEM,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = (md or "").strip()
            if text:
                return {"success": True, "markdown": text, "error": None}
            last_err = "empty output"
        except Exception as e:
            last_err = str(e)[:200]
            # Don't burn the whole retry budget on a definite content-filter
            # block — those are deterministic; fail fast and hand off to
            # the next layer.
            if "content filter" in last_err.lower() or "filtering policy" in last_err.lower():
                return {"success": False, "markdown": "", "error": f"sonnet: {last_err}"}
        if attempt < retries:
            time.sleep(2.0 * (2 ** attempt))

    return {"success": False, "markdown": "", "error": f"sonnet: {last_err}"}


# ── Layer 2: OpenAI GPT-4o ──────────────────────────────────────────


def _try_gpt4o(
    pdf_path: str | Path,
    page_num: int,
    dpi: int,
    max_tokens: int,
    retries: int,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Attempt GPT-4o vision extraction."""
    if client is None:
        if not OPENAI_API_KEY:
            return {"success": False, "markdown": "",
                    "error": "gpt4o: OPENAI_API_KEY missing"}
        client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        png = render_page_to_png(pdf_path, page_num, dpi=dpi)
    except Exception as e:
        return {"success": False, "markdown": "", "error": f"render failed: {e}"}

    b64 = base64.standard_b64encode(png).decode("ascii")
    img_url = f"data:image/png;base64,{b64}"

    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _VLM_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _GPT4O_PROMPT},
                            {"type": "image_url",
                             "image_url": {"url": img_url, "detail": "high"}},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            md = (resp.choices[0].message.content or "").strip()
            # Record cost in the global usage_tracker so the budget guard
            # in scripts/ingest_curated_12.py sees GPT-4o spend (raw
            # OpenAI client bypasses llm.call's auto-tracking).
            try:
                t_in  = resp.usage.prompt_tokens if resp.usage else 0
                t_out = resp.usage.completion_tokens if resp.usage else 0
                _record_usage(UsageRecord(
                    tokens_in=t_in, tokens_out=t_out,
                    cost_usd=_compute_cost(MODELS["gpt4o"], t_in, t_out),
                    model=MODELS["gpt4o"],
                ))
            except Exception:
                pass
            if md:
                return {"success": True, "markdown": md, "error": None}
            last_err = "empty output"
        except Exception as e:
            last_err = str(e)[:200]
        if attempt < retries:
            time.sleep(2.0 * (2 ** attempt))

    return {"success": False, "markdown": "", "error": f"gpt4o: {last_err}"}


# ── Layer 3: PyMuPDF raw text ───────────────────────────────────────


def _try_pymupdf(
    pdf_path: str | Path,
    page_num: int,
) -> dict[str, Any]:
    """Last-resort extraction. No LLM, no network.

    PyMuPDF's text mode preserves reading order reasonably but has
    known limitations: math equations break into floating tokens,
    tables collapse to whitespace-separated text, figure captions
    may drift. Better than nothing though — the raw words are there,
    and BM25 keyword retrieval can still find them even if vector
    similarity suffers.
    """
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[page_num - 1]
            text = page.get_text("text").strip()
        finally:
            doc.close()
    except Exception as e:
        return {"success": False, "markdown": "", "error": f"pymupdf: {e}"}

    if not text:
        return {"success": False, "markdown": "", "error": "pymupdf: empty text"}

    # Wrap in a minimal markdown header so downstream chunker still
    # gets something to segment on. Tag the page clearly so nobody
    # mistakes this for VLM-quality output downstream.
    md = f"## Page {page_num} (raw text extraction)\n\n{text}\n"
    return {"success": True, "markdown": md, "error": None}


# ── Public API ──────────────────────────────────────────────────────


def parse_page_cascade(
    pdf_path: str | Path,
    page_num: int,
    dpi: int = 180,
    max_tokens: int = 8000,
    sonnet_retries: int = 2,
    gpt4o_retries: int = 2,
) -> dict[str, Any]:
    """Try Sonnet → GPT-4o → PyMuPDF and return the first success.

    Returns:
        {
          "page":          int,
          "markdown":      str,
          "success":       bool,
          "error":         str | None,
          "extracted_via": "sonnet" | "gpt4o" | "pymupdf" | "none",
        }

    The `extracted_via` field is what ingestion should write to
    ChromaDB metadata so future retrievers can weight chunks by
    extraction quality.
    """
    # Layer 1: Sonnet
    r = _try_sonnet(pdf_path, page_num, dpi, max_tokens, sonnet_retries)
    if r["success"]:
        return {
            "page": page_num, "markdown": r["markdown"],
            "success": True, "error": None, "extracted_via": "sonnet",
        }
    sonnet_err = r.get("error") or ""

    # Layer 2: GPT-4o
    r = _try_gpt4o(pdf_path, page_num, dpi, max_tokens, gpt4o_retries)
    if r["success"]:
        return {
            "page": page_num, "markdown": r["markdown"],
            "success": True, "error": None, "extracted_via": "gpt4o",
        }
    gpt4o_err = r.get("error") or ""

    # Layer 3: PyMuPDF
    r = _try_pymupdf(pdf_path, page_num)
    if r["success"]:
        return {
            "page": page_num, "markdown": r["markdown"],
            "success": True, "error": None, "extracted_via": "pymupdf",
        }
    pymupdf_err = r.get("error") or ""

    # Everything failed — return a dead result that the caller can
    # skip. Chain the individual layer errors for debugging.
    return {
        "page": page_num,
        "markdown": "",
        "success": False,
        "error": f"all layers failed | sonnet:[{sonnet_err}] gpt4o:[{gpt4o_err}] pymupdf:[{pymupdf_err}]",
        "extracted_via": "none",
    }


def parse_paper_cascade(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    dpi: int = 180,
) -> list[dict[str, Any]]:
    """Sequentially parse every page of a PDF through the cascade.

    Mirrors the `parse_paper()` signature in the original VLM parser
    so this is a drop-in replacement for ingestion scripts. Sequential
    by design — quality over throughput, one clean API-pressure line
    per paper, deterministic progress output.
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    if pages is None:
        pages = list(range(1, total_pages + 1))
    pages = [p for p in pages if 1 <= p <= total_pages]

    results: list[dict[str, Any]] = []
    for p in pages:
        results.append(parse_page_cascade(pdf_path, p, dpi=dpi))
    return results
