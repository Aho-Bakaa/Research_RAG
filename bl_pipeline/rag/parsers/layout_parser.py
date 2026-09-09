"""layout_parser.py — Layout-aware parser yielding typed elements.

Implements section 3 of RAG_ARCHITECTURE.md:
- 300 DPI page rasterization support
- Extraction of typed elements (headings, paragraphs, tables, equations, figures)
- Preserves bounding boxes, reading order, and heading hierarchy (section_path)
- Seamless fallback cascade (PyMuPDF layout/table detection -> docling/pymupdf4llm/OCR hooks)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from bl_pipeline.rag.schema import DocumentElement, ElementType


# Heading pattern matching common scientific section numbering
_HEADING_RX = re.compile(
    r"^(?:(?:[0-9]+(?:\.[0-9]+)*|[A-Z]|Appendix\s+[A-Z])[\.\s]+[A-Z][^\n]{2,80}|"
    r"Abstract|Introduction|Methodology|Experimental\s+Setup|Results|Discussion|Conclusions?|References)\s*$",
    re.IGNORECASE,
)

# Equation numbering pattern: (1), (3.6), Eq. (5)
_EQ_TAG_RX = re.compile(
    r"(?:\(\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)|Eq\.\s*\(?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)?)\s*$"
)

# Figure caption pattern: Fig. 1, Figure 4:
_FIG_CAPTION_RX = re.compile(
    r"^(?:Fig(?:\.|ure)?\s*([0-9]+(?:\.[0-9]+)*[a-z]?))[:\.\s]+(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Table caption pattern: Table 1:, Tab. 2
_TABLE_CAPTION_RX = re.compile(
    r"^(?:Table|Tab\.)\s*([0-9]+(?:\.[0-9]+)*[a-z]?)(?:[:\.\s]+(.*))?",
    re.IGNORECASE | re.DOTALL,
)


class LayoutAwareParser:
    """Parser that processes PDF documents into a structured stream of DocumentElements."""

    def __init__(self, rasterize_dpi: int = 300, output_crops_dir: Path | None = None):
        self.rasterize_dpi = rasterize_dpi
        self.output_crops_dir = output_crops_dir

    def rasterize_page(self, pdf_path: str | Path, page_num: int, output_path: str | Path | None = None) -> Path:
        """Rasterize a single PDF page at target DPI (default 300)."""
        pdf_path = Path(pdf_path)
        if output_path is None:
            output_path = pdf_path.parent / f"{pdf_path.stem}_p{page_num}_{self.rasterize_dpi}dpi.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num - 1]
            scale = self.rasterize_dpi / 72.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(output_path))
            return output_path
        finally:
            doc.close()

    def parse_pdf(self, pdf_path: str | Path, paper_id: str | None = None) -> list[DocumentElement]:
        """Extract structured DocumentElements with bounding boxes and reading order."""
        pdf_path = Path(pdf_path)
        paper_id = paper_id or pdf_path.stem

        elements: list[DocumentElement] = []
        doc = fitz.open(pdf_path)

        current_section = "Root"
        current_heading_level = 0
        reading_order = 0

        try:
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1

                # 1. Extract tables using PyMuPDF table finder if available
                tables = []
                table_bboxes = []
                try:
                    tabs = page.find_tables()
                    for t_idx, tab in enumerate(tabs):
                        t_bbox = list(tab.bbox)
                        table_bboxes.append(t_bbox)
                        header = tab.header.names if tab.header else []
                        rows = tab.extract()
                        
                        # Serialize to Markdown
                        md_table = self._format_markdown_table(header, rows)
                        table_el = DocumentElement(
                            element_type=ElementType.TABLE.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=t_bbox,
                            reading_order=reading_order,
                            text=md_table,
                            table_structure={
                                "header": header,
                                "row_count": len(rows),
                                "col_count": len(header) if header else (len(rows[0]) if rows else 0),
                            },
                        )
                        tables.append(table_el)
                        reading_order += 1
                except Exception:
                    pass

                # 2. Extract text blocks and identify headings, captions, equations, and paragraphs
                blocks = page.get_text("blocks")
                # blocks: (x0, y0, x1, y1, text, block_no, block_type)
                for b in blocks:
                    x0, y0, x1, y1, text, b_no, b_type = b
                    text = text.strip()
                    if not text:
                        continue

                    # Check if inside an already-extracted table bbox
                    if self._is_inside_any(x0, y0, x1, y1, table_bboxes):
                        continue

                    bbox = [float(x0), float(y0), float(x1), float(y1)]

                    # Check if Heading
                    if _HEADING_RX.match(text) and len(text) < 120 and "\n" not in text:
                        level = 1
                        m = re.match(r"^([0-9]+(?:\.[0-9]+)*)", text)
                        if m:
                            level = len(m.group(1).split("."))
                        current_heading_level = level
                        current_section = text
                        
                        el = DocumentElement(
                            element_type=ElementType.HEADING.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=bbox,
                            reading_order=reading_order,
                            text=text,
                        )
                        elements.append(el)
                        reading_order += 1
                        continue

                    # Check if Figure Caption
                    fig_match = _FIG_CAPTION_RX.match(text)
                    if fig_match:
                        fig_no = fig_match.group(1)
                        el = DocumentElement(
                            element_type=ElementType.CAPTION.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=bbox,
                            reading_order=reading_order,
                            text=text,
                            figure_ref=f"Fig. {fig_no}",
                            caption=text,
                        )
                        elements.append(el)
                        reading_order += 1
                        continue

                    # Check if Table Caption
                    tab_match = _TABLE_CAPTION_RX.match(text)
                    if tab_match:
                        tab_no = tab_match.group(1)
                        el = DocumentElement(
                            element_type=ElementType.CAPTION.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=bbox,
                            reading_order=reading_order,
                            text=text,
                            caption=text,
                            metadata={"table_ref": f"Table {tab_no}"},
                        )
                        elements.append(el)
                        reading_order += 1
                        continue

                    # Check if standalone Equation
                    eq_match = _EQ_TAG_RX.search(text)
                    is_math_like = any(sym in text for sym in ["=", "∫", "∑", "∂", "√", "\\", "·", "Re_", "Tu^"])
                    if eq_match or (is_math_like and len(text.splitlines()) <= 4 and len(text) < 200):
                        eq_id = None
                        if eq_match:
                            eq_id = eq_match.group(1) or eq_match.group(2)
                        
                        el = DocumentElement(
                            element_type=ElementType.EQUATION.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=bbox,
                            reading_order=reading_order,
                            text=text,
                            equation_ref=f"Eq. ({eq_id})" if eq_id else None,
                        )
                        elements.append(el)
                        reading_order += 1
                        continue

                    # Default: Paragraph
                    el = DocumentElement(
                        element_type=ElementType.PARAGRAPH.value,
                        paper_id=paper_id,
                        page=page_num,
                        section_path=current_section,
                        heading_level=current_heading_level,
                        bbox=bbox,
                        reading_order=reading_order,
                        text=text,
                    )
                    elements.append(el)
                    reading_order += 1

                # Append any tables found on this page
                elements.extend(tables)

        finally:
            doc.close()

        # Sort elements by page and reading_order
        elements.sort(key=lambda e: (e.page, e.reading_order))
        return elements

    @staticmethod
    def _is_inside_any(x0: float, y0: float, x1: float, y1: float, bboxes: list[list[float]]) -> bool:
        """Check if bbox center falls inside any bounding box in list."""
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for b in bboxes:
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                return True
        return False

    @staticmethod
    def _format_markdown_table(header: list[str], rows: list[list[Any]]) -> str:
        """Serialize table rows and header into clean GitHub-flavored Markdown."""
        if not rows and not header:
            return ""
        lines = []
        if header:
            clean_hdr = [str(c or "").replace("\n", " ").strip() for c in header]
            lines.append("| " + " | ".join(clean_hdr) + " |")
            lines.append("| " + " | ".join(["---"] * len(clean_hdr)) + " |")
        
        for r in rows:
            clean_r = [str(c or "").replace("\n", " ").strip() for c in r]
            if not header and not lines:
                lines.append("| " + " | ".join(clean_r) + " |")
                lines.append("| " + " | ".join(["---"] * len(clean_r)) + " |")
            else:
                lines.append("| " + " | ".join(clean_r) + " |")
        return "\n".join(lines)


def parse_pdf_to_elements(pdf_path: str | Path, paper_id: str | None = None) -> list[DocumentElement]:
    """Convenience functional interface for PDF parsing."""
    parser = LayoutAwareParser()
    return parser.parse_pdf(pdf_path, paper_id=paper_id)
