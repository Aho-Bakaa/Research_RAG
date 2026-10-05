"""layout_parser.py — Layout-aware parser yielding typed elements.

Implements section 3 of RAG_ARCHITECTURE.md:
- 300 DPI page rasterization support
- Extraction of typed elements (headings, paragraphs, tables, equations, figures)
- Preserves bounding boxes, reading order, and heading hierarchy (section_path)
- Seamless fallback cascade (PyMuPDF layout/table detection -> docling/pymupdf4llm/OCR hooks)
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any, Tuple

import fitz  # PyMuPDF

from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
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


def is_garbled_or_math_dense(text: str) -> Tuple[bool, str]:
    """Evaluates whether a text block contains garbled math patterns or dense formulas.
    
    Checks:
    1. Repeated = or corrupted operator sequences
    2. Broken fraction-like sequences or slash artifacts
    3. Isolated stray symbols (~, ^, &, \\, |, *, ·, ', _)
    4. Truncated equations or bare equation numbers e.g. '(16)'
    5. Low ratio of dictionary/alphabetic words to total tokens in math-dense context
    """
    if not text or not text.strip():
        return False, "empty"
    stripped = text.strip()

    # 1. Repeated = or corrupted operator sequences
    if re.search(r'={2,}|=\s*=|_\s*_|—\s*—|-{3,}|[0-9]=[0-9]', stripped):
        return True, "repeated_equals_or_operators"

    # 2. Broken fraction-like sequences
    if re.search(r'(?:^|\n)\s*[a-zA-Z0-9\'-]{1,10}\s*\n\s*[_–—=]{2,}\s*\n\s*[a-zA-Z0-9\'-]{1,10}', stripped):
        return True, "broken_fraction_sequence"
    if re.search(r'[a-zA-Z0-9]\s*\/\s*[\/\'~]{1,}', stripped):
        return True, "broken_slash_sequence"

    # 3. Isolated stray symbols & corrupted glyphs
    stray_matches = re.findall(r'(?:^|\s)[~^&\\|*·\'_]{1,3}(?:\s|$)', stripped)
    corrupted_glyphs = re.findall(r'[~^\\|·§©®ðÞ∂∫∑√]', stripped)
    if len(stray_matches) >= 2 or len(corrupted_glyphs) >= 1:
        return True, "isolated_stray_symbols_or_glyphs"

    # 4. Incomplete / truncated equation or bare equation tag
    if re.match(r'^\s*(?:\(\s*\d+\s*\)|Eq\.\s*\(?\s*\d+\s*\)?)\s*$', stripped):
        return True, "bare_equation_number"
    if re.search(r'^[∂∫∑√]\s*[a-zA-Z0-9~_\s]{1,12}$', stripped):
        return True, "dangling_derivative_or_integral"
    if re.search(r'^[a-zA-Z0-9\s_]{0,5}[=−-]\s*[0-9−-]{1,5}$', stripped) and len(stripped) < 15:
        return True, "truncated_short_equation"

    # 5. Low ratio of dictionary words in math-dense context
    tokens = stripped.split()
    if len(tokens) >= 2:
        words = [t for t in tokens if t.isalpha() and len(t) >= 2]
        has_math_cues = any(c in stripped for c in ['=', '+', '-', '/', '∫', '∑', '∂', '√', 'Re_', 'Tu', '^', '(', ')', '%', '*', 'ε'])
        word_ratio = len(words) / len(tokens)
        if has_math_cues and word_ratio < 0.50:
            return True, f"low_word_ratio_{word_ratio:.2f}"

    return False, "clean"


class LayoutAwareParser:
    """Parser that processes PDF documents into a structured stream of DocumentElements."""

    def __init__(
        self,
        rasterize_dpi: int = 300,
        output_crops_dir: Path | None = None,
        enable_equation_ocr: bool = False,
        enable_math_reconstruct: bool = True,
        enable_vision_math: bool = False,
    ):
        self.rasterize_dpi = rasterize_dpi
        self.output_crops_dir = output_crops_dir
        self.enable_equation_ocr = enable_equation_ocr
        self.enable_math_reconstruct = enable_math_reconstruct
        self.enable_vision_math = enable_vision_math
        self._ocr_engine = None
        self._latex_ocr = None

    def _init_latex_ocr(self):
        """Lazy-load pix2tex LatexOCR engine for hard equations."""
        if not self.enable_vision_math:
            return None
        if self._latex_ocr is None:
            try:
                from pix2tex.cli import LatexOCR
                self._latex_ocr = LatexOCR()
            except Exception:
                self._latex_ocr = None
        return self._latex_ocr

    def _init_ocr_engine(self):
        """Lazy-load equation-aware OCR engine (PaddleOCR-VL)."""
        if not self.enable_equation_ocr:
            return None
        if self._ocr_engine is None:
            try:
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
                from paddleocr import PaddleOCRVL
                self._ocr_engine = PaddleOCRVL(pipeline_version="v1", use_chart_recognition=False, use_seal_recognition=False)
            except Exception:
                try:
                    from paddleocr import PaddleOCRVL
                    self._ocr_engine = PaddleOCRVL
                except Exception:
                    self._ocr_engine = None
        return self._ocr_engine

    def _ocr_equation_crop(self, page: fitz.Page, bbox: list[float]) -> str | None:
        """Rasterize equation crop and extract formula representation via PaddleOCR-VL or EasyOCR fallback."""
        if not self.enable_equation_ocr:
            return None
        rect = fitz.Rect(bbox)
        padded = fitz.Rect(max(0, rect.x0 - 4), max(0, rect.y0 - 4), min(page.rect.width, rect.x1 + 4), min(page.rect.height, rect.y1 + 4))
        pix = page.get_pixmap(clip=padded, dpi=200)
        img_bytes = pix.tobytes("png")

        # 1. Primary: PaddleOCR-VL
        engine = self._init_ocr_engine()
        if engine:
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if hasattr(engine, "predict"):
                    ocr_res = engine.predict(img)
                    if ocr_res:
                        text_parts = []
                        for item in ocr_res:
                            if hasattr(item, "markdown"):
                                text_parts.append(str(item.markdown))
                            elif hasattr(item, "formula"):
                                text_parts.append(str(item.formula))
                            elif hasattr(item, "text"):
                                text_parts.append(str(item.text))
                        if text_parts:
                            return "\n".join(text_parts).strip()
            except Exception:
                pass

        # 2. Robust Local Fallback: EasyOCR
        try:
            import easyocr
            if not hasattr(self, "_easyocr_reader"):
                self._easyocr_reader = easyocr.Reader(["en"], gpu=False)
            res = self._easyocr_reader.readtext(img_bytes, detail=0)
            if res:
                return " ".join(res).strip()
        except Exception:
            pass

        return None

    @staticmethod
    def extract_equation_band(page: fitz.Page, bbox: list[float]) -> tuple[str, list[float]]:
        """Expands an isolated equation tag/fragment across its column band to assemble the full formula."""
        x0, y0, x1, y1 = bbox
        p_width = page.rect.width
        mid = p_width / 2.0

        if x1 < mid:
            col_x0, col_x1 = max(0.0, page.rect.x0 + 30.0), mid - 5.0
        elif x0 > mid:
            col_x0, col_x1 = mid + 5.0, page.rect.x1 - 30.0
        else:
            col_x0, col_x1 = max(0.0, page.rect.x0 + 30.0), page.rect.x1 - 30.0

        # Detect prose paragraph boundaries in this column to prevent bleeding into adjacent text
        blocks = page.get_text("blocks")
        prev_y1 = 0.0
        next_y0 = page.rect.height

        for b in blocks:
            ibx0, iby0, ibx1, iby1, itxt = b[:5]
            if not (ibx1 < col_x0 or ibx0 > col_x1):
                clean_txt = itxt.strip()
                is_tag = bool(re.search(r"^[\(\x03]\s*\d+[\.\da-z]*\s*[\)\x04]$", clean_txt))
                is_math = any(sym in clean_txt for sym in ["=", "∂", "∫", "∑", "√", "∕"])
                is_prose = (len(clean_txt) >= 80 or (len(clean_txt) >= 40 and not is_math and " " in clean_txt))
                if is_prose and not is_tag:
                    if iby1 <= y0 - 10.0 and iby1 > prev_y1:
                        prev_y1 = iby1
                    elif iby0 >= y1 + 5.0 and iby0 < next_y0:
                        next_y0 = iby0

        band_y0 = max(prev_y1 + 1.5, y0 - 45.0)
        band_y1 = min(next_y0 - 1.5, y1 + 15.0)

        words = page.get_text("words")
        band_words = [w for w in words if col_x0 <= w[0] <= col_x1 and band_y0 <= w[1] and w[3] <= next_y0 - 0.5]

        if not band_words:
            return "", bbox

        band_words_sorted = sorted(band_words, key=lambda w: (round(w[1] / 7.0), w[0]))
        raw_assembled = " ".join(w[4] for w in band_words_sorted).strip()
        assembled = normalize_scientific_text(raw_assembled)

        # Compute tight envelope encompassing words and drawings in this band
        drawings = page.get_drawings()
        eq_drawings = [d["rect"] for d in drawings if col_x0 - 5 <= d["rect"][0] and d["rect"][2] <= col_x1 + 5 and band_y0 - 2 <= d["rect"][1] and d["rect"][3] <= band_y1 + 2]
        all_rects = [fitz.Rect(w[:4]) for w in band_words] + [fitz.Rect(r) for r in eq_drawings]
        if all_rects:
            u = all_rects[0]
            for r in all_rects[1:]:
                u |= r
            expanded_bbox = [max(0.0, u.x0 - 2.0), max(0.0, u.y0 - 2.0), min(page.rect.width, u.x1 + 2.0), min(page.rect.height, u.y1 + 2.0)]
        else:
            expanded_bbox = [col_x0, band_y0, col_x1, band_y1]

        return assembled, expanded_bbox

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
                    text = normalize_scientific_text(text)
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
                        
                        eq_text = text
                        target_bbox = bbox

                        # Recover complete formula if equation tag is isolated or truncated
                        if eq_match and (len(text.strip()) < 40 or is_garbled_or_math_dense(text)[0]):
                            band_text, band_bbox = self.extract_equation_band(page, bbox)
                            if band_text and len(band_text) > len(eq_text):
                                eq_text = band_text
                                target_bbox = band_bbox

                                # Remove previously added orphan math fragments on this page subsumed by this full equation band
                                elements[:] = [
                                    el for el in elements
                                    if not (
                                        el.element_type == ElementType.EQUATION.value
                                        and el.page == page_num
                                        and el.equation_ref is None
                                        and target_bbox[0] - 2.0 <= el.bbox[0]
                                        and el.bbox[2] <= target_bbox[2] + 2.0
                                        and target_bbox[1] - 2.0 <= el.bbox[1]
                                        and el.bbox[3] <= target_bbox[3] + 2.0
                                    )
                                ]

                        meta: dict[str, Any] = {}
                        final_bbox = target_bbox

                        # Math Reconstruction & Complexity Routing
                        if getattr(self, "enable_math_reconstruct", True):
                            try:
                                from math_reconstruct import (
                                    classify_equation,
                                    get_tight_formula_rect,
                                    reconstruct_simple_equation,
                                )

                                tight_rect = get_tight_formula_rect(page, target_bbox, tag_id=eq_id)
                                final_bbox = [round(c, 1) for c in list(tight_rect)]

                                cls_info = classify_equation(doc, page_num - 1, tuple(tight_rect))
                                is_simple = cls_info["classification"] == "simple"
                                meta["classification"] = cls_info["classification"]
                                meta["classification_reasons"] = cls_info["reasons"]
                                meta["tight_bbox"] = final_bbox

                                if is_simple:
                                    rec_latex = reconstruct_simple_equation(doc, page_num - 1, tuple(tight_rect))
                                    if rec_latex and len(rec_latex.strip()) > 1:
                                        meta["raw_text"] = eq_text
                                        meta["latex_source"] = "rule_based_reconstructor"
                                        eq_text = rec_latex.strip()
                                elif getattr(self, "enable_vision_math", False):
                                    ocr_model = self._init_latex_ocr()
                                    if ocr_model:
                                        from PIL import Image
                                        padded = fitz.Rect(final_bbox)
                                        pix = page.get_pixmap(clip=padded, dpi=250)
                                        img_bytes = pix.tobytes("png")
                                        pred_latex = ocr_model(Image.open(io.BytesIO(img_bytes)))
                                        if pred_latex and len(pred_latex.strip()) > 3:
                                            meta["raw_text"] = eq_text
                                            meta["latex_source"] = "pix2tex_vision"
                                            eq_text = pred_latex.strip()
                            except Exception as e:
                                meta["math_reconstruct_error"] = str(e)
                        elif is_garbled_or_math_dense(eq_text)[0]:
                            ocr_result = self._ocr_equation_crop(page, target_bbox)
                            if ocr_result and len(ocr_result.strip()) > 3:
                                eq_text = ocr_result.strip()
                                meta = {
                                    "equation_ocr": True,
                                    "ocr_engine": "PaddleOCR-VL",
                                    "original_raw_text": text,
                                }

                        el = DocumentElement(
                            element_type=ElementType.EQUATION.value,
                            paper_id=paper_id,
                            page=page_num,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=final_bbox,
                            reading_order=reading_order,
                            text=eq_text,
                            equation_ref=f"Eq. ({eq_id})" if eq_id else None,
                            metadata=meta if meta else None,
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
            clean_hdr = [normalize_scientific_text(str(c or "")).replace("\n", " ").strip() for c in header]
            lines.append("| " + " | ".join(clean_hdr) + " |")
            lines.append("| " + " | ".join(["---"] * len(clean_hdr)) + " |")
        
        for r in rows:
            clean_r = [normalize_scientific_text(str(c or "")).replace("\n", " ").strip() for c in r]
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
