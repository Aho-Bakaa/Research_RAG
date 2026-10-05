"""cell_dissector.py — Cell-level semantic classifier and canonical table dissector.

Deconstructs tables into structured 2D canonical representations:
- Classifies each cell: TEXT, NUMERIC, FORMULA, MIXED, or EMPTY
- Detects mathematical variables, Greek symbols, superscripts, and relations
- Wraps cell formulas into clean LaTeX syntax ($ ... $)
- Produces CanonicalTable instances deriving Markdown, HTML, CSV, and summaries
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import fitz

from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
from bl_pipeline.rag.schema import (
    CanonicalTable,
    CanonicalTableCell,
    CellContentType,
)

# Regex for pure numeric data (with optional signs, commas, decimals, units like %, deg)
_NUMERIC_RX = re.compile(r"^[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:%|°|deg|cm|mm|m|s|kg|Pa|kPa|MPa)?$", re.IGNORECASE)

# Regex for equation syntax / relations
_EQUATION_REL_RX = re.compile(r"(?:[=><≈≤≥∝±]|[/×÷·]\s*[a-zA-Z0-9])")

# Mathematical symbols
_MATH_SYMBOLS = set("²³√∫∑∂∝≈≤≥±×÷·λθγνωρτμφψχη")


class CellSemanticClassifier:
    """Classifies table cell contents into semantic types and formats LaTeX math."""

    def classify_cell(self, raw_text: str) -> tuple[CellContentType, str, str | None]:
        """Classify cell text. Returns (content_type, normalized_text, math_latex)."""
        stripped = raw_text.strip()
        if not stripped:
            return CellContentType.EMPTY, "", None

        normalized = normalize_scientific_text(stripped)

        # 1. Pure Numeric Check
        if _NUMERIC_RX.match(normalized):
            return CellContentType.NUMERIC, normalized, None

        # 2. Check for Mathematical Symbols & Relations
        has_symbols = any(sym in normalized for sym in _MATH_SYMBOLS) or any(c in normalized for c in "^_")
        has_relation = bool(_EQUATION_REL_RX.search(normalized))
        has_var_index = bool(re.search(r"\b[A-Za-z]+[0-9_]\b|\b[uv][0-9]\b|\bn[12]\b|\bRe_[a-z0-9]+\b", normalized))

        # Check word count
        words = normalized.split()
        word_count = len(words)

        if has_symbols or has_relation or has_var_index:
            # Distinguish pure FORMULA vs MIXED vs TEXT
            # If predominantly words with a single symbol (e.g. "Parameter theta description"), treat as TEXT with inline math
            if word_count > 4 and not has_relation:
                return CellContentType.TEXT, normalized, None

            # Check if MIXED (e.g. "Re_theta = 420" or "M = 2.54 cm" or "u2-decay &")
            has_digits = any(c.isdigit() for c in normalized)
            has_letters = any(c.isalpha() for c in normalized)

            # Format as LaTeX math
            math_latex = f"${normalized}$" if not normalized.startswith("$") else normalized

            if has_digits and has_letters and (word_count > 1 or has_relation):
                return CellContentType.MIXED, normalized, math_latex
            else:
                return CellContentType.FORMULA, normalized, math_latex

        # Default to TEXT
        return CellContentType.TEXT, normalized, None


class TableDissector:
    """Dissects Docling or raw table structures into rich CanonicalTable representations."""

    def __init__(self):
        self.classifier = CellSemanticClassifier()

    def dissect_docling_table(
        self,
        table_item: Any,
        page: fitz.Page | None = None,
        caption: str | None = None,
    ) -> CanonicalTable:
        """Convert a Docling TableItem into a CanonicalTable with cell-level classifications."""
        data = getattr(table_item, "data", None)
        if not data:
            return CanonicalTable(bbox=[0.0, 0.0, 0.0, 0.0], num_rows=0, num_cols=0)

        num_rows = getattr(data, "num_rows", 0)
        num_cols = getattr(data, "num_cols", 0)
        table_cells = getattr(data, "table_cells", [])

        prov = getattr(table_item, "prov", [])
        tbl_bbox = [0.0, 0.0, 0.0, 0.0]
        if prov and hasattr(prov[0], "bbox"):
            b = prov[0].bbox
            tbl_bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]

        canonical_cells: list[CanonicalTableCell] = []

        for cell in table_cells:
            r_idx = getattr(cell, "start_row_offset_idx", 0)
            c_idx = getattr(cell, "start_col_offset_idx", 0)
            r_span = getattr(cell, "row_span", 1)
            c_span = getattr(cell, "col_span", 1)
            raw_text = getattr(cell, "text", "") or ""
            is_header = bool(getattr(cell, "column_header", False) or getattr(cell, "row_header", False))

            c_bbox = [0.0, 0.0, 0.0, 0.0]
            if hasattr(cell, "bbox") and cell.bbox:
                cb = cell.bbox
                c_bbox = [float(cb.l), float(cb.t), float(cb.r), float(cb.b)]

            # Classify Cell Content
            content_type, norm_text, math_latex = self.classifier.classify_cell(raw_text)

            canonical_cells.append(
                CanonicalTableCell(
                    row_idx=r_idx,
                    col_idx=c_idx,
                    row_span=r_span,
                    col_span=c_span,
                    raw_text=raw_text,
                    normalized_text=norm_text,
                    content_type=content_type,
                    math_latex=math_latex,
                    bbox=c_bbox,
                    is_header=is_header,
                )
            )

        return CanonicalTable(
            bbox=tbl_bbox,
            num_rows=num_rows,
            num_cols=num_cols,
            cells=canonical_cells,
            caption=caption,
        )

    def dissect_raw_grid(
        self,
        grid: list[list[str]],
        bbox: list[float] | None = None,
        caption: str | None = None,
    ) -> CanonicalTable:
        """Build CanonicalTable from a 2D string grid (e.g. from PyMuPDF find_tables)."""
        num_rows = len(grid)
        num_cols = max((len(row) for row in grid), default=0)
        canonical_cells: list[CanonicalTableCell] = []

        for r_idx, row in enumerate(grid):
            for c_idx, raw_text in enumerate(row):
                content_type, norm_text, math_latex = self.classifier.classify_cell(raw_text)
                canonical_cells.append(
                    CanonicalTableCell(
                        row_idx=r_idx,
                        col_idx=c_idx,
                        row_span=1,
                        col_span=1,
                        raw_text=raw_text,
                        normalized_text=norm_text,
                        content_type=content_type,
                        math_latex=math_latex,
                        is_header=(r_idx == 0),
                    )
                )

        return CanonicalTable(
            bbox=bbox or [0.0, 0.0, 0.0, 0.0],
            num_rows=num_rows,
            num_cols=num_cols,
            cells=canonical_cells,
            caption=caption,
        )
