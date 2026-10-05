"""schema.py — Internal Element Schema for layout-aware PDF parsing and routing.

Implements the research-grade 6-stage scientific parsing architecture:
- Typed elements (headings, paragraphs, equations, tables, figures, algorithms, pseudocode, code)
- Confidence vectors tracking detection, extraction, structural, and cross-modal certainty
- Canonical table structures preserving cell-level math vs numeric vs text
- Canonical figure structures preserving plot axes, units, series, and approximate trends
- Suppressed element retention for auditability and provenance
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    ALGORITHM = "algorithm"
    PSEUDOCODE = "pseudocode"
    CODE = "code"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


class PageMode(str, Enum):
    DIGITAL = "DIGITAL"
    HYBRID = "HYBRID"
    SCANNED = "SCANNED"


class CellContentType(str, Enum):
    TEXT = "TEXT"
    NUMERIC = "NUMERIC"
    FORMULA = "FORMULA"
    MIXED = "MIXED"
    EMPTY = "EMPTY"


class FigureType(str, Enum):
    PLOT = "PLOT"
    SCHEMATIC = "SCHEMATIC"
    ARCHITECTURE = "ARCHITECTURE"
    FLOWCHART = "FLOWCHART"
    PHOTOGRAPH = "PHOTOGRAPH"
    MICROSCOPY = "MICROSCOPY"
    DIAGRAM = "DIAGRAM"
    OTHER = "OTHER"


@dataclass
class ConfidenceVector:
    """Multidimensional certainty score for each extracted element."""
    detection_confidence: float = 1.0
    extraction_confidence: float = 1.0
    structural_confidence: float = 1.0
    cross_modal_agreement: float = 1.0
    overall_confidence: float = 1.0

    def compute_overall(self) -> float:
        self.overall_confidence = round(
            0.30 * self.detection_confidence
            + 0.30 * self.extraction_confidence
            + 0.20 * self.structural_confidence
            + 0.20 * self.cross_modal_agreement,
            3,
        )
        return self.overall_confidence

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PageTriageResult:
    """Stage 0 Triage result providing continuous evidence for page classification."""
    page_num: int
    page_mode: PageMode
    confidence: float
    text_density: float
    image_coverage: float
    vector_density: float
    text_layer_quality: float
    raster_resolution_dpi: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["page_mode"] = self.page_mode.value
        return d


@dataclass
class CanonicalTableCell:
    """Individual cell representation preserving raw text, normalized math, and semantic type."""
    row_idx: int
    col_idx: int
    row_span: int = 1
    col_span: int = 1
    raw_text: str = ""
    normalized_text: str = ""
    content_type: CellContentType = CellContentType.TEXT
    math_latex: str | None = None
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    is_header: bool = False

    def get_display_text(self) -> str:
        if self.content_type in (CellContentType.FORMULA, CellContentType.MIXED) and self.math_latex:
            return self.math_latex
        return self.normalized_text or self.raw_text


@dataclass
class CanonicalTable:
    """Rich canonical table model enabling Markdown, HTML, CSV, and summary derivations."""
    bbox: list[float]
    num_rows: int
    num_cols: int
    cells: list[CanonicalTableCell] = field(default_factory=list)
    caption: str | None = None
    source_image_path: str | None = None

    def get_grid(self) -> list[list[str]]:
        grid: list[list[str]] = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row_idx < self.num_rows and 0 <= cell.col_idx < self.num_cols:
                disp = cell.get_display_text().replace("\n", " ").strip()
                for r in range(cell.row_idx, min(self.num_rows, cell.row_idx + cell.row_span)):
                    for c in range(cell.col_idx, min(self.num_cols, cell.col_idx + cell.col_span)):
                        grid[r][c] = disp
        return grid

    def to_markdown(self) -> str:
        grid = self.get_grid()
        if not grid:
            return ""
        lines = []
        header = grid[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in grid[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        grid = self.get_grid()
        if not grid:
            return ""
        lines = ["<table>"]
        if self.caption:
            lines.append(f"  <caption>{self.caption}</caption>")
        lines.append("  <thead><tr>" + "".join(f"<th>{c}</th>" for c in grid[0]) + "</tr></thead>")
        lines.append("  <tbody>")
        for row in grid[1:]:
            lines.append("    <tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        lines.append("  </tbody>")
        lines.append("</table>")
        return "\n".join(lines)

    def to_csv_representation(self) -> str:
        grid = self.get_grid()
        return "\n".join(",".join(f'"{c}"' for c in row) for row in grid)

    def to_natural_language_summary(self) -> str:
        grid = self.get_grid()
        if not grid or len(grid) < 2:
            return "Empty or single-row table."
        header = grid[0]
        row_count = len(grid) - 1
        col_count = len(header)
        summary = f"Table with {row_count} entries across {col_count} columns: {', '.join(header)}."
        if self.caption:
            summary = f"{self.caption}. {summary}"
        return summary


@dataclass
class PlotMetadata:
    """Specialized metadata for scientific plots and diagrams."""
    x_axis: str | None = None
    y_axis: str | None = None
    units: str | None = None
    legend: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    approximate_trends: list[str] = field(default_factory=list)


@dataclass
class CanonicalFigure:
    """Rich figure representation preserving type, crop, caption, and plot metadata."""
    figure_type: FigureType = FigureType.OTHER
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    caption: str | None = None
    image_crop_path: str | None = None
    plot_metadata: PlotMetadata | None = None
    vlm_description: str | None = None


@dataclass
class DocumentElement:
    """A single layout-aware document element extracted by the router.

    Attributes
    ----------
    element_id : str
        Unique identifier (UUID4 string).
    element_type : str
        One of ElementType values.
    paper_id : str
        Identifier of the source paper.
    page : int
        1-based page number.
    section_path : str
        Hierarchical section heading.
    heading_level : int
        Depth of heading.
    bbox : list[float]
        Page coordinates [x0, y0, x1, y1].
    reading_order : int
        Sequential reading order on page / document.
    text : str
        Verbatim text or primary serialized representation.
    confidence : ConfidenceVector
        Multidimensional certainty vector.
    suppressed : bool
        True if element is noise (page header, footer) suppressed from context.
    suppression_reason : str | None
        Provenance rationale if suppressed.
    canonical_table : CanonicalTable | None
        Full canonical table representation if element_type == 'table'.
    canonical_figure : CanonicalFigure | None
        Full canonical figure representation if element_type == 'figure'.
    """
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    element_type: str = ElementType.PARAGRAPH.value
    paper_id: str = ""
    page: int = 1
    section_path: str = ""
    heading_level: int = 0
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    reading_order: int = 0
    text: str = ""
    confidence: ConfidenceVector = field(default_factory=ConfidenceVector)
    suppressed: bool = False
    suppression_reason: str | None = None
    table_structure: dict[str, Any] | None = None
    canonical_table: CanonicalTable | None = None
    figure_ref: str | None = None
    caption: str | None = None
    canonical_figure: CanonicalFigure | None = None
    equation_ref: str | None = None
    symbol_definitions: dict[str, str] | None = None
    units: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DocumentElement:
        known_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known_keys}
        if "confidence" in filtered and isinstance(filtered["confidence"], dict):
            filtered["confidence"] = ConfidenceVector(**filtered["confidence"])
        return cls(**filtered)
