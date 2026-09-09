"""schema.py — Internal Element Schema for layout-aware PDF parsing.

Implements section 3.1 of RAG_ARCHITECTURE.md:
Typed elements with bounding boxes and reading order, decoupling parser
backends (Unlimited-OCR, DeepSeek-OCR, PyMuPDF, Docling) from downstream
chunking, indexing, and retrieval.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


@dataclass
class DocumentElement:
    """A single layout-aware document element extracted by a parser.

    Attributes
    ----------
    element_id : str
        Unique identifier (UUID4 string).
    element_type : str
        One of 'heading', 'paragraph', 'table', 'figure', 'equation', 'caption', 'footnote'.
    paper_id : str
        Identifier of the source paper (e.g. 'abu_ghannam_shaw_1980').
    page : int
        1-based page number.
    section_path : str
        Hierarchical section heading (e.g. "2.3.2 Test Configuration").
    heading_level : int
        Depth of heading (0 for root/body, 1 for H1, 2 for H2, etc.).
    bbox : list[float]
        Page coordinates [x0, y0, x1, y1] normalized or points.
    reading_order : int
        Sequential reading order on page / document.
    text : str
        Verbatim text or serialized representation (e.g. Markdown for tables, LaTeX for equations).
    table_structure : dict[str, Any] | None
        Optional structured table metadata (header row, rows, column count, raw matrix).
    figure_ref : str | None
        Optional figure identifier (e.g. "Fig. 4", image crop path).
    caption : str | None
        Associated caption text if linked.
    equation_ref : str | None
        Optional equation number / tag (e.g. "Eq. (3.6)").
    symbol_definitions : dict[str, str] | None
        Optional mapping of symbols defined within this element.
    units : dict[str, str] | None
        Optional mapping of symbols to physical units.
    metadata : dict[str, Any] = field(default_factory=dict)
        Extra parser-specific payload (confidence scores, font sizes, etc.).
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
    table_structure: dict[str, Any] | None = None
    figure_ref: str | None = None
    caption: str | None = None
    equation_ref: str | None = None
    symbol_definitions: dict[str, str] | None = None
    units: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DocumentElement:
        known_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)
