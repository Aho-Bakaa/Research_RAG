"""structured_chunker.py — Structure-preserving child-parent multimodal chunker.

Implements section 4 of RAG_ARCHITECTURE.md:
1. Atomic tables with repeated headers if split across row boundaries.
2. Equations bundled with symbol definitions, units, and defining paragraph.
3. Figure chunks bundling caption, surrounding paragraphs, and VLM description context.
4. Small-to-big parent-child linking: child chunks link to a section-level parent chunk.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from bl_pipeline.rag.schema import DocumentElement, ElementType


@dataclass
class StructuredChunk:
    """A semantic, structure-preserving chunk ready for indexing in Qdrant and Neo4j.

    Attributes
    ----------
    chunk_id : str
        Unique identifier.
    parent_id : str | None
        UUID of the enclosing section-level parent chunk (enables small-to-big retrieval).
    paper_id : str
        Source paper identifier.
    page : int
        Primary page number.
    section_path : str
        Hierarchical heading path (e.g. "3. Experimental Results").
    heading_level : int
        Depth in heading tree.
    element_type : str
        'paragraph', 'table', 'figure', 'equation', or 'parent'.
    content : str
        The textual or serialized representation for embedding and retrieval.
    metadata : dict[str, Any]
        Rich metadata including bbox, reading_order, equation_ref, figure_ref, table_no, etc.
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    paper_id: str = ""
    page: int = 1
    section_path: str = ""
    heading_level: int = 0
    element_type: str = "paragraph"
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Noise detection patterns
NOISE_HEADINGS = {
    "acknowledgements", "acknowledgments", "references", "nomenclature",
    "contents", "table of contents", "author details", "biography",
}

ISOLATED_PAGE_NUM_RX = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]+)\s*$", re.IGNORECASE)
RUNNING_HEADER_RX = re.compile(
    r"^(?:(?:\d+\s+[A-Z0-9-]+|[A-Z\.\s]+(?:ET AL\.|VOL\.|NO\.|PAGE|\d{4}))|"
    r"(?:[A-Za-z\s]+(?:Journal|Transactions|AIAA|ASME|Elsevier|Springer|Conference)\b.*))\s*$",
    re.IGNORECASE,
)


def is_noise_fragment(text: str, bbox: list[float] | None = None, page_height: float = 792.0) -> bool:
    """Detect isolated page numbers, headers, single-word noise, and orphan coordinates."""
    stripped = text.strip()
    if not stripped or len(stripped) == 1:
        return True

    lower = stripped.lower()
    if lower in NOISE_HEADINGS:
        return True

    if ISOLATED_PAGE_NUM_RX.match(stripped):
        return True

    # Header / footer margin filter (top 5% or bottom 5% of page)
    if bbox is not None and len(bbox) == 4:
        y0, y1 = bbox[1], bbox[3]
        if (y0 < page_height * 0.05 or y1 > page_height * 0.95) and len(stripped) < 120:
            if RUNNING_HEADER_RX.match(stripped) or ISOLATED_PAGE_NUM_RX.match(stripped):
                return True

    # Standalone isolated numbers or numeric coordinates (< 12 chars, e.g. '10 000', '1.5', '0.008')
    if len(stripped) <= 12 and re.match(r"^[\d\s\.,\-\+\*\/()=]+$", stripped):
        return True

    return False


class StructurePreservingChunker:
    """Chunker that converts a stream of DocumentElements into StructuredChunks with micro-chunk merging."""

    def __init__(
        self,
        min_paragraph_chars: int = 200,
        max_paragraph_chars: int = 1500,
        max_table_rows_per_chunk: int = 25,
        max_paragraph_words: int = 400,
    ):
        self.min_paragraph_chars = min_paragraph_chars
        self.max_paragraph_chars = max_paragraph_chars
        self.max_table_rows_per_chunk = max_table_rows_per_chunk
        self.max_paragraph_words = max_paragraph_words

    def chunk_elements(
        self,
        elements: Sequence[DocumentElement],
        paper_id: str | None = None,
    ) -> list[StructuredChunk]:
        """Convert elements into parent and child chunks preserving reading order and relations."""
        if not elements:
            return []

        paper_id = paper_id or elements[0].paper_id or "unknown_paper"
        chunks: list[StructuredChunk] = []

        # 1. Group elements by section_path
        sections: dict[str, list[DocumentElement]] = {}
        for el in elements:
            if getattr(el, "suppressed", False):
                continue
            sec = el.section_path or "Root"
            sections.setdefault(sec, []).append(el)

        # 2. Process each section: generate Parent chunk + merged Child chunks
        for section_path, sec_elements in sections.items():
            first_el = sec_elements[0]
            heading_level = first_el.heading_level

            # Create Section-level Parent Chunk
            parent_chunk_id = str(uuid.uuid4())
            parent_text_parts = [e.text for e in sec_elements if e.text]

            parent_chunk = StructuredChunk(
                chunk_id=parent_chunk_id,
                parent_id=None,
                paper_id=paper_id,
                page=first_el.page,
                section_path=section_path,
                heading_level=heading_level,
                element_type="parent",
                content=f"# {section_path}\n\n" + "\n\n".join(parent_text_parts[:10]),
                metadata={
                    "total_elements": len(sec_elements),
                    "element_types": list({e.element_type for e in sec_elements}),
                    "start_page": min(e.page for e in sec_elements),
                    "end_page": max(e.page for e in sec_elements),
                },
            )
            chunks.append(parent_chunk)

            # Accumulator buffer for merging adjacent paragraph elements
            acc_elements: list[DocumentElement] = []
            consumed_indices: set[int] = set()

            def _flush_accumulator() -> None:
                nonlocal acc_elements
                if not acc_elements:
                    return

                merged_text = "\n\n".join(e.text for e in acc_elements)
                first = acc_elements[0]
                last = acc_elements[-1]

                if all(e.page == first.page for e in acc_elements):
                    union_bbox = [
                        min(e.bbox[0] for e in acc_elements),
                        min(e.bbox[1] for e in acc_elements),
                        max(e.bbox[2] for e in acc_elements),
                        max(e.bbox[3] for e in acc_elements),
                    ]
                else:
                    union_bbox = first.bbox

                p_chunk = StructuredChunk(
                    parent_id=parent_chunk_id,
                    paper_id=paper_id,
                    page=first.page,
                    section_path=section_path,
                    heading_level=heading_level,
                    element_type="paragraph",
                    content=merged_text,
                    metadata={
                        "bbox": union_bbox,
                        "all_bboxes": [e.bbox for e in acc_elements],
                        "reading_order": first.reading_order,
                        "reading_order_end": last.reading_order,
                        "start_page": first.page,
                        "end_page": last.page,
                        "element_count": len(acc_elements),
                        "confidence": first.confidence.to_dict() if first.confidence else {},
                    },
                )
                chunks.append(p_chunk)
                acc_elements = []

            for idx, el in enumerate(sec_elements):
                if idx in consumed_indices:
                    continue

                if el.element_type == ElementType.HEADING.value:
                    _flush_accumulator()
                    continue

                elif el.element_type == ElementType.TABLE.value:
                    _flush_accumulator()
                    table_chunks = self._chunk_table(el, parent_chunk_id, paper_id)
                    chunks.extend(table_chunks)

                elif el.element_type == ElementType.EQUATION.value:
                    # Boundary Stitching around Equations
                    prev_p = ""
                    if acc_elements:
                        acc_text = "\n\n".join(e.text for e in acc_elements)
                        if len(acc_text) < self.min_paragraph_chars or acc_text.rstrip().endswith((":", "as:", "follows:", "given by:")):
                            prev_p = acc_text
                            acc_elements = []  # Absorbed: suppress standalone micro-chunk
                        else:
                            prev_p = acc_text[-300:]
                            _flush_accumulator()

                    next_p = ""
                    if idx + 1 < len(sec_elements) and sec_elements[idx + 1].element_type == ElementType.PARAGRAPH.value:
                        cand_next = sec_elements[idx + 1].text
                        cand_lead = cand_next.strip().lower()
                        is_def = cand_lead.startswith(("where", "in which", "here", "with", "and ")) or len(cand_next) < self.min_paragraph_chars
                        if is_def and len(cand_next) < 350:
                            next_p = cand_next
                            consumed_indices.add(idx + 1)  # Absorbed: suppress standalone micro-chunk
                        else:
                            next_p = cand_next[:300]

                    eq_chunk = self._chunk_equation(el, prev_p, next_p, parent_chunk_id, paper_id)
                    chunks.append(eq_chunk)

                elif el.element_type in (ElementType.CAPTION.value, ElementType.FIGURE.value):
                    _flush_accumulator()
                    prev_p = sec_elements[idx - 1].text[-250:] if idx > 0 and sec_elements[idx - 1].element_type == ElementType.PARAGRAPH.value else ""
                    next_p = sec_elements[idx + 1].text[:250] if idx + 1 < len(sec_elements) and sec_elements[idx + 1].element_type == ElementType.PARAGRAPH.value else ""
                    fig_chunk = self._chunk_figure(el, prev_p, next_p, parent_chunk_id, paper_id)
                    chunks.append(fig_chunk)

                elif el.element_type in (ElementType.ALGORITHM.value, ElementType.PSEUDOCODE.value, ElementType.CODE.value):
                    _flush_accumulator()
                    code_chunk = StructuredChunk(
                        parent_id=parent_chunk_id,
                        paper_id=paper_id,
                        page=el.page,
                        section_path=section_path,
                        heading_level=heading_level,
                        element_type=el.element_type,
                        content=f"```{el.element_type}\n{el.text}\n```",
                        metadata={
                            "bbox": el.bbox,
                            "reading_order": el.reading_order,
                            "confidence": el.confidence.to_dict() if el.confidence else {},
                        },
                    )
                    chunks.append(code_chunk)

                elif el.element_type == ElementType.PARAGRAPH.value:
                    if is_noise_fragment(el.text, el.bbox):
                        continue

                    acc_len = sum(len(e.text) for e in acc_elements)
                    if (acc_len + len(el.text) > self.max_paragraph_chars) and (acc_len >= self.min_paragraph_chars):
                        _flush_accumulator()

                    acc_elements.append(el)

            # Flush any remaining paragraph elements in section
            _flush_accumulator()

        return chunks

    def _chunk_table(
        self,
        el: DocumentElement,
        parent_id: str,
        paper_id: str,
    ) -> list[StructuredChunk]:
        """Atomic table chunking: repeat headers if table rows exceed threshold."""
        tbl_content = el.text
        tbl_meta = {
            "table_structure": el.table_structure,
            "bbox": el.bbox,
            "reading_order": el.reading_order,
            "confidence": el.confidence.to_dict() if el.confidence else {},
        }
        if el.canonical_table:
            tbl_meta["canonical_table"] = el.canonical_table.to_dict()
            summary = el.canonical_table.to_natural_language_summary()
            if summary:
                tbl_meta["summary"] = summary
                tbl_content += f"\n\nTable Summary: {summary}"

        lines = [ln for ln in tbl_content.splitlines() if ln.strip()]
        if len(lines) <= self.max_table_rows_per_chunk + 2:
            return [
                StructuredChunk(
                    parent_id=parent_id,
                    paper_id=paper_id,
                    page=el.page,
                    section_path=el.section_path,
                    heading_level=el.heading_level,
                    element_type="table",
                    content=tbl_content,
                    metadata=tbl_meta,
                )
            ]

        # Multi-part split with repeated header
        header_lines = lines[:2] if len(lines) >= 2 and "---" in lines[1] else lines[:1]
        data_rows = lines[len(header_lines):]

        chunks = []
        for i in range(0, len(data_rows), self.max_table_rows_per_chunk):
            chunk_rows = data_rows[i : i + self.max_table_rows_per_chunk]
            part_content = "\n".join(header_lines + chunk_rows)
            part_meta = dict(tbl_meta)
            part_meta["part"] = i // self.max_table_rows_per_chunk + 1
            part_meta["is_split"] = True
            chunks.append(
                StructuredChunk(
                    parent_id=parent_id,
                    paper_id=paper_id,
                    page=el.page,
                    section_path=el.section_path,
                    heading_level=el.heading_level,
                    element_type="table",
                    content=part_content,
                    metadata=part_meta,
                )
            )
        return chunks

    def _chunk_equation(
        self,
        el: DocumentElement,
        prev_p: str,
        next_p: str,
        parent_id: str,
        paper_id: str,
    ) -> StructuredChunk:
        """Equation chunking: bundles LaTeX equation + surrounding defining context."""
        parts = []
        if prev_p:
            parts.append(f"Context before:\n{prev_p[:300]}")
        
        eq_header = f"Equation {el.equation_ref}:" if el.equation_ref else "Equation:"
        parts.append(f"{eq_header}\n```latex\n{el.text}\n```")

        if next_p:
            parts.append(f"Context after (variable definitions):\n{next_p[:300]}")

        content = "\n\n".join(parts)
        return StructuredChunk(
            parent_id=parent_id,
            paper_id=paper_id,
            page=el.page,
            section_path=el.section_path,
            heading_level=el.heading_level,
            element_type="equation",
            content=content,
            metadata={
                "equation_ref": el.equation_ref,
                "raw_formula": el.text,
                "bbox": el.bbox,
                "reading_order": el.reading_order,
                "confidence": el.confidence.to_dict() if el.confidence else {},
                "is_certified": (el.metadata or {}).get("is_certified", True),
            },
        )

    def _chunk_figure(
        self,
        el: DocumentElement,
        prev_p: str,
        next_p: str,
        parent_id: str,
        paper_id: str,
    ) -> StructuredChunk:
        """Figure chunking: bundles caption, VLM description anchor, and context."""
        parts = []
        fig_ref = el.figure_ref or "Figure"
        parts.append(f"[{fig_ref}]")
        if el.caption or el.text:
            parts.append(f"Caption: {el.caption or el.text}")
        if prev_p:
            parts.append(f"Preceding context: {prev_p[:250]}")
        if next_p:
            parts.append(f"Following discussion: {next_p[:250]}")

        content = "\n\n".join(parts)
        fig_meta = {
            "figure_ref": el.figure_ref,
            "caption": el.caption or el.text,
            "bbox": el.bbox,
            "reading_order": el.reading_order,
            "confidence": el.confidence.to_dict() if el.confidence else {},
        }
        if el.canonical_figure:
            fig_meta["canonical_figure"] = el.canonical_figure.to_dict()
            fig_meta["figure_type"] = el.canonical_figure.figure_type.value
            if el.canonical_figure.plot_metadata:
                fig_meta["plot_metadata"] = el.canonical_figure.plot_metadata.__dict__

        return StructuredChunk(
            parent_id=parent_id,
            paper_id=paper_id,
            page=el.page,
            section_path=el.section_path,
            heading_level=el.heading_level,
            element_type="figure",
            content=content,
            metadata=fig_meta,
        )
