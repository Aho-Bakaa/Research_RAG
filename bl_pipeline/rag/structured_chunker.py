"""structured_chunker.py — Structure-preserving child-parent multimodal chunker.

Implements section 4 of RAG_ARCHITECTURE.md:
1. Atomic tables with repeated headers if split across row boundaries.
2. Equations bundled with symbol definitions, units, and defining paragraph.
3. Figure chunks bundling caption, surrounding paragraphs, and VLM description context.
4. Small-to-big parent-child linking: child chunks link to a section-level parent chunk.
"""
from __future__ import annotations

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


class StructurePreservingChunker:
    """Chunker that converts a stream of DocumentElements into StructuredChunks."""

    def __init__(
        self,
        max_table_rows_per_chunk: int = 25,
        max_paragraph_words: int = 400,
    ):
        self.max_table_rows_per_chunk = max_table_rows_per_chunk
        self.max_paragraph_words = max_paragraph_words

    def chunk_elements(
        self,
        elements: Sequence[DocumentElement],
        paper_id: str | None = None,
    ) -> list[StructuredChunk]:
        """Convert elements into parent and child chunks preserving relations."""
        if not elements:
            return []

        paper_id = paper_id or elements[0].paper_id or "unknown_paper"
        chunks: list[StructuredChunk] = []

        # 1. Group elements by section_path
        sections: dict[str, list[DocumentElement]] = {}
        for el in elements:
            sec = el.section_path or "Root"
            sections.setdefault(sec, []).append(el)

        # 2. Process each section: generate Parent chunk + specialized Child chunks
        for section_path, sec_elements in sections.items():
            first_el = sec_elements[0]
            heading_level = first_el.heading_level

            # Create Section-level Parent Chunk
            parent_chunk_id = str(uuid.uuid4())
            parent_text_parts = []
            for el in sec_elements:
                if el.text:
                    parent_text_parts.append(el.text)
            
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

            # Process individual elements in this section
            for idx, el in enumerate(sec_elements):
                if el.element_type == ElementType.HEADING.value:
                    continue  # Headings are represented in parent chunk and metadata

                elif el.element_type == ElementType.TABLE.value:
                    table_chunks = self._chunk_table(el, parent_chunk_id, paper_id)
                    chunks.extend(table_chunks)

                elif el.element_type == ElementType.EQUATION.value:
                    # Look for surrounding paragraph for context binding
                    prev_p = sec_elements[idx - 1].text if idx > 0 and sec_elements[idx - 1].element_type == ElementType.PARAGRAPH.value else ""
                    next_p = sec_elements[idx + 1].text if idx + 1 < len(sec_elements) and sec_elements[idx + 1].element_type == ElementType.PARAGRAPH.value else ""
                    
                    eq_chunk = self._chunk_equation(el, prev_p, next_p, parent_chunk_id, paper_id)
                    chunks.append(eq_chunk)

                elif el.element_type == ElementType.CAPTION.value or el.element_type == ElementType.FIGURE.value:
                    prev_p = sec_elements[idx - 1].text if idx > 0 and sec_elements[idx - 1].element_type == ElementType.PARAGRAPH.value else ""
                    next_p = sec_elements[idx + 1].text if idx + 1 < len(sec_elements) and sec_elements[idx + 1].element_type == ElementType.PARAGRAPH.value else ""
                    
                    fig_chunk = self._chunk_figure(el, prev_p, next_p, parent_chunk_id, paper_id)
                    chunks.append(fig_chunk)

                elif el.element_type == ElementType.PARAGRAPH.value:
                    p_chunk = StructuredChunk(
                        parent_id=parent_chunk_id,
                        paper_id=paper_id,
                        page=el.page,
                        section_path=section_path,
                        heading_level=heading_level,
                        element_type="paragraph",
                        content=el.text,
                        metadata={
                            "bbox": el.bbox,
                            "reading_order": el.reading_order,
                        },
                    )
                    chunks.append(p_chunk)

        return chunks

    def _chunk_table(
        self,
        el: DocumentElement,
        parent_id: str,
        paper_id: str,
    ) -> list[StructuredChunk]:
        """Atomic table chunking: repeat headers if table rows exceed threshold."""
        lines = [ln for ln in el.text.splitlines() if ln.strip()]
        if len(lines) <= self.max_table_rows_per_chunk + 2:
            return [
                StructuredChunk(
                    parent_id=parent_id,
                    paper_id=paper_id,
                    page=el.page,
                    section_path=el.section_path,
                    heading_level=el.heading_level,
                    element_type="table",
                    content=el.text,
                    metadata={
                        "table_structure": el.table_structure,
                        "bbox": el.bbox,
                        "reading_order": el.reading_order,
                    },
                )
            ]

        # Multi-part split with repeated header
        header_lines = lines[:2] if len(lines) >= 2 and "---" in lines[1] else lines[:1]
        data_rows = lines[len(header_lines):]

        chunks = []
        for i in range(0, len(data_rows), self.max_table_rows_per_chunk):
            chunk_rows = data_rows[i : i + self.max_table_rows_per_chunk]
            part_content = "\n".join(header_lines + chunk_rows)
            chunks.append(
                StructuredChunk(
                    parent_id=parent_id,
                    paper_id=paper_id,
                    page=el.page,
                    section_path=el.section_path,
                    heading_level=el.heading_level,
                    element_type="table",
                    content=part_content,
                    metadata={
                        "table_structure": el.table_structure,
                        "part": i // self.max_table_rows_per_chunk + 1,
                        "is_split": True,
                        "bbox": el.bbox,
                        "reading_order": el.reading_order,
                    },
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
        return StructuredChunk(
            parent_id=parent_id,
            paper_id=paper_id,
            page=el.page,
            section_path=el.section_path,
            heading_level=el.heading_level,
            element_type="figure",
            content=content,
            metadata={
                "figure_ref": el.figure_ref,
                "caption": el.caption or el.text,
                "bbox": el.bbox,
                "reading_order": el.reading_order,
            },
        )
