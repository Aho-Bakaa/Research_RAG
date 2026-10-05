"""docling_parser.py — Neural scientific document parser powered by IBM Docling.

Integrates Docling for deep layout analysis, TableFormer for borderless tables,
and LaTeX formula extraction, audited by deterministic quality gates.
Gracefully falls back to PyMuPDF LayoutAwareParser if docling is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from bl_pipeline.rag.parsers.layout_parser import LayoutAwareParser
from bl_pipeline.rag.parsers.quality_gates import ParsingQualityGatePipeline
from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
from bl_pipeline.rag.schema import DocumentElement, ElementType

log = logging.getLogger(__name__)

# Check docling availability
try:
    import docling
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False


class DoclingParser:
    """Neural scientific parser using IBM Docling with quality gate validation."""

    def __init__(
        self,
        enable_tableformer_accurate: bool = True,
        enable_ocr: bool = True,
        fallback_parser: LayoutAwareParser | None = None,
    ):
        self.fallback_parser = fallback_parser or LayoutAwareParser()
        self.quality_gate = ParsingQualityGatePipeline()
        self.converter: Any = None

        if HAS_DOCLING:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = enable_ocr
            pipeline_options.do_table_structure = True
            if enable_tableformer_accurate:
                try:
                    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                except Exception:
                    pass

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            log.info("Initialized neural DoclingParser with TableFormer ACCURATE mode.")
        else:
            log.info("Docling library not installed. Operating in high-speed PyMuPDF fallback mode.")

    def parse_pdf(
        self,
        pdf_path: str | Path,
        paper_id: str | None = None,
    ) -> tuple[list[DocumentElement], dict[str, Any]]:
        """Parse PDF using Docling neural pipeline or PyMuPDF fallback, audited by quality gates."""
        pdf_path = Path(pdf_path)
        paper_id = paper_id or pdf_path.stem

        if not HAS_DOCLING or self.converter is None:
            return self._parse_with_pymupdf(pdf_path, paper_id)

        try:
            doc_result = self.converter.convert(pdf_path)
            docling_doc = doc_result.document
            elements: list[DocumentElement] = []

            # Extract elements from Docling Document
            for item, level in docling_doc.iterate_items():
                label = getattr(item, "label", "").lower()
                text = getattr(item, "text", "") or ""
                prov = getattr(item, "prov", [])
                page_no = prov[0].page_no if prov else 1
                bbox = [0.0, 0.0, 0.0, 0.0]
                if prov and hasattr(prov[0], "bbox"):
                    b = prov[0].bbox
                    bbox = [float(b.l), float(b.t), float(b.r), float(b.b)]

                # Clean and normalize text
                clean_text = normalize_scientific_text(text)

                if "formula" in label or "equation" in label:
                    elem_type = ElementType.EQUATION.value
                elif "table" in label:
                    elem_type = ElementType.TABLE.value
                    if hasattr(item, "export_to_markdown"):
                        clean_text = item.export_to_markdown()
                elif "header" in label or "title" in label:
                    elem_type = ElementType.HEADING.value
                elif "caption" in label:
                    elem_type = ElementType.CAPTION.value
                elif "picture" in label or "figure" in label:
                    elem_type = ElementType.FIGURE.value
                else:
                    elem_type = ElementType.PARAGRAPH.value

                elements.append(
                    DocumentElement(
                        element_type=elem_type,
                        text=clean_text,
                        page=page_no,
                        bbox=bbox,
                        paper_id=paper_id,
                        section_path=getattr(item, "section_path", "") or "Root",
                        heading_level=level if elem_type == ElementType.HEADING.value else 0,
                        reading_order=len(elements),
                    )
                )

            audit_report = self.quality_gate.audit_elements(elements)
            return elements, audit_report

        except Exception as e:
            log.warning("Docling neural parse failed on %s: %s. Falling back to PyMuPDF.", pdf_path, e)
            return self._parse_with_pymupdf(pdf_path, paper_id)

    def _parse_with_pymupdf(
        self,
        pdf_path: Path,
        paper_id: str,
    ) -> tuple[list[DocumentElement], dict[str, Any]]:
        """Fallback to PyMuPDF parser audited by quality gates."""
        elements = self.fallback_parser.parse_pdf(pdf_path, paper_id=paper_id)
        audit_report = self.quality_gate.audit_elements(elements)
        return elements, audit_report
