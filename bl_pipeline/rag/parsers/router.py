"""router.py — Stage 1 to Stage 4 ScientificDocumentRouter orchestrator.

Implements the research-grade 6-stage scientific parsing lifecycle:
1. Inspect: Stage 0 Page Triage (digital vs hybrid vs scanned).
2. Detect: Neural Layout Detection (Docling layout model, e.g. Heron).
3. Route: Modality-Dedicated Dispatcher (digital text, formula reconciler, cell dissector, figure analyzer, code extractor, noise filter).
4. Extract: Bounded digital glyph extraction, TableFormer 2D grid, LaTeX formatting.
5. Reconcile: Canonical tree assembly, reading order stitching, provenance preservation.
6. Certify: Deterministic quality gates, reference-link verification, and confidence calibration.
"""
from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Sequence

import fitz

from bl_pipeline.rag.parsers.cell_dissector import TableDissector
from bl_pipeline.rag.parsers.figure_analyzer import FigureAnalyzer, VisionBackend
from bl_pipeline.rag.parsers.formula_reconciler import FormulaReconciler
from bl_pipeline.rag.parsers.quality_gates import ParsingQualityGatePipeline
from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
from bl_pipeline.rag.parsers.triage import DocumentTriageEngine
from bl_pipeline.rag.schema import (
    ConfidenceVector,
    DocumentElement,
    ElementType,
    PageMode,
    PageTriageResult,
)

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


class ScientificDocumentRouter:
    """Master orchestrator implementing inspect -> detect -> route -> extract -> reconcile -> certify."""

    def __init__(
        self,
        vision_backend: VisionBackend | None = None,
        enable_tableformer_fast: bool = True,
    ):
        self.triage_engine = DocumentTriageEngine()
        self.formula_reconciler = FormulaReconciler()
        self.table_dissector = TableDissector()
        self.figure_analyzer = FigureAnalyzer(vision_backend=vision_backend)
        self.quality_gate = ParsingQualityGatePipeline()

        self.converter: Any = None
        if HAS_DOCLING:
            opts = PdfPipelineOptions()
            opts.do_ocr = False  # Digital vector stream first; zero OCR text pollution
            opts.do_table_structure = True
            opts.generate_page_images = False
            opts.generate_table_images = False
            opts.generate_picture_images = False
            if enable_tableformer_fast:
                try:
                    opts.table_structure_options.mode = TableFormerMode.FAST
                except Exception:
                    pass

            self.converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
            log.info("Initialized ScientificDocumentRouter with Docling Neural Layout & TableFormer.")
        else:
            log.warning("Docling not available; operating in fallback layout mode.")

    def parse_pdf(
        self,
        pdf_path: str | Path,
        paper_id: str | None = None,
    ) -> tuple[list[DocumentElement], dict[str, Any]]:
        """Execute the full 6-stage lifecycle across the document."""
        pdf_path = Path(pdf_path)
        paper_id = paper_id or pdf_path.stem

        # =====================================================================
        # STAGE 0: INSPECT (Document & Page Triage)
        # =====================================================================
        triage_results = self.triage_engine.triage_document(pdf_path)
        triage_by_page = {r.page_num: r for r in triage_results}

        # Open PDF via PyMuPDF for precise digital glyph extraction
        doc = fitz.open(pdf_path)
        elements: list[DocumentElement] = []

        try:
            # =================================================================
            # STAGE 1: DETECT (Neural Layout Detection via Docling)
            # =================================================================
            if HAS_DOCLING and self.converter is not None:
                doc_res = self.converter.convert(pdf_path)
                docling_doc = doc_res.document

                # Track captions to pair with figures
                captions_by_page: dict[int, list[tuple[list[float], str]]] = {}
                for item, _ in docling_doc.iterate_items():
                    lbl = getattr(item, "label", "").lower()
                    if "caption" in lbl:
                        prov = getattr(item, "prov", [])
                        p_no = prov[0].page_no if prov else 1
                        txt = getattr(item, "text", "") or ""
                        b_l = [0.0, 0.0, 0.0, 0.0]
                        if prov and hasattr(prov[0], "bbox"):
                            b = prov[0].bbox
                            b_l = [float(b.l), float(b.t), float(b.r), float(b.b)]
                        captions_by_page.setdefault(p_no, []).append((b_l, txt))

                current_section = "Root"
                current_heading_level = 0

                # =============================================================
                # STAGE 2 & 3: ROUTE, EXTRACT & RECONCILE
                # =============================================================
                for item, lvl in docling_doc.iterate_items():
                    label = str(getattr(item, "label", "")).lower()
                    prov = getattr(item, "prov", [])
                    page_no = prov[0].page_no if prov else 1
                    page = doc[page_no - 1]
                    p_height = page.rect.height
                    p_width = page.rect.width

                    # Convert Docling bottom-left bbox to PyMuPDF top-left bbox
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    if prov and hasattr(prov[0], "bbox"):
                        b = prov[0].bbox
                        # Docling: l, b, r, t (where b is bottom, t is top in bottom-left origin)
                        x0 = float(b.l)
                        x1 = float(b.r)
                        y0 = p_height - max(float(b.t), float(b.b))
                        y1 = p_height - min(float(b.t), float(b.b))
                        bbox = [round(x0, 2), round(max(0.0, y0), 2), round(x1, 2), round(min(p_height, y1), 2)]

                    clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                    triage_info = triage_by_page.get(page_no)

                    # ---------------------------------------------------------
                    # ROUTE: Running Headers & Footers (Noise Filter)
                    # ---------------------------------------------------------
                    if "header" in label and "section" not in label and "title" not in label:
                        raw_h = page.get_text("text", clip=clip_rect).strip()
                        elements.append(
                            DocumentElement(
                                element_type=ElementType.PAGE_HEADER.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=0,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=raw_h,
                                suppressed=True,
                                suppression_reason="PAGE_HEADER",
                                confidence=ConfidenceVector(detection_confidence=0.98),
                            )
                        )
                        continue

                    if "footer" in label:
                        raw_f = page.get_text("text", clip=clip_rect).strip()
                        elements.append(
                            DocumentElement(
                                element_type=ElementType.PAGE_FOOTER.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=0,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=raw_f,
                                suppressed=True,
                                suppression_reason="PAGE_FOOTER",
                                confidence=ConfidenceVector(detection_confidence=0.98),
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Headings & Titles
                    # ---------------------------------------------------------
                    if "section_header" in label or "title" in label or label == "heading":
                        raw_heading = page.get_text("text", clip=clip_rect).strip()
                        if not raw_heading:
                            raw_heading = getattr(item, "text", "") or ""
                        norm_heading = normalize_scientific_text(raw_heading)

                        current_section = norm_heading
                        current_heading_level = lvl if lvl > 0 else 1

                        elements.append(
                            DocumentElement(
                                element_type=ElementType.HEADING.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=norm_heading,
                                confidence=ConfidenceVector(
                                    detection_confidence=0.95,
                                    extraction_confidence=1.0,
                                    structural_confidence=1.0,
                                ),
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Standalone Equations & Formulas
                    # ---------------------------------------------------------
                    if "formula" in label or "equation" in label:
                        # Extract exact digital glyphs strictly bounded within formula bbox
                        native_formula_raw = page.get_text("text", clip=clip_rect).strip()
                        neural_text = getattr(item, "text", "") or None

                        reconciled = self.formula_reconciler.reconcile(
                            native_text=native_formula_raw,
                            neural_text=neural_text,
                            detection_confidence=0.95,
                        )

                        elements.append(
                            DocumentElement(
                                element_type=ElementType.EQUATION.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=reconciled.best_latex,
                                equation_ref=reconciled.equation_ref,
                                confidence=reconciled.confidence,
                                metadata={
                                    "is_certified": reconciled.is_certified,
                                    "reconciliation_notes": reconciled.reconciliation_notes,
                                },
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Tables & Mixed Regions
                    # ---------------------------------------------------------
                    if "table" in label:
                        canonical_tbl = self.table_dissector.dissect_docling_table(
                            table_item=item,
                            page=page,
                            caption=current_section,
                        )
                        md_table = canonical_tbl.to_markdown()

                        elements.append(
                            DocumentElement(
                                element_type=ElementType.TABLE.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=md_table,
                                canonical_table=canonical_tbl,
                                confidence=ConfidenceVector(
                                    detection_confidence=0.96,
                                    structural_confidence=0.95 if canonical_tbl.num_rows > 1 else 0.50,
                                ),
                                metadata={
                                    "num_rows": canonical_tbl.num_rows,
                                    "num_cols": canonical_tbl.num_cols,
                                    "summary": canonical_tbl.to_natural_language_summary(),
                                },
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Figures, Charts & Plots
                    # ---------------------------------------------------------
                    if "picture" in label or "figure" in label or "chart" in label:
                        # Find closest caption on same page
                        closest_caption = ""
                        p_caps = captions_by_page.get(page_no, [])
                        if p_caps:
                            # Pick caption closest vertically below the picture
                            valid_below = [(c_box, c_txt) for c_box, c_txt in p_caps if c_box[1] >= bbox[3] - 20.0]
                            if valid_below:
                                closest_caption = valid_below[0][1]
                            else:
                                closest_caption = p_caps[0][1]

                        is_vec = triage_info.metadata.get("is_vector_plot_likely", False) if triage_info else False
                        canonical_fig = self.figure_analyzer.analyze_figure(
                            bbox=bbox,
                            caption_text=closest_caption,
                            is_vector_likely=is_vec,
                        )

                        fig_content = f"[{canonical_fig.figure_type.value}]"
                        if canonical_fig.caption:
                            fig_content += f" Caption: {canonical_fig.caption}"
                        if canonical_fig.plot_metadata and canonical_fig.plot_metadata.approximate_trends:
                            fig_content += f"\nTrends: {'; '.join(canonical_fig.plot_metadata.approximate_trends)}"

                        elements.append(
                            DocumentElement(
                                element_type=ElementType.FIGURE.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=fig_content,
                                caption=closest_caption,
                                canonical_figure=canonical_fig,
                                confidence=ConfidenceVector(
                                    detection_confidence=0.94,
                                    structural_confidence=0.90,
                                ),
                                metadata={"figure_type": canonical_fig.figure_type.value},
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Algorithms, Pseudocode & Code Blocks
                    # ---------------------------------------------------------
                    if "code" in label:
                        raw_code = page.get_text("text", clip=clip_rect)
                        if not raw_code:
                            raw_code = getattr(item, "text", "") or ""

                        # Classify algorithm vs pseudocode vs code
                        if re.search(r"\b(?:algorithm|procedure|input:|output:)\b", raw_code, re.IGNORECASE):
                            c_type = ElementType.ALGORITHM.value
                        elif re.search(r"\b(?:while|for each|repeat until)\b", raw_code, re.IGNORECASE):
                            c_type = ElementType.PSEUDOCODE.value
                        else:
                            c_type = ElementType.CODE.value

                        elements.append(
                            DocumentElement(
                                element_type=c_type,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=raw_code,
                                confidence=ConfidenceVector(detection_confidence=0.92, extraction_confidence=0.95),
                            )
                        )
                        continue

                    # ---------------------------------------------------------
                    # ROUTE: Algorithms, Pseudocode or Standard Paragraph Prose
                    # ---------------------------------------------------------
                    raw_p = page.get_text("text", clip=clip_rect).strip()
                    if not raw_p:
                        raw_p = getattr(item, "text", "") or ""
                    norm_p = normalize_scientific_text(raw_p)

                    is_algo = bool(re.search(r"^(?:algorithm\s+\d+|procedure\s+\w+|input\s*:|output\s*:)", norm_p, re.IGNORECASE))
                    is_pseudo = bool(re.search(r"\b(?:step\s+1\s*:|while\s+.*\bdo\b|for\s+each\s+.*\bdo\b|repeat\s+until\b)", norm_p, re.IGNORECASE))
                    if is_algo or is_pseudo:
                        elements.append(
                            DocumentElement(
                                element_type=ElementType.ALGORITHM.value if is_algo else ElementType.PSEUDOCODE.value,
                                paper_id=paper_id,
                                page=page_no,
                                section_path=current_section,
                                heading_level=current_heading_level,
                                bbox=bbox,
                                reading_order=len(elements),
                                text=norm_p,
                                confidence=ConfidenceVector(
                                    detection_confidence=0.92,
                                    extraction_confidence=0.98,
                                    structural_confidence=0.95,
                                ),
                            )
                        )
                        continue

                    elements.append(
                        DocumentElement(
                            element_type=ElementType.PARAGRAPH.value,
                            paper_id=paper_id,
                            page=page_no,
                            section_path=current_section,
                            heading_level=current_heading_level,
                            bbox=bbox,
                            reading_order=len(elements),
                            text=norm_p,
                            confidence=ConfidenceVector(
                                detection_confidence=0.95,
                                extraction_confidence=0.98,
                                structural_confidence=1.0,
                            ),
                        )
                    )

                # -------------------------------------------------------------
                # STAGE 3 (Reconciliation): Intercept any uncaptured margin noise
                # (Running headers, footers, page numbers) for 100% auditability
                # -------------------------------------------------------------
                for page_idx in range(len(doc)):
                    page_no = page_idx + 1
                    page = doc[page_idx]
                    p_height = page.rect.height
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        if len(b) > 6 and b[6] != 0:
                            continue
                        txt = b[4].strip()
                        if not txt:
                            continue
                        y0, y1 = b[1], b[3]
                        is_top_margin = y1 <= p_height * 0.08
                        is_bottom_margin = y0 >= p_height * 0.92
                        if is_top_margin or is_bottom_margin:
                            b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
                            already_covered = any(
                                e.page == page_no and fitz.Rect(e.bbox).intersects(b_rect)
                                for e in elements
                            )
                            if not already_covered:
                                elem_type = ElementType.PAGE_HEADER.value if is_top_margin else ElementType.PAGE_FOOTER.value
                                elements.append(
                                    DocumentElement(
                                        element_type=elem_type,
                                        paper_id=paper_id,
                                        page=page_no,
                                        section_path="Root",
                                        heading_level=0,
                                        bbox=[round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                                        reading_order=len(elements),
                                        text=txt,
                                        suppressed=True,
                                        suppression_reason="PAGE_HEADER" if is_top_margin else "PAGE_FOOTER",
                                        confidence=ConfidenceVector(detection_confidence=0.98),
                                    )
                                )

            # Fallback if Docling not installed
            else:
                log.info("Parsing via PyMuPDF fallback layout engine...")
                from bl_pipeline.rag.parsers.layout_parser import LayoutAwareParser
                elements = LayoutAwareParser().parse_pdf(pdf_path, paper_id=paper_id)

            # =================================================================
            # STAGE 4: CERTIFY (Deterministic Quality Gate Certification)
            # =================================================================
            audit_report = self.quality_gate.audit_elements(elements)
            return elements, audit_report

        finally:
            doc.close()
