"""triage.py — Stage 0 Document and Page Triage Engine.

Evaluates continuous physical and structural page evidence:
- Character density & digital text layer quality
- Raster image coverage & estimated DPI
- Vector graphics path complexity & spatial clustering
Yields PageTriageResult with continuous confidence scores across DIGITAL, HYBRID, and SCANNED modes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import fitz

from bl_pipeline.rag.schema import PageMode, PageTriageResult


class DocumentTriageEngine:
    """Stage 0 Document & Page Triage Engine evaluating multi-modal evidence."""

    def __init__(
        self,
        digital_density_threshold: float = 1.5,
        scanned_density_threshold: float = 0.4,
        high_image_coverage_threshold: float = 0.55,
    ):
        self.digital_density_threshold = digital_density_threshold
        self.scanned_density_threshold = scanned_density_threshold
        self.high_image_coverage_threshold = high_image_coverage_threshold

    def triage_page(self, page: fitz.Page, page_num: int = 1) -> PageTriageResult:
        """Inspect a single PDF page and compute continuous evidence and classification."""
        rect = page.rect
        page_area = max(rect.width * rect.height, 1.0)
        area_kpt2 = page_area / 1000.0  # normalized per 1,000 pt^2

        # 1. Digital Text Analysis
        text = page.get_text()
        char_count = len(text.strip())
        text_density = round(char_count / max(area_kpt2, 0.1), 2)

        # Text layer quality: printable ratio and non-replacement characters
        if char_count > 0:
            printable_count = sum(1 for c in text if c.isprintable() and ord(c) >= 32 and ord(c) != 0xFFFD)
            printable_ratio = printable_count / char_count
            unique_chars = len(set(text))
            # Entropy approximation
            char_entropy = math.log2(max(unique_chars, 2)) / 8.0  # ~0.0 to 1.0
            text_layer_quality = round(min(1.0, printable_ratio * 0.7 + min(1.0, char_entropy) * 0.3), 3)
        else:
            text_layer_quality = 0.0

        # 2. Raster Image Coverage & Resolution
        image_infos = page.get_image_info()
        total_img_area = 0.0
        max_dpi = 72.0

        for img in image_infos:
            b = img.get("bbox", [0.0, 0.0, 0.0, 0.0])
            w_pt = max(b[2] - b[0], 1.0)
            h_pt = max(b[3] - b[1], 1.0)
            total_img_area += w_pt * h_pt

            # Estimate DPI if pixel width available
            px_w = img.get("width", 0)
            if px_w > 0:
                dpi = (px_w / w_pt) * 72.0
                if dpi > max_dpi:
                    max_dpi = dpi

        image_coverage = round(min(1.0, total_img_area / page_area), 3)
        raster_resolution_dpi = round(max_dpi, 1)

        # 3. Vector Graphics Complexity
        drawings = page.get_drawings()
        vector_count = len(drawings)
        vector_density = round(vector_count / max(area_kpt2, 0.1), 2)

        # Classify vector drawings: font/border outlines vs. complex plots
        rect_like = sum(1 for d in drawings if len(d.get("items", [])) <= 4)
        curve_like = sum(1 for d in drawings if len(d.get("items", [])) > 4)
        is_plot_likely = (curve_like > 15) and (vector_density > 2.0)

        # 4. Continuous Classifier & Confidence Calculation
        # Digital score: high text density + high text quality - image coverage
        digital_score = (
            min(1.0, text_density / self.digital_density_threshold) * 0.50
            + text_layer_quality * 0.35
            + (1.0 - image_coverage) * 0.15
        )

        # Scanned score: low text density + high image coverage
        scanned_score = (
            max(0.0, 1.0 - text_density / max(self.scanned_density_threshold, 0.1)) * 0.50
            + image_coverage * 0.50
        )

        if text_density >= self.digital_density_threshold and image_coverage < self.high_image_coverage_threshold:
            page_mode = PageMode.DIGITAL
            confidence = round(min(0.99, max(0.65, digital_score)), 3)
        elif text_density >= self.scanned_density_threshold and image_coverage >= self.high_image_coverage_threshold:
            # High text + High raster = Hybrid (searchable scanned layer OR digital text with full-page figures/plots)
            page_mode = PageMode.HYBRID
            confidence = round(min(0.98, max(0.70, (digital_score + image_coverage) / 2.0)), 3)
        elif text_density < self.scanned_density_threshold and image_coverage >= 0.30:
            page_mode = PageMode.SCANNED
            confidence = round(min(0.99, max(0.65, scanned_score)), 3)
        else:
            # Edge case: sparse vector page or title cover
            page_mode = PageMode.DIGITAL
            confidence = 0.60

        return PageTriageResult(
            page_num=page_num,
            page_mode=page_mode,
            confidence=confidence,
            text_density=text_density,
            image_coverage=image_coverage,
            vector_density=vector_density,
            text_layer_quality=text_layer_quality,
            raster_resolution_dpi=raster_resolution_dpi,
            metadata={
                "char_count": char_count,
                "images_count": len(image_infos),
                "vector_drawings_count": vector_count,
                "vector_curve_paths": curve_like,
                "vector_rect_paths": rect_like,
                "is_vector_plot_likely": is_plot_likely,
                "digital_evidence_score": round(digital_score, 3),
                "scanned_evidence_score": round(scanned_score, 3),
            },
        )

    def triage_document(self, pdf_path: str | Path) -> list[PageTriageResult]:
        """Triage all pages in a document and compute overall document-level profile."""
        doc = fitz.open(pdf_path)
        try:
            results = []
            for i, page in enumerate(doc):
                results.append(self.triage_page(page, page_num=i + 1))
            return results
        finally:
            doc.close()
