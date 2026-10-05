"""figure_analyzer.py — Scientific figure classifier, plot analyzer, and VLM dispatcher.

Classifies scientific figures (PLOT, SCHEMATIC, FLOWCHART, PHOTOGRAPH, DIAGRAM, etc.),
associates spatially adjacent captions, extracts approximate trends for plots (x-axis, y-axis, units),
and provides a pluggable VisionBackend interface.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from bl_pipeline.rag.schema import CanonicalFigure, FigureType, PlotMetadata

# Caption keywords indicating plot vs schematic vs photo
_PLOT_RX = re.compile(
    r"\b(?:vs\.?|versus|profile|distribution|variation|dependence|effect of|decay of|curve|measured|calculated|computed|function of)\b",
    re.IGNORECASE,
)
_SCHEMATIC_RX = re.compile(
    r"\b(?:schematic|setup|layout|apparatus|test section|geometry|grid|wind tunnel|model|diagram)\b",
    re.IGNORECASE,
)
_FLOWCHART_RX = re.compile(
    r"\b(?:flowchart|workflow|algorithm|decision tree|process flow|block diagram)\b",
    re.IGNORECASE,
)
_PHOTO_RX = re.compile(
    r"\b(?:photograph|visualization|schlieren|shadowgraph|smoke|oil flow|particle image|piv)\b",
    re.IGNORECASE,
)

# Figure identifier regex
_FIG_REF_RX = re.compile(r"^(?:Fig(?:ure|\.)?\s*([0-9]+[a-z]?))[:\.\s]+(.*)", re.IGNORECASE | re.DOTALL)


class VisionBackend:
    """Configurable interface for Multi-Modal Vision-Language Models."""

    def __init__(
        self,
        provider: str = "disabled",  # 'openai', 'groq', 'local', or 'disabled'
        model_name: str | None = None,
        custom_caller: Callable[[str, str], str] | None = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.custom_caller = custom_caller

    def analyze_crop(self, image_path: str | Path, prompt: str) -> str | None:
        """Analyze an image crop using the configured vision backend."""
        if self.provider == "disabled" and self.custom_caller is None:
            return None

        if self.custom_caller:
            try:
                return self.custom_caller(str(image_path), prompt)
            except Exception:
                return None

        # Return None if no live provider is active; avoids hardcoding deprecated models
        return None


class FigureAnalyzer:
    """Analyzes scientific figures, classifies figure types, and extracts plot metadata."""

    def __init__(self, vision_backend: VisionBackend | None = None):
        self.vision_backend = vision_backend or VisionBackend()

    def parse_caption_reference(self, caption_text: str) -> tuple[str | None, str]:
        """Separate figure reference (e.g. 'Fig. 1') from the description."""
        stripped = caption_text.strip()
        m = _FIG_REF_RX.match(stripped)
        if m:
            fig_ref = f"Fig. {m.group(1)}"
            desc = m.group(2).strip()
            return fig_ref, desc
        return None, stripped

    def classify_figure_type(self, caption_text: str, is_vector_likely: bool = False) -> FigureType:
        """Determine figure type based on caption semantics and vector evidence."""
        if not caption_text:
            return FigureType.PLOT if is_vector_likely else FigureType.OTHER

        text = caption_text.lower()
        if _PLOT_RX.search(text) or is_vector_likely:
            return FigureType.PLOT
        elif _FLOWCHART_RX.search(text):
            return FigureType.FLOWCHART
        elif _SCHEMATIC_RX.search(text):
            return FigureType.SCHEMATIC
        elif _PHOTO_RX.search(text):
            return FigureType.PHOTOGRAPH
        return FigureType.DIAGRAM

    def extract_plot_metadata_from_caption(self, caption_text: str) -> PlotMetadata:
        """Extract approximate axes, units, and directional trends deterministically from caption."""
        meta = PlotMetadata()
        text = caption_text.strip()

        # Check for 'Y vs X' pattern
        vs_match = re.search(r"([A-Za-z0-9_\-\s]+)\s+(?:vs\.?|versus)\s+([A-Za-z0-9_\-\s\(\)]+)", text, re.IGNORECASE)
        if vs_match:
            meta.y_axis = vs_match.group(1).strip()
            meta.x_axis = vs_match.group(2).strip()

        # Check for units in parentheses (e.g. [cm], (m/s), (deg))
        units_match = re.findall(r"[\(\[]([a-zA-Z0-9_\-/\s%°]+)[\)\]]", text)
        if units_match:
            meta.units = ", ".join(units_match)

        # Detect directional trends
        trends: list[str] = []
        if re.search(r"\b(?:decay|decrease|reduction|drop|fall)\b", text, re.IGNORECASE):
            trends.append("Approximate trend: monotonic decay or decrease across domain")
        if re.search(r"\b(?:growth|increase|rise|amplification)\b", text, re.IGNORECASE):
            trends.append("Approximate trend: monotonic increase or growth across domain")
        if re.search(r"\b(?:peak|maximum|optimum)\b", text, re.IGNORECASE):
            trends.append("Approximate trend: reaches distinct peak or local extremum")
        if re.search(r"\b(?:asymptotic|leveling|plateau|saturation)\b", text, re.IGNORECASE):
            trends.append("Approximate trend: approaches asymptotic plateau")

        meta.approximate_trends = trends
        return meta

    def analyze_figure(
        self,
        bbox: list[float],
        caption_text: str = "",
        image_crop_path: str | None = None,
        is_vector_likely: bool = False,
    ) -> CanonicalFigure:
        """Build CanonicalFigure with typed classification and approximate plot metadata."""
        fig_ref, clean_caption = self.parse_caption_reference(caption_text)
        fig_type = self.classify_figure_type(caption_text, is_vector_likely=is_vector_likely)

        plot_meta = None
        if fig_type == FigureType.PLOT:
            plot_meta = self.extract_plot_metadata_from_caption(caption_text)

        # Optional VLM call if crop and backend available
        vlm_desc = None
        if image_crop_path and self.vision_backend:
            vlm_desc = self.vision_backend.analyze_crop(
                image_crop_path,
                prompt=f"Describe the key visual relationships in this {fig_type.value.lower()}: {caption_text}",
            )

        return CanonicalFigure(
            figure_type=fig_type,
            bbox=bbox,
            caption=caption_text if caption_text else None,
            image_crop_path=image_crop_path,
            plot_metadata=plot_meta,
            vlm_description=vlm_desc,
        )
