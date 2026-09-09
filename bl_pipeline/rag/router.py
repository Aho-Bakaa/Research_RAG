"""router.py — Query routing for multi-track scientific retrieval.

Implements section 6.1 of RAG_ARCHITECTURE.md:
Classifies user queries into optimal tracks ('text', 'table', 'figure', 'equation', 'graph')
and assigns track search weights.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class QueryRoute:
    """Routing decision for an incoming query."""
    primary_tracks: list[str] = field(default_factory=lambda: ["text", "equation"])
    is_structural_or_citation: bool = False
    section_scoped: str | None = None
    target_equation_ref: str | None = None
    target_figure_ref: str | None = None
    track_weights: dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "equation": 1.0,
        "table": 0.5,
        "figure": 0.4,
    })


# Fast heuristic pattern matching
_FIGURE_KEYWORDS = re.compile(r"\b(plot|figure|fig|curve|profile|distribution|graph|contour|visual)\b", re.I)
_TABLE_KEYWORDS = re.compile(r"\b(table|tabular|values|matrix|tabulated|dataset|constants|list of values)\b", re.I)
_EQUATION_KEYWORDS = re.compile(r"\b(equation|formula|eq|re_theta|re_xt|correlation|power law|exponent|decay law)\b", re.I)
_CITATION_KEYWORDS = re.compile(r"\b(who cites|citing|references|cited by|bibliography|prior work of|papers referencing)\b", re.I)
_SECTION_KEYWORDS = re.compile(r"\b(?:section|§)\s*([0-9]+(?:\.[0-9]+)*)", re.I)
_SPECIFIC_EQ_RX = re.compile(r"\b(?:eq(?:uation|\.)?|formula)\s*\(?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)?", re.I)
_SPECIFIC_FIG_RX = re.compile(r"\b(?:fig(?:ure|\.)?)\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\b", re.I)


class QueryRouter:
    """Lightweight rule-based and intent-driven query router."""

    def route_query(self, query: str) -> QueryRoute:
        q = query.strip()
        route = QueryRoute()
        tracks = set()

        # Check citation / structure queries
        if _CITATION_KEYWORDS.search(q):
            route.is_structural_or_citation = True
            tracks.add("graph")

        # Check section scoping
        sec_m = _SECTION_KEYWORDS.search(q)
        if sec_m:
            route.section_scoped = sec_m.group(1)
            route.is_structural_or_citation = True

        # Check specific equation ref
        eq_m = _SPECIFIC_EQ_RX.search(q)
        if eq_m:
            route.target_equation_ref = eq_m.group(1)
            tracks.add("equation")

        # Check specific figure ref
        fig_m = _SPECIFIC_FIG_RX.search(q)
        if fig_m:
            route.target_figure_ref = fig_m.group(1)
            tracks.add("figure")

        # Topic matches
        has_fig = bool(_FIGURE_KEYWORDS.search(q))
        has_tab = bool(_TABLE_KEYWORDS.search(q))
        has_eq = bool(_EQUATION_KEYWORDS.search(q))

        if has_fig:
            tracks.add("figure")
            route.track_weights["figure"] = 1.2

        if has_tab:
            tracks.add("table")
            route.track_weights["table"] = 1.2

        if has_eq:
            tracks.add("equation")
            route.track_weights["equation"] = 1.2

        # Always include text as fallback baseline
        tracks.add("text")
        if not has_fig and not has_tab and not has_eq:
            tracks.add("equation")

        route.primary_tracks = sorted(list(tracks))
        return route
