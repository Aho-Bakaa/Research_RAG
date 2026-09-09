"""crag_gate.py — Corrective RAG (CRAG) retrieval quality gate and re-query loop.

Implements section 6.3 of RAG_ARCHITECTURE.md:
- Judges whether retrieved chunks are sufficient to answer the scientific query.
- Detects physics gaps (missing decay laws, onset correlations, length scale formulas).
- Suggests structured rewrites when retrieval is weak.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

log = logging.getLogger(__name__)


@dataclass
class CRAGVerdict:
    """Outcome of the retrieval quality gate."""
    is_sufficient: bool = True
    confidence: float = 1.0
    detected_gaps: list[str] = field(default_factory=list)
    suggested_rewrites: list[str] = field(default_factory=list)
    verdict: str = "PASS"  # PASS | INSUFFICIENT | IRRELEVANT


class RetrievalQualityGate:
    """Evaluates candidate chunks and triggers corrective query rewrites if gaps exist."""

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: Sequence[dict[str, Any]],
        flow_conditions: Any = None,
    ) -> CRAGVerdict:
        """Evaluate retrieval coverage against the physics requirements of the query."""
        if not retrieved_chunks:
            return CRAGVerdict(
                is_sufficient=False,
                confidence=0.0,
                detected_gaps=["No chunks retrieved"],
                suggested_rewrites=[f"{query} transition onset correlation Re_theta"],
                verdict="INSUFFICIENT",
            )

        all_text = " ".join(
            str(c.get("content", "") or c.get("payload", {}).get("content", ""))
            for c in retrieved_chunks
        ).lower()

        gaps: list[str] = []
        rewrites: list[str] = []

        # Check 1: Transition Onset Correlation
        has_onset = any(term in all_text for term in ["re_theta", "re_xt", "re_x,t", "transition onset", "start of transition"])
        if not has_onset:
            gaps.append("Missing transition onset correlation formula (e.g., AGS or Mayle)")
            rewrites.append("Abu-Ghannam Shaw Mayle transition onset correlation Re_theta_t")

        # Check 2: Free-stream turbulence / decay handling
        has_decay = any(term in all_text for term in ["decay", "grid", "tu(x)", "fransson", "comte-bellot", "mesh size"])
        has_tu_in_query = "tu" in query.lower() or "grid" in query.lower() or "turbulence" in query.lower()
        if has_tu_in_query and not has_decay:
            gaps.append("Missing turbulence decay formula (e.g., Fransson Eq. 3.1 or Comte-Bellot)")
            rewrites.append("grid turbulence decay law Tu(x) Fransson mesh size")

        # Check 3: Length-scale aware correlation if Lambda_x mentioned
        lambda_in_query = "lambda" in query.lower() or "length scale" in query.lower()
        has_fs20 = "shahinfar" in all_text or "lambda_x" in all_text or "re_fst" in all_text
        if lambda_in_query and not has_fs20:
            gaps.append("Missing integral length scale Lambda_x transition correlation (Fransson-Shahinfar 2020 Eq. 3.6)")
            rewrites.append("fransson_shahinfar_2020 Eq.3.6 Re_FST Lambda_x integral length scale")

        if gaps:
            return CRAGVerdict(
                is_sufficient=False,
                confidence=round(1.0 - (len(gaps) * 0.25), 2),
                detected_gaps=gaps,
                suggested_rewrites=rewrites,
                verdict="INSUFFICIENT",
            )

        return CRAGVerdict(
            is_sufficient=True,
            confidence=0.95,
            verdict="PASS",
        )
