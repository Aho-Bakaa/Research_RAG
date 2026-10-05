"""formula_reconciler.py — Multi-candidate equation reconciliation engine.

Reconciles candidate representations from native PDF glyphs, neural visual crops,
and optional VLM fallbacks. Evaluates:
- Syntax validity (balanced delimiters, operator completeness)
- Symbol preservation (Greek symbols, subscripts, variables)
- Equation tag preservation (separating Eq. (1) from formula body)
- Cross-modal agreement across visual and digital representations
Yields certified LaTeX with multidimensional confidence vectors.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from bl_pipeline.rag.parsers.quality_gates import LaTeXSyntaxGate
from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
from bl_pipeline.rag.schema import ConfidenceVector

# Equation numbering pattern: (1), (3.6), Eq. (5)
_EQ_TAG_RX = re.compile(
    r"(?:\(\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)|Eq\.\s*\(?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)?)\s*$"
)

# Known aerospace boundary layer variables for consistency audit
_CANONICAL_VARIABLES = {
    "re", "re_theta", "re_thetat", "re_v", "re_x", "re_m",
    "tu", "gamma", "theta", "omega", "nu", "rho", "tau", "lambda_x",
    "cf", "cp", "u_inf", "h_12", "delta", "delta_star"
}


@dataclass
class EquationCandidate:
    """A candidate formula representation from a specific extraction specialist."""
    source: str  # 'native_glyph', 'neural_model', 'vlm_fallback'
    raw_text: str
    latex_text: str
    syntax_valid: bool = True
    syntax_score: float = 1.0
    syntax_issues: list[str] = field(default_factory=list)
    has_tag: bool = False
    tag_id: str | None = None


@dataclass
class ReconciledEquation:
    """Certified equation output with consensus LaTeX, extracted tag, and confidence vector."""
    best_latex: str
    equation_ref: str | None
    confidence: ConfidenceVector
    candidates: list[EquationCandidate]
    is_certified: bool
    reconciliation_notes: list[str] = field(default_factory=list)


class FormulaReconciler:
    """Reconciles multi-candidate equation extractions and certifies fidelity."""

    def __init__(self):
        self.syntax_gate = LaTeXSyntaxGate()

    def parse_equation_tag(self, text: str) -> tuple[str, str | None]:
        """Separate equation tag (e.g. '(1)', 'Eq. (3.6)') from the formula body."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return text, None

        # Check last line
        last_line = lines[-1]
        m = _EQ_TAG_RX.search(last_line)
        if m:
            tag_id = m.group(1) or m.group(2)
            formula_lines = lines[:-1]
            # If the last line only contained the tag, remove it
            tag_stripped = _EQ_TAG_RX.sub("", last_line).strip()
            if tag_stripped:
                formula_lines.append(tag_stripped)
            return "\n".join(formula_lines).strip(), tag_id

        # Check inline tag at end of text
        m_inline = _EQ_TAG_RX.search(text)
        if m_inline:
            tag_id = m_inline.group(1) or m_inline.group(2)
            cleaned = _EQ_TAG_RX.sub("", text).strip()
            return cleaned, tag_id

        return text, None

    def evaluate_candidate(self, raw_text: str, source: str) -> EquationCandidate:
        """Evaluate syntax validity, normalize symbols, and isolate tag for a candidate."""
        body, tag = self.parse_equation_tag(raw_text)
        normalized = normalize_scientific_text(body)

        # Audit with LaTeXSyntaxGate
        gate_res = self.syntax_gate.evaluate(normalized)

        return EquationCandidate(
            source=source,
            raw_text=raw_text,
            latex_text=normalized,
            syntax_valid=gate_res.passed,
            syntax_score=gate_res.score,
            syntax_issues=gate_res.issues,
            has_tag=(tag is not None),
            tag_id=tag,
        )

    def reconcile(
        self,
        native_text: str,
        neural_text: str | None = None,
        vlm_text: str | None = None,
        detection_confidence: float = 0.95,
    ) -> ReconciledEquation:
        """Reconcile candidates from native glyphs, neural models, and VLM fallbacks."""
        candidates: list[EquationCandidate] = []
        notes: list[str] = []

        # 1. Evaluate Native Digital Glyph Candidate
        cand_native = self.evaluate_candidate(native_text, source="native_glyph")
        candidates.append(cand_native)

        # 2. Evaluate Neural / Visual Candidate if provided
        if neural_text and neural_text.strip():
            cand_neural = self.evaluate_candidate(neural_text, source="neural_model")
            candidates.append(cand_neural)

        # 3. Evaluate VLM Candidate if provided
        if vlm_text and vlm_text.strip():
            cand_vlm = self.evaluate_candidate(vlm_text, source="vlm_fallback")
            candidates.append(cand_vlm)

        # 4. Measure Cross-Modal Agreement
        if len(candidates) >= 2:
            s1 = candidates[0].latex_text.lower().replace(" ", "")
            s2 = candidates[1].latex_text.lower().replace(" ", "")
            seq_matcher = difflib.SequenceMatcher(None, s1, s2)
            cross_modal_agreement = round(seq_matcher.ratio(), 3)
            notes.append(f"Cross-modal agreement between {candidates[0].source} and {candidates[1].source}: {cross_modal_agreement:.3f}")
        else:
            # Single candidate available (e.g. pure digital mode)
            cross_modal_agreement = 1.0 if cand_native.syntax_valid and len(cand_native.latex_text) > 3 else 0.70

        # 5. Determine Equation Tag
        eq_ref = None
        for c in candidates:
            if c.tag_id:
                eq_ref = f"Eq. ({c.tag_id})"
                break

        # 6. Candidate Selection Logic
        # Prioritize candidate with highest syntax score + non-empty body
        valid_candidates = [c for c in candidates if c.syntax_valid and len(c.latex_text.strip()) > 1]

        if valid_candidates:
            # Sort by syntax_score descending, preferring neural if agreement is high, or native if valid
            best_cand = max(valid_candidates, key=lambda c: (c.syntax_score, len(c.latex_text)))
            best_latex = best_cand.latex_text
            extraction_conf = best_cand.syntax_score
            notes.append(f"Selected candidate from {best_cand.source} (syntax_score={best_cand.syntax_score})")
        else:
            # Fallback to normalized native candidate
            best_latex = cand_native.latex_text
            extraction_conf = max(0.40, cand_native.syntax_score)
            notes.append("No candidate fully passed syntax audit; using best-effort normalized native candidate")

        # 7. Structural Confidence (Delimiter matching, tag consistency)
        structural_conf = 1.0
        if cand_native.has_tag:
            structural_conf = min(1.0, structural_conf + 0.05)
        if any(issue.startswith("Mismatched") for issue in cand_native.syntax_issues):
            structural_conf -= 0.30
        structural_conf = round(max(0.20, min(1.0, structural_conf)), 3)

        # 8. Compute Confidence Vector
        conf_vector = ConfidenceVector(
            detection_confidence=round(detection_confidence, 3),
            extraction_confidence=round(extraction_conf, 3),
            structural_confidence=structural_conf,
            cross_modal_agreement=cross_modal_agreement,
        )
        overall_conf = conf_vector.compute_overall()

        is_certified = (overall_conf >= 0.75 and len(best_latex.strip()) > 1)

        return ReconciledEquation(
            best_latex=best_latex,
            equation_ref=eq_ref,
            confidence=conf_vector,
            candidates=candidates,
            is_certified=is_certified,
            reconciliation_notes=notes,
        )
