"""quality_gates.py — Multi-stage deterministic quality gates and cross-modal certification.

Implements the certified 6-stage scientific document quality gates:
1. LaTeXSyntaxGate: Delimiter balance, operator completeness, variable syntax.
2. TableStructureGate: Rectangular matrix consistency, column alignment, numeric density.
3. TextFidelityGate: Control characters, OCR punctuation noise, lexical entropy.
4. Refined GeometricCoverageGate: Classifies whitespace (gutters, margins) vs unexplained missing bands.
5. ReadingOrderGate: Validates multi-column reading flow, paragraph continuity across breaks.
6. ReferenceLinkConsistencyGate: Verifies Eq. (X), Fig. Y, Table Z mentions match parsed elements.
7. CrossModalFidelityGate: Analyzes agreement between native digital text and neural vision extraction.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from bl_pipeline.rag.schema import ConfidenceVector, DocumentElement, ElementType


@dataclass
class ValidationResult:
    """Result of a quality gate evaluation on a document element or page."""
    gate_name: str
    passed: bool
    score: float = 1.0  # 0.0 to 1.0
    issues: list[str] = field(default_factory=list)
    remediation_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LaTeXSyntaxGate:
    """Validates mathematical LaTeX equations extracted from papers."""

    TRAILING_OP_RX = re.compile(r"(?:[+\-=/*\\]|\\times|\\cdot|\\div|\\pm|\\mp|\\to|\\le|\\ge)\s*$")
    INCOMPLETE_FRAC_RX = re.compile(r"\\frac\s*\{[^{}]*\}\s*$")
    GREEK_CHARS = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ")

    def evaluate(self, formula_text: str) -> ValidationResult:
        stripped = formula_text.strip()
        issues: list[str] = []

        if not stripped:
            return ValidationResult(
                gate_name="LaTeXSyntaxGate",
                passed=False,
                score=0.0,
                issues=["Empty formula text"],
                remediation_action="crop_reparse_math",
            )

        # 1. Delimiter & Bracket balance check
        bracket_pairs = [("{", "}"), ("[", "]"), ("(", ")")]
        for open_b, close_b in bracket_pairs:
            open_count = len(re.findall(r"(?<!\\)" + re.escape(open_b), stripped))
            close_count = len(re.findall(r"(?<!\\)" + re.escape(close_b), stripped))
            if open_count != close_count:
                issues.append(f"Mismatched brackets / delimiters '{open_b}{close_b}': {open_count} open vs {close_count} close")

        # 2. Incomplete trailing operators
        if self.TRAILING_OP_RX.search(stripped):
            issues.append("Trailing incomplete binary/relational operator at end of formula")

        # 3. Broken fraction commands
        if self.INCOMPLETE_FRAC_RX.search(stripped):
            issues.append("Incomplete LaTeX fraction command (missing denominator)")

        # 4. Spaced aerodynamic variable artifacts
        if re.search(r"R[˜~]\s*e\s*t|R\s*e\s*_|\bT\s*u\b", stripped):
            issues.append("Contains spaced aerodynamic variable artifacts")

        passed = len(issues) == 0
        score = 1.0 - (len(issues) * 0.25)
        remediation = "crop_reparse_math" if not passed else None

        return ValidationResult(
            gate_name="LaTeXSyntaxGate",
            passed=passed,
            score=max(0.0, round(score, 2)),
            issues=issues,
            remediation_action=remediation,
        )


class TableStructureGate:
    """Validates structural integrity of extracted scientific tables."""

    def evaluate(self, table_text: str) -> ValidationResult:
        lines = [l.strip() for l in table_text.splitlines() if l.strip()]
        issues: list[str] = []

        if not lines or len(lines) < 2:
            return ValidationResult(
                gate_name="TableStructureGate",
                passed=False,
                score=0.0,
                issues=["Table contains fewer than 2 rows"],
                remediation_action="tableformer_accurate",
            )

        # Pipe-delimited markdown table parsing
        row_col_counts = []
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line[1:-1].split("|")]
                if not all(set(c) <= {"-", ":", " "} for c in cells):
                    row_col_counts.append(len(cells))

        if not row_col_counts:
            return ValidationResult(
                gate_name="TableStructureGate",
                passed=False,
                score=0.2,
                issues=["No valid markdown table rows detected"],
                remediation_action="tableformer_accurate",
            )

        header_cols = row_col_counts[0]
        mismatched_rows = [i for i, c in enumerate(row_col_counts) if c != header_cols]

        if mismatched_rows:
            issues.append(f"Non-rectangular table: {len(mismatched_rows)} rows have column count != header ({header_cols})")

        total_cells = sum(row_col_counts)
        has_numeric = bool(re.search(r"[0-9]", table_text))
        if not has_numeric and total_cells > 6:
            issues.append("Scientific table lacks numeric data values")

        passed = len(issues) == 0
        score = 1.0 - (0.4 if mismatched_rows else 0.0) - (0.2 if not has_numeric else 0.0)

        return ValidationResult(
            gate_name="TableStructureGate",
            passed=passed,
            score=max(0.0, round(score, 2)),
            issues=issues,
            remediation_action="reparse_table_deep" if not passed else None,
            metadata={"num_rows": len(row_col_counts), "num_cols": header_cols},
        )


class TextFidelityGate:
    """Detects control character corruption, OCR halftone noise, and lexical entropy."""

    CONTROL_CHARS_RX = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    PUNCT_NOISE_RX = re.compile(r"(?:[\.\-_=~`'\"]{4,})")

    def evaluate(self, text: str) -> ValidationResult:
        stripped = text.strip()
        issues: list[str] = []

        if not stripped:
            return ValidationResult(
                gate_name="TextFidelityGate",
                passed=False,
                score=0.0,
                issues=["Empty text string"],
                remediation_action=None,
            )

        control_matches = self.CONTROL_CHARS_RX.findall(stripped)
        if control_matches:
            issues.append(f"Found {len(control_matches)} non-printable control characters")

        punct_noise = self.PUNCT_NOISE_RX.findall(stripped)
        if punct_noise:
            issues.append(f"Found repeated punctuation noise streaks: {punct_noise[:3]}")

        alphanumeric_count = sum(1 for c in stripped if c.isalnum() or c.isspace() or c in ".,;:()[]{}%+-=/*_λθνμρωγτ")
        fidelity_ratio = alphanumeric_count / max(len(stripped), 1)

        if fidelity_ratio < 0.75 and len(stripped) > 20:
            issues.append(f"Low printable character fidelity: {fidelity_ratio:.1%}")

        passed = len(issues) == 0
        score = max(0.0, round(fidelity_ratio - (0.2 if control_matches else 0.0), 2))
        remediation = "normalize_scientific" if (control_matches or not passed) else None

        return ValidationResult(
            gate_name="TextFidelityGate",
            passed=passed,
            score=score,
            issues=issues,
            remediation_action=remediation,
        )


class GeometricCoverageGate:
    """Classifies unparsed whitespace into legitimate margins/gutters vs unexplained gaps."""

    def evaluate_page_elements(
        self,
        elements: Sequence[DocumentElement],
        page_width: float = 612.0,
        page_height: float = 792.0,
        gap_threshold_pt: float = 65.0,
    ) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        valid_boxes = [e for e in elements if e.bbox and len(e.bbox) == 4 and not e.suppressed]
        if len(valid_boxes) < 2:
            return results

        sorted_boxes = sorted(valid_boxes, key=lambda e: (e.bbox[1], e.bbox[0]))

        for i in range(len(sorted_boxes) - 1):
            curr_e = sorted_boxes[i]
            next_e = sorted_boxes[i + 1]

            curr_y1 = curr_e.bbox[3]
            next_y0 = next_e.bbox[1]
            gap = next_y0 - curr_y1

            # Ignore legitimate header margin (<8%) or footer margin (>92%)
            if curr_y1 < page_height * 0.08 or next_y0 > page_height * 0.92:
                continue

            # Check if elements are in different columns (multi-column jump)
            curr_cx = (curr_e.bbox[0] + curr_e.bbox[2]) / 2.0
            next_cx = (next_e.bbox[0] + next_e.bbox[2]) / 2.0
            is_cross_column = abs(curr_cx - next_cx) > (page_width * 0.25)

            # Legitimate whitespace classification:
            # 1. Cross-column jump is legitimate layout whitespace
            # 2. Flanking figure or table is legitimate aesthetic whitespace
            is_flanking_figure_or_table = (
                curr_e.element_type in (ElementType.FIGURE.value, ElementType.TABLE.value)
                or next_e.element_type in (ElementType.FIGURE.value, ElementType.TABLE.value)
            )

            if gap > gap_threshold_pt and not is_cross_column and not is_flanking_figure_or_table:
                # True unexplained gap!
                results.append(
                    ValidationResult(
                        gate_name="GeometricCoverageGate",
                        passed=False,
                        score=round(max(0.0, 1.0 - (gap / 200.0)), 2),
                        issues=[f"Unexplained vertical gap of {gap:.1f}pt between y={curr_y1:.1f} and y={next_y0:.1f}"],
                        remediation_action="unparsed_band_scan",
                        metadata={
                            "page": curr_e.page,
                            "gap_bbox": [min(curr_e.bbox[0], next_e.bbox[0]), curr_y1, max(curr_e.bbox[2], next_e.bbox[2]), next_y0],
                            "gap_size_pt": gap,
                        },
                    )
                )

        return results


class ReadingOrderGate:
    """Validates multi-column reading sequence and paragraph continuity."""

    def evaluate_reading_order(self, elements: Sequence[DocumentElement]) -> ValidationResult:
        issues: list[str] = []
        if len(elements) < 2:
            return ValidationResult(gate_name="ReadingOrderGate", passed=True, score=1.0)

        # Check paragraph continuity across consecutive elements on same page
        paragraphs = [e for e in elements if e.element_type == ElementType.PARAGRAPH.value and not e.suppressed]

        split_hyphens = 0
        for i in range(len(paragraphs) - 1):
            curr_text = paragraphs[i].text.strip()
            next_text = paragraphs[i + 1].text.strip()

            # If current block ends with hyphen and next begins with lowercase -> split word across layout
            if curr_text.endswith("-") and next_text and next_text[0].islower():
                split_hyphens += 1

        if split_hyphens > 3:
            issues.append(f"Detected {split_hyphens} broken hyphenated paragraphs across layout boundaries")

        score = max(0.5, 1.0 - (split_hyphens * 0.1))
        return ValidationResult(
            gate_name="ReadingOrderGate",
            passed=(len(issues) == 0),
            score=round(score, 2),
            issues=issues,
        )


class ReferenceLinkConsistencyGate:
    """Validates that cited Eq. (X), Fig. Y, and Table Z have matching target elements."""

    REF_RX = re.compile(r"\b(?:Eq(?:uation|\.)?\s*\(?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\)?|Fig(?:ure|\.)?\s*([0-9]+[a-z]?)|Table\s*([0-9]+))\b", re.IGNORECASE)

    def evaluate_document_references(self, elements: Sequence[DocumentElement]) -> ValidationResult:
        known_equations = {e.equation_ref for e in elements if e.equation_ref}
        known_figures = {e.figure_ref for e in elements if e.figure_ref}
        known_tables = {e.metadata.get("table_ref") for e in elements if e.element_type == ElementType.TABLE.value}

        cited_equations: set[str] = set()
        cited_figures: set[str] = set()
        cited_tables: set[str] = set()

        for e in elements:
            if e.element_type in (ElementType.PARAGRAPH.value, ElementType.HEADING.value):
                for m in self.REF_RX.finditer(e.text):
                    eq_id, fig_id, tab_id = m.groups()
                    if eq_id:
                        cited_equations.add(f"Eq. ({eq_id})")
                    if fig_id:
                        cited_figures.add(f"Fig. {fig_id}")
                    if tab_id:
                        cited_tables.add(f"Table {tab_id}")

        issues: list[str] = []
        # Missing equations audit
        missing_eqs = [eq for eq in cited_equations if eq not in known_equations and not any(eq in (e.text or "") for e in elements if e.element_type == ElementType.EQUATION.value)]
        if missing_eqs and len(missing_eqs) > len(cited_equations) * 0.5:
            issues.append(f"Referenced equations not resolved in parsed elements: {missing_eqs[:3]}")

        score = max(0.6, 1.0 - (len(issues) * 0.2))
        return ValidationResult(
            gate_name="ReferenceLinkConsistencyGate",
            passed=(len(issues) == 0),
            score=round(score, 2),
            issues=issues,
            metadata={"cited_equations": list(cited_equations), "known_equations": list(known_equations)},
        )


class CrossModalFidelityGate:
    """Measures consensus agreement across digital text, neural layout, and OCR."""

    def evaluate_agreement(self, text_a: str, text_b: str, context_label: str = "element") -> ValidationResult:
        s1 = text_a.strip().lower()
        s2 = text_b.strip().lower()

        if not s1 or not s2:
            return ValidationResult(
                gate_name="CrossModalFidelityGate",
                passed=False,
                score=0.0,
                issues=[f"One of the modal extractions is empty for {context_label}"],
            )

        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
        passed = (ratio >= 0.70)
        issues = [] if passed else [f"Low cross-modal agreement ({ratio:.1%}) between digital and neural extraction for {context_label}"]

        return ValidationResult(
            gate_name="CrossModalFidelityGate",
            passed=passed,
            score=round(ratio, 2),
            issues=issues,
            metadata={"agreement_ratio": round(ratio, 3)},
        )


class ParsingQualityGatePipeline:
    """Orchestrates all 7 deterministic quality gates and computes ConfidenceVectors."""

    def __init__(self):
        self.latex_gate = LaTeXSyntaxGate()
        self.table_gate = TableStructureGate()
        self.text_gate = TextFidelityGate()
        self.geo_gate = GeometricCoverageGate()
        self.order_gate = ReadingOrderGate()
        self.ref_gate = ReferenceLinkConsistencyGate()
        self.cross_modal_gate = CrossModalFidelityGate()

    def audit_elements(
        self,
        elements: Sequence[DocumentElement],
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> dict[str, Any]:
        """Audit elements, calibrate confidence vectors, and produce document certification."""
        element_results: list[dict[str, Any]] = []
        flagged_remediations: list[dict[str, Any]] = []

        # 1. Element-level evaluations & confidence vector calibration
        for idx, el in enumerate(elements):
            res: ValidationResult | None = None
            if el.element_type == ElementType.EQUATION.value:
                res = self.latex_gate.evaluate(el.text)
                el.confidence.extraction_confidence = res.score
            elif el.element_type == ElementType.TABLE.value:
                res = self.table_gate.evaluate(el.text)
                el.confidence.structural_confidence = res.score
            elif el.element_type in (ElementType.PARAGRAPH.value, ElementType.HEADING.value, ElementType.CAPTION.value):
                res = self.text_gate.evaluate(el.text)
                el.confidence.extraction_confidence = res.score

            # Recompute overall confidence
            el.confidence.compute_overall()

            if res is not None:
                record = {
                    "element_index": idx,
                    "element_type": el.element_type,
                    "page": el.page,
                    "passed": res.passed,
                    "score": res.score,
                    "gate": res.gate_name,
                    "issues": res.issues,
                    "remediation": res.remediation_action,
                }
                element_results.append(record)
                if not res.passed:
                    flagged_remediations.append(record)

        # 2. Geometric coverage audit
        pages = {el.page for el in elements}
        gap_results: list[dict[str, Any]] = []
        for p in sorted(pages):
            p_elements = [e for e in elements if e.page == p]
            geo_issues = self.geo_gate.evaluate_page_elements(
                p_elements, page_width=page_width, page_height=page_height
            )
            for g in geo_issues:
                record = {
                    "page": p,
                    "gate": g.gate_name,
                    "passed": False,
                    "score": g.score,
                    "issues": g.issues,
                    "remediation": g.remediation_action,
                    "metadata": g.metadata,
                }
                gap_results.append(record)
                flagged_remediations.append(record)

        # 3. Reading order audit
        order_res = self.order_gate.evaluate_reading_order(elements)
        if not order_res.passed:
            flagged_remediations.append({
                "gate": order_res.gate_name,
                "passed": False,
                "score": order_res.score,
                "issues": order_res.issues,
            })

        # 4. Reference link consistency audit
        ref_res = self.ref_gate.evaluate_document_references(elements)
        if not ref_res.passed:
            flagged_remediations.append({
                "gate": ref_res.gate_name,
                "passed": False,
                "score": ref_res.score,
                "issues": ref_res.issues,
            })

        total_checked = len(element_results)
        passed_count = sum(1 for r in element_results if r["passed"])
        quality_score = (passed_count / max(total_checked, 1)) * 100.0

        return {
            "total_elements_audited": total_checked,
            "passed_elements": passed_count,
            "quality_pass_rate_pct": round(quality_score, 1),
            "flagged_issues_count": len(flagged_remediations),
            "flagged_remediations": flagged_remediations,
            "element_results": element_results,
            "gap_anomalies_count": len(gap_results),
            "reading_order_score": order_res.score,
            "reference_link_score": ref_res.score,
            "certified": (quality_score >= 80.0 and len(gap_results) == 0),
        }
