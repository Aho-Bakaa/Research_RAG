"""
anchor_validator.py — task #148

Forces ``lookup_equation`` dispatch for every anchor paper in the
optimizer's plan.

What this protects against
--------------------------
The OPTIMIZER produces ``inventory_partition.anchors`` listing the
primary papers the analysis MUST cite. The REASONER then runs the
ReAct loop and produces ``key_findings`` with citations. The post-hoc
``_enrich_findings_with_glossary`` helper in ``reason.py`` attempts a
``lookup_equation`` for every finding's (paper_id × eq_num) combo and
attaches ``glossary_formula`` on success.

BUT: there is no guarantee that for every anchor paper_id, at least one
finding actually cites it with a verified glossary entry. If the
reasoner skips an anchor entirely, or cites it without an equation tag,
the writer downstream will reproduce a formula from training memory
(the "Sonnet retyped the formula" attack surface) — exactly the
fabrication mode that has caused multiple regressions in this codebase.

This module is the deterministic check: every anchor paper must appear
in ``key_findings`` at least once WITH ``glossary_formula`` populated.
Any violation is reported as a structured ``AnchorCoverageReport`` so
the orchestrator can emit an event, mark the manuscript suspect, or
trigger a revision pass.

No LLM, no fabrication risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class AnchorViolation:
    """One uncovered anchor with the reason."""
    paper_id: str
    reason: str           # 'no_finding' | 'no_glossary_formula'
    # When reason == 'no_glossary_formula', `findings_citing` lists the
    # findings that *named* the paper but had no verified equation.
    findings_citing: tuple = ()


@dataclass
class AnchorCoverageReport:
    """Output of ``validate_anchor_coverage``."""
    anchors:        tuple             = ()        # input anchor paper_ids
    covered:        tuple             = ()        # anchors with ≥1 verified finding
    violations:     tuple             = ()        # anchors NOT covered
    is_clean:       bool              = True      # convenience: no violations

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_clean", len(self.violations) == 0)


def _normalise(s: Any) -> str:
    return (str(s) if s is not None else "").strip().lower()


def _paper_id_variants(paper_id: str) -> tuple[str, ...]:
    """Forms a paper_id can appear in inside a citation string.

    ``inventory_partition.anchors`` always uses snake_case paper_ids
    (``roach_1987``), but the citation field commonly uses
    ``"Roach 1987"`` (space, capitalised) or ``"Roach (1987)"``. We
    accept any of those.
    """
    pid = _normalise(paper_id)
    if not pid:
        return ()
    variants = {pid, pid.replace("_", " "), pid.replace("_", "-")}
    return tuple(variants)


def _citation_mentions_paper(citation: str, paper_id: str) -> bool:
    cit = _normalise(citation)
    if not cit:
        return False
    return any(v in cit for v in _paper_id_variants(paper_id))


def _findings_citing_paper(
    paper_id: str,
    findings: Iterable[dict],
) -> list[dict]:
    """All findings whose `citation` string mentions paper_id (any
    accepted variant) OR whose ``glossary_paper_id`` matches."""
    if not _normalise(paper_id):
        return []
    matches: list[dict] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        if _citation_mentions_paper(f.get("citation"), paper_id):
            matches.append(f)
            continue
        if _normalise(f.get("glossary_paper_id")) == _normalise(paper_id):
            matches.append(f)
    return matches


def _has_verified_glossary(finding: dict) -> bool:
    """A finding is *verified* iff lookup_equation succeeded and
    attached a non-empty ``glossary_formula`` (LaTeX from the
    deterministic glossary, not Sonnet's restatement)."""
    formula = finding.get("glossary_formula")
    return isinstance(formula, str) and formula.strip() != ""


def validate_anchor_coverage(
    anchors:  Iterable[str],
    findings: Iterable[dict],
) -> AnchorCoverageReport:
    """Check that every anchor paper has ≥1 finding with a verified
    ``glossary_formula`` attached.

    Parameters
    ----------
    anchors
        ``inventory_partition.anchors`` from the optimizer (list of
        paper_id strings).
    findings
        ``reasoner.final.key_findings`` after
        ``_enrich_findings_with_glossary`` has run.

    Returns
    -------
    AnchorCoverageReport with structured violations.
    """
    anchor_list = [a for a in anchors if a]
    finding_list = list(findings)

    covered: list[str] = []
    violations: list[AnchorViolation] = []

    for paper_id in anchor_list:
        cites = _findings_citing_paper(paper_id, finding_list)
        if not cites:
            violations.append(AnchorViolation(
                paper_id=paper_id,
                reason="no_finding",
            ))
            continue
        if not any(_has_verified_glossary(f) for f in cites):
            violations.append(AnchorViolation(
                paper_id=paper_id,
                reason="no_glossary_formula",
                findings_citing=tuple(
                    _normalise(f.get("citation"))[:120]
                    for f in cites
                ),
            ))
            continue
        covered.append(paper_id)

    return AnchorCoverageReport(
        anchors    = tuple(anchor_list),
        covered    = tuple(covered),
        violations = tuple(violations),
    )


# ---------------------------------------------------------------------------
# Human-readable diagnostic for logs / frontend trace
# ---------------------------------------------------------------------------


def format_violation_block(report: AnchorCoverageReport) -> str:
    """Render the report as a short markdown block for the trace UI."""
    if report.is_clean:
        return (
            f"✓ Anchor coverage clean: every one of "
            f"{len(report.anchors)} anchor paper(s) was cited at least "
            f"once with a verified glossary formula attached "
            f"(`lookup_equation` succeeded).\n"
        )
    lines = [
        f"⚠ Anchor coverage incomplete — "
        f"{len(report.violations)}/{len(report.anchors)} anchor paper(s) "
        f"failed the `lookup_equation` dispatch check.\n",
        "",
        "| Anchor paper_id | Reason | Findings citing |",
        "|---|---|---|",
    ]
    for v in report.violations:
        citing = (
            "; ".join(s for s in v.findings_citing) if v.findings_citing
            else "(none)"
        )
        lines.append(f"| `{v.paper_id}` | {v.reason} | {citing} |")
    lines.append("")
    lines.append(
        "These anchors must be re-fetched via `lookup_equation` before "
        "the writer renders the manuscript, OR the optimizer must remove "
        "them from `anchors` and document the reason."
    )
    return "\n".join(lines)
