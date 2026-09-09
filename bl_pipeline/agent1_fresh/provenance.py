"""provenance.py — final-number provenance gate (ENFORCEMENT, not instruction).

WHY THIS EXISTS
───────────────
prompts.py §3 ("NO MENTAL ARITHMETIC") *asks* the reasoner to make every
number in its FINAL block come from a compute() call.  An instruction can
only ask — it cannot enforce.  Run 1e84051e proved the gap: every compute()
was blocked by the (old) all-or-nothing verifier, the reasoner fell back to
mental arithmetic, and shipped L_tr = 1.948 m (correct value ≈ 0.34 m — it
evaluated 109618**0.8 as ~61,200 in its head instead of 10,763).  Every
"verified" label in the ledger was a lie because the verifier only checked
the *code*, never the final answer.

This module closes the hole structurally: after the reasoner finishes, every
numeric value in `reasoner.final.key_findings` is matched against the set of
numbers actually PRINTED by successful compute() calls this run.  A value
that doesn't trace to an execution is flagged `provenance="UNVERIFIED"` so
the writer can never present LLM mental-math as a computed result.

Instructions don't hold; gates do.  This is the gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Matches integers / decimals / thousands-separated / scientific notation.
_NUM_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?"   # 1,028,000
    r"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"                  # 220.0, 1.96e6
)

# Common unit-scale factors so a finding in cm/mm still matches a compute()
# stdout printed in m (and vice-versa): ×1, ×10⁻², ×10², ×10⁻³, ×10³.
_UNIT_SCALES = (1.0, 1e-2, 1e2, 1e-3, 1e3)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute-or-key accessor so the gate works on both the live
    dataclass tree (ReasonerTrace/ToolCall) and a JSON-loaded dict tree."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_numbers(text: str) -> list[float]:
    """Extract every numeric token from text as a float (commas stripped)."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0).replace(",", "")
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _collect_from_obj(o: Any) -> list[float]:
    """Recursively pull numeric values out of a compute() `result` object."""
    out: list[float] = []
    if isinstance(o, bool):
        return out
    if isinstance(o, (int, float)):
        out.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            out.extend(_collect_from_obj(v))
    elif isinstance(o, (list, tuple)):
        for v in o:
            out.extend(_collect_from_obj(v))
    elif isinstance(o, str):
        out.extend(_parse_numbers(o))
    return out


def _collect_computed_numbers(reasoner: Any) -> list[float]:
    """Every number printed/returned by a compute() call that ACTUALLY ran.

    A compute() that was empty-coded or rejected by the formula-verifier
    (literal hardcode) executed nothing, so its (empty) output is skipped —
    only genuine executions count as provenance.
    """
    nums: list[float] = []
    for tc in (_get(reasoner, "tool_calls") or []):
        if _get(tc, "tool_name") != "compute":
            continue
        res = _get(tc, "result") or {}
        # Skip calls that never executed any code.
        if res.get("_formula_verifier_rejected") or res.get("_empty_code"):
            continue
        nums.extend(_parse_numbers(res.get("stdout") or ""))
        nums.extend(_collect_from_obj(res.get("result")))
    return nums


def _matches(value: float, pool: list[float], rel_tol: float, abs_tol: float) -> bool:
    """True if `value` (or a common unit-scaling of it) is within tolerance
    of any number in the computed pool."""
    for scale in _UNIT_SCALES:
        v = value * scale
        for p in pool:
            if abs(v - p) <= max(abs_tol, rel_tol * max(abs(v), abs(p))):
                return True
    return False


@dataclass
class ProvenanceReport:
    n_findings: int = 0
    n_verified: int = 0
    n_unverified: int = 0
    n_qualitative: int = 0
    n_computed_numbers: int = 0
    unverified: list[tuple[str, str]] = field(default_factory=list)  # (quantity, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_findings": self.n_findings,
            "n_verified": self.n_verified,
            "n_unverified": self.n_unverified,
            "n_qualitative": self.n_qualitative,
            "n_computed_numbers": self.n_computed_numbers,
            "unverified": [{"quantity": q, "value": v} for q, v in self.unverified],
        }


def enforce_provenance(
    reasoner: Any, *, rel_tol: float = 0.02, abs_tol: float = 1e-9,
) -> ProvenanceReport:
    """Tag each finding in `reasoner.final.key_findings` with `provenance`:

        "computed"     — every number in the value matched a compute() output
        "qualitative"  — the value has no substantive number (e.g. "bypass")
        "UNVERIFIED"   — at least one number did NOT trace to an execution

    Mutates findings in place (adds `provenance` + `provenance_detail`).
    Returns a ProvenanceReport.  Never raises.
    """
    report = ProvenanceReport()
    final = _get(reasoner, "final")
    if not isinstance(final, dict):
        return report
    findings = final.get("key_findings") or []
    if not isinstance(findings, list):
        return report

    pool = _collect_computed_numbers(reasoner)
    report.n_computed_numbers = len(pool)

    for f in findings:
        if not isinstance(f, dict):
            continue
        report.n_findings += 1
        nums = [n for n in _parse_numbers(str(f.get("value", ""))) if abs(n) > 1e-12]
        if not nums:
            # No substantive number → qualitative finding (regime, etc.)
            f["provenance"] = "qualitative"
            report.n_qualitative += 1
            report.n_verified += 1
            continue
        if pool and all(_matches(n, pool, rel_tol, abs_tol) for n in nums):
            f["provenance"] = "computed"
            report.n_verified += 1
        else:
            f["provenance"] = "UNVERIFIED"
            f["provenance_detail"] = (
                "value not traced to any compute() stdout this run "
                "(possible mental-arithmetic) — do not present as computed"
            )
            report.n_unverified += 1
            report.unverified.append(
                (str(f.get("quantity", f.get("name", "?"))), str(f.get("value", "")))
            )
    return report


# ══════════════════════════════════════════════════════════════════
# Value reconciliation — the CORRECTNESS counterpart to the gate
# ══════════════════════════════════════════════════════════════════
#
# The reasoner often COMPUTES the right value via compute() (it lands in the
# tool's `result` dict and stdout) but then RETYPES a different, WRONG value
# into its FINAL key_findings (run 609afa43: computed Re_θt=168.18 but
# reported 172.8).  The provenance gate DETECTS the mismatch; this step
# FIXES it — it pulls the computed value back into the finding by matching
# the finding's quantity to a labeled compute output.
#
# Conservative by design: substitutes ONLY on a confident base+qualifier
# match; ambiguous findings are left for the gate to flag.  It never invents
# a value — every substitution comes verbatim from a compute() result.

_STDOUT_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)

# Correlation-method qualifiers.  Reconciliation must never cross these
# (e.g. substitute a Mayle-computed value into an AGS finding) — see the
# run-609afa43 bug where the generic Re_theta_t key (Mayle) was wrongly
# pulled into an "AGS cross-check" finding.
_METHODS = {"ags", "mayle", "fransson", "suzen"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower().replace("θ", "theta"))


def _base_symbol(name: str) -> str | None:
    """The physical quantity a finding/result-key refers to.  Ordered
    specific->general (tu / re_x before x_t, since their normalised forms
    contain 'xt')."""
    n = _norm(name)
    if "rethetat" in n or "rethetas" in n:                       return "re_theta_t"
    if "retr" in n:                                              return "re_tr"
    if "rex" in n:                                               return "re_x"
    if "tulocal" in n or "tuatxt" in n or "tuxt" in n or "tudecay" in n: return "tu_xt"
    if "xend" in n or "xcomplete" in n or "xturb" in n or n == "xe":     return "x_end"
    if "ltr" in n or "ltrans" in n:                             return "l_tr"
    if "xt" in n or "xtr" in n or "xonset" in n:               return "x_t"
    if "lambda" in n:                                           return "lambda"
    if "delta" in n:                                            return "delta"
    return None


def _qualifiers(name: str) -> set[str]:
    n = (name or "").lower()
    q: set[str] = set()
    if ("inlet" in n or "no decay" in n or "no-decay" in n or "first" in n
            or "lower bound" in n or "undecayed" in n):
        q.add("inlet")
    if ("converg" in n or "decay-correct" in n or "decay correct" in n
            or "decay-corr" in n or "primary" in n or "corrected" in n):
        q.add("decay")
    if "ags" in n or "abu" in n or "ghannam" in n:  q.add("ags")
    if "mayle" in n:                                q.add("mayle")
    if "fransson" in n:                             q.add("fransson")
    return q


def _fmt_num(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 10000 else f"{v:.4g}"


def _fmt_with_units(v: float, original: str) -> str:
    """Format the computed value, preserving any unit suffix from the
    reasoner's original string (so '0.1015 m' -> '0.0962 m')."""
    m = re.match(r"\s*[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?\s*(.*)$", original or "")
    unit = (m.group(1).strip() if m else "")
    s = _fmt_num(v)
    return f"{s} {unit}".strip() if unit else s


def _collect_labeled(reasoner: Any) -> dict[str, float]:
    """Every labeled value compute() produced this run: result-dict keys
    (clean, primary) plus `name = value` pairs parsed from stdout (fallback,
    start-anchored so 'x_t = 0.14 m = 14 cm' yields x_t, not the mid-line
    'm').  Last result-dict assignment wins; stdout never overrides a
    result-dict key."""
    out: dict[str, float] = {}
    for tc in (_get(reasoner, "tool_calls") or []):
        if _get(tc, "tool_name") != "compute":
            continue
        res = _get(tc, "result") or {}
        if res.get("_formula_verifier_rejected") or res.get("_empty_code"):
            continue
        rd = res.get("result")
        if isinstance(rd, dict):
            for k, v in rd.items():
                fv = _to_float(v)
                if fv is not None:
                    out[str(k)] = fv
        for m in _STDOUT_ASSIGN_RE.finditer(res.get("stdout") or ""):
            fv = _to_float(m.group(2).replace(",", ""))
            if fv is not None:
                out.setdefault(m.group(1), fv)
    return out


@dataclass
class ReconcileReport:
    n_reconciled: int = 0
    reconciled: list = field(default_factory=list)  # (quantity, old, new)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reconciled": self.n_reconciled,
            "reconciled": [
                {"quantity": q, "from": o, "to": n} for q, o, n in self.reconciled
            ],
        }


def reconcile_findings(reasoner: Any, *, rel_tol: float = 0.02) -> ReconcileReport:
    """Substitute compute()-produced values into FINAL key_findings whose
    reported value doesn't match what was actually computed.  Mutates the
    findings in place (value + provenance='computed' + records the original
    in `original_llm_value`).  Conservative — only on a confident match.
    Run this BEFORE enforce_provenance().  Never raises."""
    report = ReconcileReport()
    final = _get(reasoner, "final")
    if not isinstance(final, dict):
        return report
    findings = final.get("key_findings") or []
    if not isinstance(findings, list):
        return report

    computed = _collect_labeled(reasoner)
    if not computed:
        return report

    by_base: dict[str, list] = {}
    for key, val in computed.items():
        b = _base_symbol(key)
        if b:
            by_base.setdefault(b, []).append((key, val, _qualifiers(key)))

    for f in findings:
        if not isinstance(f, dict):
            continue
        raw = str(f.get("quantity", f.get("name", "")))
        base = _base_symbol(raw)
        if not base or base not in by_base:
            continue
        cands = by_base[base]
        fq = _qualifiers(raw)
        fm = fq & _METHODS
        # Method-agreement guard: never substitute across correlations.
        # Eligible = both generic (no named method), or both share a method.
        eligible = []
        for (k, v, cq) in cands:
            cm = cq & _METHODS
            if bool(fm) != bool(cm):
                continue                      # one names a method, the other doesn't
            if fm and cm and not (fm & cm):
                continue                      # conflicting methods (ags vs mayle)
            eligible.append((k, v, cq))
        if not eligible:
            continue
        # Among eligible, rank by STAGE-qualifier overlap (inlet vs decay).
        sfq = fq - _METHODS
        scored = sorted(eligible, key=lambda c: len((c[2] - _METHODS) & sfq),
                        reverse=True)
        best_key, best_val, best_q = scored[0]
        best_score = len((best_q - _METHODS) & sfq)
        # Ambiguity guard: if the top two eligible tie on stage overlap we
        # can't safely pick the variant — skip (the gate still flags it).
        if len(scored) > 1 and len((scored[1][2] - _METHODS) & sfq) == best_score:
            continue
        cur = _parse_numbers(str(f.get("value", "")))
        if cur and _matches(cur[0], [best_val], rel_tol, 1e-9):
            continue  # reasoner already reported the computed value
        old = str(f.get("value", ""))
        f["original_llm_value"] = old
        f["value"] = _fmt_with_units(best_val, old)
        f["reconciled_from"] = best_key
        f["provenance"] = "computed"
        report.n_reconciled += 1
        report.reconciled.append((raw[:80], old, f["value"]))
    return report
