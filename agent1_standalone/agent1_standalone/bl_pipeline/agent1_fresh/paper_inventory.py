"""paper_inventory.py — load paper summaries and format for prompts.

A "paper inventory" is the compact listing the reasoner gets in its kickoff
message: one block per paper, showing the algorithm class, what closed-form
quantities the paper supplies, what regimes it covers, and what it
explicitly DOES NOT cover.  This lets Sonnet "skim the forest" before
diving into individual chunks.

Summaries are produced by `scripts/summarize_algebraic_papers.py` and
stored as `runs/_logs/paper_summaries/<paper_id>.json`.  For now we
inject them all into every query's kickoff message — once we have
summaries for all 54 papers and the corpus grows, we'll switch to
retrieving the top-K by semantic match.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _REPO_ROOT / "runs" / "_logs" / "paper_summaries"


def load_all_paper_summaries(
    summaries_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load every <paper_id>.json in the summaries directory.

    Skips files whose summary failed to parse (`_parse_failed=True`)
    so we never inject broken data into a downstream prompt.
    """
    d = Path(summaries_dir) if summaries_dir else _DEFAULT_DIR
    if not d.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(d.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record = data.get("summary_record") or {}
        if not isinstance(record, dict) or record.get("_parse_failed"):
            continue
        out.append({
            "paper_id": data.get("paper_id") or path.stem,
            "summary_record": record,
        })
    return out


def format_paper_inventory_for_prompt(
    summaries: list[dict[str, Any]],
    *,
    max_quantities_per_paper: int = 8,
    max_exclusions_per_paper: int = 4,
    max_pitfalls_per_paper: int = 5,
    max_relations_per_category: int = 3,
) -> str:
    """Render a compact inventory block for OPTIMIZER, PLANNER, REASONER.

    Format (per paper, with x_t-focused enhanced schema):

        [paper_id]  (algorithm_class) [x_t role: ONSET_CORRELATION]
        summary: <prose paragraph framed around x_t contribution>
        how to use for x_t: <specific recipe — eq labels + assumptions>
        numeric ranges: Tu_pct=[1.8,6.2] Lambda_x_mm=[16,26] ...
        required inputs: required=[U,Tu,nu], optional=[Lambda_x], facility=[]
        regimes: <comma-separated regimes>
        closed-form quantities (verbatim from paper):
          - Re_theta,t (Mayle) [label: Eq.9] [formula: 400*Tu^(-5/8)]
            [role: ONSET] [Tu>=3%, ZPG]
          ...
        known pitfalls for x_t:
          - <recurring-bug warning 1>
          - ...
        paper relationships:
          praises:        <pid (reason)>; ...
          critiques:      <pid (reason)>; ...
          compares_with:  <pid (reason)>; ...
        not covered:
          - <exclusion>

    All NEW fields (x_t_contribution, how_to_use_for_x_t, numeric_ranges,
    required_inputs, known_pitfalls_for_x_t, paper_relationships, plus the
    formula/role_in_x_t sub-fields on quantities) are surfaced if present.
    Old-schema summaries (without these fields) still render the legacy
    subset — graceful degradation.
    """
    if not summaries:
        return "(no paper inventory available)"

    out_lines: list[str] = []
    for entry in summaries:
        paper_id = entry["paper_id"]
        rec = entry["summary_record"]
        algo_class = rec.get("algorithm_class", "?")
        x_t_role   = (rec.get("x_t_contribution") or "").strip()

        # Header line: paper_id, algorithm class, x_t role tag
        header = f"[{paper_id}]  ({algo_class})"
        if x_t_role:
            header += f"  [x_t role: {x_t_role}]"
        out_lines.append(header)

        # Summary paragraph (capped to keep token cost bounded)
        summary = (rec.get("summary") or "").strip()
        if len(summary) > 600:
            summary = summary[:600] + "…[trim]"
        if summary:
            out_lines.append(f"  summary: {summary}")

        # NEW: how_to_use_for_x_t — the planner-actionable recipe
        recipe = (rec.get("how_to_use_for_x_t") or "").strip()
        if recipe:
            if len(recipe) > 400:
                recipe = recipe[:400] + "…[trim]"
            out_lines.append(f"  how to use for x_t: {recipe}")

        # NEW: numeric_ranges — compact one-liner for programmatic filtering
        ranges = rec.get("numeric_ranges") or {}
        if isinstance(ranges, dict) and ranges:
            range_parts: list[str] = []
            for key in ("Tu_pct", "Lambda_x_mm", "Re_x", "Re_theta",
                        "Mach", "geometry", "pressure_grad"):
                val = ranges.get(key)
                if val is None:
                    continue
                if isinstance(val, list) and val:
                    # Numeric [min,max] pairs render compactly.
                    if len(val) == 2 and all(
                        isinstance(v, (int, float)) for v in val
                    ):
                        range_parts.append(f"{key}=[{val[0]},{val[1]}]")
                    else:
                        # String list (geometry, pressure_grad)
                        joined = ",".join(str(v) for v in val[:4])
                        range_parts.append(f"{key}=[{joined}]")
            if range_parts:
                out_lines.append(f"  numeric ranges: {' '.join(range_parts)}")

        # NEW: required_inputs — what FLOW the method needs
        inputs = rec.get("required_inputs") or {}
        if isinstance(inputs, dict) and inputs:
            input_parts: list[str] = []
            for key in ("required", "optional", "facility"):
                vals = inputs.get(key) or []
                if isinstance(vals, list) and vals:
                    joined = ",".join(str(v) for v in vals[:5])
                    input_parts.append(f"{key}=[{joined}]")
            if input_parts:
                out_lines.append(f"  required inputs: {', '.join(input_parts)}")

        # Regimes covered — comma-joined, short
        regimes = rec.get("regimes_covered") or []
        if regimes:
            shortened = []
            for r in regimes[:6]:
                rs = str(r)
                if len(rs) > 90:
                    rs = rs[:90] + "…"
                shortened.append(rs)
            out_lines.append(f"  regimes: {' ; '.join(shortened)}")

        # Closed-form quantities — capped per paper.  New schema adds
        # `formula` (verbatim) and `role_in_x_t` (ONSET/DECAY/etc.).
        quantities = rec.get("quantities_with_closed_forms") or []
        if quantities:
            out_lines.append("  closed-form quantities (verbatim from paper):")
            for q in quantities[:max_quantities_per_paper]:
                if not isinstance(q, dict):
                    continue
                qname   = str(q.get("quantity", "?"))[:120]
                label   = str(q.get("label_in_paper", ""))[:100]
                formula = str(q.get("formula", ""))[:160]
                role    = str(q.get("role_in_x_t", ""))[:30]
                applies = str(q.get("applies_when", ""))[:120]
                line = f"    - {qname}"
                if label:
                    line += f"  [label: {label}]"
                if formula:
                    line += f"  [formula: {formula}]"
                if role:
                    line += f"  [role: {role}]"
                if applies:
                    line += f"  [{applies}]"
                out_lines.append(line)
            if len(quantities) > max_quantities_per_paper:
                out_lines.append(
                    f"    (… {len(quantities) - max_quantities_per_paper} more "
                    f"quantities in this paper — call search() if needed)"
                )

        # NEW: known_pitfalls_for_x_t — recurring-bug surface
        pitfalls = rec.get("known_pitfalls_for_x_t") or []
        if pitfalls:
            out_lines.append("  known pitfalls for x_t:")
            for p in pitfalls[:max_pitfalls_per_paper]:
                ps = str(p)
                if len(ps) > 220:
                    ps = ps[:220] + "…"
                out_lines.append(f"    - {ps}")

        # NEW: paper_relationships — citation graph (praises/critiques/compares_with)
        relations = rec.get("paper_relationships") or {}
        if isinstance(relations, dict) and relations:
            rel_lines: list[str] = []
            for cat in ("praises", "critiques", "compares_with"):
                items = relations.get(cat) or []
                if not isinstance(items, list) or not items:
                    continue
                trimmed: list[str] = []
                for item in items[:max_relations_per_category]:
                    if isinstance(item, dict):
                        pid = str(item.get("paper_id", "?"))[:60]
                        reason = str(item.get("reason", ""))[:80]
                        trimmed.append(f"{pid} ({reason})" if reason else pid)
                    else:
                        trimmed.append(str(item)[:80])
                if trimmed:
                    rel_lines.append(f"    {cat}: {'; '.join(trimmed)}")
            if rel_lines:
                out_lines.append("  paper relationships:")
                out_lines.extend(rel_lines)

        # Explicit exclusions — capped
        exclusions = rec.get("explicit_exclusions") or []
        if exclusions:
            out_lines.append("  not covered (per paper's own scope statement):")
            for ex in exclusions[:max_exclusions_per_paper]:
                exs = str(ex)
                if len(exs) > 200:
                    exs = exs[:200] + "…"
                out_lines.append(f"    - {exs}")

        out_lines.append("")   # blank line between papers

    return "\n".join(out_lines).rstrip()
