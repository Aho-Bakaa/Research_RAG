"""output_loader.py — read a persisted agent1_fresh AgentResult from disk.

Mirrors the legacy `agent1_transition.output_loader.load_agent1_results_from_disk`
contract so Agent 3's `input_resolver.py` can consume either backend's
output through a single uniform interface.

CONTRACT
────────
`load_agent1_fresh_results_from_disk(run_id, runs_dir=None)` returns:

    None                                — file not found / unreadable
    dict with at minimum these keys     — agent3 input_resolver consumes these:
      • flow_conditions   : FlowConditions dict (velocity_ms, Tu pct, ν, chord, …)
      • transition_type   : "bypass" | "natural" | "separation" | "roughness"
      • manuscript_section: the full markdown manuscript (or "")
      • loaded_from_disk  : True
      • agent_backend     : "fresh"
      • agent_version     : "fresh-1.0"

    Plus passthroughs from the AgentResult for any caller that wants them:
      • run_id, user_query, flow, retrieval, plan, reasoner, coverage,
        critique, process_log, manuscript, total_cost_usd, total_elapsed_s,
        status, error

The `transition_type` field is derived from the AgentResult by looking
for the `regime` entry in `reasoner.final.key_findings` (the fresh agent
writes regime as a key finding, value like "bypass (FST-dominated)").

PERSISTENCE PATH
────────────────
Reads from:
    runs/_logs/<run_id>/agent1_output/agent_result.json

This file is written by `bl_pipeline.agent1_fresh.run.run()` in its
finally-block (best-effort, never blocks the run).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _runs_dir_default() -> Path:
    """Resolve the project's RUNS_DIR.  We do this lazily to avoid
    importing the heavy shared.config module at file load.
    """
    try:
        from bl_pipeline.shared.config import RUNS_DIR
        return Path(RUNS_DIR)
    except Exception:
        return Path("runs/_logs")


_REGIME_NAMES = {
    "bypass": "bypass",
    "natural": "natural",
    "ts-wave": "natural",
    "tollmien-schlichting": "natural",
    "separation": "separation",
    "separation-induced": "separation",
    "roughness": "roughness",
    "roughness-tripped": "roughness",
}


def _infer_transition_type(agent_result: dict[str, Any]) -> str:
    """Walk reasoner.final.key_findings looking for a 'regime' entry.
    Falls back to 'bypass' for high-Tu cases (Tu > 1%), 'natural' otherwise.
    """
    final = (agent_result.get("reasoner") or {}).get("final") or {}
    for f in final.get("key_findings") or []:
        if not isinstance(f, dict):
            continue
        qname = str(f.get("quantity") or "").lower()
        if "regime" in qname:
            val = str(f.get("value") or "").lower()
            for k, v in _REGIME_NAMES.items():
                if k in val:
                    return v
            # If no canonical name matches, return the raw value
            return val.split()[0] if val else "bypass"
    # Heuristic fallback: Tu threshold
    flow = agent_result.get("flow") or {}
    tu = flow.get("turbulence_intensity_pct") or 0
    if isinstance(tu, (int, float)) and tu > 1.0:
        return "bypass"
    return "natural"


def load_agent1_fresh_results_from_disk(
    run_id: str,
    runs_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Read the AgentResult written by `agent1_fresh.run()` and return
    a dict compatible with what Agent 3's input_resolver consumes.

    Returns None when the file is missing or unreadable.

    The returned dict carries BOTH the legacy-compatible keys
    (flow_conditions / transition_type / manuscript_section) AND the
    full AgentResult tree (flow / reasoner / plan / critique / …) so
    callers that want richer access can dig in.
    """
    base = (runs_dir or _runs_dir_default()) / run_id / "agent1_output"
    result_path = base / "agent_result.json"
    if not result_path.exists():
        return None

    try:
        ar = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(ar, dict):
        return None

    flow = ar.get("flow") or {}
    if not isinstance(flow, dict):
        flow = {}

    # Legacy-compatible flow_conditions field.  Agent 3 reads
    # flow.velocity_ms + flow.turbulence_intensity (note: legacy used
    # `turbulence_intensity` without _pct suffix; the fresh schema uses
    # `turbulence_intensity_pct`).  We map both for safety.
    flow_conditions = dict(flow)
    if "turbulence_intensity_pct" in flow and "turbulence_intensity" not in flow_conditions:
        flow_conditions["turbulence_intensity"] = flow["turbulence_intensity_pct"]

    transition_type = _infer_transition_type(ar)

    out: dict[str, Any] = {
        # ── Legacy-compatible fields (agent3 input_resolver reads these) ──
        "flow_conditions":    flow_conditions,
        "transition_type":    transition_type,
        "manuscript_section": ar.get("manuscript") or "",
        "loaded_from_disk":   True,
        "agent_backend":      ar.get("agent_backend") or "fresh",
        "agent_version":      ar.get("agent_version") or "fresh-1.0",
        # Legacy placeholders agent3 may inspect (kept empty for the
        # fresh backend; agent3 doesn't strictly require them):
        "recommended_models":      [],
        "correlations":            [],
        "conflicts":               [],
        "gaps_detected":           [],
        "papers_suggested":        [],
        "clarification_requests":  [],
        "assumed_defaults":        {},
        "physics_gap_result":      {},
        "paper_gap_result":        {},
        "scope_status":            ar.get("status"),
        "best_model":              None,
        "per_model_packets":       [],

        # ── Full AgentResult passthrough (for richer consumers) ────
        "run_id":           ar.get("run_id"),
        "user_query":       ar.get("user_query"),
        "flow":             flow,
        "retrieval":        ar.get("retrieval"),
        "plan":             ar.get("plan"),
        "reasoner":         ar.get("reasoner"),
        "coverage":         ar.get("coverage"),
        "critique":         ar.get("critique"),
        "process_log":      ar.get("process_log"),
        "manuscript":       ar.get("manuscript"),
        "total_cost_usd":   ar.get("total_cost_usd"),
        "total_elapsed_s":  ar.get("total_elapsed_s"),
        "status":           ar.get("status"),
        "error":            ar.get("error"),
        # Provenance report (gate) — so the a1->a3 handoff confidence
        # counts resolve via the disk path too.
        "provenance":       ar.get("provenance"),
    }
    # (Standalone A1: no downstream Agent-3 handoff in this project.)
    out["agent3_handoff"] = None
    return out
