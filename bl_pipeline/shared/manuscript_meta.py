"""Sidecar metadata for per-agent manuscripts.

Every agent writes its `manuscript.md` into
`data/runs/a<N>_<pipeline_id>/` and calls
``write_manuscript_meta()`` once at the end.  The sidecar JSON file
``manuscript_meta.json`` lets Agent 6 discover, validate, and stitch
all five upstream sections by globbing
``data/runs/*_<pipeline_id>/manuscript_meta.json``.

The schema is deliberately tiny — only the fields A6 needs to find
the section, attribute it, and embed any plots.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


# ── Agent → display label (used by A6 when introducing each section).
_AGENT_LABELS: dict[str, str] = {
    "agent1": "Theoretical framework (empirical correlations)",
    "agent2": "Reynolds-averaged CFD with transition-sensitive closure",
    "agent3": "Hot-wire pilot acquisition and intermittency detection",
    "agent4": "Full streamwise + wall-normal traverse acquisition",
    "agent5": "Hot-wire calibration and boundary-layer integral parameters",
    "agent6": "Synthesis manuscript",
}


def write_manuscript_meta(
    *,
    agent_id: str,
    pipeline_id: str,
    output_dir: Path | str,
    plot_files: Optional[Iterable[str]] = None,
    upstream_run_ids: Optional[dict[str, str]] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Write ``manuscript_meta.json`` next to a freshly-written
    ``manuscript.md``.

    Parameters
    ──────────
    agent_id          : "agent1" .. "agent6".
    pipeline_id       : the shared 8-char pipeline ID that anchors
                        this paper.  Every agent in a single pipeline
                        run stamps the same value.
    output_dir        : the directory that contains the agent's
                        ``manuscript.md`` (typically
                        ``data/runs/a<N>_<pipeline_id>/``).
    plot_files        : optional iterable of relative paths to PNG
                        plots Agent 6 may embed.  Paths are stored
                        verbatim — relative to ``output_dir`` or
                        absolute, caller's choice.
    upstream_run_ids  : optional map of upstream agent run-IDs this
                        section depended on (e.g.
                        ``{"agent1": "a1_7bc148b5"}`` when A2 was
                        seeded from A1).
    extra             : optional free-form dict carried verbatim into
                        the sidecar; useful for run-specific notes
                        without growing this helper's schema.

    Returns
    ───────
    Absolute path to the written ``manuscript_meta.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "agent_id":         agent_id,
        "agent_label":      _AGENT_LABELS.get(agent_id, agent_id),
        "pipeline_id":      pipeline_id,
        "agent_run_id":     output_dir.name,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "plot_files":       list(plot_files) if plot_files else [],
        "upstream_run_ids": dict(upstream_run_ids or {}),
    }
    if extra:
        payload["extra"] = extra

    meta_path = output_dir / "manuscript_meta.json"
    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta_path


def derive_agent_run_id(agent_id: str, pipeline_id: str) -> str:
    """Return the canonical run-ID for ``agent_id`` inside
    ``pipeline_id``.

    Convention: ``a<N>_<pipeline_id>`` — e.g. ``a1_7bc148b5``,
    ``a2_7bc148b5``.  Every agent's output folder under
    ``data/runs/`` uses this name.

    Pass ``pipeline_id`` = the agent's own standalone run-ID when the
    agent runs outside a pipeline; the convention still works
    (``a2_<standalone_id>`` is unique and self-anchoring).
    """
    short = agent_id.removeprefix("agent")
    return f"a{short}_{pipeline_id}"
