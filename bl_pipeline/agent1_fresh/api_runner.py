"""api_runner.py — entry point the FastAPI backend calls to run fresh Agent 1.

DESIGN
──────
The agent core (`run.run()`) is pure-Python and event-driven.  It emits
events through `bl_pipeline.shared.event_bus`, which `api/main.py`
forwards to React over WebSocket via `_WSBridgeListener` (already
installed at run start).  This means we DON'T need to build any event
forwarding here — events flow automatically.

What this module provides is a single async-friendly entry point the
API calls to:

  1. Run the agent (sync — it's CPU+API-bound but blocks the event
     loop, so caller uses run_in_executor).
  2. Serialise the resulting `AgentResult` dataclass tree into a plain
     dict so FastAPI can `JSONResponse(...)` and the WebSocket bridge
     can broadcast a final `agent_done` message.

The serialised shape is the NATIVE AgentResult schema (NOT the legacy
`Agent1Output` shape).  The new React `Agent1.jsx` component is
designed around this native shape; see `docs/AGENT1_FINAL.md` §2 for
the field-by-field contract.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from bl_pipeline.agent1_fresh.run import run as _run_agent
from bl_pipeline.agent1_fresh.state import AgentResult


# ──────────────────────────────────────────────────────────────────────
# Serialisation — AgentResult tree → JSON-safe dict
# ──────────────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, sets, tuples, custom objects
    into JSON-safe primitives.  Pydantic-free; doesn't depend on
    anything besides stdlib.

    Handles:
      • dataclasses (FlowConditions, Plan, ReasonerTrace, ...) → dict
      • dicts → dict with values recursively converted
      • lists/tuples/sets → list
      • primitives → as-is
      • everything else → str(obj) (last resort, prevents JSON error)
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Pydantic-style .dict() / .model_dump() (we don't depend on
    # pydantic here, but if a future state schema uses it, this works)
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    # Last resort — stringify so the JSON response doesn't fail
    return str(obj)


def agent_result_to_dict(result: AgentResult) -> dict[str, Any]:
    """Convert AgentResult into a JSON-safe dict the API can return."""
    return _to_jsonable(result)


# ──────────────────────────────────────────────────────────────────────
# Public entry point — what api/main.py calls
# ──────────────────────────────────────────────────────────────────────

def run_agent1_fresh_for_api(
    query: str,
    run_id: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fresh Agent 1 and return a JSON-safe dict.

    Events are forwarded to WebSocket automatically by the API's
    `_WSBridgeListener` (installed before this function is called).

    The `config` dict accepts the same keys as `run.run()`:
        max_retrieval_iterations  (default 3)
        max_react_turns           (default 12)
        max_revisions             (default 1)
        max_cost_usd              (default 3.00; top-level run() cap is $5.00 — see run.py)

    Returns a dict with the AgentResult fields plus:
        agent_backend: "fresh"        (which code path ran — "fresh" vs
                                       legacy "transition")
        agent_version: "fresh-1.0"    (schema/contract version of the
                                       fresh-agent line; bump when the
                                       AgentResult shape changes
                                       — e.g. fresh-1.1 for additive
                                       changes, fresh-2.0 for breaking)

    Never raises — on catastrophic failure returns
        {"status": "error", "error": "...", "run_id": run_id,
         "agent_backend": "fresh"}
    """
    cfg = dict(config or {})
    # Whitelist + default the kwargs run() accepts.  Anything else in
    # config gets silently dropped — keeps the API tolerant of UI
    # sending extra fields.
    kwargs: dict[str, Any] = {"run_id": run_id}
    for k in (
        "max_retrieval_iterations",
        "max_react_turns",
        "max_revisions",
        "max_cost_usd",
        "facility_overrides",   # NEW — {lambda_x_mm, grid_solidity, ...}
    ):
        if k in cfg:
            kwargs[k] = cfg[k]

    try:
        result: AgentResult = _run_agent(query, **kwargs)
    except Exception as e:
        # run() itself catches its own exceptions and returns an
        # AgentResult with status="error".  This handler covers the
        # extreme case where even importing/initialising the agent
        # blows up before that defensive layer runs.
        return {
            "status":         "error",
            "error":          f"{type(e).__name__}: {e}",
            "run_id":         run_id,
            "user_query":     query,
            "agent_backend":  "fresh",
            "agent_version":  "fresh-1.0",
        }

    out = agent_result_to_dict(result)
    out["agent_backend"] = "fresh"
    out["agent_version"] = "fresh-1.0"
    return out
