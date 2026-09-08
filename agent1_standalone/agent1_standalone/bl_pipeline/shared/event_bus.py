"""event_bus.py — live pipeline observability.

Why this exists
───────────────
Agent 1 is a multi-step pipeline that takes ~10 minutes to produce
a 22k-char manuscript. The user shouldn't have to stare at a blank
terminal and wait. Every meaningful action — a node starting, an LLM
call firing, a chunk being retrieved, a math step being derived —
emits a structured event through this bus.

Two listeners are installed by default at run start:

  1. JSONLFileWriter — appends every event to
     data/runs/{run_id}/agent1_output/events.jsonl.
     A web frontend can tail this file and render live cards.

  2. TerminalPrettyPrinter — pretty-prints each event to stdout.
     A researcher running Agent 1 in a shell sees every step as it
     happens, not a 10-minute stall followed by a dump.

Design constraints
──────────────────
  • Thread-safe. The orchestrator uses ThreadPoolExecutor for the
    per-model loop; three parallel Sonnet workers must not interleave
    their events on a single file handle.
  • Listener failures MUST NOT break the pipeline. Each listener call
    is wrapped in try/except with a visible warning.
  • Emit is O(1) — listeners do one append each, no scanning.
  • Every emit is flushed so the frontend sees events immediately,
    not on process exit.

Usage
─────
At run start (orchestrator):

    from bl_pipeline.shared.event_bus import install_default_listeners, emit
    install_default_listeners(run_id, events_file=run_root / "events.jsonl")
    emit("run_started", run_id=run_id, query=query)

At each meaningful action:

    emit("step_started", run_id=run_id, step_n=5, name="retrieval",
         title="Literature retrieval")
    # ... work ...
    emit("step_completed", run_id=run_id, step_n=5, name="retrieval",
         summary="15 chunks across 3 papers", duration_s=12.3)

At run end:

    close_default_listeners()  # flush and close file handles
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ══════════════════════════════════════════════════════════════════
# Event record
# ══════════════════════════════════════════════════════════════════


@dataclass
class Event:
    """One pipeline event.

    event_type is a short snake_case string like "step_started" or
    "derivation_step". Payload is an arbitrary JSON-serialisable dict.
    Timestamp is UNIX seconds (float, millisecond precision).
    """
    event_type: str
    run_id: str
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# ══════════════════════════════════════════════════════════════════
# Event bus
# ══════════════════════════════════════════════════════════════════


class EventBus:
    """Thread-safe fan-out of events to all registered listeners.

    Listeners are called serially under a lock so a file writer and a
    terminal printer can't interleave mid-event. Listener calls are
    fast (single line write, single print); serial dispatch is fine.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[Event], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def clear_listeners(self) -> None:
        with self._lock:
            self._listeners.clear()

    def emit(self, event: Event) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception as e:
                # Listener errors MUST NOT break the pipeline.
                name = getattr(fn, "__qualname__", None) or repr(fn)
                print(f"[event_bus] listener {name} failed: {e}",
                      flush=True)


# Module-level singleton. Import this from callers.
bus = EventBus()


# ══════════════════════════════════════════════════════════════════
# Built-in listeners
# ══════════════════════════════════════════════════════════════════


class JSONLFileWriter:
    """Appends each event as one JSON line to a file.

    Open once, keep file handle for the lifetime of the run. Flush
    after every write so a frontend tailing the file sees events
    immediately.
    """

    __slots__ = ("path", "_fh", "_lock", "_run_id")

    def __init__(self, path: Path, run_id: str | None = None) -> None:
        """Open a writer.

        `run_id`, when set, makes this writer FILTER events: only events
        whose `event.run_id == run_id` are written.  This is critical
        for concurrent pipeline runs — without filtering, two writers
        installed simultaneously (one per run) would each receive every
        event from every run and double-write everything to both files.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on open — one file per run, not appended across runs.
        self._fh = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._run_id = run_id

    def __call__(self, event: Event) -> None:
        # Per-run filter: silently drop events that aren't for this run.
        if self._run_id is not None and event.run_id != self._run_id:
            return
        line = json.dumps(event.to_dict(), default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                # fsync may not be supported on all filesystems; fine.
                pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class TerminalPrettyPrinter:
    """Human-readable stdout rendering of events.

    Kept intentionally simple — each event type has a format in the
    dispatch dict. Unknown event types fall through to a generic
    renderer so nothing is lost.
    """

    __slots__ = ("_lock", "_step_start_times", "_run_id")

    def __init__(self, run_id: str | None = None) -> None:
        """Console-friendly event renderer.

        `run_id`, when set, filters output to events for THIS run only —
        without it, concurrent runs interleave their step banners on
        stdout and you can't tell which run printed which line.
        """
        self._lock = threading.Lock()
        # step_n → start timestamp, so step_completed can compute duration
        self._step_start_times: dict[Any, float] = {}
        self._run_id = run_id

    def __call__(self, event: Event) -> None:
        # Per-run filter
        if self._run_id is not None and event.run_id != self._run_id:
            return
        et = event.event_type
        p = event.payload
        with self._lock:
            # Dispatch. Default: generic one-liner.
            formatter = _FORMATTERS.get(et)
            if formatter is None:
                msg = self._format_generic(et, p)
            else:
                msg = formatter(self, event, p)
            if msg:
                print(msg, flush=True)

    # ── Internal helpers used by formatters ──────────────────────

    def _format_generic(self, et: str, p: dict[str, Any]) -> str:
        # Compact dict preview for unknown event types
        preview = ", ".join(f"{k}={str(v)[:60]}" for k, v in p.items())
        return f"  • {et}  {preview[:200]}"

    def _track_step_start(self, step_n: Any, ts: float) -> None:
        self._step_start_times[step_n] = ts

    def _step_duration(self, step_n: Any, now: float) -> float:
        t0 = self._step_start_times.get(step_n, now)
        return now - t0


# ── Per-event-type formatters ────────────────────────────────────
#
# Each formatter takes (printer, event, payload) and returns a string
# to print (or "" / None to suppress). Keep these terse — this is the
# terminal, not the manuscript.

def _fmt_run_started(pp, e, p):
    q = str(p.get("query", ""))
    return (
        "\n" + "═" * 66 +
        f"\n  RUN STARTED  run_id={p.get('run_id', e.run_id)}" +
        "\n" + "═" * 66 +
        f"\n  Query: {q[:200]}" + ("…" if len(q) > 200 else "")
    )


def _fmt_run_completed(pp, e, p):
    return (
        "\n" + "═" * 66 +
        f"\n  RUN COMPLETED  elapsed={p.get('elapsed_s', '?')}s" +
        f"  cost=${p.get('cost_usd', 0):.3f}" +
        f"  calls={p.get('calls', '?')}" +
        "\n" + "═" * 66
    )


def _fmt_step_started(pp, e, p):
    pp._track_step_start(p.get("step_n"), e.timestamp)
    n = p.get("step_n", "?")
    title = p.get("title") or p.get("name", "?")
    return f"\n▶ [{n}] {title} ..."


def _fmt_step_completed(pp, e, p):
    dur = pp._step_duration(p.get("step_n"), e.timestamp)
    summary = p.get("summary", "")
    return f"  ✓ {p.get('name', '?')} done ({dur:.0f}s) — {summary}"


def _fmt_step_failed(pp, e, p):
    return f"  ✗ {p.get('name', '?')} FAILED — {p.get('error', '')}"


def _fmt_llm_call_started(pp, e, p):
    task = p.get("task", "?")
    model = p.get("model", "")
    return f"    → LLM[{task}] {model}"


def _fmt_llm_call_completed(pp, e, p):
    task = p.get("task", "?")
    tin = p.get("tokens_in", 0)
    tout = p.get("tokens_out", 0)
    cost = p.get("cost_usd", 0.0)
    return f"    ← LLM[{task}] {tin}→{tout} tok  ${cost:.4f}"


def _fmt_chunk_retrieved(pp, e, p):
    pid = p.get("paper_id", "?")
    page = p.get("page", "?")
    score = p.get("score", 0.0)
    preview = str(p.get("preview", ""))[:80]
    return f"      chunk [{pid}::p{page} s={score:.2f}]  {preview}"


def _fmt_model_shortlisted(pp, e, p):
    return f"      model: {p.get('name', '?')}  [{p.get('type', '?')}]"


def _fmt_equation_extracted(pp, e, p):
    eq = str(p.get("equation", ""))[:120]
    return f"      eq: {eq}"


def _fmt_code_executed(pp, e, p):
    mn = p.get("model_name", "?")
    ok = p.get("ok", True)
    return f"      code[{mn}] {'OK' if ok else 'FAIL'}"


def _fmt_derivation_step(pp, e, p):
    lhs = p.get("lhs", "?")
    expr = p.get("expression", "")
    result = p.get("result", "?")
    return f"      {lhs} = {expr} = {result}"


def _fmt_physics_check(pp, e, p):
    model = p.get("model", "?")
    verdict = p.get("verdict", "?")
    return f"      physics[{model}] {verdict}"


def _fmt_critic_issue(pp, e, p):
    sev = p.get("severity", "?")
    kind = p.get("type", "?")
    desc = str(p.get("description", ""))[:120]
    return f"      ! [{sev}] {kind}: {desc}"


def _fmt_section_written(pp, e, p):
    key = p.get("section_key", "?")
    n_chars = p.get("n_chars", 0)
    return f"      section {key} written ({n_chars} chars)"


def _fmt_citation_rejected(pp, e, p):
    ref = p.get("reference", "?")
    reason = p.get("reason", "?")
    return f"    ⚠ citation_rejected: {ref} ({reason})"


_FORMATTERS: dict[str, Callable[..., str]] = {
    "run_started":         _fmt_run_started,
    "run_completed":       _fmt_run_completed,
    "step_started":        _fmt_step_started,
    "step_completed":      _fmt_step_completed,
    "step_failed":         _fmt_step_failed,
    "llm_call_started":    _fmt_llm_call_started,
    "llm_call_completed":  _fmt_llm_call_completed,
    "chunk_retrieved":     _fmt_chunk_retrieved,
    "model_shortlisted":   _fmt_model_shortlisted,
    "equation_extracted":  _fmt_equation_extracted,
    "code_executed":       _fmt_code_executed,
    "derivation_step":     _fmt_derivation_step,
    "physics_check":       _fmt_physics_check,
    "critic_issue":        _fmt_critic_issue,
    "section_written":     _fmt_section_written,
    "citation_rejected":   _fmt_citation_rejected,
}


# ══════════════════════════════════════════════════════════════════
# Default-listener management
# ══════════════════════════════════════════════════════════════════
#
# The orchestrator calls install_default_listeners() at run start and
# close_default_listeners() at run end.
#
# CONCURRENT-SAFE since the cluster fix.  Previously the module held a
# single-slot global writer + pretty-printer, so when Run B started
# while Run A was still running, the install_default_listeners call
# at the top of Run B would CLOSE Run A's writer mid-write — Run A's
# events.jsonl was truncated, and the pretty-printer for Run A stopped
# emitting to stdout.  We now store listeners in a dict keyed by
# run_id; each install adds a per-run filtered listener pair, each
# close removes ONLY that run's pair.

_installed_listeners: dict[str, tuple[JSONLFileWriter, TerminalPrettyPrinter]] = {}
_installed_lock:      threading.Lock                                            = threading.Lock()


def install_default_listeners(run_id: str, events_file: Path) -> None:
    """Wire up file + terminal listeners for `run_id`.

    Safe under concurrent runs: each (run_id) gets its own listener
    pair, and the writer/printer filter on `event.run_id` so they only
    receive events for their own run.  Calling for the same run_id a
    second time replaces the previous listener pair for that run only
    (other runs are unaffected).
    """
    with _installed_lock:
        # Same-run re-install: drop the old pair first.  Other runs'
        # listeners stay in place.
        old = _installed_listeners.pop(run_id, None)
        if old is not None:
            fw_old, pp_old = old
            bus.remove_listener(fw_old)
            try:
                fw_old.close()
            except Exception:
                pass
            bus.remove_listener(pp_old)

        fw = JSONLFileWriter(events_file, run_id=run_id)
        pp = TerminalPrettyPrinter(run_id=run_id)
        bus.add_listener(fw)
        bus.add_listener(pp)
        _installed_listeners[run_id] = (fw, pp)


def close_default_listeners(run_id: str | None = None) -> None:
    """Remove and close listeners for `run_id` (or ALL runs when None).

    `run_id=None` is the legacy "close everything" behaviour — preserved
    so test fixtures and one-shot scripts still work.  Production
    orchestrators should pass their run_id so they don't kill listeners
    belonging to other concurrently-running runs.
    """
    with _installed_lock:
        if run_id is None:
            # Legacy / test path: close every installed pair.
            for rid in list(_installed_listeners.keys()):
                fw, pp = _installed_listeners.pop(rid)
                bus.remove_listener(fw)
                try:
                    fw.close()
                except Exception:
                    pass
                bus.remove_listener(pp)
            return
        pair = _installed_listeners.pop(run_id, None)
        if pair is None:
            return
        fw, pp = pair
        bus.remove_listener(fw)
        try:
            fw.close()
        except Exception:
            pass
        bus.remove_listener(pp)


# ══════════════════════════════════════════════════════════════════
# Convenience emit helper
# ══════════════════════════════════════════════════════════════════


def emit(event_type: str, *, run_id: str | None = None, **payload: Any) -> None:
    """Shorthand: bus.emit(Event(event_type, run_id, payload=payload)).

    If `run_id` is omitted, uses the value from the current context
    (set by set_current_run). This is how the LLM router and any node
    that doesn't carry run_id explicitly can still emit events —
    the orchestrator sets the run_id at the top of the pipeline, and
    every downstream emit picks it up.

    Call sites look like:

        emit("step_started", step_n=5, name="retrieval",
             title="Literature retrieval")
    """
    rid = run_id if run_id is not None else current_run_id.get()
    if rid is None:
        # No run context set — silently drop rather than crash.
        # This happens in unit tests and in the smoke script, neither
        # of which has a WebSocket listener anyway.
        return
    bus.emit(Event(event_type=event_type, run_id=rid, payload=payload))


# ══════════════════════════════════════════════════════════════════
# Current-run context (ContextVar so it survives ThreadPoolExecutor
# context-copy on Python 3.11+)
# ══════════════════════════════════════════════════════════════════

current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bl_pipeline_current_run_id", default=None,
)


def set_current_run(run_id: str) -> None:
    """Mark which run is currently active. Call once at run start.
    After this, any `emit(...)` without explicit run_id picks this up.
    """
    current_run_id.set(run_id)


def clear_current_run() -> None:
    """Reset the active-run context. Call after the run completes."""
    current_run_id.set(None)


# Agent role for the current execution context.  Used by helpers like
# `_emit_a3_step()` in agent3_experiment/api_adapter.py to tag their
# step_started events with the correct agent name when the SAME Python
# code path is reached from two different API entry points:
#   /api/agent3/run → pilot wrapper → sets current_agent_role="agent3"
#   /api/agent4/run → full wrapper  → sets current_agent_role="agent4"
# Default "agent3" preserves pre-split behaviour for callers (CLI, tests,
# direct imports) that don't bother setting the context.
current_agent_role: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bl_pipeline_current_agent_role", default="agent3",
)


def set_current_agent_role(role: str) -> None:
    """Set the agent label used by emit() helpers for the current run.
    Pair with reset_current_agent_role() in a try/finally."""
    current_agent_role.set(role)


def reset_current_agent_role() -> None:
    """Restore default ("agent3"); call after the agent's work finishes."""
    current_agent_role.set("agent3")
