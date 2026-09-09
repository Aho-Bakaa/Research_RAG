"""tools.py — tool registry for the ReAct reasoner.

The reasoner LLM has three tools it can call mid-reasoning.  Each tool is
a pure Python function:

    compute(code)              → run sandboxed Python; return result + stdout
    lookup_glossary(symbol,    → resolve a symbol's canonical meaning + units
                    paper_id)    using canonical_symbols.yaml + paper-aware
                                 normalizer
    search(query)              → one-shot hybrid retrieval; return chunks

Production-grade contract for every tool:
  • Never raises.  All exceptions are caught and returned as
    {"error": "...", "_tool": tool_name} so the reasoner can react.
  • Returns a JSON-serializable dict.  No numpy types, no datetimes.
  • Bounded cost — `search` returns at most 8 chunks; `compute` runs in
    a sandbox with a wall-clock cap; `lookup_glossary` is purely local.
  • Logged via event_bus — every call emits start + done events so the
    UI shows the live tool trace.

The reasoner calls a tool by emitting:

    <tool_call>
    {"tool": "compute", "args": {"code": "..."}}
    </tool_call>

The orchestrator in reason.py routes that through `dispatch(tool_call)`
to the matching function below.
"""
from __future__ import annotations

import time
from typing import Any

from bl_pipeline.agent1_fresh.state import emit_event


# ══════════════════════════════════════════════════════════════════════
# Tool registry — name → (callable, doc-string snippet for prompt help)
# ══════════════════════════════════════════════════════════════════════

# The registry is built at the bottom of this file once the tool
# functions are defined.  Keep imports local-ish for production hygiene.


# ──────────────────────────────────────────────────────────────────────
# Tool 1: compute(code)
# ──────────────────────────────────────────────────────────────────────

def compute(code: str) -> dict[str, Any]:
    """Run Python in a sandbox and return the result dict + stdout.

    Conventions
    ───────────
    The code MUST put its final answers in a top-level dict named `result`.
    Available imports: numpy, scipy, sympy, math.  Other imports are
    blocked by the safety AST check.

    Returns
    ───────
    {
      "stdout":  str,           # captured stdout
      "result":  dict | None,   # the `result` dict from the code, if defined
      "error":   str | None,    # exception message if anything failed
      "elapsed_s": float,
    }

    Never raises — exceptions inside the code become "error" strings so
    the reasoner can decide to fix and retry.
    """
    # Moved to bl_pipeline/shared/ on 2026-05-17 to unblock the
    # archive of bl_pipeline/agent1_transition/.  The original copy
    # lives in agent1_transition/tools/ for the legacy backend path;
    # agent1_fresh now uses the shared copy so the two are
    # decoupled.  Both files are byte-identical; if you change one,
    # update the other (or, once transition is archived, delete the
    # old copy).
    from bl_pipeline.shared.code_executor import (
        execute_code, validate_code_safety,
    )
    from bl_pipeline.shared.json_utils import strip_trailing_prose

    t0 = time.time()
    # Self-heal any trailing prose the LLM might have appended (paranoia
    # — the prompt tells it to fence; but if it forgets we still survive).
    code = strip_trailing_prose(code or "")
    if not code.strip():
        # Bug 2 from run #4: Sonnet sometimes emits empty code blocks
        # (likely due to its own confusion about what to do next, or a
        # truncated tool_call payload).  The previous one-word "empty
        # code" error didn't tell it how to recover, and it sent ANOTHER
        # empty block on the next turn.  This message is action-oriented
        # so Sonnet immediately knows what to do.
        return {
            "stdout": "", "result": None,
            "error": (
                "EMPTY_CODE: the compute() tool received no code.  This "
                "usually means your <tool_call> block had `args: {\"code\": \"\"}` "
                "or no `code` field at all.  Recover by re-emitting the "
                "tool_call with the full Python source in args.code.  "
                "Remember the JSON inside <tool_call> must be valid — "
                "if your code contains backticks or quotes, escape them "
                "or use a here-doc style triple-quoted string."
            ),
            "_empty_code": True,
            "elapsed_s": 0.0,
        }

    # Safety AST check (forbidden imports, exec, etc.)
    safe, reason = validate_code_safety(code)
    if not safe:
        return {
            "stdout": "", "result": None,
            "error": f"safety check failed: {reason}",
            "elapsed_s": round(time.time() - t0, 3),
        }

    # Formula-verifier pre-execution gate.
    #
    # WHY: in run #12 Sonnet hardcoded `Re_theta_t = 172.80` at the top
    # of a compute() script and decorated it with a print statement that
    # *looked* like a Mayle Eq.(9) evaluation but never actually
    # multiplied anything by 400.  The manuscript shipped the wrong
    # number.  This gate AST-walks the code, rejects literal assignments
    # to tracked result variables, requires every formula to carry a
    # `# paper_id::Eq.(N)` tag, fetches the glossary entry for that
    # tag, and asks a Haiku judge whether the Python expression matches
    # the paper formula.  Mismatch → reject with side-by-side comparison
    # so Sonnet can fix and re-emit.
    #
    # Opt-out env: BL_FORMULA_VERIFIER=off skips the gate (useful for
    # the legacy non-fresh agents whose code style doesn't use tags yet).
    from bl_pipeline.shared.config import formula_verifier_enabled
    vresult = None
    if formula_verifier_enabled():
        try:
            from bl_pipeline.agent1_fresh.formula_verifier import verify
            vresult = verify(code)
        except Exception as _e:
            # Verifier infra failure — fail OPEN (run the code anyway)
            # so a bug in the verifier doesn't brick the pipeline.
            # The event_bus log captures the failure for debugging.
            emit_event(
                "formula_verifier_infra_error",
                error=f"{type(_e).__name__}: {_e}",
            )
            vresult = None

        if vresult is not None:
            emit_event(
                "formula_verifier_done",
                passed=vresult.passed,
                n_sites=len(vresult.sites),
                n_literals=sum(1 for s in vresult.sites if s.is_literal),
                n_untagged=sum(
                    1 for s in vresult.sites
                    if not s.is_literal and (not s.paper_id or not s.eq_id)
                ),
                n_judge_mismatches=sum(
                    1 for s in vresult.sites
                    if s.judge_verdict in ("mismatch", "uncertain")
                ),
                elapsed_s=vresult.elapsed_s,
                judge_cost_usd=vresult.judge_cost_usd,
            )
            if not vresult.passed:
                return {
                    "stdout": "", "result": None,
                    "error": vresult.error_message,
                    "_formula_verifier_rejected": True,
                    "elapsed_s": round(time.time() - t0, 3),
                    "judge_cost_usd": vresult.judge_cost_usd,
                }

    try:
        exec_result = execute_code(code)
    except Exception as e:
        return {
            "stdout": "", "result": None,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t0, 3),
        }

    out = {
        "stdout":  exec_result.get("stdout", "") or "",
        "result":  exec_result.get("result"),
        "error":   exec_result.get("error"),
        "elapsed_s": round(time.time() - t0, 3),
    }
    # Attach formula-verification provenance (execute-&-flag): which
    # computed quantities were matched to a cited paper-formula ("match" /
    # "derived") vs flagged ("mismatch" / "unverified_*").  The
    # final-number provenance gate and the writer's ledger use this so an
    # UNVERIFIED value can never ship looking trustworthy.  Warnings are
    # also surfaced to the reasoner so it can fix the formula on a later
    # turn if the quantity matters.
    if vresult is not None:
        if vresult.warnings:
            out["_formula_warnings"] = list(vresult.warnings)
        if vresult.sites:
            out["_formula_sites"] = {
                s.variable: (s.judge_verdict or "unchecked")
                for s in vresult.sites
            }
    return out


# ──────────────────────────────────────────────────────────────────────
# Tool 2: lookup_glossary(symbol, paper_id=None)
# ──────────────────────────────────────────────────────────────────────

# Lazy singleton for the normalizer — it loads canonical_symbols.yaml
# (~3000 lines).  Reuse across calls.
_NORMALIZER = None


def _get_normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        from bl_pipeline.rag.symbol_normalization.normalizer import (
            SymbolNormalizer,
        )
        _NORMALIZER = SymbolNormalizer()
    return _NORMALIZER


def lookup_glossary(
    symbol: str, paper_id: str | None = None,
) -> dict[str, Any]:
    """Resolve `symbol`'s canonical meaning and unit.

    With `paper_id`, the paper-scoped aliases run first — useful when a
    paper uses non-standard notation (e.g. AGS uses `τ_t` to mean Tu in
    percent; Mayle uses `τ_0` to mean wall shear; Walters uses `k_T` to
    mean turbulent KE).

    Returns
    ───────
    {
      "symbol":            the input symbol (unchanged),
      "canonical_form":    canonical name in our glossary, or "" if unknown,
      "unit":              units string, e.g. "m/s", "-", "%"
      "description":       1-2 sentence plain-language definition,
      "paper_scoped":      bool — whether paper_id-specific aliasing fired,
      "error":             None on success
    }
    """
    sym = (symbol or "").strip()
    if not sym:
        return {
            "symbol": "", "canonical_form": "",
            "unit": "", "description": "",
            "paper_scoped": False, "error": "empty symbol",
        }

    try:
        normalizer = _get_normalizer()
    except Exception as e:
        return {
            "symbol": sym, "canonical_form": "",
            "unit": "", "description": "",
            "paper_scoped": False,
            "error": f"normalizer load failed: {e}",
        }

    # Normalize the input symbol (paper-aware) and look up canonical info.
    normalized, _log = normalizer.normalize_text_with_log(
        sym, paper_id=paper_id,
    )
    paper_scoped = bool(_log and any(entry.get("paper_scoped") for entry in _log))

    info = normalizer.canonical_info(normalized) if hasattr(
        normalizer, "canonical_info",
    ) else {}
    # Fallback: derive canonical entry by scanning the YAML directly.
    if not info:
        for entry in (getattr(normalizer, "_canonicals", []) or []):
            if entry.get("canonical") == normalized:
                info = entry
                break

    return {
        "symbol":         sym,
        "canonical_form": str(info.get("canonical", normalized) or normalized),
        "unit":           str(info.get("unit", "") or ""),
        "description":    str(info.get("description", "") or ""),
        "paper_scoped":   paper_scoped,
        "error":          None,
    }


# ──────────────────────────────────────────────────────────────────────
# Tool 3: search(query)
# ──────────────────────────────────────────────────────────────────────

# Lazy singleton for the RAG engine — instantiating it loads the
# Nemotron-8B embedder (~6s) and the Chroma client.  Reuse across every
# search() call instead of rebuilding per-tool-call.  Saved ~15s/run
# (the agent kept reloading Nemotron on each mid-reasoning search).
_RAG_ENGINE = None


def _get_rag_engine():
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        from bl_pipeline.rag.engine import RAGEngine
        _RAG_ENGINE = RAGEngine()
    return _RAG_ENGINE


def search(query: str, top_k: int = 8) -> dict[str, Any]:
    """Pull additional chunks for a mid-reasoning query.

    Hybrid search (BM25 + vector + metadata weighting) across all active
    collections, then return the top-`top_k` ranked results — without
    invoking the LLM judge or cross-encoder rerank (those belong to the
    main retrieval pipeline, not mid-reasoning lookups).

    The reasoner uses this when it discovers a gap mid-reasoning and
    doesn't want to bail out and trigger a full retrieval iteration.

    Returns
    ───────
    {
      "query":     the input query,
      "n":         number of chunks returned,
      "chunks":    [ {chunk_id, paper_id, page, content, collection,
                      raw_distance}, ... ],
      "error":     None on success
    }
    """
    q = (query or "").strip()
    if not q:
        return {"query": "", "n": 0, "chunks": [], "error": "empty query"}

    try:
        from bl_pipeline.agent1_fresh.retrieve import _hybrid_search
        rag = _get_rag_engine()    # singleton — no model reload per call
        chunks = _hybrid_search(rag, [q], top_k_per_query=top_k)
    except Exception as e:
        return {
            "query": q, "n": 0, "chunks": [],
            "error": f"{type(e).__name__}: {e}",
        }

    out = []
    for c in chunks[:top_k]:
        out.append({
            "chunk_id":   c.chunk_id,
            "paper_id":   c.paper_id,
            "page":       c.page,
            "content":    (c.content or "")[:1500],   # cap for token budget
            "collection": c.collection,
        })
    return {"query": q, "n": len(out), "chunks": out, "error": None}


# ──────────────────────────────────────────────────────────────────────
# Tool 4: lookup_equation(paper_id, eq_id=None, content_match=None)
# ──────────────────────────────────────────────────────────────────────

# Lazy cache for per-paper equation indices.  Each entry is a list of
# {paper_id, eq_id, page, formula_latex_verbatim, tagged, ...}.
_EQUATION_INDICES: dict[str, list[dict]] = {}

# Lazy-loaded alias map: { paper_id: { canonical_cited_eq_id: glossary_eq_id } }
# Used to bridge the gap when a paper publishes an equation without a
# \tag{N} macro (so our extractor files it as "Eq. (untagged pN #M)")
# but Sonnet naturally cites it by its paper-text number, e.g.
# "Mayle Eq.(9)" for the 400·Tu^(-5/8) onset formula.
# See runs/_logs/equation_indices/_aliases.json
_EQUATION_ALIASES: dict[str, dict[str, str]] | None = None


def _load_equation_aliases() -> dict[str, dict[str, str]]:
    """Load the alias map once.  Returns {} if the file is missing or malformed."""
    global _EQUATION_ALIASES
    if _EQUATION_ALIASES is not None:
        return _EQUATION_ALIASES
    import json
    from pathlib import Path
    path = Path("runs/_logs/equation_indices/_aliases.json")
    if not path.exists():
        _EQUATION_ALIASES = {}
        return _EQUATION_ALIASES
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Drop the underscore-prefixed metadata keys (_description, _notes)
        _EQUATION_ALIASES = {
            k: v for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict)
        }
        return _EQUATION_ALIASES
    except Exception:
        _EQUATION_ALIASES = {}
        return _EQUATION_ALIASES


def _load_equation_index(paper_id: str) -> list[dict]:
    """Load runs/_logs/equation_indices/<paper_id>.json with caching."""
    if paper_id in _EQUATION_INDICES:
        return _EQUATION_INDICES[paper_id]
    import json
    from pathlib import Path
    path = Path(f"runs/_logs/equation_indices/{paper_id}.json")
    if not path.exists():
        _EQUATION_INDICES[paper_id] = []
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            _EQUATION_INDICES[paper_id] = []
            return []
        _EQUATION_INDICES[paper_id] = data
        return data
    except Exception:
        _EQUATION_INDICES[paper_id] = []
        return []


def lookup_equation(
    paper_id: str,
    eq_id: str | None = None,
    content_match: str | None = None,
) -> dict[str, Any]:
    """Look up a specific equation from a paper's deterministic index.

    Built once at ingestion time by scripts/extract_equations_oneshot.py;
    reads from runs/_logs/equation_indices/<paper_id>.json.  Zero
    embedding cost, zero LLM cost — returns the verbatim formula
    (including the paper's original symbol notation, e.g. `R_XS` not the
    symbol-normalized `Re_xt`).

    Three lookup modes:

      1. paper_id + eq_id  (preferred):
            lookup_equation("abu_ghannam_shaw_1980", "Eq.17")
            → returns that specific equation verbatim.
            Accepts also tag-less ids the extractor assigned:
            "Eq.untagged-p10-#1" for formulas that lacked `\\tag{N}`.

      2. paper_id + content_match:
            lookup_equation("mayle_1991", content_match="400")
            → returns first equation whose verbatim formula contains
            the substring.  Useful when the eq_id is unknown (most
            common for Mayle Eq.9 which had no `\\tag{9}`).

      3. paper_id only:
            lookup_equation("abu_ghannam_shaw_1980")
            → returns the index header (eq_id list with page numbers,
            NO formulas) so the reasoner can pick which to fetch in a
            second call.  Cheaper than dumping every equation.

    Returns
    ───────
    Single match:
      {
        "paper_id": str,
        "eq_id":    str,
        "page":     int,
        "formula_latex_verbatim": str,
        "tagged":   bool,           # had explicit \\tag{} in source
        "describes": str,            # from paper_summary if available
        "validity":  str,            # from paper_summary if available
        "error":    None,
      }

    Header-only (mode 3) or multi-match (mode 2 returns up to 5):
      {
        "paper_id": str,
        "matches": [ {eq_id, page, formula_latex_verbatim (only if mode 2)}, ... ],
        "n_total":  int,             # total equations in the paper
        "error":    None,
      }

    Error:
      {"error": "...", "paper_id": str}
    """
    # Normalise paper_id: lowercase + strip.  This makes lookups
    # portable across filesystems (Windows is case-insensitive but
    # Linux/Mac glossaries are case-sensitive — without this normaliser
    # `MAYLE_1991` would accidentally resolve on Windows and silently
    # fail on Linux).
    pid = (paper_id or "").strip().lower()
    if not pid:
        return {"error": "paper_id is required", "paper_id": ""}

    equations = _load_equation_index(pid)
    if not equations:
        return {
            "error": (f"no equation index for paper_id={pid!r} — "
                      f"either the paper isn't in the priority-10 list or "
                      f"the extractor hasn't been run on it yet."),
            "paper_id": pid,
        }

    # Mode 1: eq_id specified — case-insensitive, whitespace/parens-tolerant
    # Bug 4 from run #4: Sonnet called lookup_equation with
    # eq_id="Eq. (9)" (using the original paper's parenthesised notation
    # for citing equations).  My normaliser stripped spaces but left the
    # "()" so the match failed, even though "Eq.9" was in the index.
    # Strip BOTH parens AND spaces from BOTH sides of the comparison.
    def _norm_eq_id(s: str) -> str:
        return (s.strip().lower()
                  .replace(" ", "")
                  .replace("(", "")
                  .replace(")", ""))
    if eq_id:
        eq_norm = _norm_eq_id(eq_id)
        for eq in equations:
            if _norm_eq_id(eq.get("eq_id", "")) == eq_norm:
                return {
                    "paper_id":               eq.get("paper_id", pid),
                    "eq_id":                  eq.get("eq_id", ""),
                    "page":                   eq.get("page"),
                    "formula_latex_verbatim": eq.get("formula_latex_verbatim", ""),
                    "tagged":                 bool(eq.get("tagged")),
                    "describes":              eq.get("describes", ""),
                    "validity":               eq.get("validity", ""),
                    "error":                  None,
                }

        # Alias fallback — bridge "paper-cited Eq.(N)" → "extractor's
        # untagged-pN-#M" filing for equations that the source PDF
        # emitted without a \tag{N} macro.  Hand-curated in
        # runs/_logs/equation_indices/_aliases.json.
        aliases = _load_equation_aliases().get(pid, {})
        # Match aliases case-insensitively, whitespace-tolerant, like the
        # main matcher.
        for alias_key, alias_target in aliases.items():
            if _norm_eq_id(alias_key) == eq_norm:
                alias_target_norm = _norm_eq_id(alias_target)
                for eq in equations:
                    if _norm_eq_id(eq.get("eq_id", "")) == alias_target_norm:
                        return {
                            "paper_id":               eq.get("paper_id", pid),
                            "eq_id":                  eq.get("eq_id", ""),
                            "page":                   eq.get("page"),
                            "formula_latex_verbatim": eq.get("formula_latex_verbatim", ""),
                            "tagged":                 bool(eq.get("tagged")),
                            "describes":              eq.get("describes", ""),
                            "validity":               eq.get("validity", ""),
                            "_resolved_via_alias":    f"{alias_key} → {alias_target}",
                            "error":                  None,
                        }
                # Alias points to a glossary key that doesn't exist —
                # fall through to the standard not-found response so
                # Sonnet sees a useful message.
                break

        # Not found by exact match or alias — helpful error
        avail = sorted({e.get("eq_id", "") for e in equations})[:30]
        return {
            "error": (f"eq_id={eq_id!r} not found in paper {pid!r}. "
                      f"Available eq_ids include: {avail}. "
                      f"Try content_match=<substring> if you don't know the eq_id."),
            "paper_id": pid,
        }

    # Mode 2: content substring search
    if content_match:
        needle = content_match.strip()
        if not needle:
            return {"error": "empty content_match", "paper_id": pid}
        hits = []
        for eq in equations:
            if needle in eq.get("formula_latex_verbatim", ""):
                hits.append({
                    "paper_id":               eq.get("paper_id", pid),
                    "eq_id":                  eq.get("eq_id", ""),
                    "page":                   eq.get("page"),
                    "formula_latex_verbatim": eq.get("formula_latex_verbatim", ""),
                    "tagged":                 bool(eq.get("tagged")),
                    "describes":              eq.get("describes", ""),
                    "validity":               eq.get("validity", ""),
                })
                if len(hits) >= 5:
                    break
        if not hits:
            return {
                "paper_id": pid,
                "matches":  [],
                "n_total":  len(equations),
                "error":    f"no equation in {pid!r} contains substring {needle!r}",
            }
        return {
            "paper_id": pid,
            "matches":  hits,
            "n_total":  len(equations),
            "error":    None,
        }

    # Mode 3: header only — list all eq_ids with pages
    return {
        "paper_id": pid,
        "matches":  [
            {"eq_id": e.get("eq_id"), "page": e.get("page"),
             "tagged": bool(e.get("tagged")),
             "describes": e.get("describes", "")[:80]}
            for e in equations
        ],
        "n_total":  len(equations),
        "error":    None,
    }


# ══════════════════════════════════════════════════════════════════════
# Dispatcher — single entry point for the reasoner
# ══════════════════════════════════════════════════════════════════════

TOOLS: dict[str, Any] = {
    "compute":          compute,
    "lookup_glossary":  lookup_glossary,
    "lookup_equation":  lookup_equation,
    "search":           search,
}

TOOL_DESCRIPTIONS_FOR_PROMPT: dict[str, str] = {
    "compute": (
        "compute(code: str) → {stdout, result, error, elapsed_s}.  "
        "Run sandboxed Python.  Available: numpy, scipy, sympy, math.  "
        "Put final answers in a top-level dict named `result`."
    ),
    "lookup_glossary": (
        "lookup_glossary(symbol: str, paper_id: str | None = None) "
        "→ {canonical_form, unit, description, paper_scoped, error}.  "
        "Resolve a symbol's meaning; pass paper_id for paper-specific "
        "aliasing (e.g. τ_t means Tu in AGS but Reynolds-stress shear "
        "elsewhere)."
    ),
    "lookup_equation": (
        "lookup_equation(paper_id: str, eq_id: str | None = None, "
        "content_match: str | None = None) → {paper_id, eq_id, page, "
        "formula_latex_verbatim, tagged, describes, validity, error}.  "
        "PREFER THIS over `search` when you know the paper and equation "
        "you need (e.g. AGS Eq.17, Mayle Eq.9, Fransson Eq.5.5).  "
        "Returns the VERBATIM formula from the paper's markdown cache — "
        "no embeddings, no rerank, no LLM cost.  Three modes: (1) eq_id "
        "for exact lookup; (2) content_match for substring search "
        "(useful for un-tagged equations like Mayle Eq.9 which is "
        "indexed as 'Eq.untagged-p10-#1' but contains '400'); (3) "
        "paper_id only to list available eq_ids."
    ),
    "search": (
        "search(query: str, top_k: int = 8) → {query, n, chunks, error}.  "
        "Pull additional chunks for a mid-reasoning query.  Use when you "
        "discover an evidence gap and need more context without bailing.  "
        "If you just need ONE specific equation, prefer `lookup_equation`."
    ),
}


def dispatch(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Single dispatch entry the reasoner uses.

    Production-grade: never raises.  Unknown tool → error response.
    Bad-shape args → error response.  Tool exception → error response.
    """
    t0 = time.time()
    # Build a safely-displayable args dict for telemetry — `args` may
    # be a non-dict if the caller mis-passes (e.g. a stringified JSON
    # that wasn't parsed).  Tolerate both for production sanity.
    if isinstance(args, dict):
        safe_args = {k: (str(v)[:200] if isinstance(v, str) else v)
                     for k, v in args.items()}
    else:
        safe_args = {"_raw_args_type": type(args).__name__,
                     "_raw_args_repr": str(args)[:200]}
    emit_event("tool_call_started", tool_name=tool_name, args=safe_args)

    fn = TOOLS.get(tool_name)
    if fn is None:
        result = {"error": f"unknown tool: {tool_name!r}"}
    elif not isinstance(args, dict):
        result = {"error": f"args must be a dict, got {type(args).__name__}"}
    else:
        try:
            result = fn(**args)
        except TypeError as e:
            # Bad keyword arguments
            result = {"error": f"bad args for {tool_name}: {e}"}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

    result.setdefault("_tool", tool_name)
    result.setdefault("_elapsed_s", round(time.time() - t0, 3))

    emit_event(
        "tool_call_done",
        tool_name=tool_name,
        error=result.get("error"),
        elapsed_s=result["_elapsed_s"],
    )
    return result
