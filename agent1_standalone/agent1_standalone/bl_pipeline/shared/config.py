"""Pipeline-level configuration loaded from environment and defaults."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv(override=True)    # override=True so an empty pre-set env var
                              # (e.g. inherited from a parent shell) gets
                              # replaced by the .env file's actual value

# ── Paths ─────────────────────────────────────────────────────────
# Standalone Agent-1 project.  TWO roots:
#   • PROJECT_ROOT     — THIS sub-project; owns the editable papers, the
#                        glossary, and any LOCALLY re-ingested vector DB.
#   • SHARED_DATA_ROOT — the main bl_transition_pipeline, used READ-ONLY as a
#                        fallback vector DB so A1 runs out of the box before
#                        you have re-ingested anything locally.  Override with
#                        the BL_DATA_ROOT env var.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DATA_ROOT = Path(
    os.environ.get("BL_DATA_ROOT") or r"C:\Projects\bl_transition_pipeline"
)
DATA_DIR = PROJECT_ROOT / "data"           # LOCAL — this sub-project's data
PAPERS_DIR = DATA_DIR / "papers"           # legacy / raw drop zone
EXPERIMENTAL_DATA_DIR = DATA_DIR / "experimental_data"
RUNS_DIR = DATA_DIR / "runs"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# ── RAG version selector ──────────────────────────────────────────
# RAG storage is versioned so multiple embedding-model corpora can
# coexist on disk and be A/B compared with a single env-var flip.
# Set BL_RAG_VERSION="" (or unset) to use the legacy unversioned
# layout (chroma_db/, primary_rag/, secondary_rag/) — preserves
# existing behaviour for anyone who hasn't migrated.
#
# When BL_RAG_VERSION="<tag>" is set, paths become
#   data/chroma_db_<tag>/
#   data/primary_rag_<tag>/
#   data/secondary_rag_<tag>/
#
# Recommended tags:
#   v1_openai_small  — text-embedding-3-small  (the legacy embedder)
#   v2_nemotron_8b   — nvidia/llama-embed-nemotron-8b
#
# Swap embedding versions live without overwriting the previous
# build by setting the env var and re-launching the worker:
#   $env:BL_RAG_VERSION = "v2_nemotron_8b"
#   uvicorn api.main:app --port 8124 --reload
#
# Semantics are deliberately strict to prevent accidental overwrite:
#   BL_RAG_VERSION=""        → legacy unversioned paths (chroma_db/,
#                              primary_rag/, secondary_rag/)
#   BL_RAG_VERSION="<tag>"   → versioned paths ONLY
#                              (chroma_db_<tag>/, etc.).  No fallback
#                              to legacy — a fresh ingest into a new
#                              tag is guaranteed to create its own
#                              directory rather than overwrite the
#                              legacy one.
#
# To migrate the existing legacy corpus to a tagged version:
#   1. Stop the running worker.
#   2. Rename data/chroma_db → data/chroma_db_v1_openai_small
#      (same for primary_rag, secondary_rag).
#   3. Set $env:BL_RAG_VERSION = "v1_openai_small".
#   4. Restart the worker — it now reads from the versioned paths.
RAG_VERSION: str = os.getenv("BL_RAG_VERSION", "").strip()


def _rag_path(base_name: str) -> Path:
    """Resolve a RAG storage directory under the active version
    selector.  See the BL_RAG_VERSION docstring above for semantics.

    Strict: when RAG_VERSION is set, ALWAYS returns the versioned
    path even if the directory doesn't exist yet (so a fresh ingest
    creates the new labelled directory rather than silently writing
    into the legacy one).
    """
    if RAG_VERSION:
        return DATA_DIR / f"{base_name}_{RAG_VERSION}"
    return DATA_DIR / base_name


PRIMARY_RAG_DIR   = _rag_path("primary_rag")     # LOCAL — your editable papers
SECONDARY_RAG_DIR = _rag_path("secondary_rag")

# ── Vector DB: safe local/shared split (standalone A1) ────────────
# LOCAL_CHROMA  — the ONLY vector store this project ever WRITES to, so a
#                 re-ingest here can never touch the main thesis project's DB.
# SHARED_CHROMA — the main project's prebuilt DB, used READ-ONLY as a fallback
#                 so retrieval works before you've built a local index.
# Rule: use LOCAL if it holds a built index (or BL_A1_FORCE_LOCAL_CHROMA=1,
#       which the ingest entry point sets); otherwise fall back to SHARED.
LOCAL_CHROMA  = _rag_path("chroma_db")
SHARED_CHROMA = (SHARED_DATA_ROOT / "data" /
                 (f"chroma_db_{RAG_VERSION}" if RAG_VERSION else "chroma_db"))


def _local_chroma_has_index() -> bool:
    try:
        return LOCAL_CHROMA.is_dir() and any(LOCAL_CHROMA.iterdir())
    except OSError:
        return False


if os.environ.get("BL_A1_FORCE_LOCAL_CHROMA") == "1" or _local_chroma_has_index():
    CHROMA_DIR = LOCAL_CHROMA
else:
    CHROMA_DIR = SHARED_CHROMA

# ── API keys ──────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
# Groq's free tier — used by the "thesis-development" routing for
# all schema-validated extraction / planning / critique tasks.  See
# llm_router.TASK_MODEL_MAP for the per-task assignment.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "")

# ── Budget ────────────────────────────────────────────────────────
BUDGET_LIMIT_USD: float = float(os.getenv("BUDGET_LIMIT_USD", "10.0"))

# ── LLM model IDs (pinned) ───────────────────────────────────────
MODELS = {
    "haiku":           "claude-haiku-4-5-20251001",
    "sonnet":          "claude-sonnet-4-6",
    "opus":            "claude-opus-4-6",
    "gpt4o":           "gpt-4o",
    "gpt4o_mini":      "gpt-4o-mini",
    "grok2":           "grok-2-latest",
    "groq_llama_70b":  "llama-3.3-70b-versatile",
    "groq_llama_8b":   "llama-3.1-8b-instant",
    # ── Local models (Ollama-hosted) ──────────────────────────────
    # Qwen2.5-Coder 32B is the primary thesis-pipeline reasoner:
    #   - Specifically trained for code generation (matches gpt-4o on
    #     HumanEval, beats it on instruction following at this size).
    #   - Fits in ~23 GB at Q5_K_M, well within 128 GB system RAM.
    #   - Fully local: reproducible (weights don't change underneath
    #     us), no API rate limits, no per-call cost, no vendor lockin.
    # To enable, install Ollama and `ollama pull qwen2.5-coder:32b`.
    # The router dispatches any model name starting with "qwen" to
    # the Ollama OpenAI-compatible endpoint at OLLAMA_BASE_URL.
    #
    # For thesis reproducibility, replace the floating "32b" tag with
    # a digest pin once you've verified your run, e.g.
    #     "qwen2.5-coder:32b@sha256:..."
    "qwen_coder_32b":  "qwen2.5-coder:32b",
}

# ── Ollama (local LLM) endpoint ─────────────────────────────────
# Default = the standard Ollama service URL.  Override only if you've
# launched Ollama on a non-default port or moved it to another host.
OLLAMA_BASE_URL: str = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1",
)

# ── RAG settings ──────────────────────────────────────────────────
# Embedding model used for NEW ingest into the active RAG version.
# Each RAG version is bound to one embedding model — switching models
# requires re-ingesting into a new BL_RAG_VERSION (see _rag_path above).
# The default keeps backwards compatibility; override via env var:
#   $env:BL_EMBEDDING_MODEL = "nvidia/llama-embed-nemotron-8b"
EMBEDDING_MODEL: str = os.getenv(
    "BL_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

# Map known RAG versions → the embedding model that produced them.
# Used for sanity-checking at retrieval time: if the active embedder
# doesn't match the corpus that wrote the active RAG version, we
# warn the operator instead of returning silently-mismatched vectors.
RAG_VERSION_EMBEDDING_MODELS: dict[str, str] = {
    "":                "text-embedding-3-small",       # legacy unversioned
    "v1_openai_small": "text-embedding-3-small",
    # HuggingFace repo id (dash, not Ollama's colon-prefixed tag).
    # Loaded via sentence-transformers — see engine._make_embedding_fn
    # dispatch.  Requires `pip install sentence-transformers torch`.
    "v2_nemotron_8b":  "nvidia/llama-embed-nemotron-8b",
}

# ── Cross-encoder reranker — SINGLE SOURCE OF TRUTH ───────────────
# Both the A1 retrieval loop (agent1_fresh/retrieve.py) and the A2/A3
# helper (rag/reranker.py) read these, so exactly ONE model + ONE
# max_length + ONE enable switch govern reranking across every agent.
# (Before 2026-07-03 A1 hardcoded bge-reranker-v2-m3 in retrieve.py,
# reranker.py defaulted to bge-reranker-base, and this constant held a
# dead cross-encoder/ms-marco string — three models, none unified.)
RERANKER_MODEL = os.getenv("BL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = int(os.getenv("BL_RERANKER_MAX_LENGTH", "1024"))
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7
RETRIEVAL_TOP_K = 15
RERANK_TOP_K = 5
RELEVANCE_THRESHOLD = 0.35


def reranker_enabled() -> bool:
    """Whether cross-encoder reranking runs — the ONE canonical switch.

    Default ON (reflects the pipeline's actual behaviour + the downstream
    tuning that assumes reranked input).  Set ``BL_RAG_USE_RERANKER=0``
    (or false/no/off) for a deterministic no-rerank baseline.  This is the
    only lever that governs reranking on every path; the old
    ``BL_DISABLE_RERANKER`` never controlled the live A1 reranker.
    """
    return os.getenv("BL_RAG_USE_RERANKER", "1").strip().lower() not in (
        "0", "false", "no", "off")


def hyde_enabled_for(phase: str) -> bool:
    """Whether HyDE (hypothetical-document expansion) runs for ``phase``.

    HyDE is a deliberate PER-PHASE choice, not a global default: it helps
    the retrospective *diagnostic* retrievals (A1 Phase 2, A2 Phase 2)
    where we search for explanations of why results differ, but it adds an
    LLM call + run-to-run variation that we do NOT want on the forward
    Phase-1 retrievals (A1 forward, A2/A3 forward).

    ``phase`` is one of: "a1_forward", "a1_phase2", "a2_phase2",
    "a2a3_forward".  The env var ``BL_RAG_USE_HYDE`` is a GLOBAL override:
    set it to 1/true to force HyDE on everywhere, or 0/false to force it
    off everywhere; leave it unset for the per-phase defaults below.
    """
    override = os.getenv("BL_RAG_USE_HYDE")
    if override is not None and override.strip() != "":
        return override.strip().lower() in ("1", "true", "yes", "on")
    # Per-phase defaults: ON only for the Phase-2 diagnostics.
    return phase in ("a1_phase2", "a2_phase2")

# ── OpenFOAM Docker ──────────────────────────────────────────────
# Single source of truth for the image `run_solver` mounts the case
# into. Override with the BLP_OPENFOAM_IMAGE env var if your host
# pulled a different tag (e.g. `openfoam/openfoam2312-dev` for the
# foundation build, or a pinned-digest image for reproducibility).
OPENFOAM_IMAGE = os.getenv("BLP_OPENFOAM_IMAGE", "opencfd/openfoam-default:2312")

# ── Orchestrator thresholds ───────────────────────────────────────
AUTO_ACCEPT_THRESHOLD = 0.10   # <10% discrepancy → auto-accept
AUTO_ITERATE_THRESHOLD = 0.25  # 10-25% → auto-iterate; >25% → human review

# ── Ensure data directories exist ─────────────────────────────────
for _d in (DATA_DIR, PAPERS_DIR, PRIMARY_RAG_DIR, SECONDARY_RAG_DIR,
           EXPERIMENTAL_DATA_DIR, LOCAL_CHROMA, RUNS_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Central env-var accessors — SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════════
# Every runtime env-var flag/value the pipeline reads is DEFINED here as a
# typed accessor that PRESERVES THE EXACT historical truthiness convention
# of its original read site.  Seven distinct conventions exist across the
# codebase; they differ on purpose (e.g. `!= "false"` disables only on the
# literal "false", `== "1"` enables only on "1") — do NOT "normalise" them,
# that would silently change behaviour.
#
# Accessors read at CALL TIME (not snapshotted at import) so live toggles
# and test monkeypatching keep working.  Consumers migrate from inline
# `os.getenv(...)` to these; `self_check()` validates + logs at startup.
#
# Boolean flag already defined above (kept there next to their feature):
#   reranker_enabled()  — BL_RAG_USE_RERANKER  (default ON; off on 0/false/no/off)
#   hyde_enabled_for()  — BL_RAG_USE_HYDE      (per-phase + global override)

# ── Small convention helpers (named so the intent is explicit) ────
def _optin_1(name: str) -> bool:
    """True ONLY if the var == '1' (strict opt-in)."""
    return os.environ.get(name) == "1"


def _flag_on(name: str, default: str = "") -> bool:
    """True if the var (stripped/lowered) is one of 1/true/yes/on."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── LLM trace persistence (bl_pipeline/shared/llm_router) ─────────
def persist_llm_trace() -> bool:
    """BLP_PERSIST_LLM_TRACE — default ON; OFF only on the literal 'false'."""
    return os.environ.get("BLP_PERSIST_LLM_TRACE", "true").lower() != "false"


# ── Per-sub-query reranker (bl_pipeline/rag/weighted_retriever) ───
def subquery_reranker_enabled() -> bool:
    """BL_RAG_USE_SUBQUERY_RERANKER — strict opt-in (== '1').  Distinct from
    the canonical reranker (reranker_enabled / BL_RAG_USE_RERANKER)."""
    return _optin_1("BL_RAG_USE_SUBQUERY_RERANKER")


# ── Formula verifier (bl_pipeline/agent1_fresh/tools) ─────────────
def formula_verifier_enabled() -> bool:
    """BL_FORMULA_VERIFIER — default ON; OFF only on the literal 'off'."""
    return os.environ.get("BL_FORMULA_VERIFIER", "on").lower() != "off"


# ── A6/A7 manuscript modes (agent6_manuscript/nodes/manuscript_writer) ──
def a7_deterministic() -> bool:
    """BLP_A7_DETERMINISTIC — cheap deterministic-assembly manuscript path."""
    return _flag_on("BLP_A7_DETERMINISTIC")


def a7_polish() -> bool:
    """BLP_A7_POLISH — post-assembly polish pass on the deterministic path."""
    return _flag_on("BLP_A7_POLISH")


# ── Tavily web search (agent1_fresh/phase2_llm, agent5 commentary) ─
def tavily_disabled() -> bool:
    """BL_DISABLE_TAVILY — strict opt-out (== '1' disables web search)."""
    return _optin_1("BL_DISABLE_TAVILY")


TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


# ── Bootstrap suppressors (agent3_experiment/wt_ai layers) ────────
def agent3_no_bootstrap() -> bool:
    """AGENT3_NO_BOOTSTRAP — any non-empty value suppresses bootstrap."""
    return bool(os.environ.get("AGENT3_NO_BOOTSTRAP"))


def a4_no_bootstrap() -> bool:
    """A4_NO_BOOTSTRAP — any non-empty value suppresses bootstrap."""
    return bool(os.environ.get("A4_NO_BOOTSTRAP"))


# ── ParaView auto-launch (agent2_cfd/paraview_launcher) ───────────
def auto_launch_paraview() -> bool:
    """BLP_AUTO_LAUNCH_PARAVIEW — default ON; OFF on 0/false/no/off/''
    (note the empty string is in the OFF set, unlike reranker_enabled)."""
    raw = os.environ.get("BLP_AUTO_LAUNCH_PARAVIEW", "true").strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def paraview_bin() -> str:
    """BLP_PARAVIEW_BIN — explicit ParaView executable path ('' if unset)."""
    return os.environ.get("BLP_PARAVIEW_BIN", "").strip()


# ── Pipeline graph mode / Agent-1 backend / CORS (api/main) ───────
def pipeline_use_graph() -> bool:
    """BLP_PIPELINE_USE_GRAPH — run the pipeline via the LangGraph orchestrator."""
    return _flag_on("BLP_PIPELINE_USE_GRAPH")


def agent1_backend() -> str:
    """BL_AGENT1_BACKEND — 'fresh' (default) or 'transition' (deprecated)."""
    return os.environ.get("BL_AGENT1_BACKEND", "fresh").strip().lower()


def cors_extra_origins() -> list[str]:
    """BLP_CORS_ORIGINS — comma-separated extra CORS origins (empties dropped)."""
    return [o.strip() for o in os.environ.get("BLP_CORS_ORIGINS", "").split(",")
            if o.strip()]


# ── Embedding / RAG backend (bl_pipeline/rag/engine) ──────────────
def embed_server() -> str:
    """BL_EMBED_SERVER — persistent embed-server URL ('' → in-process)."""
    return os.getenv("BL_EMBED_SERVER", "").strip()


def nemotron_attn() -> str:
    """BL_NEMOTRON_ATTN — attention implementation for the Nemotron embedder."""
    return os.environ.get("BL_NEMOTRON_ATTN", "eager")


def ollama_host() -> str:
    """OLLAMA_HOST — Ollama server URL (trailing slash trimmed)."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


# ── Orchestrator query-tailor model + costs (orchestrator/prompt_orc) ──
def orch_tailor_model() -> str:
    """BL_ORCH_TAILOR_MODEL — model for the per-agent query tailorer."""
    return os.environ.get("BL_ORCH_TAILOR_MODEL", "claude-sonnet-4-5-20250929")


def orch_tailor_max_tokens() -> int:
    """BL_ORCH_TAILOR_MAX_TOKENS — max output tokens for the tailorer."""
    return int(os.environ.get("BL_ORCH_TAILOR_MAX_TOKENS", "16000"))


def orch_tailor_input_usd_per_mtok() -> float:
    """BL_ORCH_TAILOR_INPUT_USD — input $/Mtok for tailorer cost accounting."""
    return float(os.environ.get("BL_ORCH_TAILOR_INPUT_USD", "3.00"))


def orch_tailor_output_usd_per_mtok() -> float:
    """BL_ORCH_TAILOR_OUTPUT_USD — output $/Mtok for tailorer cost accounting."""
    return float(os.environ.get("BL_ORCH_TAILOR_OUTPUT_USD", "15.00"))


def self_check() -> list[str]:
    """Validate the resolved config at process startup.

    RAISES ``RuntimeError`` on hard misconfiguration (a numeric env var that
    won't parse, or a non-sensical value) so the process fails loud with a
    clear message instead of a cryptic downstream crash.  Returns a list of
    soft WARNINGS (missing-but-optional secrets) — safe to call in CI where
    keys are absent.  Wire this into the API/CLI startup path.
    """
    # ── Hard errors — numeric parses must succeed and be sane ─────
    if BUDGET_LIMIT_USD <= 0:
        raise RuntimeError(
            f"config self-check: BUDGET_LIMIT_USD must be > 0; got {BUDGET_LIMIT_USD}")
    try:
        if orch_tailor_max_tokens() <= 0:
            raise ValueError("must be > 0")
    except Exception as exc:
        raise RuntimeError(
            f"config self-check: BL_ORCH_TAILOR_MAX_TOKENS invalid: {exc}") from exc
    for _fn, _name in (
        (orch_tailor_input_usd_per_mtok,  "BL_ORCH_TAILOR_INPUT_USD"),
        (orch_tailor_output_usd_per_mtok, "BL_ORCH_TAILOR_OUTPUT_USD"),
        (lambda: RERANKER_MAX_LENGTH,     "BL_RERANKER_MAX_LENGTH"),
    ):
        try:
            _fn()
        except Exception as exc:
            raise RuntimeError(
                f"config self-check: {_name} invalid: {exc}") from exc

    # ── Soft warnings — missing-but-optional secrets ──────────────
    warnings: list[str] = []
    if not ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY not set — Anthropic LLM calls will fail.")
    if not tavily_disabled() and not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY not set and BL_DISABLE_TAVILY!=1 — "
                        "Phase-2 web search will no-op.")
    for _w in warnings:
        log.warning("config self-check: %s", _w)
    log.info("config self-check OK — reranker=%s (model=%s), budget=$%.2f",
             reranker_enabled(), RERANKER_MODEL, BUDGET_LIMIT_USD)
    return warnings
