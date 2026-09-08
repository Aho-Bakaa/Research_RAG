#!/usr/bin/env python
"""Standalone Agent-1 runner.

Runs the literature -> transition-prediction reasoning agent end to end on a
single query and prints the result location.

Data model (see bl_pipeline/shared/config.py):
  * Papers + glossary are LOCAL to this project (data/, runs/_logs/).
  * The vector DB is read from the main bl_transition_pipeline (shared,
    READ-ONLY) UNTIL you build a local one with `python ingest.py`.  After a
    local ingest, retrieval switches to the local DB automatically.

A full grounded run needs an embedder for the active RAG version — either the
shared nemotron embed server on :8765 (started in the main project) or a local
sentence-transformers install.  `--parse-only` needs no RAG at all.

Usage:
    python run_agent.py "your query"
    python run_agent.py --parse-only "your query"   # cheap smoke, no RAG
"""
import os
import sys
from pathlib import Path

# Run from THIS project's root so the CWD-relative glossary paths
# (runs/_logs/equation_indices/...) resolve to the local copy.
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Shared corpus fallback + models.  Override any of these in your shell.
os.environ.setdefault("BL_DATA_ROOT", r"C:\Projects\bl_transition_pipeline")
os.environ.setdefault("BL_RAG_VERSION", "v2_nemotron_8b")
os.environ.setdefault("BL_EMBED_SERVER", "http://localhost:8765")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> None:
    args = sys.argv[1:]
    parse_only = False
    if args and args[0] in ("--parse-only", "-p"):
        parse_only = True
        args = args[1:]
    query = args[0] if args else (
        "Estimate transition onset for a flat plate at U=6.2 m/s, Tu=2.7%, "
        "chord 1.6 m, zero pressure gradient, turbulence grid solidity 0.3, "
        "bar 5 mm, 1000 mm upstream of the leading edge."
    )

    from bl_pipeline.shared import config
    print(f"[a1] project root  = {PROJECT_ROOT}")
    print(f"[a1] papers (local)= {config.PRIMARY_RAG_DIR}")
    print(f"[a1] vector DB     = {config.CHROMA_DIR}")
    print(f"[a1]   (local DB   = {config.LOCAL_CHROMA})")
    print(f"[a1] query         = {query[:90]}{'...' if len(query) > 90 else ''}")

    if parse_only:
        from bl_pipeline.agent1_fresh.run import _parse_query
        flow, cost, _ = _parse_query(query)
        print(f"\n[a1] PARSE-ONLY OK  U={flow.velocity_ms}  "
              f"Tu%={flow.turbulence_intensity_pct}  chord_m={flow.chord_m}  "
              f"grid_solidity={flow.grid_solidity}  cost=${cost}")
        return

    from bl_pipeline.agent1_fresh.run import run
    result = run(query)
    print(f"\n[a1] status = {result.status}   run_id = {result.run_id}")
    if result.status == "error":
        print(f"[a1] error  = {result.error}")
    else:
        print(f"[a1] output = {PROJECT_ROOT / 'data' / 'runs' / result.run_id / 'agent1_output'}")


if __name__ == "__main__":
    main()
