#!/usr/bin/env python
"""Re-ingest THIS project's LOCAL papers into a LOCAL vector DB.

Reads the PDFs in data/primary_rag_<version>/ (your editable copy) and builds
a fresh Chroma index under data/chroma_db_<version>/ (the LOCAL store).

SAFETY: this sets BL_A1_FORCE_LOCAL_CHROMA=1, so ingestion writes ONLY to the
local vector DB.  It can NEVER write into the main bl_transition_pipeline's
vector store.  A hard assert below refuses to run if the target isn't local.

After a successful ingest the local DB exists, so `run_agent.py` retrieval
switches from the shared thesis DB to your local one automatically.

Needs an embedder for the active RAG version (nemotron): the shared embed
server on :8765, or a local sentence-transformers install.

Usage:
    python ingest.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Force the LOCAL vector DB as the write target — thesis DB is untouchable.
os.environ["BL_A1_FORCE_LOCAL_CHROMA"] = "1"
os.environ.setdefault("BL_RAG_VERSION", "v2_nemotron_8b")
os.environ.setdefault("BL_EMBED_SERVER", "http://localhost:8765")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> None:
    from bl_pipeline.shared import config
    # Guardrail: never let a re-ingest touch the shared thesis DB.
    if config.CHROMA_DIR != config.LOCAL_CHROMA:
        raise RuntimeError(
            f"Safety: ingest must target LOCAL_CHROMA ({config.LOCAL_CHROMA}), "
            f"not {config.CHROMA_DIR}")
    print(f"[ingest] papers  = {config.PRIMARY_RAG_DIR}")
    print(f"[ingest] target  = {config.CHROMA_DIR}   (LOCAL — thesis DB untouched)")

    from bl_pipeline.rag.ingestion import ingest_all_papers
    results = ingest_all_papers()
    total = sum(sum(v.values()) for v in results.values()) if results else 0
    print(f"\n[ingest] done — {len(results)} papers, {total} chunks into the local DB.")
    print("[ingest] run_agent.py will now retrieve from the LOCAL DB.")


if __name__ == "__main__":
    main()
