"""Test whether AGS Eq.11 chunk now retrieves for the canonical Tu/Re_θ,t query.

This is the keystone test for the Wave-2 glossary expansion.  Before the
fix, AGS used paper-private notation (R_{θ,S}, τ_t) that didn't match
canonical query tokens, so the AGS onset chunk ranked below Mayle / DN
chunks even though it is the most relevant.

After normalization at ingest, AGS chunks embed under the canonical
tokens — so a query about "Re_θ,t correlation Tu" should now surface
the AGS Eq.11 chunk near the top.

Run from repo root:
    BL_RAG_VERSION=v2_nemotron_8b python -m bl_pipeline.rag.symbol_normalization.test_ags_retrieval
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )
    except Exception:
        pass

os.environ.setdefault("BL_RAG_VERSION", "v2_nemotron_8b")

from bl_pipeline.rag.engine import RAGEngine
from bl_pipeline.rag.collections import ALL_COLLECTIONS

QUERIES = [
    "Re_θ,t correlation with Tu freestream turbulence intensity",
    "transition onset Reynolds number momentum thickness turbulence intensity correlation",
    "Abu-Ghannam Shaw onset correlation",
    "163 + exp pressure gradient onset formula",  # Targeting AGS Eq.11 explicitly
]


def dump_ags_eq11_chunk(rag) -> None:
    """Find and print the AGS chunk(s) containing the 163-formula directly."""
    col = rag.get_or_create_collection("algebraic_onset")
    # Get all AGS chunks and search for the 163+exp pattern
    result = col.get(where={"paper_id": "abu_ghannam_shaw_1980"})
    ids, docs = result["ids"], result["documents"]
    hits = [(i, d) for i, d in zip(ids, docs) if "163" in d and ("exp" in d.lower() or "ln" in d.lower())]
    print(f"\n{'=' * 78}")
    print(f"DIRECT SCAN: AGS chunks containing '163' + 'exp/ln'  →  {len(hits)} hit(s)")
    print("=" * 78)
    for i, (cid, doc) in enumerate(hits[:5], 1):
        print(f"\n[{i}] chunk id: {cid}")
        print("-" * 78)
        # Show ~400 chars around the first "163" occurrence
        idx = doc.find("163")
        start = max(0, idx - 100)
        end = min(len(doc), idx + 400)
        print(doc[start:end])


def main() -> int:
    rag = RAGEngine()
    print("=" * 78)
    print(f"Chroma version: {os.environ.get('BL_RAG_VERSION', '(empty)')}")
    print("=" * 78)

    for query in QUERIES:
        print(f"\nQUERY: {query}")
        print("-" * 78)

        # Query algebraic_onset (the collection AGS lives in) directly.
        col = rag.get_or_create_collection("algebraic_onset")
        result = col.query(query_texts=[query], n_results=5)
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
            paper = meta.get("paper_id", "?")
            page = meta.get("page", "?")
            preview = (doc[:140].replace("\n", " ") + "…") if len(doc) > 140 else doc.replace("\n", " ")
            marker = "  AGS!" if "abu_ghannam" in paper else "       "
            print(f"  #{rank}  dist={dist:.3f}  {paper}  p{page} {marker}")
            print(f"        {preview}")

    dump_ags_eq11_chunk(rag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
