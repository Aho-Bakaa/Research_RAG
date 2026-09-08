"""PDF → RAG ingestion pipeline (cascade VLM, equation-aware chunking).

Pipeline per paper:

  1. Idempotency check       — skip via sha256 if already ingested
  2. Markdown extraction     — page_parser_cascade: Sonnet vision →
                                GPT-4o vision → PyMuPDF (fallback chain)
                                Per-page markdown is cached to disk so
                                future re-chunking experiments are free.
  3. Chunking                — markdown_chunker: equation-aware, never
                                splits inside $$…$$ display blocks or
                                tables; respects collection chunk_size.
  4. Symbol normalization    — Tu / FSTI / turbulence_intensity → Tu
  5. Deterministic metadata  — chunk_metadata.derive_chunk_metadata
                                (regex-based: content_type, section,
                                figure/table refs, equation counts).
                                NO per-chunk LLM call.
  6. ChromaDB write          — bulk insert into target collection(s)
  7. Registry mark           — sha256 → skip on next run

This module replaces an older PyMuPDF-only path. The cascade is the
single ingestion path used by both `POST /api/rag/ingest` and the
`scripts/wipe_and_reingest_all.py` script.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from bl_pipeline.rag.collections import COLLECTION_MAP
from bl_pipeline.rag.engine import RAGEngine
from bl_pipeline.rag.symbol_normalization.normalizer import SymbolNormalizer


# ── Markdown cache helpers (per-page, with extraction provenance) ──
# Per-page markdown lives at data/markdown_cache/{paper_id}/p001.md
# and p001.meta.json (extracted_via, success, error). The expensive
# VLM step only runs once; subsequent re-chunks are free as long as
# every page's .md + .meta.json pair exists.

def _markdown_cache_dir(paper_id: str) -> Path:
    root = (Path(__file__).resolve().parents[2]
            / "data" / "markdown_cache" / paper_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_page_markdown(paper_id: str, page_result: dict) -> None:
    page_num = page_result["page"]
    d = _markdown_cache_dir(paper_id)
    (d / f"p{page_num:03d}.md").write_text(
        page_result.get("markdown") or "", encoding="utf-8",
    )
    (d / f"p{page_num:03d}.meta.json").write_text(
        json.dumps({
            "page":          page_num,
            "success":       page_result.get("success", False),
            "extracted_via": page_result.get("extracted_via", "unknown"),
            "error":         page_result.get("error"),
        }, indent=2),
        encoding="utf-8",
    )


def _load_cached_pages(paper_id: str) -> list[dict] | None:
    """Return cached pages iff every .md has a matching .meta.json.

    A partial cache returns None so we never serve half-stale output —
    instead we re-run the VLM cleanly on the next ingest call.

    Marker-extracted fallback
    ─────────────────────────
    `marker_ingest.py` writes a whole-document markdown to
    `<cache>/<paper_id>/full.md` plus a `marker_meta.json` describing
    provenance.  If no per-page `pNNN.md` files exist but `full.md`
    does, return it as a single synthetic "page 1" so the downstream
    chunker (which is heading-aware and already concatenates pages
    anyway) treats it uniformly with Sonnet/GPT-4o output.

    Provenance: `extracted_via` is set to `"marker"` in that case so
    downstream retrievers / metadata can distinguish Marker-extracted
    chunks from Sonnet/GPT-4o ones.
    """
    d = _markdown_cache_dir(paper_id)
    md_files = sorted(d.glob("p*.md"))

    if md_files:
        # ─── Sonnet / GPT-4o cascade output (per-page) ───
        pages: list[dict] = []
        for md_path in md_files:
            meta_path = md_path.with_suffix(".meta.json")
            if not meta_path.exists():
                return None
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return None
            pages.append({
                "page":          meta.get("page", int(md_path.stem[1:])),
                "markdown":      md_path.read_text(encoding="utf-8"),
                "success":       meta.get("success", True),
                "error":         meta.get("error"),
                "extracted_via": meta.get("extracted_via", "unknown"),
            })
        return pages

    # ─── Marker fallback: full.md + marker_meta.json ───
    full_md = d / "full.md"
    marker_meta = d / "marker_meta.json"
    if full_md.exists() and marker_meta.exists():
        try:
            meta = json.loads(marker_meta.read_text(encoding="utf-8"))
        except Exception:
            return None
        return [{
            "page":          1,
            "markdown":      full_md.read_text(encoding="utf-8"),
            "success":       True,
            "error":         None,
            "extracted_via": meta.get("extractor", "marker"),
        }]

    return None


# ── Main ingestion pipeline ──────────────────────────────────────

def ingest_paper(
    pdf_path: str | Path,
    target_collections: list[str],
    rag_engine: RAGEngine,
    normalizer: SymbolNormalizer | None = None,
    use_cache: bool = True,
) -> dict[str, int]:
    """Ingest a single PDF into one or more RAG collections.

    Pipeline:
      1. Skip if sha256 already in registry.
      2. Try the per-page markdown cache; if complete, use it.
      3. Otherwise run the cascade parser (Sonnet → GPT-4o → PyMuPDF)
         and persist each page's markdown + provenance metadata.
      4. For each target collection, chunk the markdown with
         markdown_chunker (equation-aware), normalize symbols, derive
         deterministic metadata via chunk_metadata.derive_chunk_metadata
         (no LLM), then bulk-write to ChromaDB.
      5. Mark the paper as ingested.

    Returns dict of {collection_name: chunks_added}.
    """
    # Local imports — these modules pull in the OpenAI client and a
    # markdown segmenter that aren't needed for the helpers above. Keep
    # them inside the function so a stripped test harness importing
    # only `_load_cached_pages` doesn't pay for them.
    from bl_pipeline.rag.chunk_metadata import derive_chunk_metadata
    from bl_pipeline.rag.collections import ALL_COLLECTIONS
    from bl_pipeline.rag.markdown_chunker import chunk_markdown
    from bl_pipeline.rag.page_parser_cascade import parse_paper_cascade

    pdf_path = Path(pdf_path)
    paper_id = pdf_path.stem
    if normalizer is None:
        normalizer = SymbolNormalizer()

    # 1. Idempotency
    if rag_engine.is_paper_ingested(pdf_path):
        print(f"  [skip] {paper_id} already ingested (sha256 match)", flush=True)
        return {c: 0 for c in target_collections}

    # 2. Cache → 3. Cascade
    pages: list[dict] | None = None
    if use_cache:
        pages = _load_cached_pages(paper_id)
        if pages is not None:
            print(f"  using cached markdown ({len(pages)} pages) — no VLM call",
                  flush=True)

    if pages is None:
        try:
            pages = parse_paper_cascade(pdf_path)
        except Exception as e:
            print(f"  [error] cascade parse failed for {paper_id}: {e}",
                  flush=True)
            return {c: 0 for c in target_collections}
        # Persist per-page so a future re-chunk is free.
        for p in pages:
            try:
                _save_page_markdown(paper_id, p)
            except Exception as e:
                print(f"  [warn] cache save failed for {paper_id} "
                      f"p{p.get('page')}: {e}", flush=True)

    ok_pages = [p for p in pages
                if p.get("success") and p.get("markdown")]
    if not ok_pages:
        print(f"  [skip] {paper_id} — zero pages parsed", flush=True)
        return {c: 0 for c in target_collections}

    # 4. Chunk + normalize + metadata + write per collection
    #
    # WHOLE-PAPER CHUNKING (changed from per-page).  The previous
    # implementation called chunk_markdown() once per page, which meant
    # any equation whose anchoring prose sat on the previous page (e.g.
    # Mayle's canonical Re_θ,t correlation, where the "we correlate the
    # data as ..." setup ends one page and the `$$ ... $$` block opens
    # the next) was emitted as a context-light orphan chunk.  The
    # orphan-merge post-pass in markdown_chunker only sees adjacent
    # chunks WITHIN a single page, so cross-page orphans never get
    # rejoined.  Concatenating the whole paper first lets the chunker's
    # equation/figure/table protection + orphan-merge logic operate
    # across the entire document.  Page provenance is preserved via
    # inline page-break markers that survive chunking and are stripped
    # before embedding.
    PAGE_MARKER_FMT = "\n\n<!-- page_break page={p} -->\n\n"
    PAGE_MARKER_RE  = re.compile(r"<!-- page_break page=(\d+) -->")

    # Concatenate all OK pages with markers between them.  Markers are
    # HTML comments — markdown-neutral, won't perturb heading detection
    # or paragraph splitting in chunk_markdown.
    full_md_parts: list[str] = []
    for p in ok_pages:
        full_md_parts.append(PAGE_MARKER_FMT.format(p=p["page"]))
        full_md_parts.append(p["markdown"])
    full_md = "".join(full_md_parts)

    # Dominant extractor across pages — collapse to a single value for
    # the chunk_id format.  Almost always "sonnet" for the current
    # corpus; fallback handles mixed VLM cascade outputs.
    _via_counts: dict[str, int] = {}
    for p in ok_pages:
        _via_counts[p.get("extracted_via", "unknown")] = (
            _via_counts.get(p.get("extracted_via", "unknown"), 0) + 1
        )
    via_dominant = max(_via_counts, key=_via_counts.get) if _via_counts else "unknown"

    results: dict[str, int] = {}
    t0 = time.time()
    for col_name in target_collections:
        config = COLLECTION_MAP.get(col_name)
        if config is None:
            continue

        chunk_rows: list[dict[str, Any]] = []
        # Carry-forward page pointer: when a chunk contains zero page
        # markers (e.g. continuation prose after an equation/figure
        # block swept the marker into the prior chunk), inherit the
        # last-seen page so provenance never reads "?".
        _last_page: int | None = None

        for idx, ct in enumerate(chunk_markdown(
            full_md,
            target_size=config.chunk_size,
            max_size=int(config.chunk_size * 1.5),
            min_size=max(200, config.chunk_size // 3),
        )):
            # Recover the page span from the markers inside this chunk.
            pages_seen = sorted({
                int(m.group(1)) for m in PAGE_MARKER_RE.finditer(ct)
            })
            if pages_seen:
                _last_page = pages_seen[-1]
                if len(pages_seen) == 1:
                    page_str  = str(pages_seen[0])
                    page_stem = f"p{pages_seen[0]:03d}"
                else:
                    page_str  = f"{pages_seen[0]}-{pages_seen[-1]}"
                    page_stem = f"p{pages_seen[0]:03d}-{pages_seen[-1]:03d}"
            elif _last_page is not None:
                # Continuation chunk — inherit the most recent page seen.
                page_str  = str(_last_page)
                page_stem = f"p{_last_page:03d}"
            else:
                # Pre-first-marker prose (only at the very head of the
                # stream); attribute to page 1 as a defensive default.
                page_str  = "1"
                page_stem = "p001"

            # Strip the markers BEFORE embedding — they're metadata, not
            # content the LLM (or the embedder) should see.
            ct_clean = PAGE_MARKER_RE.sub("", ct).strip()
            if not ct_clean:
                continue
            # Paper-aware normalization: pass paper_id so paper-scoped
            # aliases fire (Mayle's τ_0 → τ_w only inside mayle_1991,
            # AGS's R_{θ,S} → Re_θ,t only inside abu_ghannam_shaw_1980,
            # Narasimha's K → k only inside narasimha_1985, etc.).  This
            # is the line that makes the 2026-05 paper-aware glossary
            # actually take effect at ingest time.
            ct_norm = normalizer.normalize_text(ct_clean, paper_id=paper_id)
            extra_meta = derive_chunk_metadata(ct_norm, page_markdown=ct_norm)
            chunk_rows.append({
                "id":   f"{paper_id}_{col_name}_{via_dominant}_{page_stem}_{idx:02d}",
                "text": ct_norm,
                "metadata": {
                    "paper_id":      paper_id,
                    "page_number":   page_str,
                    "extracted_via": via_dominant,
                    **extra_meta,
                },
            })

        if chunk_rows:
            results[col_name] = rag_engine.ingest_chunks(col_name, chunk_rows)
        else:
            results[col_name] = 0

    # 5. Mark
    rag_engine.mark_paper_ingested(pdf_path)
    dt = time.time() - t0
    total = sum(results.values())
    print(f"  done in {dt:.0f}s — {total} chunks across "
          f"{sum(1 for v in results.values() if v > 0)} col(s) "
          f"({len(ok_pages)}/{len(pages)} pages OK)", flush=True)
    return results


def ingest_all_papers(
    papers_dir: str | Path | None = None,
    target_collections: list[str] | None = None,
    rag_engine: RAGEngine | None = None,
    normalizer: SymbolNormalizer | None = None,
) -> dict[str, dict[str, int]]:
    """Ingest all PDFs from PRIMARY, SECONDARY, and EXPERIMENTAL_DATA folders.

    Scans three source folders (primary_rag, secondary_rag, experimental_data)
    and routes each paper to the correct ChromaDB collections via PAPER_COLLECTION_MAP.
    Falls back to scanning papers_dir if the structured folders are empty.
    """
    from bl_pipeline.rag.collections import ALL_COLLECTIONS, PAPER_COLLECTION_MAP
    from bl_pipeline.shared.config import (
        EXPERIMENTAL_DATA_DIR,
        PAPERS_DIR,
        PRIMARY_RAG_DIR,
        SECONDARY_RAG_DIR,
    )

    if rag_engine is None:
        rag_engine = RAGEngine()
    results: dict[str, dict[str, int]] = {}

    # Collect PDFs from structured folders first, fallback to papers_dir
    search_dirs = [PRIMARY_RAG_DIR, SECONDARY_RAG_DIR, EXPERIMENTAL_DATA_DIR]
    if papers_dir is not None:
        search_dirs.append(Path(papers_dir))

    # If structured folders empty, add legacy papers/ folder
    has_structured = any(list(d.glob("*.pdf")) for d in search_dirs[:3])
    if not has_structured:
        search_dirs.append(PAPERS_DIR)

    seen: set[str] = set()  # avoid double-ingesting same file
    for folder in search_dirs:
        folder = Path(folder)
        if not folder.exists():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            if pdf.name in seen:
                continue
            seen.add(pdf.name)

            paper_id = pdf.stem
            if target_collections:
                cols = target_collections
            else:
                cols = PAPER_COLLECTION_MAP.get(paper_id, [c.name for c in ALL_COLLECTIONS])

            total = len(seen)
            print(f"  [{total:02d}] {pdf.name}  -> {cols}", flush=True)
            result = ingest_paper(pdf, cols, rag_engine, normalizer)
            chunks_added = sum(result.values()) if result else 0
            print(f"       {chunks_added} chunks added", flush=True)
            results[pdf.name] = result

    return results
