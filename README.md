# agent1_standalone

A **clean, standalone copy of Agent 1** — the literature → transition-prediction
reasoning agent — carved out of the `bl_transition_pipeline` thesis project so it
can be developed and run on its own.

Only the **core Agent-1 flow** is here:

```
parse query → retrieve (hybrid + rerank) → reason (ReAct + tools) → verify → write
```

The downstream / thesis-specific parts were **stripped**: the Agent-1 → Agent-3
handoff, the Phase-2 retrospective diagnostics (`phase2`, `phase2_llm`,
`theory_envelope`), and the thesis-building scripts. This is A1 and nothing else.

## Data model (important)

| Data | Where | Notes |
|---|---|---|
| **Papers** (39 PDFs) | `data/primary_rag_v2_nemotron_8b/` — **LOCAL, yours** | Edit / add / remove freely, then re-ingest. |
| **Glossary** (`paper_summaries`, `equation_indices`) | `runs/_logs/` — **LOCAL, yours** | Domain data you own. |
| **Vector DB** (embeddings) | **shared, read-only** from the main project | Not copied (it's ~481 MB). |
| **API keys** | `.env` | Copied from the main project. |

### The safe local/shared vector-DB rule

Retrieval reads the **local** vector DB *if you've built one*; otherwise it falls
back to the **main project's** DB, **read-only**. Re-ingestion **always writes to
the local DB** — it can never touch the thesis project's store (enforced by
`BL_A1_FORCE_LOCAL_CHROMA=1` + a hard assert in `ingest.py`). So:

- **Out of the box:** retrieval uses the shared thesis DB → A1 runs immediately.
- **After you edit papers + re-ingest:** a local DB appears and retrieval switches
  to it automatically. The thesis DB is never modified.

Point the shared fallback elsewhere with the `BL_DATA_ROOT` env var (default
`C:\Projects\bl_transition_pipeline`).

## Run

```bash
# cheap smoke — parse a query, no RAG / no embed server needed
python run_agent.py --parse-only "flat plate, U=6.2 m/s, Tu=2.7%, chord 1.6 m, ZPG"

# full grounded run (needs an embedder for the nemotron RAG version:
# the shared embed server on :8765, or a local sentence-transformers install)
python run_agent.py "your query here"
```

## Edit the corpus and re-ingest

1. Add / edit / remove PDFs in `data/primary_rag_v2_nemotron_8b/`.
2. Build a local index (writes locally only):

   ```bash
   python ingest.py
   ```
3. `run_agent.py` now retrieves from your local DB.

Re-ingestion needs the nemotron embedder — start the shared embed server in the
main project (`:8765`), or install `sentence-transformers` + `torch` locally.

## Requirements

Uses the same Python dependencies as the main project (chromadb, anthropic,
sentence-transformers, python-dotenv, …). Run with the main project's virtualenv
or any environment that has those installed.

## Layout

```
agent1_standalone/
├─ run_agent.py                 # run A1 on a query (+ --parse-only smoke)
├─ ingest.py                    # re-ingest LOCAL papers → LOCAL vector DB
├─ .env                         # API keys
├─ bl_pipeline/
│  ├─ agent1_fresh/             # the Agent-1 core (downstream modules stripped)
│  ├─ shared/                   # llm_router, config, event_bus, json_utils, …
│  └─ rag/                      # engine, hybrid retrieval, reranker, HyDE, ingestion
├─ data/
│  └─ primary_rag_v2_nemotron_8b/   # 39 source PDFs (editable)
└─ runs/_logs/
   ├─ paper_summaries/          # per-paper glossary
   └─ equation_indices/         # per-paper equation index
```
