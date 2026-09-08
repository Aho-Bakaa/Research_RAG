"""Hybrid RAG engine — ChromaDB vector search + BM25 keyword search.

Implements:
- Hybrid retrieval (weighted BM25 + vector similarity)
- Query decomposition for complex questions
- Corrective RAG loop (re-retrieve if confidence low)
- Relevance threshold filtering
- Order-preserving retrieval (OP-RAG) for sequential content
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)


# ── Network retry helper ────────────────────────────────────────────
# Any call going over the public internet (OpenAI embeddings, ChromaDB
# adds that trigger embeddings, etc.) can hiccup on a single packet.
# Wrap those calls so a transient blip doesn't kill a 3-hour ingest
# job. We retry on any exception — lenient on purpose, since these are
# idempotent for embeddings, and ChromaDB `add` with the same IDs is a
# no-op on the second attempt (upsert semantics).

T = TypeVar("T")


def _retry_with_backoff(
    fn: Callable[[], T],
    tries: int = 4,
    initial_delay: float = 2.0,
    backoff: float = 2.5,
    label: str = "network op",
) -> T:
    """Call fn() up to `tries` times with exponential backoff.

    Retries on ANY exception. Raises the final exception if all tries
    exhaust. First-failure log happens silently; retries print a short
    note so progress logs stay readable.
    """
    delay = initial_delay
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:   # noqa: BLE001 — intentional lenient catch
            last_err = e
            if attempt < tries:
                # Short, readable error — trim tracebacks and giant reprs.
                err_short = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"    [retry {attempt}/{tries - 1}] {label}: {err_short}")
                time.sleep(delay)
                delay *= backoff
    assert last_err is not None   # for type checker
    raise last_err

import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi


# ── BM25 tokenizer (LaTeX-aware) ───────────────────────────────────
# The default `.lower().split()` treats `400\,` as a single token,
# never matching user queries that say `400`. Scientific text is
# full of LaTeX scaffolding (`\cdot`, `\,`, `^{...}`, `_{...}`, etc.)
# that we need to split on so the useful parts (numbers, variable
# names, equation numbers) become individual tokens BM25 can match.
#
# Strategy:
#   1. Lowercase the whole document.
#   2. Split on anything that isn't a word-char, dash, slash, or dot.
#      (Dash/slash/dot kept so '5/8', '-5/8', 'eq. 9' survive as
#       meaningful fragments.)
#   3. Drop the standalone LaTeX command words (\cdot, \frac, etc.)
#      because they're noise — any chunk with math has them, so they
#      add no discriminative power.
#
# Tested on the Mayle Eq. 9 chunk: produces tokens
#   ['re', 'θt', 'θ', 't', '400', 'tu', '-5/8', '9', 'if', 'only', ...]
# which lets queries matching any of these score.

_LATEX_COMMANDS = {
    "cdot", "cdots", "frac", "sqrt", "sum", "int", "infty",
    "partial", "nabla", "alpha", "beta", "gamma", "delta",
    "epsilon", "theta", "lambda", "mu", "nu", "pi", "rho",
    "sigma", "tau", "phi", "psi", "omega",
    "left", "right", "begin", "end", "text", "textit",
    "mathrm", "mathbf", "vec", "hat", "bar", "tilde", "dot",
    "quad", "qquad", "overline", "leq", "geq", "approx",
    "times", "simeq", "equiv",
}

# Token regex: allow tokens to START with '-' (so negative numbers /
# fractions like '-5/8' survive as their own token), then word chars
# and meaningful math glue (slash, dot, hyphen) can continue.
#
# The leading '-' is OPTIONAL via the `-?` group; body characters
# include letters (Latin + Greek), digits, dot, slash, hyphen.
_TOKEN_RX = re.compile(
    r"-?[A-Za-zΑ-Ωα-ω0-9][A-Za-zΑ-Ωα-ω0-9\-/.]*"
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizer shared by indexing and querying.

    Lowercases, strips LaTeX punctuation scaffolding, keeps numerical
    and variable tokens including signed fragments like '5/8', '-5/8',
    '-0.69'. Same pass on indexing and querying keeps both in sync.
    """
    if not text:
        return []
    # Drop backslash-prefixed LaTeX commands first so they don't split
    # into their letter tails (`\theta` shouldn't leave `theta` behind).
    text_nobackslash = re.sub(r"\\[A-Za-z]+", " ", text)
    lowered = text_nobackslash.lower()
    raw = _TOKEN_RX.findall(lowered)
    out: list[str] = []
    for tok in raw:
        # Drop pure-punctuation remnants ('.', '/' etc.) — but keep
        # leading '-' because it's meaningful (signed numbers).
        if not any(ch.isalnum() for ch in tok):
            continue
        # Strip TRAILING punctuation only. Preserve the leading '-'
        # because `-5/8` and `5/8` are genuinely different tokens in
        # scientific text.
        tok = tok.rstrip("./-")
        if not tok or tok == "-":
            continue
        # Skip LaTeX command words (cdot, frac, etc.) — noise in BM25
        if tok in _LATEX_COMMANDS:
            continue
        out.append(tok)
    return out

from bl_pipeline.rag.collections import ALL_COLLECTIONS, COLLECTION_MAP, CollectionConfig
from bl_pipeline.shared.config import (
    BM25_WEIGHT,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RERANK_TOP_K,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_TOP_K,
    VECTOR_WEIGHT,
    embed_server,
    nemotron_attn,
    ollama_host,
)


class RAGEngine:
    """Unified hybrid retrieval across all 7 collections."""

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._persist_dir = Path(persist_dir or CHROMA_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # BM25 indices per collection (rebuilt from stored docs)
        self._bm25: dict[str, BM25Okapi | None] = {}
        self._bm25_docs: dict[str, list[dict[str, Any]]] = {}

        # Paper registry for incremental ingestion
        self._registry_path = self._persist_dir / "paper_registry.json"
        self._registry: dict[str, str] = self._load_registry()

        # Embedding function — uses OpenAI for consistency
        self._embed_fn = self._make_embedding_fn()

    # ── Embedding function ────────────────────────────────────────

    def _make_embedding_fn(self):
        """Create a ChromaDB-compatible embedding function.

        Dispatches on the active EMBEDDING_MODEL string:

          - "text-embedding-*"       → OpenAI hosted (existing behaviour)
          - "nvidia/llama-embed-*"   → Ollama local (Nemotron family)
          - "nvidia/nv-embed-*"      → Ollama local (NV-Embed family)
          - anything containing "nemotron" or "ollama:" → Ollama local

        Override the model via the BL_EMBEDDING_MODEL env var.  See
        bl_pipeline.shared.config.RAG_VERSION_EMBEDDING_MODELS for the
        recommended (RAG_VERSION → embedding model) mapping.
        """
        from chromadb.utils.embedding_functions import EmbeddingFunction

        model = EMBEDDING_MODEL
        model_lower = model.lower()

        # Explicit prefix wins: "ollama:..." => Ollama, "st:..." or
        # "sentence-transformers:..." => SentenceTransformer.
        if model_lower.startswith("ollama:"):
            return self._make_ollama_embedding_fn(model)
        if (model_lower.startswith("st:")
                or model_lower.startswith("sentence-transformers:")):
            return self._make_sentence_transformer_embedding_fn(model)

        # Heuristic dispatch by model-name family.
        # Nemotron embedding models (Llama-Embed-Nemotron / NV-Embed)
        # ship as HuggingFace repos and are loaded via the
        # sentence-transformers library — they're NOT in the Ollama
        # registry as of May 2026, so we route them locally.
        if (model_lower.startswith("nvidia/llama-embed")
                or model_lower.startswith("nvidia/nv-embed")
                or "llama-embed-nemotron" in model_lower
                or "nv-embed" in model_lower):
            return self._make_sentence_transformer_embedding_fn(model)

        # Other "nvidia/*" or "*nemotron*" names default to Ollama for
        # backwards compatibility with the earlier wiring.
        if (model_lower.startswith("nvidia/")
                or "nemotron" in model_lower):
            return self._make_ollama_embedding_fn(model)

        # Fallback: OpenAI hosted embeddings (the legacy default).
        return self._make_openai_embedding_fn(model)

    def _make_openai_embedding_fn(self, model: str):
        """OpenAI hosted-embedding function. Used for `text-embedding-*`."""
        from chromadb.utils.embedding_functions import EmbeddingFunction
        from openai import OpenAI

        api_key = OPENAI_API_KEY

        class OpenAIEmbedV1(EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                # Retry the whole embed call — transient OpenAI
                # APIConnectionError / timeout / 5xx should not kill a
                # long-running ingestion job.
                def _do() -> list[list[float]]:
                    client = OpenAI(api_key=api_key)
                    response = client.embeddings.create(input=input, model=model)
                    return [item.embedding for item in response.data]

                return _retry_with_backoff(
                    _do,
                    tries=4,
                    initial_delay=2.0,
                    backoff=2.5,
                    label=f"openai embed (n={len(input)})",
                )

        return OpenAIEmbedV1()

    def _make_sentence_transformer_embedding_fn(self, model: str):
        """SentenceTransformer-based local embedding function.

        Loads any HuggingFace embedding model via the `sentence-
        transformers` library.  Used for `nvidia/llama-embed-nemotron-8b`
        (the headline target — top of multilingual MTEB) and other
        Nemotron-family / NV-Embed weights that don't ship through
        Ollama's registry.

        First call downloads the weights to the local HuggingFace
        cache (`~/.cache/huggingface/hub/` or %USERPROFILE%\\.cache\\...
        on Windows).  Subsequent calls reuse the cached files — no
        network access.

        Asymmetric prompting
        ────────────────────
        Llama-Embed-Nemotron is an ASYMMETRIC model: queries are
        expected to carry a task-instruction prefix, documents are
        not.  ChromaDB's `EmbeddingFunction` interface is symmetric —
        the same function is called whether the input is a chunk being
        stored or a user query being matched against stored chunks.
        We resolve this by:

          - Always treating __call__ input as DOCUMENTS during ingest
            (the common path).  At ingest time, `encode_document(...)`
            is correct.
          - Exposing an `encode_query` method on the embedding-fn
            instance for the retrieval layer to call explicitly when
            it has a user query in hand.  The weighted_retriever and
            HyDE paths can pick this up to apply the correct prefix.

        The default __call__ path stays document-mode for two reasons:
        (1) ingest is the throughput-critical case, and (2) ChromaDB
        will call __call__ on a query string by default — at which
        point treating it as a document is still better than crashing,
        even if 1-2 MTEB points worse than the asymmetric path.

        Strips an optional `st:` / `sentence-transformers:` prefix so
        the configured model string can be tagged for the dispatcher.
        """
        from chromadb.utils.embedding_functions import EmbeddingFunction

        # Late-import — sentence-transformers + torch is a heavy
        # dependency (~3 GB on disk).  Only pay for it when this fn is
        # actually requested.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for HuggingFace "
                "embedding models.  Install with: "
                "`pip install sentence-transformers`"
            ) from e

        # Strip dispatcher prefixes.
        if model.lower().startswith("sentence-transformers:"):
            hf_repo = model.split(":", 1)[1]
        elif model.lower().startswith("st:"):
            hf_repo = model.split(":", 1)[1]
        else:
            hf_repo = model

        # If a persistent embedding server is configured, route embedding to
        # it and skip the local model load — pay the ~14 GB load once (in the
        # server), not per run. The server reuses the same model.encode_query
        # / encode_document, so vectors are identical (guarded by a vector-
        # match check). Default off: unset = unchanged behaviour.
        _server = embed_server()
        if _server:
            print(f"  [embedding] BL_EMBED_SERVER set; routing to {_server} "
                  f"(no local model load).", flush=True)
            return self._make_remote_embedding_fn(_server, hf_repo)

        print(f"  [embedding] loading SentenceTransformer({hf_repo!r}) "
              f"— first run downloads weights, may take 15-30 min "
              f"depending on connection (~15 GB for 8B models).",
              flush=True)

        # Llama-Embed-Nemotron specifically requires:
        #   - trust_remote_code (custom encode_document / encode_query
        #     implementations live in the repo)
        #   - bfloat16 dtype (matches release weights)
        #   - left-side padding (per the model card)
        #   - attn_implementation of "eager" (CPU) or "flash_attention_2"
        #     (CUDA only).  The repo's custom llama_bidirectional_model
        #     hard-asserts on attn_implementation in {"eager",
        #     "flash_attention_2"} and rejects PyTorch's default "sdpa"
        #     — observed failure: "Unsupported attention implementation:
        #     sdpa, only support flash_attention_2 or eager".  We pick
        #     "eager" since the path is CPU-first; users with CUDA can
        #     override via BL_NEMOTRON_ATTN env var.
        # Other ST-compatible models will pass these kwargs without
        # issue — they're widely supported.
        st_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        _attn_impl = nemotron_attn()
        try:
            import torch  # late import to keep startup light
            st_kwargs["model_kwargs"] = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": _attn_impl,
            }
        except Exception:
            # Torch missing or import-broken — let SentenceTransformer
            # raise its own error from the constructor call below.
            st_kwargs["model_kwargs"] = {
                "attn_implementation": _attn_impl,
            }
        st_kwargs["tokenizer_kwargs"] = {"padding_side": "left"}

        st_model = SentenceTransformer(hf_repo, **st_kwargs)

        class SentenceTransformerEmbedFn(EmbeddingFunction):
            """ChromaDB-compatible wrapper.  __call__ defaults to
            document-mode encoding; `encode_query` is a separate
            method the retrieval layer can call for query strings."""

            def __init__(self) -> None:
                self._st_model = st_model
                self._repo = hf_repo

            def __call__(self, input: list[str]) -> list[list[float]]:
                # Document-mode (ingest path).
                def _do() -> list[list[float]]:
                    # encode_document is the asymmetric-aware doc path
                    # when the model exposes it (Llama-Embed-Nemotron
                    # does).  Otherwise fall back to encode().
                    encoder = getattr(self._st_model, "encode_document",
                                       None) or self._st_model.encode
                    arr = encoder(input, batch_size=4,
                                   show_progress_bar=False)
                    # SentenceTransformer returns numpy arrays; convert
                    # to plain lists so ChromaDB's typed-storage layer
                    # serialises cleanly.
                    return [list(map(float, v)) for v in arr]

                return _retry_with_backoff(
                    _do,
                    tries=2,
                    initial_delay=2.0,
                    backoff=2.0,
                    label=f"st embed ({self._repo}, n={len(input)})",
                )

            def encode_query(
                self,
                queries: list[str],
                task_instruction: str | None = None,
            ) -> list[list[float]]:
                """Apply the asymmetric query prefix and return query
                embeddings.  Retrieval code paths should use this in
                preference to __call__ when the input is a user query.
                """
                encoder = getattr(self._st_model, "encode_query", None)
                if encoder is not None:
                    arr = encoder(queries, batch_size=4,
                                   show_progress_bar=False)
                else:
                    # Manual prefix if encode_query is not exposed.
                    instr = (task_instruction or
                             "Given a question, retrieve relevant "
                             "passages that answer it.")
                    prefixed = [f"Instruct: {instr}\nQuery: {q}"
                                for q in queries]
                    arr = self._st_model.encode(
                        prefixed, batch_size=4, show_progress_bar=False,
                    )
                return [list(map(float, v)) for v in arr]

        return SentenceTransformerEmbedFn()

    def _make_remote_embedding_fn(self, server_url: str, hf_repo: str):
        """Embedding fn that calls a persistent embed server over HTTP rather
        than loading the model in-process.  Mirrors the local
        SentenceTransformer fn: ``__call__`` -> ``/embed_document`` (ingest
        path), ``encode_query`` -> ``/embed_query`` (retrieval path).
        See scripts/embed_server.py.
        """
        from chromadb.utils.embedding_functions import EmbeddingFunction
        import requests

        base = server_url.rstrip("/")

        class RemoteEmbedFn(EmbeddingFunction):
            def __init__(self) -> None:
                self._base = base
                self._repo = hf_repo

            def _post(self, route: str, texts: list[str]) -> list[list[float]]:
                resp = requests.post(
                    f"{self._base}/{route}",
                    json={"texts": list(texts)},
                    timeout=600,
                )
                resp.raise_for_status()
                return resp.json()["vectors"]

            def __call__(self, input: list[str]) -> list[list[float]]:
                return self._post("embed_document", input)

            def encode_query(
                self,
                queries: list[str],
                task_instruction: str | None = None,
            ) -> list[list[float]]:
                return self._post("embed_query", queries)

        return RemoteEmbedFn()

    def _make_ollama_embedding_fn(self, model: str):
        """Ollama local-embedding function.

        Talks to the Ollama HTTP server on `localhost:11434` (override
        via OLLAMA_HOST env var).  One HTTP request per input string —
        Ollama's /api/embeddings endpoint is single-input only, so we
        loop client-side.  Ingest throughput on a modern CPU is roughly
        10-30 chunks/sec for the 8B Nemotron embedder, which means
        ~1-3 min for a full corpus re-ingest.

        Strips the `ollama:` prefix if present, so the user can write
        either `nvidia/llama-embed-nemotron:8b` or
        `ollama:nvidia/llama-embed-nemotron:8b` — both resolve to the
        same model.
        """
        from chromadb.utils.embedding_functions import EmbeddingFunction
        import os
        import requests

        host = ollama_host()
        # Strip optional "ollama:" prefix.
        api_model = model[len("ollama:"):] if model.lower().startswith("ollama:") else model
        endpoint = f"{host}/api/embeddings"

        class OllamaEmbedFn(EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                def _embed_one(text: str) -> list[float]:
                    resp = requests.post(
                        endpoint,
                        json={"model": api_model, "prompt": text},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    vec = data.get("embedding")
                    if not vec:
                        raise RuntimeError(
                            f"Ollama returned no embedding for "
                            f"model={api_model!r} (response keys: "
                            f"{list(data.keys())})"
                        )
                    return vec

                def _do() -> list[list[float]]:
                    return [_embed_one(t or " ") for t in input]

                return _retry_with_backoff(
                    _do,
                    tries=3,
                    initial_delay=1.5,
                    backoff=2.0,
                    label=f"ollama embed (model={api_model}, n={len(input)})",
                )

        return OllamaEmbedFn()

    # ── Collection management ─────────────────────────────────────

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        config = COLLECTION_MAP.get(name)
        if config is None:
            raise ValueError(f"Unknown collection: {name}")
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embed_fn,
            metadata={"description": config.description},
        )

    def collection_count(self, name: str) -> int:
        try:
            col = self._client.get_collection(name, embedding_function=self._embed_fn)
            return col.count()
        except Exception as exc:
            # Collection missing / not yet created is expected on first run
            # (returns 0 = empty).  A connection/backend error is NOT — log
            # it at WARNING so a masked backend failure is visible rather
            # than silently reported as an empty collection.
            log.warning("collection_count(%r) failed, treating as empty: "
                        "%s: %s", name, type(exc).__name__, exc)
            return 0

    # ── Paper registry (incremental ingestion) ────────────────────

    def _load_registry(self) -> dict[str, str]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )

    def is_paper_ingested(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._registry.get(path.name) == file_hash

    def mark_paper_ingested(self, file_path: str | Path) -> None:
        path = Path(file_path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self._registry[path.name] = file_hash
        self._save_registry()

    # ── Ingestion ─────────────────────────────────────────────────

    def ingest_chunks(
        self,
        collection_name: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Add pre-chunked documents to a collection.

        Each chunk dict must have:
          - "text": str
          - "metadata": dict (matching the collection's metadata_fields)
          - "id": str (unique)
        """
        col = self.get_or_create_collection(collection_name)

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [c["id"] for c in chunks]

        # ChromaDB batches of 5000
        batch_size = 5000
        added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # ChromaDB.add triggers an embedding call under the hood,
            # so it inherits the network risk. Wrap once more in case
            # something in the ChromaDB path (not the embed fn itself)
            # raises — e.g. a transient SQLite lock or a second
            # embedding call ChromaDB makes internally.
            _retry_with_backoff(
                lambda bt=batch_texts, bm=batch_meta, bi=batch_ids: col.add(
                    documents=bt, metadatas=bm, ids=bi
                ),
                tries=3,
                initial_delay=3.0,
                backoff=2.5,
                label=f"chroma add (n={len(batch_texts)})",
            )
            added += len(batch_texts)

        # Rebuild BM25 index for this collection
        self._rebuild_bm25(collection_name)

        return added

    def _rebuild_bm25(self, collection_name: str) -> None:
        """Rebuild the BM25 index from all documents in a collection."""
        col = self.get_or_create_collection(collection_name)
        count = col.count()
        if count == 0:
            self._bm25[collection_name] = None
            self._bm25_docs[collection_name] = []
            return

        # Fetch all documents (paginated)
        all_docs: list[dict[str, Any]] = []
        offset = 0
        while offset < count:
            batch = col.get(
                limit=min(1000, count - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc_text, meta, doc_id in zip(
                batch["documents"], batch["metadatas"], batch["ids"]
            ):
                all_docs.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta or {},
                })
            offset += len(batch["ids"])

        # LaTeX-aware tokenization so chunks like "$$Re_θt = 400 Tu^{-5/8}$$"
        # produce matchable tokens (400, tu, 5/8, -5/8, θt) instead of
        # collapsing into one unmatchable glob.
        tokenized = [tokenize_for_bm25(doc["text"]) for doc in all_docs]
        self._bm25[collection_name] = BM25Okapi(tokenized)
        self._bm25_docs[collection_name] = all_docs

    # ── BM25 helpers exposed for the weighted retriever ───────────

    def bm25_scores_for_collection(
        self,
        collection_name: str,
        query: str,
    ) -> dict[str, float]:
        """Return {doc_id -> BM25 score in [0,1]} for every indexed doc
        in the collection.

        Used by `weighted_retriever` so it can fuse BM25 scores with
        vector similarities BEFORE applying metadata weights. Math-heavy
        chunks (naked equations) often have weak English embeddings but
        strong literal-token BM25 scores; fusing both recovers them.

        Scores are normalised by the max so they're comparable across
        collections and fusion weights.
        """
        if (collection_name not in self._bm25
                or self._bm25[collection_name] is None):
            # Lazy rebuild in case the collection was touched without
            # going through ingest_chunks (e.g. direct col.add()).
            self._rebuild_bm25(collection_name)

        bm25 = self._bm25.get(collection_name)
        docs = self._bm25_docs.get(collection_name, [])
        if bm25 is None or not docs:
            return {}

        tokens = tokenize_for_bm25(query)
        raw_scores = bm25.get_scores(tokens)
        peak = max(raw_scores) if len(raw_scores) and max(raw_scores) > 0 else 1.0

        out: dict[str, float] = {}
        for i, score in enumerate(raw_scores):
            if score > 0:
                out[docs[i]["id"]] = score / peak
        return out

    # ── Hybrid retrieval ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        collection_names: list[str] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + vector retrieval across one or more collections.

        Returns list of dicts with keys: text, metadata, score, collection.
        """
        targets = collection_names or [c.name for c in ALL_COLLECTIONS]
        all_results: list[dict[str, Any]] = []

        for col_name in targets:
            col = self.get_or_create_collection(col_name)
            if col.count() == 0:
                continue

            # Vector search
            vector_kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(top_k * 2, col.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if metadata_filter:
                vector_kwargs["where"] = metadata_filter

            try:
                vec_results = col.query(**vector_kwargs)
            except Exception as exc:
                # A single collection's vector query failed — skip it but
                # make the failure VISIBLE (fail-loud rule, UPGRADE_NOTES
                # §1.3).  One bad collection must not silently degrade
                # retrieval quality across the whole corpus.
                log.error("RAG vector query failed on collection %r: %s: %s",
                          col_name, type(exc).__name__, exc)
                continue

            # Build vector score map (ChromaDB returns L2 distance; convert)
            vec_scores: dict[str, float] = {}
            vec_docs: dict[str, dict] = {}
            if vec_results["ids"] and vec_results["ids"][0]:
                for doc_id, text, meta, dist in zip(
                    vec_results["ids"][0],
                    vec_results["documents"][0],
                    vec_results["metadatas"][0],
                    vec_results["distances"][0],
                ):
                    # Convert L2 distance to similarity score [0,1]
                    sim = 1.0 / (1.0 + dist)
                    vec_scores[doc_id] = sim
                    vec_docs[doc_id] = {"text": text, "metadata": meta or {}}

            # BM25 search
            bm25_scores: dict[str, float] = {}
            if col_name not in self._bm25 or self._bm25[col_name] is None:
                self._rebuild_bm25(col_name)

            bm25 = self._bm25.get(col_name)
            bm25_docs = self._bm25_docs.get(col_name, [])

            if bm25 is not None and bm25_docs:
                tokens = tokenize_for_bm25(query)
                scores = bm25.get_scores(tokens)
                max_score = max(scores) if max(scores) > 0 else 1.0
                for idx, score in enumerate(scores):
                    doc_id = bm25_docs[idx]["id"]
                    bm25_scores[doc_id] = score / max_score  # normalise to [0,1]
                    if doc_id not in vec_docs:
                        vec_docs[doc_id] = {
                            "text": bm25_docs[idx]["text"],
                            "metadata": bm25_docs[idx]["metadata"],
                        }

            # Combine scores
            all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
            for doc_id in all_ids:
                vs = vec_scores.get(doc_id, 0.0)
                bs = bm25_scores.get(doc_id, 0.0)
                combined = VECTOR_WEIGHT * vs + BM25_WEIGHT * bs

                if combined >= RELEVANCE_THRESHOLD:
                    all_results.append({
                        "id": doc_id,
                        "text": vec_docs[doc_id]["text"],
                        "metadata": vec_docs[doc_id]["metadata"],
                        "score": combined,
                        "collection": col_name,
                    })

        # Sort by score descending and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # ── Query decomposition ───────────────────────────────────────

    def decompose_and_retrieve(
        self,
        sub_queries: list[str],
        collection_names: list[str] | None = None,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve for multiple sub-queries and merge results.

        The LLM decomposes a complex question into sub-queries before
        calling this.  Results are deduplicated by document ID.
        """
        seen_ids: set[str] = set()
        merged: list[dict[str, Any]] = []

        for sq in sub_queries:
            results = self.retrieve(sq, collection_names, top_k=top_k)
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    merged.append(r)

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    # ── Corrective RAG loop ───────────────────────────────────────

    def retrieve_with_correction(
        self,
        query: str,
        collection_names: list[str] | None = None,
        top_k: int = RERANK_TOP_K,
        min_confidence: float = 0.4,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Retrieve and flag if results seem insufficient.

        Returns (results, needs_correction).  The caller should either
        reformulate or broaden the query if needs_correction is True.
        """
        results = self.retrieve(query, collection_names, top_k=top_k * 2)

        if not results:
            return [], True

        avg_score = sum(r["score"] for r in results) / len(results)
        top_results = results[:top_k]

        needs_correction = avg_score < min_confidence or len(results) < 3
        return top_results, needs_correction

    # ── Primary-first retrieval ──────────────────────────────────

    def retrieve_tiered(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        primary_min: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Retrieve from PRIMARY first, then SECONDARY for explanation.

        Returns (primary_results, secondary_results).
        Primary gives actionable answers; secondary gives physics reasoning.
        """
        from bl_pipeline.rag.collections import PRIMARY_COLLECTIONS, SECONDARY_COLLECTIONS

        primary_names = [c.name for c in PRIMARY_COLLECTIONS]
        secondary_names = [c.name for c in SECONDARY_COLLECTIONS]

        primary = self.retrieve(query, primary_names, top_k=top_k, metadata_filter=metadata_filter)

        # Only hit secondary if primary is thin
        sec_k = top_k if len(primary) < primary_min else max(3, top_k // 2)
        secondary = self.retrieve(query, secondary_names, top_k=sec_k, metadata_filter=metadata_filter)

        return primary, secondary

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return document counts per collection."""
        return {c.name: self.collection_count(c.name) for c in ALL_COLLECTIONS}

    def stats_by_tier(self) -> dict[str, dict[str, int]]:
        """Return counts grouped by PRIMARY / SECONDARY."""
        from bl_pipeline.rag.collections import PRIMARY_COLLECTIONS, SECONDARY_COLLECTIONS
        return {
            "primary": {c.name: self.collection_count(c.name) for c in PRIMARY_COLLECTIONS},
            "secondary": {c.name: self.collection_count(c.name) for c in SECONDARY_COLLECTIONS},
        }
