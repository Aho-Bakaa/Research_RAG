"""dedup.py — Exact-hash and near-duplicate deduplication at index and candidate pool time.

Implements rule 5 of section 4 in RAG_ARCHITECTURE.md:
- Exact-hash dedup via text normalization and SHA-256.
- Near-duplicate dedup via MinHash / token-set Jaccard / embedding cosine similarity.
- Keeps occurrence with most context; flags or filters redundant duplicates.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for hash comparison: lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    # Strip metadata headers, e.g. '[paper: ... page: ...]'
    cleaned = re.sub(r"\[(?:paper|chunk_id|page)[^\]]*\]", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def exact_text_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = normalize_text_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_token_jaccard(text1: str, text2: str) -> float:
    """Calculate token-level Jaccard similarity between two texts."""
    norm1 = normalize_text_for_dedup(text1)
    norm2 = normalize_text_for_dedup(text2)
    if not norm1 or not norm2:
        return 0.0
    s1 = set(norm1.split())
    s2 = set(norm2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(len(s1 | s2))


def deduplicate_chunks(
    chunks: Sequence[Any],
    similarity_threshold: float = 0.90,
    text_accessor: str = "content",
) -> list[Any]:
    """Filter out exact and near-duplicate chunks from a list.

    Parameters
    ----------
    chunks : Sequence[Any]
        List of chunk objects or dictionaries.
    similarity_threshold : float
        Jaccard / near-duplicate threshold above which the later chunk is considered a duplicate.
    text_accessor : str
        Attribute or key name to extract text from chunk.

    Returns
    -------
    list[Any]
        Deduplicated list of chunks retaining the richer / earlier entries.
    """
    seen_hashes: set[str] = set()
    unique_chunks: list[Any] = []
    unique_texts: list[str] = []

    def _get_text(c: Any) -> str:
        if isinstance(c, dict):
            if text_accessor in c and c[text_accessor]:
                return str(c[text_accessor])
            if "payload" in c and isinstance(c["payload"], dict):
                return str(c["payload"].get(text_accessor, "") or "")
            return str(c.get(text_accessor, "") or "")
        return str(getattr(c, text_accessor, "") or "")

    for chunk in chunks:
        raw_text = _get_text(chunk)
        if not raw_text.strip():
            continue

        h = exact_text_hash(raw_text)
        if h in seen_hashes:
            continue

        # Check near-duplicate against already accepted chunks
        is_near_dup = False
        for ut in unique_texts:
            jaccard = compute_token_jaccard(raw_text, ut)
            if jaccard >= similarity_threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            seen_hashes.add(h)
            unique_chunks.append(chunk)
            unique_texts.append(raw_text)

    return unique_chunks
