"""test_hybrid_and_reranker.py — Unit tests for BM25, RRF fusion, and Cross-Encoder re-ranking."""
import pytest
from bl_pipeline.rag.hybrid_search import (
    SimpleBM25,
    apply_hybrid_search,
    rrf_fusion,
    tokenize_scientific_text,
)
from bl_pipeline.rag.reranker import CrossEncoderReranker


def test_tokenize_scientific_text():
    tokens = tokenize_scientific_text("Abu-Ghannam & Shaw (1980) correlation: Re_theta_t = 163 + exp(6.91 - Tu)")
    assert "abu-ghannam" in tokens or "abu" in tokens
    assert "163" in tokens
    assert "6.91" in tokens
    assert "re_theta_t" in tokens
    assert "tu" in tokens


def test_simple_bm25_exact_formula_boost():
    candidates = [
        {"chunk_id": "c1", "content": "General boundary layer transition on flat plates is influenced by many parameters."},
        {"chunk_id": "c2", "content": "Abu-Ghannam and Shaw (1980) correlation states that Re_theta_t = 163 + exp(6.91 - Tu)."},
        {"chunk_id": "c3", "content": "The flow is turbulent when the Reynolds number exceeds critical values."},
    ]
    query = "What is the Abu-Ghannam and Shaw correlation with 163 and 6.91?"
    tokens = tokenize_scientific_text(query)

    bm25 = SimpleBM25()
    scores = bm25.score_candidates(tokens, candidates)

    assert len(scores) == 3
    # Candidate c2 contains the exact formula numbers and names, so it must have the highest BM25 score
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_rrf_fusion():
    dense = [
        {"chunk_id": "c1", "content": "Dense Rank 1"},
        {"chunk_id": "c2", "content": "Dense Rank 2"},
    ]
    sparse = [
        {"chunk_id": "c2", "content": "Sparse Rank 1 (BM25 top hit)"},
        {"chunk_id": "c1", "content": "Sparse Rank 2"},
    ]

    fused = rrf_fusion(dense, sparse, k=60)
    assert len(fused) == 2
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > 0.0


def test_apply_hybrid_search():
    dense_candidates = [
        {"chunk_id": "doc_general", "content": "General boundary layer discussion without formulas."},
        {"chunk_id": "doc_exact", "content": "Exact correlation formula: Re_theta_t = 163 + exp(6.91 - Tu)."},
    ]
    query = "Abu-Ghannam Shaw onset Re_theta 163"

    fused = apply_hybrid_search(query, dense_candidates)
    assert len(fused) == 2
    # Even if dense search returned doc_general first, doc_exact should be elevated by BM25
    assert fused[0]["chunk_id"] == "doc_exact"


def test_cross_encoder_reranker_fallback_or_live():
    reranker = CrossEncoderReranker(enabled=False)  # test fallback mode first
    candidates = [
        {"chunk_id": "c1", "content": "Text 1"},
        {"chunk_id": "c2", "content": "Text 2"},
    ]
    out = reranker.rerank("query", candidates, top_k=2)
    assert len(out) == 2
