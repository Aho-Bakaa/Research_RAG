"""Unit tests for Phase 5: QueryRouter, MMR, and CRAG Quality Gate.
"""
from bl_pipeline.rag.router import QueryRouter
from bl_pipeline.rag.mmr import select_diverse_chunks_mmr
from bl_pipeline.rag.crag_gate import RetrievalQualityGate


def test_query_router():
    router = QueryRouter()

    r1 = router.route_query("Show me the skin friction plot from the Fransson experiment")
    assert "figure" in r1.primary_tracks
    assert r1.track_weights["figure"] > 1.0

    r2 = router.route_query("What does Eq. (3.6) predict for onset?")
    assert "equation" in r2.primary_tracks
    assert r2.target_equation_ref == "3.6"

    r3 = router.route_query("Who cites Abu-Ghannam & Shaw 1980 in this corpus?")
    assert r3.is_structural_or_citation is True
    assert "graph" in r3.primary_tracks

    r4 = router.route_query("Tabulated onset values across different Tu percentages")
    assert "table" in r4.primary_tracks

    print("PASS  test_query_router")


def test_mmr_diversity():
    candidates = [
        {"id": 1, "score": 0.95, "content": "Abu-Ghannam Shaw 1980 correlation for onset"},
        {"id": 2, "score": 0.94, "content": "Abu-Ghannam Shaw 1980 correlation for onset identical"},
        {"id": 3, "score": 0.85, "content": "Fransson-Shahinfar length scale aware correlation"},
    ]
    # Vectors: item 1 and 2 are almost identical, item 3 is orthogonal
    embeddings = [
        [1.0, 0.05, 0.0],
        [0.99, 0.04, 0.0],
        [0.0, 0.0, 1.0],
    ]

    selected = select_diverse_chunks_mmr(candidates, embeddings=embeddings, limit=2, lambda_param=0.7)
    assert len(selected) == 2
    selected_ids = [c["id"] for c in selected]
    # MMR must pick item 1 (highest score) and item 3 (diverse), rejecting item 2 (redundant clone)
    assert selected_ids == [1, 3], f"MMR should select diverse candidates, got {selected_ids}"
    print("PASS  test_mmr_diversity")


def test_crag_quality_gate():
    gate = RetrievalQualityGate()

    # Incomplete retrieval (missing decay law when query has Tu)
    query = "Estimate onset for flat plate at Tu=2.7% with grid bar 5 mm"
    chunks_incomplete = [
        {"content": "Abu-Ghannam & Shaw correlation gives Re_theta_t = 163 + exp(...)"}
    ]
    verdict1 = gate.evaluate_retrieval(query, chunks_incomplete)
    assert verdict1.is_sufficient is False
    assert any("decay" in g.lower() for g in verdict1.detected_gaps)
    assert len(verdict1.suggested_rewrites) > 0

    # Complete retrieval
    chunks_complete = [
        {"content": "Abu-Ghannam & Shaw correlation gives Re_theta_t = 163 + exp(...) for transition onset"},
        {"content": "Fransson grid turbulence decay law Tu(x) = C * (x - x0)^(-b) with mesh size M"},
    ]
    verdict2 = gate.evaluate_retrieval(query, chunks_complete)
    assert verdict2.is_sufficient is True
    assert verdict2.verdict == "PASS"

    print("PASS  test_crag_quality_gate")


if __name__ == "__main__":
    test_query_router()
    test_mmr_diversity()
    test_crag_quality_gate()
    print("------------------------------------------------------------")
    print("Phase 5 retrieval pipeline tests passed!")
