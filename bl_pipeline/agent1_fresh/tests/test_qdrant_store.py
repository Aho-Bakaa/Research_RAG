"""Unit test for Phase 3: QdrantMultiTrackStore with payload filtering.
"""
import uuid
from bl_pipeline.rag.qdrant_store import QdrantMultiTrackStore
from bl_pipeline.rag.structured_chunker import StructuredChunk


def test_qdrant_multitrack_upsert_and_filtered_search():
    store = QdrantMultiTrackStore(location=":memory:", vector_size=4)

    # 1. Create dummy chunks across two papers
    c1 = StructuredChunk(
        chunk_id=str(uuid.uuid4()),
        paper_id="abu_ghannam_shaw_1980",
        page=3,
        section_path="2. Experiments",
        element_type="paragraph",
        content="Transition onset is governed by free stream turbulence.",
    )
    v1 = [1.0, 0.0, 0.0, 0.0]

    c2 = StructuredChunk(
        chunk_id=str(uuid.uuid4()),
        paper_id="mayle_1991",
        page=5,
        section_path="3. Correlation",
        element_type="equation",
        content="Re_theta_t = 400 * Tu^(-5/8)",
        metadata={"equation_ref": "Eq. (9)"},
    )
    v2 = [0.0, 1.0, 0.0, 0.0]

    # Upsert into 'text' and 'equation' tracks
    n1 = store.upsert_chunks("text", [c1], [v1])
    n2 = store.upsert_chunks("equation", [c2], [v2])
    assert n1 == 1
    assert n2 == 1

    # 2. Search without filter
    res = store.search("text", [1.0, 0.0, 0.0, 0.0], limit=5)
    assert len(res) == 1
    assert res[0]["chunk_id"] == c1.chunk_id
    assert res[0]["score"] > 0.99
    assert res[0]["payload"]["paper_id"] == "abu_ghannam_shaw_1980"

    # 3. Search with payload filter (filtering for mayle_1991 in equation track)
    res_eq = store.search("equation", [0.0, 1.0, 0.0, 0.0], limit=5, payload_filter={"paper_id": "mayle_1991"})
    assert len(res_eq) == 1
    assert res_eq[0]["payload"]["equation_ref"] == "Eq. (9)"

    # 4. Search with negative payload filter (should return 0)
    res_none = store.search("equation", [0.0, 1.0, 0.0, 0.0], limit=5, payload_filter={"paper_id": "nonexistent_paper"})
    assert len(res_none) == 0

    # 5. Multi-track search
    multi_res = store.search_multitrack(["text", "equation"], [1.0, 1.0, 0.0, 0.0], limit_per_track=2)
    assert len(multi_res) == 2
    tracks = {r["track"] for r in multi_res}
    assert "text" in tracks and "equation" in tracks

    print("PASS  test_qdrant_multitrack_upsert_and_filtered_search")


if __name__ == "__main__":
    test_qdrant_multitrack_upsert_and_filtered_search()
    print("------------------------------------------------------------")
    print("Phase 3 Qdrant store tests passed!")
