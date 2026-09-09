"""Unit tests for Phase 2: StructurePreservingChunker and dedup.py.
"""
from pathlib import Path
from bl_pipeline.rag.dedup import deduplicate_chunks, exact_text_hash, compute_token_jaccard
from bl_pipeline.rag.parsers.layout_parser import LayoutAwareParser
from bl_pipeline.rag.structured_chunker import StructurePreservingChunker, StructuredChunk
from bl_pipeline.rag.schema import DocumentElement, ElementType


def test_dedup_exact_and_near_duplicates():
    t1 = "Abu-Ghannam and Shaw (1980) investigated boundary-layer transition under free-stream turbulence."
    t2 = "Abu-Ghannam and Shaw (1980) investigated boundary-layer transition under free-stream turbulence." # exact
    t3 = "Abu-Ghannam and Shaw (1980) investigated boundary layer transition under free-stream turbulence!" # near-dup
    t4 = "Mayle (1991) developed an algebraic correlation for turbomachinery boundary layers."

    chunks = [
        {"content": t1, "id": 1},
        {"content": t2, "id": 2},
        {"content": t3, "id": 3},
        {"content": t4, "id": 4},
    ]

    deduped = deduplicate_chunks(chunks, similarity_threshold=0.85)
    assert len(deduped) == 2, f"Expected 2 unique concepts, got {len(deduped)}"
    assert deduped[0]["id"] == 1
    assert deduped[1]["id"] == 4
    print("PASS  test_dedup_exact_and_near_duplicates")


def test_structured_chunking_on_parsed_elements():
    pdf_path = Path("data/primary_rag_v2_nemotron_8b/abu_ghannam_shaw_1980.pdf")
    if not pdf_path.exists():
        print("SKIP  test_structured_chunking_on_parsed_elements (PDF not found)")
        return

    parser = LayoutAwareParser()
    elements = parser.parse_pdf(pdf_path, paper_id="abu_ghannam_shaw_1980")

    chunker = StructurePreservingChunker()
    chunks = chunker.chunk_elements(elements, paper_id="abu_ghannam_shaw_1980")

    assert len(chunks) > 0, "Must produce structured chunks"

    parent_chunks = [c for c in chunks if c.element_type == "parent"]
    child_chunks = [c for c in chunks if c.element_type != "parent"]

    assert len(parent_chunks) > 0, "Must create section-level parent chunks"
    assert len(child_chunks) > 0, "Must create child chunks"

    parent_ids = {p.chunk_id for p in parent_chunks}
    for child in child_chunks:
        assert child.parent_id in parent_ids, f"Child {child.chunk_id} parent_id must link to existing parent"

    types = {c.element_type for c in chunks}
    print(f"Generated {len(chunks)} chunks (Parents: {len(parent_chunks)}, Children: {len(child_chunks)}) across types: {types}")
    print("PASS  test_structured_chunking_on_parsed_elements")


def test_table_atomic_and_header_repeat():
    header = "| Re_theta | Tu% | M_mm |"
    sep = "| --- | --- | --- |"
    rows = [f"| {i*100} | {i*0.5} | {i*2} |" for i in range(40)]
    table_text = "\n".join([header, sep] + rows)

    el = DocumentElement(
        element_type=ElementType.TABLE.value,
        paper_id="tab_test",
        page=3,
        section_path="4. Tables",
        text=table_text,
    )

    chunker = StructurePreservingChunker(max_table_rows_per_chunk=15)
    chunks = chunker._chunk_table(el, parent_id="pid_123", paper_id="tab_test")

    assert len(chunks) >= 2, "Long table must split into multiple chunks"
    for c in chunks:
        assert "Re_theta" in c.content, "Every split table chunk must repeat the header!"
        assert c.parent_id == "pid_123"
    print("PASS  test_table_atomic_and_header_repeat")


if __name__ == "__main__":
    test_dedup_exact_and_near_duplicates()
    test_table_atomic_and_header_repeat()
    test_structured_chunking_on_parsed_elements()
    print("------------------------------------------------------------")
    print("Phase 2 chunker and dedup tests passed!")
