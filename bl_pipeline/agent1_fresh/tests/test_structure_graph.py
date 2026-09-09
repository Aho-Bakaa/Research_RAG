"""Unit tests for Phase 4: DocumentStructureGraph.
"""
from pathlib import Path
import tempfile
from bl_pipeline.rag.graph.structure_graph import DocumentStructureGraph
from bl_pipeline.rag.structured_chunker import StructuredChunk


def test_structure_graph_cross_references_and_citations():
    graph = DocumentStructureGraph()

    # 1. Add papers
    graph.add_paper("fransson_2005", {"title": "Transition under FST"})
    graph.add_paper("fransson_shahinfar_2020", {"title": "Lambda-aware onset"})
    graph.add_paper("abu_ghannam_shaw_1980", {"title": "Natural and bypass transition"})

    # Citation edge: FS20 cites Fransson 2005 and AGS 1980
    graph.add_citation("fransson_shahinfar_2020", "fransson_2005", ref_num="[14]")
    graph.add_citation("fransson_shahinfar_2020", "abu_ghannam_shaw_1980", ref_num="[1]")

    assert "fransson_shahinfar_2020" in graph.get_citing_papers("fransson_2005")
    assert "fransson_2005" in graph.get_cited_papers("fransson_shahinfar_2020")
    assert "abu_ghannam_shaw_1980" in graph.get_cited_papers("fransson_shahinfar_2020")

    # 2. Add sections and chunks with cross-references
    c1 = StructuredChunk(
        chunk_id="chunk_fs20_eq36",
        paper_id="fransson_shahinfar_2020",
        page=12,
        section_path="3. Transition Formulation",
        content="The onset location is calculated using Eq. (3.6) and compares against Fig. 4.",
    )
    c2 = StructuredChunk(
        chunk_id="chunk_fs20_discussion",
        paper_id="fransson_shahinfar_2020",
        page=13,
        section_path="4. Discussion",
        content="We observe that formula (3.6) accounts for the integral length scale.",
    )

    graph.add_chunk(c1)
    graph.add_chunk(c2)

    # 3. Query chunks referencing Eq. (3.6)
    referencing = graph.get_chunks_referencing_equation("fransson_shahinfar_2020", "3.6")
    assert "chunk_fs20_eq36" in referencing
    assert "chunk_fs20_discussion" in referencing
    assert len(referencing) == 2

    # 4. Query chunks referencing Fig. 4
    fig_referencing = graph.get_chunks_referencing_figure("fransson_shahinfar_2020", "4")
    assert fig_referencing == ["chunk_fs20_eq36"]

    # 5. Section query
    sec_chunks = graph.get_section_chunks("fransson_shahinfar_2020", "3. Transition Formulation")
    assert "chunk_fs20_eq36" in sec_chunks

    # 6. Persistence roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "graph.json"
        graph.save_to_file(tmp_path)
        assert tmp_path.exists()

        loaded_graph = DocumentStructureGraph()
        loaded_graph.load_from_file(tmp_path)
        assert "chunk_fs20_eq36" in loaded_graph.get_chunks_referencing_equation("fransson_shahinfar_2020", "3.6")

    print("PASS  test_structure_graph_cross_references_and_citations")


if __name__ == "__main__":
    test_structure_graph_cross_references_and_citations()
    print("------------------------------------------------------------")
    print("Phase 4 document structure graph tests passed!")
