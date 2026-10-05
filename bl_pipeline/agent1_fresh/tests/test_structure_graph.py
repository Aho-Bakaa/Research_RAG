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


def test_structure_graph_hierarchical_tree_and_siblings():
    from bl_pipeline.rag.schema import DocumentElement, ElementType
    from bl_pipeline.rag.hierarchical_chunker import HierarchicalTreeChunker

    chunker = HierarchicalTreeChunker()
    elements = [
        DocumentElement(
            element_type=ElementType.HEADING.value,
            text="3. Transition Modeling",
            page=1,
            heading_level=1,
        ),
        DocumentElement(
            element_type=ElementType.PARAGRAPH.value,
            text="We begin with the transport equation for intermittency given in Eq. (3.1).",
            page=1,
        ),
        DocumentElement(
            element_type=ElementType.EQUATION.value,
            text=r"\frac{\partial \gamma}{\partial t} = P_\gamma - E_\gamma",
            equation_ref="3.1",
            page=1,
        ),
        DocumentElement(
            element_type=ElementType.PARAGRAPH.value,
            text="The experimental data in Table 2 supports this correlation.",
            page=1,
        ),
    ]

    tree = chunker.build_tree(elements, paper_id="test_paper_tree")
    graph = DocumentStructureGraph()
    graph.add_hierarchical_tree(tree)

    leaves = tree.leaf_nodes
    # Note: Paragraph 1 is absorbed into the equation chunk by boundary stitching (preventing orphan stubs)
    assert len(leaves) == 2
    eq_leaf = leaves[0]
    tab_leaf = leaves[1]

    # Check sibling reading-order traversal in graph
    siblings = graph.get_sibling_nodes(eq_leaf.node_id, before=0, after=1)
    assert len(siblings) == 2
    assert siblings[0] == eq_leaf.node_id
    assert siblings[1] == tab_leaf.node_id

    # Check intra-paper cross-reference traversal for Eq. (3.1)
    eq_refs = graph.get_chunks_referencing_equation("test_paper_tree", "3.1")
    assert len(eq_refs) >= 1
    assert eq_leaf.node_id in eq_refs

    # Check intra-paper cross-reference traversal for Table 2
    tab_refs = graph.get_chunks_referencing_table("test_paper_tree", "2")
    assert len(tab_refs) >= 1
    assert tab_leaf.node_id in tab_refs


if __name__ == "__main__":
    test_structure_graph_cross_references_and_citations()
    test_structure_graph_hierarchical_tree_and_siblings()
    print("------------------------------------------------------------")
    print("All document structure graph tests passed!")
