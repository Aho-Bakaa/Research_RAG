"""Unit test for Phase 1: DocumentElement schema and LayoutAwareParser.
"""
from pathlib import Path
from bl_pipeline.rag.schema import DocumentElement, ElementType
from bl_pipeline.rag.parsers.layout_parser import LayoutAwareParser, parse_pdf_to_elements


def test_document_element_roundtrip():
    el = DocumentElement(
        element_type=ElementType.EQUATION.value,
        paper_id="test_paper",
        page=2,
        section_path="2.1 Transition Model",
        heading_level=2,
        bbox=[10.0, 20.0, 300.0, 50.0],
        reading_order=5,
        text=r"Re_{\theta,t} = 163 + \exp(6.91 \cdot (1 - Tu/6.91))",
        equation_ref="Eq. (3)",
    )
    d = el.to_dict()
    assert d["element_type"] == "equation"
    assert d["equation_ref"] == "Eq. (3)"
    assert d["bbox"] == [10.0, 20.0, 300.0, 50.0]

    el2 = DocumentElement.from_dict(d)
    assert el2.element_id == el.element_id
    assert el2.text == el.text
    print("PASS  test_document_element_roundtrip")


def test_layout_aware_parser_on_real_pdf():
    pdf_path = Path("data/primary_rag_v2_nemotron_8b/abu_ghannam_shaw_1980.pdf")
    if not pdf_path.exists():
        print("SKIP  test_layout_aware_parser_on_real_pdf (PDF not found)")
        return

    parser = LayoutAwareParser()
    elements = parser.parse_pdf(pdf_path, paper_id="abu_ghannam_shaw_1980")

    assert len(elements) > 0, "Parser must extract non-empty elements"
    types = {e.element_type for e in elements}
    print(f"Extracted {len(elements)} elements across types: {types}")

    assert ElementType.PARAGRAPH.value in types, "Paragraph elements must be extracted"
    for e in elements[:10]:
        assert e.page >= 1
        assert len(e.bbox) == 4
        assert isinstance(e.reading_order, int)
    print("PASS  test_layout_aware_parser_on_real_pdf")


if __name__ == "__main__":
    test_document_element_roundtrip()
    test_layout_aware_parser_on_real_pdf()
    print("------------------------------------------------------------")
    print("Phase 1 schema & layout parser tests passed!")
