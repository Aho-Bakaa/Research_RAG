"""hierarchical_chunker.py — Multi-tier recursive tree chunker for scientific papers.

Implements the 4-tier hierarchy:
  L0: Document (Paper Node)
    └── L1: Major Section (H1)
          └── L2: Subsection (H2)
                └── L3: Leaf Micro-Elements (Paragraph, Equation, Table, Figure)

Preserves:
- Explicit parent_id, children_ids, and ancestor_ids.
- Sequential linear reading order via sibling_prev and sibling_next pointers.
- Variable-zoom context retrieval (zooming into Subsection or Section).
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from bl_pipeline.rag.parsers.scientific_normalizer import normalize_scientific_text
from bl_pipeline.rag.schema import DocumentElement, ElementType


@dataclass
class HierarchicalNode:
    """A node in the document hierarchy tree."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    depth: int = 3  # 0: Doc, 1: H1, 2: H2, 3: Leaf
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    ancestor_ids: list[str] = field(default_factory=list)
    sibling_prev: str | None = None
    sibling_next: str | None = None
    section_path: str = ""
    heading_title: str = ""
    element_type: str = "paragraph"  # document, section, subsection, paragraph, equation, table, figure
    content: str = ""
    token_count: int = 0
    page_start: int = 1
    page_end: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalDocumentTree:
    """Complete document tree for a parsed paper."""
    paper_id: str
    root_id: str
    nodes: dict[str, HierarchicalNode] = field(default_factory=dict)

    @property
    def root(self) -> HierarchicalNode:
        return self.nodes[self.root_id]

    @property
    def leaf_nodes(self) -> list[HierarchicalNode]:
        return [n for n in self.nodes.values() if n.depth == 3]

    @property
    def section_nodes(self) -> list[HierarchicalNode]:
        return [n for n in self.nodes.values() if n.depth in (1, 2)]

    def get_node(self, node_id: str) -> HierarchicalNode | None:
        return self.nodes.get(node_id)

    def get_enclosing_section(self, node_id: str, target_depth: int = 2) -> HierarchicalNode | None:
        """Walk up ancestor tree to find enclosing section or subsection at target depth."""
        curr = self.nodes.get(node_id)
        if not curr:
            return None
        if curr.depth <= target_depth:
            return curr

        for anc_id in reversed(curr.ancestor_ids):
            anc = self.nodes.get(anc_id)
            if anc and anc.depth == target_depth:
                return anc
        # Fallback to closest available ancestor
        if curr.parent_id and curr.parent_id in self.nodes:
            return self.nodes[curr.parent_id]
        return None

    def get_sibling_window(
        self,
        node_id: str,
        before: int = 1,
        after: int = 1,
    ) -> list[HierarchicalNode]:
        """Fetch consecutive sibling leaf nodes in linear reading order."""
        target = self.nodes.get(node_id)
        if not target or target.depth != 3:
            return []

        # Walk backward
        preceding = []
        curr = target
        for _ in range(before):
            if not curr.sibling_prev or curr.sibling_prev not in self.nodes:
                break
            curr = self.nodes[curr.sibling_prev]
            preceding.append(curr)
        preceding.reverse()

        # Walk forward
        following = []
        curr = target
        for _ in range(after):
            if not curr.sibling_next or curr.sibling_next not in self.nodes:
                break
            curr = self.nodes[curr.sibling_next]
            following.append(curr)

        return preceding + [target] + following


class HierarchicalTreeChunker:
    """Constructs a 4-tier document tree from a stream of DocumentElements."""

    def __init__(
        self,
        min_paragraph_chars: int = 180,
        max_paragraph_chars: int = 1400,
    ):
        self.min_paragraph_chars = min_paragraph_chars
        self.max_paragraph_chars = max_paragraph_chars

    def build_tree(
        self,
        elements: Sequence[DocumentElement],
        paper_id: str | None = None,
    ) -> HierarchicalDocumentTree:
        """Parse elements into a hierarchical tree with full parent-child and sibling pointers."""
        paper_id = paper_id or (elements[0].paper_id if elements else "unknown_paper")
        tree = HierarchicalDocumentTree(
            paper_id=paper_id,
            root_id="",
        )

        if not elements:
            root = HierarchicalNode(
                paper_id=paper_id,
                depth=0,
                element_type="document",
                heading_title=paper_id,
                content=f"# Document: {paper_id}",
            )
            tree.root_id = root.node_id
            tree.nodes[root.node_id] = root
            return tree

        # 1. Create Document Root (L0)
        root = HierarchicalNode(
            paper_id=paper_id,
            depth=0,
            element_type="document",
            heading_title=paper_id,
            content=f"# Document: {paper_id}",
            page_start=min(e.page for e in elements),
            page_end=max(e.page for e in elements),
        )
        tree.root_id = root.node_id
        tree.nodes[root.node_id] = root

        # 2. Tracking active branch
        current_h1: HierarchicalNode | None = None
        current_h2: HierarchicalNode | None = None
        leaf_nodes: list[HierarchicalNode] = []

        # Buffer for merging small adjacent paragraphs
        acc_elements: list[DocumentElement] = []

        def _get_active_parent() -> HierarchicalNode:
            if current_h2 is not None:
                return current_h2
            if current_h1 is not None:
                return current_h1
            return root

        def _flush_paragraph_accumulator() -> None:
            nonlocal acc_elements
            if not acc_elements:
                return

            parent = _get_active_parent()
            merged_text = "\n\n".join(e.text for e in acc_elements)
            first_e = acc_elements[0]
            last_e = acc_elements[-1]

            leaf = HierarchicalNode(
                paper_id=paper_id,
                depth=3,
                parent_id=parent.node_id,
                ancestor_ids=[root.node_id] + ([current_h1.node_id] if current_h1 else []) + ([current_h2.node_id] if current_h2 else []),
                section_path=parent.section_path or parent.heading_title,
                heading_title=parent.heading_title,
                element_type="paragraph",
                content=merged_text,
                token_count=len(merged_text.split()),
                page_start=first_e.page,
                page_end=last_e.page,
                metadata={
                    "element_count": len(acc_elements),
                    "reading_order": first_e.reading_order,
                    "bbox": first_e.bbox,
                },
            )
            parent.children_ids.append(leaf.node_id)
            tree.nodes[leaf.node_id] = leaf
            leaf_nodes.append(leaf)
            acc_elements = []

        # 3. Process elements sequentially
        for idx, el in enumerate(elements):
            if getattr(el, "suppressed", False):
                continue

            clean_text = normalize_scientific_text(el.text)
            if not clean_text:
                continue

            conf_dict = el.confidence.to_dict() if el.confidence else {}

            if el.element_type == ElementType.HEADING.value:
                _flush_paragraph_accumulator()

                # Infer heading level: H1 vs H2
                # e.g., "1. Introduction" -> H1, "1.2 Boundary Conditions" -> H2
                is_sub = bool(re.match(r"^\d+\.\d+", clean_text.strip())) or el.heading_level >= 2

                if not is_sub:
                    # New Major Section (H1)
                    sec_node = HierarchicalNode(
                        paper_id=paper_id,
                        depth=1,
                        parent_id=root.node_id,
                        ancestor_ids=[root.node_id],
                        section_path=clean_text.strip(),
                        heading_title=clean_text.strip(),
                        element_type="section",
                        content=f"# {clean_text.strip()}",
                        page_start=el.page,
                        page_end=el.page,
                        metadata={"confidence": conf_dict},
                    )
                    root.children_ids.append(sec_node.node_id)
                    tree.nodes[sec_node.node_id] = sec_node
                    current_h1 = sec_node
                    current_h2 = None
                else:
                    # New Subsection (H2)
                    parent_h1 = current_h1 or root
                    subsec_path = f"{parent_h1.section_path} > {clean_text.strip()}" if parent_h1 != root else clean_text.strip()
                    subsec_node = HierarchicalNode(
                        paper_id=paper_id,
                        depth=2,
                        parent_id=parent_h1.node_id,
                        ancestor_ids=parent_h1.ancestor_ids + [parent_h1.node_id],
                        section_path=subsec_path,
                        heading_title=clean_text.strip(),
                        element_type="subsection",
                        content=f"## {clean_text.strip()}",
                        page_start=el.page,
                        page_end=el.page,
                        metadata={"confidence": conf_dict},
                    )
                    parent_h1.children_ids.append(subsec_node.node_id)
                    tree.nodes[subsec_node.node_id] = subsec_node
                    current_h2 = subsec_node

            elif el.element_type == ElementType.EQUATION.value:
                # Equation boundary stitching: absorb immediate lead-in definition
                lead_in = ""
                if acc_elements:
                    acc_text = "\n\n".join(e.text for e in acc_elements)
                    if len(acc_text) < 180 or acc_text.rstrip().endswith((":", "as:", "follows:", "given by:")):
                        lead_in = acc_text
                        acc_elements = []
                    else:
                        _flush_paragraph_accumulator()

                parent = _get_active_parent()
                eq_content = f"Context: {lead_in}\n\nEquation {el.equation_ref or ''}:\n```latex\n{clean_text}\n```" if lead_in else f"Equation {el.equation_ref or ''}:\n```latex\n{clean_text}\n```"

                leaf = HierarchicalNode(
                    paper_id=paper_id,
                    depth=3,
                    parent_id=parent.node_id,
                    ancestor_ids=[root.node_id] + ([current_h1.node_id] if current_h1 else []) + ([current_h2.node_id] if current_h2 else []),
                    section_path=parent.section_path or parent.heading_title,
                    heading_title=parent.heading_title,
                    element_type="equation",
                    content=eq_content,
                    token_count=len(eq_content.split()),
                    page_start=el.page,
                    page_end=el.page,
                    metadata={
                        "equation_ref": el.equation_ref,
                        "raw_formula": clean_text,
                        "reading_order": el.reading_order,
                        "bbox": el.bbox,
                        "confidence": conf_dict,
                        "is_certified": (el.metadata or {}).get("is_certified", True),
                    },
                )
                parent.children_ids.append(leaf.node_id)
                tree.nodes[leaf.node_id] = leaf
                leaf_nodes.append(leaf)

            elif el.element_type == ElementType.TABLE.value:
                _flush_paragraph_accumulator()
                parent = _get_active_parent()
                tbl_content = clean_text
                tbl_meta = {
                    "table_structure": el.table_structure,
                    "reading_order": el.reading_order,
                    "bbox": el.bbox,
                    "confidence": conf_dict,
                }
                if el.canonical_table:
                    tbl_meta["canonical_table"] = el.canonical_table.to_dict()
                    summary = el.canonical_table.to_natural_language_summary()
                    if summary:
                        tbl_meta["summary"] = summary
                        tbl_content += f"\n\nTable Summary: {summary}"

                leaf = HierarchicalNode(
                    paper_id=paper_id,
                    depth=3,
                    parent_id=parent.node_id,
                    ancestor_ids=[root.node_id] + ([current_h1.node_id] if current_h1 else []) + ([current_h2.node_id] if current_h2 else []),
                    section_path=parent.section_path or parent.heading_title,
                    heading_title=parent.heading_title,
                    element_type="table",
                    content=tbl_content,
                    token_count=len(tbl_content.split()),
                    page_start=el.page,
                    page_end=el.page,
                    metadata=tbl_meta,
                )
                parent.children_ids.append(leaf.node_id)
                tree.nodes[leaf.node_id] = leaf
                leaf_nodes.append(leaf)

            elif el.element_type in (ElementType.FIGURE.value, ElementType.CAPTION.value):
                _flush_paragraph_accumulator()
                parent = _get_active_parent()
                fig_meta = {
                    "figure_ref": el.figure_ref,
                    "reading_order": el.reading_order,
                    "bbox": el.bbox,
                    "confidence": conf_dict,
                }
                if el.canonical_figure:
                    fig_meta["canonical_figure"] = el.canonical_figure.to_dict()
                    fig_meta["figure_type"] = el.canonical_figure.figure_type.value
                    if el.canonical_figure.plot_metadata:
                        fig_meta["plot_metadata"] = el.canonical_figure.plot_metadata.__dict__

                leaf = HierarchicalNode(
                    paper_id=paper_id,
                    depth=3,
                    parent_id=parent.node_id,
                    ancestor_ids=[root.node_id] + ([current_h1.node_id] if current_h1 else []) + ([current_h2.node_id] if current_h2 else []),
                    section_path=parent.section_path or parent.heading_title,
                    heading_title=parent.heading_title,
                    element_type="figure",
                    content=f"[{el.figure_ref or 'Figure'}] {clean_text}",
                    token_count=len(clean_text.split()),
                    page_start=el.page,
                    page_end=el.page,
                    metadata=fig_meta,
                )
                parent.children_ids.append(leaf.node_id)
                tree.nodes[leaf.node_id] = leaf
                leaf_nodes.append(leaf)

            elif el.element_type in (ElementType.ALGORITHM.value, ElementType.PSEUDOCODE.value, ElementType.CODE.value):
                _flush_paragraph_accumulator()
                parent = _get_active_parent()
                code_content = f"```{el.element_type}\n{clean_text}\n```"
                leaf = HierarchicalNode(
                    paper_id=paper_id,
                    depth=3,
                    parent_id=parent.node_id,
                    ancestor_ids=[root.node_id] + ([current_h1.node_id] if current_h1 else []) + ([current_h2.node_id] if current_h2 else []),
                    section_path=parent.section_path or parent.heading_title,
                    heading_title=parent.heading_title,
                    element_type=el.element_type,
                    content=code_content,
                    token_count=len(code_content.split()),
                    page_start=el.page,
                    page_end=el.page,
                    metadata={
                        "reading_order": el.reading_order,
                        "bbox": el.bbox,
                        "confidence": conf_dict,
                    },
                )
                parent.children_ids.append(leaf.node_id)
                tree.nodes[leaf.node_id] = leaf
                leaf_nodes.append(leaf)

            else:
                # Standard paragraph
                acc_len = sum(len(e.text) for e in acc_elements)
                if acc_len + len(clean_text) > self.max_paragraph_chars and acc_len >= self.min_paragraph_chars:
                    _flush_paragraph_accumulator()
                acc_elements.append(el)

        # Flush any remaining paragraph elements
        _flush_paragraph_accumulator()

        # 4. Chain sequential reading-order sibling pointers across all leaf nodes
        for i in range(len(leaf_nodes)):
            if i > 0:
                leaf_nodes[i].sibling_prev = leaf_nodes[i - 1].node_id
            if i < len(leaf_nodes) - 1:
                leaf_nodes[i].sibling_next = leaf_nodes[i + 1].node_id

        # 5. Populate Section and Subsection aggregate contents
        for sec_node in tree.section_nodes:
            child_leaves = [
                tree.nodes[cid] for cid in sec_node.children_ids
                if cid in tree.nodes and tree.nodes[cid].depth == 3
            ]
            if child_leaves:
                sec_node.content = f"# {sec_node.section_path}\n\n" + "\n\n".join(c.content for c in child_leaves[:8])
                sec_node.token_count = sum(c.token_count for c in child_leaves)
                sec_node.page_start = min(c.page_start for c in child_leaves)
                sec_node.page_end = max(c.page_end for c in child_leaves)

        return tree
