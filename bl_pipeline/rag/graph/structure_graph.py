"""structure_graph.py — Deterministic, LLM-free document structure graph.

Implements section 8 of RAG_ARCHITECTURE.md:
- Heading hierarchy (Section -> Chunk via 'contains' edges)
- Cross-reference edges ('refers_to') for Eq. (X), Fig. Y, Table Z via deterministic regex
- Citation edges ('cites') between papers parsed from bibliographies
- Zero index-time LLM hallucination: structure is the document's own
- Powered by NetworkX with disk JSON serialization and Cypher/Neo4j export readiness
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

from bl_pipeline.rag.structured_chunker import StructuredChunk

log = logging.getLogger(__name__)

# Cross-reference regexes matching inline paper text citations
_EQ_XREF_RX = re.compile(
    r"(?:(?:Eq(?:uation|\.)?|formula)\s*\(?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)\s*\)?)",
    re.IGNORECASE,
)
_FIG_XREF_RX = re.compile(
    r"(?:Fig(?:ure|\.)?\s*([0-9]+(?:\.[0-9]+)*[a-z]?))",
    re.IGNORECASE,
)
_TAB_XREF_RX = re.compile(
    r"(?:Tab(?:le|\.)?\s*([0-9]+(?:\.[0-9]+)*[a-z]?))",
    re.IGNORECASE,
)
_PAPER_CIT_RX = re.compile(
    r"\[([0-9]{1,3})\]|(?:\b([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s*\(?((?:19|20)[0-9]{2})\)?)"
)


class DocumentStructureGraph:
    """Deterministic document structure graph holding sections, chunks, cross-references, and citations."""

    def __init__(self):
        self.g = nx.DiGraph()

    def add_paper(self, paper_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Register a paper node."""
        self.g.add_node(
            f"paper:{paper_id}",
            node_type="paper",
            paper_id=paper_id,
            **(metadata or {}),
        )

    def add_section(
        self,
        paper_id: str,
        section_path: str,
        heading_level: int = 1,
        page: int = 1,
        parent_section_path: str | None = None,
    ) -> str:
        """Add a section node and connect it to its parent section or paper."""
        sec_id = f"sec:{paper_id}:{section_path}"
        self.g.add_node(
            sec_id,
            node_type="section",
            paper_id=paper_id,
            section_path=section_path,
            heading_level=heading_level,
            page=page,
        )

        if parent_section_path:
            parent_sec_id = f"sec:{paper_id}:{parent_section_path}"
            if self.g.has_node(parent_sec_id):
                self.g.add_edge(parent_sec_id, sec_id, relation="contains")
        else:
            paper_node = f"paper:{paper_id}"
            if self.g.has_node(paper_node):
                self.g.add_edge(paper_node, sec_id, relation="contains")

        return sec_id

    def add_chunk(self, chunk: StructuredChunk) -> str:
        """Add a chunk node, link to its section via 'contains', and extract cross-references."""
        chunk_node_id = f"chunk:{chunk.chunk_id}"
        self.g.add_node(
            chunk_node_id,
            node_type="chunk",
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            element_type=chunk.element_type,
            page=chunk.page,
            section_path=chunk.section_path,
            content_snippet=chunk.content[:200],
        )

        # Connect section -> chunk
        sec_id = f"sec:{chunk.paper_id}:{chunk.section_path}"
        if not self.g.has_node(sec_id):
            self.add_section(chunk.paper_id, chunk.section_path, heading_level=chunk.heading_level, page=chunk.page)
        self.g.add_edge(sec_id, chunk_node_id, relation="contains")

        # Connect parent chunk -> child chunk if parent_id present
        if chunk.parent_id:
            parent_node_id = f"chunk:{chunk.parent_id}"
            self.g.add_edge(parent_node_id, chunk_node_id, relation="parent_of")

        # Deterministic Cross-Reference Extraction from text:
        self._extract_and_link_cross_references(chunk_node_id, chunk)
        return chunk_node_id

    def add_citation(self, source_paper_id: str, target_paper_id: str, ref_num: str | None = None) -> None:
        """Add a deterministic citation edge between two papers."""
        src = f"paper:{source_paper_id}"
        tgt = f"paper:{target_paper_id}"
        if not self.g.has_node(src):
            self.add_paper(source_paper_id)
        if not self.g.has_node(tgt):
            self.add_paper(target_paper_id)
        self.g.add_edge(src, tgt, relation="cites", ref_num=ref_num)

    def _extract_and_link_cross_references(self, chunk_node_id: str, chunk: StructuredChunk) -> None:
        """Scan chunk text for 'Eq. (X)', 'Fig. Y', 'Table Z' and add 'refers_to' edges."""
        text = chunk.content

        # 1. Equation cross-references
        for match in _EQ_XREF_RX.finditer(text):
            eq_id = match.group(1)
            eq_node = f"eq:{chunk.paper_id}:{eq_id}"
            if not self.g.has_node(eq_node):
                self.g.add_node(eq_node, node_type="equation", paper_id=chunk.paper_id, eq_id=eq_id)
            self.g.add_edge(chunk_node_id, eq_node, relation="refers_to")

        # 2. Figure cross-references
        for match in _FIG_XREF_RX.finditer(text):
            fig_id = match.group(1)
            fig_node = f"fig:{chunk.paper_id}:{fig_id}"
            if not self.g.has_node(fig_node):
                self.g.add_node(fig_node, node_type="figure", paper_id=chunk.paper_id, fig_id=fig_id)
            self.g.add_edge(chunk_node_id, fig_node, relation="refers_to")

        # 3. Table cross-references
        for match in _TAB_XREF_RX.finditer(text):
            tab_id = match.group(1)
            tab_node = f"tab:{chunk.paper_id}:{tab_id}"
            if not self.g.has_node(tab_node):
                self.g.add_node(tab_node, node_type="table", paper_id=chunk.paper_id, tab_id=tab_id)
            self.g.add_edge(chunk_node_id, tab_node, relation="refers_to")

    # ── Traversal Queries ──────────────────────────────────────────────

    def get_chunks_referencing_equation(self, paper_id: str, eq_id: str) -> list[str]:
        """Find all chunk IDs that refer to a given equation in a paper."""
        eq_node = f"eq:{paper_id}:{eq_id}"
        if not self.g.has_node(eq_node):
            return []
        predecessors = self.g.predecessors(eq_node)
        chunk_ids = []
        for p in predecessors:
            if self.g.nodes[p].get("node_type") == "chunk":
                chunk_ids.append(self.g.nodes[p]["chunk_id"])
        return chunk_ids

    def get_chunks_referencing_figure(self, paper_id: str, fig_id: str) -> list[str]:
        """Find all chunk IDs that refer to a given figure in a paper."""
        fig_node = f"fig:{paper_id}:{fig_id}"
        if not self.g.has_node(fig_node):
            return []
        predecessors = self.g.predecessors(fig_node)
        chunk_ids = []
        for p in predecessors:
            if self.g.nodes[p].get("node_type") == "chunk":
                chunk_ids.append(self.g.nodes[p]["chunk_id"])
        return chunk_ids

    def get_section_chunks(self, paper_id: str, section_path: str) -> list[str]:
        """Get all chunk IDs contained within a specific section."""
        sec_node = f"sec:{paper_id}:{section_path}"
        if not self.g.has_node(sec_node):
            return []
        successors = self.g.successors(sec_node)
        chunk_ids = []
        for s in successors:
            if self.g.nodes[s].get("node_type") == "chunk":
                chunk_ids.append(self.g.nodes[s]["chunk_id"])
        return chunk_ids

    def get_citing_papers(self, paper_id: str) -> list[str]:
        """Get all papers that cite this paper."""
        target = f"paper:{paper_id}"
        if not self.g.has_node(target):
            return []
        predecessors = self.g.predecessors(target)
        return [self.g.nodes[p]["paper_id"] for p in predecessors if self.g.nodes[p].get("node_type") == "paper"]

    def get_cited_papers(self, paper_id: str) -> list[str]:
        """Get all papers cited by this paper."""
        source = f"paper:{paper_id}"
        if not self.g.has_node(source):
            return []
        successors = self.g.successors(source)
        return [self.g.nodes[s]["paper_id"] for s in successors if self.g.nodes[s].get("node_type") == "paper"]

    # ── Persistence ────────────────────────────────────────────────────

    def save_to_file(self, path: str | Path) -> None:
        """Save graph to JSON node-link format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = nx.node_link_data(self.g, edges="edges")
        except TypeError:
            data = nx.node_link_data(self.g)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_from_file(self, path: str | Path) -> None:
        """Load graph from JSON node-link format."""
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            self.g = nx.node_link_graph(data, edges="edges")
        except TypeError:
            self.g = nx.node_link_graph(data)
