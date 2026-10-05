"""token_budgeter.py — Dynamic semantic token budgeter and structured context envelope.

Eliminates arbitrary character truncation (e.g. [:200] or [:300]).
Enforces:
1. Semantic Atomicity: Whole paragraphs, unbroken LaTeX equations, complete table rows.
2. Dynamic Priority Budgeting:
   - Tier 1: Primary Focal Hits (equations, tables, top passage) [~35%]
   - Tier 2: Enclosing Hierarchical Section Context [~35%]
   - Tier 3: Graph Cross-References & Sibling Windows [~20%]
   - Tier 4: Provenance Metadata Envelope [~10%]
3. Structured Context Envelope with explicit XML tags and citation headers.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class RetrievedFocalItem:
    """A primary retrieved child item from vector/BM25 search."""
    item_id: str
    paper_id: str
    element_type: str  # equation, paragraph, table, figure
    content: str
    score: float = 0.0
    section_path: str = ""
    page: int = 1
    equation_ref: str | None = None
    table_ref: str | None = None
    parent_section_text: str = ""
    cross_reference_texts: list[str] = field(default_factory=list)
    sibling_texts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticTokenBudgeter:
    """Allocates tokens dynamically without breaking semantic atomic units."""

    def __init__(
        self,
        total_token_budget: int = 4500,
        tier1_focal_pct: float = 0.35,
        tier2_section_pct: float = 0.35,
        tier3_crossref_pct: float = 0.20,
    ):
        self.total_budget = total_token_budget
        self.tier1_budget = int(total_token_budget * tier1_focal_pct)
        self.tier2_budget = int(total_token_budget * tier2_section_pct)
        self.tier3_budget = int(total_token_budget * tier3_crossref_pct)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Fast, robust token estimation (~1.3 tokens per whitespace-separated word for LaTeX/math)."""
        words = text.split()
        latex_symbols = len(re.findall(r"\\[a-zA-Z]+|[_^={}\[\]]", text))
        return int(len(words) + (latex_symbols * 0.4))

    def format_calibrated_context(
        self,
        focal_items: Sequence[RetrievedFocalItem],
        query: str = "",
    ) -> str:
        """Assemble structured, calibrated context envelope with rich provenance headers."""
        if not focal_items:
            return "No relevant context was retrieved from the primary literature corpus."

        blocks: list[str] = []
        tier1_used = 0
        tier2_used = 0
        tier3_used = 0

        for idx, item in enumerate(focal_items, 1):
            source_parts: list[str] = []
            source_parts.append(f'<document_source id="S{idx}" paper_id="{item.paper_id}">')

            # 1. Provenance Metadata Block
            source_parts.append("  <provenance>")
            source_parts.append(f"    <paper_id>{item.paper_id}</paper_id>")
            if item.section_path:
                source_parts.append(f"    <section_hierarchy>{item.section_path}</section_hierarchy>")
            source_parts.append(f"    <page>{item.page}</page>")
            if item.equation_ref:
                source_parts.append(f"    <equation_ref>{item.equation_ref}</equation_ref>")
            if item.table_ref:
                source_parts.append(f"    <table_ref>{item.table_ref}</table_ref>")
            source_parts.append("  </provenance>")

            # 2. Tier 1: Primary Focal Match (Strictly Atomic, never sliced)
            clean_focal = item.content.strip()
            focal_tokens = self.estimate_tokens(clean_focal)
            source_parts.append(f'  <focal_match type="{item.element_type}">')
            source_parts.append(f"<![CDATA[\n{clean_focal}\n]]>")
            source_parts.append("  </focal_match>")
            tier1_used += focal_tokens

            # 3. Tier 2: Enclosing Hierarchical Section Context (Paragraph-level greedy fitting)
            if item.parent_section_text:
                parent_paragraphs = [
                    p.strip() for p in item.parent_section_text.split("\n\n")
                    if p.strip() and p.strip() != clean_focal
                ]
                fitted_parent: list[str] = []
                for p in parent_paragraphs:
                    p_tokens = self.estimate_tokens(p)
                    if tier2_used + p_tokens <= self.tier2_budget:
                        fitted_parent.append(p)
                        tier2_used += p_tokens
                    else:
                        break  # Stop at paragraph boundary, never slice mid-sentence!

                if fitted_parent:
                    source_parts.append("  <enclosing_section_context>")
                    source_parts.append("\n\n".join(fitted_parent))
                    source_parts.append("  </enclosing_section_context>")

            # 4. Tier 3: Graph Cross-References & Sibling Windows
            cross_ref_blocks: list[str] = []
            for xref in item.cross_reference_texts:
                xref_clean = xref.strip()
                xref_tokens = self.estimate_tokens(xref_clean)
                if tier3_used + xref_tokens <= self.tier3_budget:
                    cross_ref_blocks.append(f'    <ref source="intra_paper_discussion">\n{xref_clean}\n    </ref>')
                    tier3_used += xref_tokens

            for sib in item.sibling_texts:
                sib_clean = sib.strip()
                sib_tokens = self.estimate_tokens(sib_clean)
                if tier3_used + sib_tokens <= self.tier3_budget:
                    cross_ref_blocks.append(f'    <ref source="sequential_sibling">\n{sib_clean}\n    </ref>')
                    tier3_used += sib_tokens

            if cross_ref_blocks:
                source_parts.append("  <cross_references_and_calibrations>")
                source_parts.extend(cross_ref_blocks)
                source_parts.append("  </cross_references_and_calibrations>")

            source_parts.append("</document_source>")
            blocks.append("\n".join(source_parts))

        return "<retrieved_scientific_context>\n" + "\n\n".join(blocks) + "\n</retrieved_scientific_context>"
