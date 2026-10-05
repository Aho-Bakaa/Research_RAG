"""Parsers package for layout-aware document extraction.
"""
from bl_pipeline.rag.parsers.layout_parser import LayoutAwareParser, parse_pdf_to_elements
from bl_pipeline.rag.parsers.router import ScientificDocumentRouter
from bl_pipeline.rag.parsers.triage import DocumentTriageEngine
from bl_pipeline.rag.parsers.cell_dissector import TableDissector
from bl_pipeline.rag.parsers.formula_reconciler import FormulaReconciler
from bl_pipeline.rag.parsers.figure_analyzer import FigureAnalyzer
from bl_pipeline.rag.parsers.quality_gates import ParsingQualityGatePipeline

__all__ = [
    "ScientificDocumentRouter",
    "LayoutAwareParser",
    "DocumentTriageEngine",
    "TableDissector",
    "FormulaReconciler",
    "FigureAnalyzer",
    "ParsingQualityGatePipeline",
    "parse_pdf_to_elements",
]
