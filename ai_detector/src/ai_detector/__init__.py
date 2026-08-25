"""
AI Text Detector - A robust framework for AI-generated text detection.

This package provides tools for detecting AI-generated text using a hybrid
approach combining statistical features, linguistic analysis, and neural
representations.

Important Notes:
- This is a probabilistic classifier, not a definitive authorship tool
- False positives are possible, especially on short or unusual text
- Results should be interpreted with caution and uncertainty estimates
- The detector requires training on appropriate datasets

PDF Analysis Feature:
- Supports analysis of PDF documents (theses, reports, academic papers)
- Extracts structural features (sections, headings, citations)
- Analyzes document flow and coherence patterns
- Detects section-level AI probability variations
"""

__version__ = "0.2.0"
__author__ = "AI Detector Team"

from .inference import AIDetector
from .preprocessing import TextPreprocessor
from .features import FeatureExtractor

# PDF analysis feature (optional imports)
try:
    from .pdf_analyzer import PDFAnalyzer, analyze_pdf, PDFAnalysisResult
    PDF_SUPPORT_AVAILABLE = True
except ImportError:
    PDF_SUPPORT_AVAILABLE = False

__all__ = [
    "AIDetector", 
    "TextPreprocessor", 
    "FeatureExtractor",
    "PDFAnalyzer",
    "analyze_pdf",
    "PDFAnalysisResult",
    "PDF_SUPPORT_AVAILABLE"
]
