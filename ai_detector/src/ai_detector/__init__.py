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
"""

__version__ = "0.1.0"
__author__ = "AI Detector Team"

from .inference import AIDetector
from .preprocessing import TextPreprocessor
from .features import FeatureExtractor

__all__ = ["AIDetector", "TextPreprocessor", "FeatureExtractor"]
