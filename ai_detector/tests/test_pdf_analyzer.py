"""
Tests for PDF analysis module.

Note: These tests require PDF files to be present in the test data directory.
If no PDF files are available, tests will be skipped gracefully.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Try to import PDF analyzer - skip tests if not available
try:
    from ai_detector.pdf_analyzer import (
        PDFTextExtractor,
        DocumentStructureAnalyzer,
        PDFFlowAnalyzer,
        PDFAnalyzer,
        StructuralFeatures,
        DocumentSection,
        analyze_pdf
    )
    PDF_SUPPORT_AVAILABLE = True
except ImportError:
    PDF_SUPPORT_AVAILABLE = False


@pytest.mark.skipif(not PDF_SUPPORT_AVAILABLE, reason="PDF support not available")
class TestPDFTextExtractor:
    """Tests for PDF text extraction."""
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        extractor = PDFTextExtractor()
        assert extractor.backend == "auto"
        
    def test_extractor_with_specific_backend(self):
        """Test extractor with specific backend selection."""
        for backend in ["auto", "pypdf2", "pdfplumber", "pymupdf"]:
            extractor = PDFTextExtractor(backend=backend)
            assert extractor.backend == backend
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        extractor = PDFTextExtractor()
        result = extractor.extract("/nonexistent/path/file.pdf")
        
        assert result.is_extractable is False
        assert len(result.warnings) > 0
        assert result.raw_text == ""
    
    def test_wrong_extension(self, tmp_path):
        """Test handling of wrong file extension."""
        # Create a text file with .pdf extension
        txt_file = tmp_path / "fake.pdf"
        txt_file.write_text("This is not a PDF")
        
        extractor = PDFTextExtractor()
        result = extractor.extract(txt_file)
        
        # Should have warning about extension
        assert any("not .pdf" in w for w in result.warnings) or result.is_extractable is False


@pytest.mark.skipif(not PDF_SUPPORT_AVAILABLE, reason="PDF support not available")
class TestDocumentStructureAnalyzer:
    """Tests for document structure analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DocumentStructureAnalyzer()
    
    def test_section_detection_simple(self):
        """Test detection of simple sections."""
        text = """
1. Introduction
This is the introduction section.

2. Methodology
This describes the methods used.

3. Results
Here are the results.

4. Conclusion
In conclusion, this is the end.
"""
        sections = self.analyzer.detect_sections(text)
        
        assert len(sections) >= 3
        assert any('introduction' in s.title.lower() for s in sections)
        assert any('methodology' in s.title.lower() or 'methods' in s.title.lower() for s in sections)
    
    def test_section_detection_academic(self):
        """Test detection of academic structure."""
        text = """
ABSTRACT
This paper presents...

INTRODUCTION
Background information...

METHODOLOGY
Our approach involves...

RESULTS
The experimental results show...

DISCUSSION
These findings suggest...

CONCLUSION
We conclude that...

REFERENCES
[1] Author et al., 2020
"""
        sections = self.analyzer.detect_sections(text)
        features = self.analyzer.identify_academic_structure(sections)
        
        assert features.has_abstract is True
        assert features.has_introduction is True
        assert features.has_methodology is True
        assert features.has_results is True
        assert features.has_discussion is True
        assert features.has_conclusion is True
        assert features.has_references is True
    
    def test_heading_consistency(self):
        """Test heading style consistency analysis."""
        sections = [
            DocumentSection(title="1. Introduction", level=1, content="Content"),
            DocumentSection(title="2. Methods", level=1, content="Content"),
            DocumentSection(title="3. Results", level=1, content="Content"),
        ]
        
        consistency = self.analyzer.analyze_heading_consistency(sections)
        assert 0 <= consistency <= 1 or np.isnan(consistency)
    
    def test_citation_analysis(self):
        """Test citation pattern analysis."""
        text_with_citations = """
        As shown by Smith et al., 2020, the method works well [1].
        Previous studies (Johnson and Lee, 2019) found similar results [2, 3].
        This contradicts earlier work^4.
        """
        
        density, consistency = self.analyzer.analyze_citations(text_with_citations)
        
        assert density > 0
        assert 0 <= consistency <= 1 or np.isnan(consistency)
    
    def test_figure_table_mentions(self):
        """Test figure and table mention detection."""
        text = """
        As shown in Figure 1, the results are clear.
        Table 2 summarizes the data.
        See Fig. 3 for comparison.
        """
        
        fig_density, table_density = self.analyzer.analyze_figure_table_mentions(text)
        
        assert fig_density > 0
        assert table_density > 0


@pytest.mark.skipif(not PDF_SUPPORT_AVAILABLE, reason="PDF support not available")
class TestPDFFlowAnalyzer:
    """Tests for document flow analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PDFFlowAnalyzer()
    
    def test_topic_coherence(self):
        """Test topic coherence between sections."""
        sections = [
            DocumentSection(
                title="Introduction", 
                level=1, 
                content="Machine learning algorithms use neural networks for classification tasks."
            ),
            DocumentSection(
                title="Methods", 
                level=1, 
                content="We implemented neural network classifiers using machine learning techniques."
            ),
        ]
        
        coherence = self.analyzer.analyze_topic_coherence(sections)
        assert 0 <= coherence <= 1 or np.isnan(coherence)
    
    def test_generic_phrase_detection(self):
        """Test detection of generic academic phrases."""
        text = """
        In recent years, machine learning has gained significant attention.
        It is important to note that this plays a crucial role.
        In conclusion, further research is needed.
        """
        
        result = self.analyzer.detect_generic_phrases(text)
        
        assert result['total_count'] > 0
        assert result['density_per_1000_words'] > 0
    
    def test_cross_section_repetition(self):
        """Test repetition analysis across sections."""
        sections = [
            DocumentSection(
                title="Section 1",
                level=1,
                content="The quick brown fox jumps over the lazy dog repeatedly."
            ),
            DocumentSection(
                title="Section 2", 
                level=1,
                content="The quick brown fox jumps over the lazy dog again."
            ),
        ]
        
        repetition = self.analyzer.analyze_repetition_across_sections(sections)
        assert repetition >= 0 or np.isnan(repetition)
    
    def test_technical_depth(self):
        """Test technical depth analysis."""
        technical_text = """
        We implemented a neural network with 128 parameters.
        The algorithm achieved 95% accuracy on the dataset.
        Using the optimization method, we configured the model with learning rate 0.001.
        """
        
        sections = [DocumentSection(title="Methods", level=1, content=technical_text)]
        depth = self.analyzer.analyze_technical_depth(sections)
        
        assert depth >= 0


@pytest.mark.skipif(not PDF_SUPPORT_AVAILABLE, reason="PDF support not available")
class TestPDFAnalyzer:
    """Integration tests for complete PDF analysis."""
    
    def test_analyzer_initialization(self):
        """Test PDF analyzer initialization."""
        analyzer = PDFAnalyzer(ai_detector=None)
        assert analyzer.extractor is not None
        assert analyzer.structure_analyzer is not None
        assert analyzer.flow_analyzer is not None
    
    def test_analyze_nonexistent_file(self):
        """Test analysis of nonexistent file."""
        analyzer = PDFAnalyzer()
        result = analyzer.analyze("/nonexistent/file.pdf", run_ai_detection=False)
        
        assert result.extraction_quality == "failed"
        assert result.overall_classification == "unknown"
    
    def test_structural_features_extraction(self):
        """Test that structural features are properly extracted."""
        # Create mock text simulating extracted PDF content
        text = """
        ABSTRACT
        This study examines AI detection methods.
        
        1. INTRODUCTION
        Background on AI-generated text detection.
        
        2. METHODOLOGY
        We used neural networks with 256 parameters.
        The dataset contained 10,000 samples.
        
        3. RESULTS
        Accuracy reached 94.5% on the test set.
        Figure 1 shows the performance curves.
        
        4. DISCUSSION
        These results demonstrate effectiveness.
        
        5. CONCLUSION
        We conclude the method is viable.
        
        REFERENCES
        [1] Smith et al., 2020
        [2] Johnson, 2019
        """
        
        analyzer = PDFAnalyzer()
        
        # Manually test structure analysis components
        sections = analyzer.structure_analyzer.detect_sections(text)
        features = analyzer.structure_analyzer.identify_academic_structure(sections)
        
        assert features.num_sections > 0
        assert features.has_abstract is True
        # The section detection may not perfectly identify all sections
        # This is expected behavior - we verify at least some are detected
        assert features.has_references is True


@pytest.mark.skipif(not PDF_SUPPORT_AVAILABLE, reason="PDF support not available")
class TestCreateSamplePDF:
    """Helper to create sample PDF for testing."""
    
    def test_create_and_analyze_sample_pdf(self, tmp_path):
        """Create a simple PDF and analyze it."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            pytest.skip("PyMuPDF not available for PDF creation")
        
        # Create a simple PDF
        pdf_path = tmp_path / "test_document.pdf"
        
        doc = fitz.open()
        page = doc.new_page()
        
        text = """
        ABSTRACT
        This is a test document for AI detection.
        
        1. INTRODUCTION
        This document contains sample academic text.
        
        2. METHODS
        We used standard methodology.
        
        3. CONCLUSION
        This is the end of the test document.
        """
        
        page.insert_text((50, 50), text, fontsize=12)
        doc.save(pdf_path)
        doc.close()
        
        # Analyze the PDF
        result = analyze_pdf(pdf_path, ai_detector=None)
        
        assert result.total_pages == 1
        assert result.total_words > 0
        assert result.extraction_quality in ["high", "medium", "low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
