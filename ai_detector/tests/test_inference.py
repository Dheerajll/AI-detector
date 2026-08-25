"""
Tests for inference module.
"""

import pytest
import numpy as np
from src.ai_detector.inference import AIDetector, DetectorConfig, PredictionResult


class TestInference:
    """Test inference functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = DetectorConfig()
        self.detector = AIDetector(self.config)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.detector.predict("")
        
        assert result.classification == "uncertain"
        assert len(result.warnings) > 0
    
    def test_short_input(self):
        """Test handling of very short input."""
        result = self.detector.predict("Hi there.")
        
        assert result.num_tokens < 10
        assert any("tokens" in w.lower() for w in result.warnings)
    
    def test_normal_input(self):
        """Test prediction on normal-length text."""
        text = """
        Artificial intelligence has transformed how we interact with technology.
        Machine learning algorithms can now recognize patterns in data that were
        previously invisible to traditional computing methods. This revolution
        continues to accelerate as researchers develop more sophisticated models.
        """
        
        result = self.detector.predict(text)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.ai_probability <= 1
        assert 0 <= result.human_probability <= 1
        assert result.confidence >= 0
        assert result.confidence <= 1
    
    def test_long_input(self):
        """Test handling of long documents."""
        # Create a long document
        paragraphs = [
            "This is paragraph number {}. ".format(i) + 
            "It contains multiple sentences to add length. "
            "The purpose is to test chunking behavior."
            for i in range(50)
        ]
        text = " ".join(paragraphs)
        
        result = self.detector.predict(text)
        
        assert result.num_chunks >= 1
        assert result.num_tokens > 500
    
    def test_unicode_input(self):
        """Test handling of Unicode text."""
        text = "Café résumé naïve coöperate 日本語 Ελληνικά"
        result = self.detector.predict(text)
        
        assert isinstance(result, PredictionResult)
    
    def test_malformed_input(self):
        """Test handling of malformed input."""
        # Very strange input
        text = "@#$%^&*() !!! ??? 123456789"
        result = self.detector.predict(text)
        
        # Should not crash, should return uncertain or warn
        assert isinstance(result, PredictionResult)
    
    def test_prediction_result_structure(self):
        """Test that prediction result has required fields."""
        text = "This is a test sentence for checking the result structure."
        result = self.detector.predict(text)
        
        # Check required fields exist
        assert hasattr(result, 'classification')
        assert hasattr(result, 'ai_probability')
        assert hasattr(result, 'human_probability')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reliability')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'evidence')
        assert hasattr(result, 'num_tokens')
        assert hasattr(result, 'num_chunks')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        text = "Sample text for testing."
        result = self.detector.predict(text)
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'classification' in result_dict
        assert 'ai_probability' in result_dict
        assert 'confidence' in result_dict
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        results = self.detector.predict_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)
    
    def test_classification_values(self):
        """Test that classification takes valid values."""
        text = "This is some sample text for classification testing."
        result = self.detector.predict(text)
        
        valid_classifications = ["likely_ai", "likely_human", "uncertain"]
        assert result.classification in valid_classifications
    
    def test_reliability_values(self):
        """Test that reliability takes valid values."""
        text = "This is some sample text for reliability testing."
        result = self.detector.predict(text)
        
        valid_reliabilities = ["high", "medium", "low"]
        assert result.reliability in valid_reliabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
