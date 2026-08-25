"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
from src.ai_detector.preprocessing import TextPreprocessor, PreprocessingConfig


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PreprocessingConfig(
            min_tokens=20,
            max_tokens=1000,
            chunk_size=100
        )
        self.preprocessor = TextPreprocessor(self.config)
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.preprocessor.preprocess("")
        assert result.is_valid == False
        assert len(result.warnings) > 0
        assert "Empty" in result.warnings[0]
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # Test with various Unicode forms
        text = "Café\u0301"  # e + combining acute accent
        result = self.preprocessor.preprocess(text)
        assert result.cleaned_text is not None
    
    def test_whitespace_handling(self):
        """Test whitespace normalization."""
        text = "Hello    world\n\n\ntest"
        result = self.preprocessor.preprocess(text)
        assert "  " not in result.cleaned_text
    
    def test_sentence_segmentation(self):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        result = self.preprocessor.preprocess(text)
        assert result.num_sentences >= 2
    
    def test_token_counting(self):
        """Test token estimation."""
        text = "This is a simple test with ten words exactly here now."
        result = self.preprocessor.preprocess(text)
        assert result.num_tokens > 0
    
    def test_short_text_warning(self):
        """Test warning for short texts."""
        text = "Short text."
        result = self.preprocessor.preprocess(text)
        assert any("tokens" in w.lower() for w in result.warnings)
    
    def test_chunking_long_text(self):
        """Test chunking of long documents."""
        # Create text longer than max_tokens
        words = ["word"] * 1500
        text = " ".join(words)
        result = self.preprocessor.preprocess(text)
        assert len(result.chunks) > 1
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        texts = ["First text.", "Second text.", "Third text."]
        results = self.preprocessor.preprocess_batch(texts)
        assert len(results) == 3
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Special chars: @#$%^&*() and unicode: 日本語"
        result = self.preprocessor.preprocess(text)
        assert result.is_valid == True or len(result.warnings) > 0
    
    def test_malformed_input(self):
        """Test handling of malformed input."""
        with pytest.raises(ValueError):
            self.preprocessor.preprocess(None)
        
        with pytest.raises(ValueError):
            self.preprocessor.preprocess(123)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
