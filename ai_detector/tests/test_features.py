"""
Tests for feature extraction module.
"""

import pytest
import numpy as np
from src.ai_detector.features import (
    FeatureExtractor, FeatureConfig, StatisticalFeatures,
    LinguisticFeatures, AllFeatures
)


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = FeatureConfig(
            use_perplexity=True,
            use_burstiness=True,
            use_vocabulary_diversity=True,
            use_ngram_repetition=True,
            use_neural_features=False  # Skip neural for faster tests
        )
        self.extractor = FeatureExtractor(self.config)
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        text = "This is a test. This is only a test. Testing testing one two three."
        sentences = ["This is a test.", "This is only a test.", "Testing testing one two three."]
        tokens = text.lower().split()
        
        features = self.extractor.extract_statistical_features(text, sentences, tokens)
        
        assert isinstance(features, StatisticalFeatures)
        assert features.type_token_ratio > 0
        assert features.type_token_ratio <= 1
    
    def test_linguistic_features(self):
        """Test linguistic feature extraction."""
        text = "The quick brown fox jumps over the lazy dog. She runs fast."
        
        features = self.extractor.extract_linguistic_features(text)
        
        assert isinstance(features, LinguisticFeatures)
    
    def test_all_features(self):
        """Test complete feature extraction."""
        text = "Natural language processing is fascinating. Machine learning enables computers to understand text."
        sentences = [
            "Natural language processing is fascinating.",
            "Machine learning enables computers to understand text."
        ]
        tokens = text.lower().split()
        
        features = self.extractor.extract_all_features(text, sentences, tokens)
        
        assert isinstance(features, AllFeatures)
        assert features.statistical is not None
        assert features.linguistic is not None
    
    def test_feature_vector(self):
        """Test conversion to feature vector."""
        text = "Sample text for feature extraction testing purposes."
        sentences = ["Sample text for feature extraction testing purposes."]
        tokens = text.lower().split()
        
        features = self.extractor.extract_all_features(text, sentences, tokens)
        vector = features.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector.shape) == 1
        assert vector.shape[0] > 0
    
    def test_empty_text(self):
        """Test handling of empty text."""
        features = self.extractor.extract_all_features("", [], [])
        vector = features.to_vector()
        
        assert isinstance(vector, np.ndarray)
        # Should contain NaN or zeros for missing features
    
    def test_short_text(self):
        """Test handling of very short text."""
        text = "Hi."
        sentences = ["Hi."]
        tokens = ["hi"]
        
        features = self.extractor.extract_all_features(text, sentences, tokens)
        vector = features.to_vector()
        
        assert isinstance(vector, np.ndarray)
    
    def test_vocabulary_metrics(self):
        """Test vocabulary diversity metrics."""
        # Text with low vocabulary diversity (repetitive)
        repetitive = "cat cat cat dog dog cat cat dog"
        # Text with high vocabulary diversity
        diverse = "The elephant wandered through the magnificent sprawling jungle yesterday"
        
        rep_tokens = repetitive.split()
        div_tokens = diverse.split()
        
        ttr_rep, _, _ = self.extractor._calculate_vocabulary_metrics(rep_tokens)
        ttr_div, _, _ = self.extractor._calculate_vocabulary_metrics(div_tokens)
        
        # Diverse text should have higher TTR
        assert ttr_div > ttr_rep
    
    def test_ngram_repetition(self):
        """Test n-gram repetition detection."""
        # Text with repeated bigrams
        repeated = "the cat and the dog and the bird and the fish"
        tokens = repeated.split()
        
        bigram_rep, trigram_rep = self.extractor._calculate_ngram_repetition(tokens)
        
        assert bigram_rep > 0  # "and the" is repeated
    
    def test_burstiness(self):
        """Test burstiness calculation."""
        # Uniform sentence lengths
        uniform = "Cat. Dog. Bird. Fish. Frog."
        uniform_sents = ["Cat.", "Dog.", "Bird.", "Fish.", "Frog."]
        
        # Varied sentence lengths
        varied = "Word. A much longer sentence with many more words in it. Medium."
        varied_sents = ["Word.", "A much longer sentence with many more words in it.", "Medium."]
        
        burst_uniform = self.extractor._calculate_burstiness(uniform_sents)
        burst_varied = self.extractor._calculate_burstiness(varied_sents)
        
        # Varied should have higher burstiness
        if not np.isnan(burst_uniform) and not np.isnan(burst_varied):
            assert burst_varied > burst_uniform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
