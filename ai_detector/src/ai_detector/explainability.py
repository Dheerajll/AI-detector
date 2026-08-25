"""
Explainability module for AI text detection.

Provides interpretable explanations for model predictions:
- Feature importance analysis
- Evidence summarization
- Chunk-level attribution

Important: Explanations are heuristic indicators, NOT proof of authorship.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureEvidence:
    """Single piece of evidence from a feature."""
    feature_name: str
    direction: str  # "AI-like", "human-like", or "neutral"
    strength: float  # 0-1, how strongly this feature indicates the direction
    value: float = np.nan  # Actual feature value
    reference_range: Tuple[float, float] = (np.nan, np.nan)  # Typical range in training data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature_name,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "value": round(self.value, 4) if self.value == self.value else None,
            "reference_range": [
                round(self.reference_range[0], 4) if self.reference_range[0] == self.reference_range[0] else None,
                round(self.reference_range[1], 4) if self.reference_range[1] == self.reference_range[1] else None
            ]
        }


@dataclass
class ExplanationResult:
    """Complete explanation for a prediction."""
    # Top evidence items
    evidence: List[FeatureEvidence] = field(default_factory=list)
    
    # Summary statistics
    ai_indicating_features: int = 0
    human_indicating_features: int = 0
    neutral_features: int = 0
    
    # Overall interpretation
    summary: str = ""
    
    # Chunk-level explanations (for long documents)
    chunk_explanations: List['ExplanationResult'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "evidence": [e.to_dict() for e in self.evidence],
            "ai_indicating_features": self.ai_indicating_features,
            "human_indicating_features": self.human_indicating_features,
            "neutral_features": self.neutral_features,
            "summary": self.summary
        }
        
        if self.chunk_explanations:
            result["chunk_explanations"] = [c.to_dict() for c in self.chunk_explanations]
            
        return result


# Feature descriptions for user-friendly explanations
FEATURE_DESCRIPTIONS = {
    "mean_perplexity": "Average token predictability (lower = more predictable)",
    "perplexity_variance": "Variation in token predictability",
    "burstiness": "Sentence length variation (higher = more varied)",
    "type_token_ratio": "Vocabulary diversity (higher = more diverse)",
    "hapax_legomena_ratio": "Ratio of words used only once",
    "bigram_repetition_rate": "Repeated two-word phrases",
    "trigram_repetition_rate": "Repeated three-word phrases",
    "mean_sentence_length": "Average sentence length in words",
    "sentence_length_std": "Sentence length consistency",
    "punctuation_ratio": "Proportion of punctuation characters",
    "comma_to_period_ratio": "Comma usage relative to periods",
    "function_word_ratio": "Proportion of common function words",
    "pos_noun_ratio": "Proportion of nouns",
    "pos_verb_ratio": "Proportion of verbs",
    "pos_adj_ratio": "Proportion of adjectives",
    "pos_adv_ratio": "Proportion of adverbs",
    "clauses_per_sentence": "Average clauses per sentence",
    "discourse_marker_frequency": "Use of transition words/phrases",
    "passive_voice_ratio": "Proportion of passive voice constructions"
}

# Reference ranges from typical training data (approximate)
REFERENCE_RANGES = {
    "mean_perplexity": (5.0, 15.0),
    "burstiness": (0.3, 0.8),
    "type_token_ratio": (0.4, 0.7),
    "bigram_repetition_rate": (0.0, 0.15),
    "trigram_repetition_rate": (0.0, 0.05),
    "mean_sentence_length": (12.0, 25.0),
    "function_word_ratio": (0.3, 0.5),
    "pos_noun_ratio": (0.15, 0.25),
    "pos_verb_ratio": (0.1, 0.2),
    "clauses_per_sentence": (1.2, 2.5)
}

# Which direction is AI-indicating for each feature
AI_INDICATING_DIRECTION = {
    "mean_perplexity": "low",  # AI text tends to have lower perplexity
    "burstiness": "low",  # AI text often has more uniform sentence lengths
    "type_token_ratio": "low",  # AI may use less diverse vocabulary
    "bigram_repetition_rate": "high",  # AI may repeat phrases
    "trigram_repetition_rate": "high",
    "function_word_ratio": "high",  # AI often uses more function words
    "pos_noun_ratio": "high",  # AI may use more nouns
    "passive_voice_ratio": "low",  # AI often uses less passive voice
}


class ExplainabilityEngine:
    """
    Generates explanations for AI detection predictions.
    
    Provides:
    - Feature-level attribution
    - Human-readable summaries
    - Chunk-level analysis for long documents
    """
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.reference_stats = {}
    
    def set_reference_statistics(self, stats: Dict[str, Dict[str, float]]):
        """
        Set reference statistics from training data.
        
        Args:
            stats: Dictionary mapping feature names to {mean, std, min, max}
        """
        self.reference_stats = stats
        logger.info(f"Set reference statistics for {len(stats)} features")
    
    def _get_direction(self, feature_name: str, value: float) -> str:
        """Determine if feature value is AI-like, human-like, or neutral."""
        if feature_name not in REFERENCE_RANGES:
            return "neutral"
        
        low, high = REFERENCE_RANGES[feature_name]
        mid = (low + high) / 2
        
        ai_direction = AI_INDICATING_DIRECTION.get(feature_name, None)
        
        if ai_direction is None:
            return "neutral"
        
        if ai_direction == "low":
            if value < low:
                return "AI-like"
            elif value > high:
                return "human-like"
            else:
                return "neutral"
        elif ai_direction == "high":
            if value > high:
                return "AI-like"
            elif value < low:
                return "human-like"
            else:
                return "neutral"
        
        return "neutral"
    
    def _calculate_strength(self, feature_name: str, value: float) -> float:
        """Calculate how strongly a feature indicates its direction."""
        if feature_name not in REFERENCE_RANGES:
            return 0.5
        
        low, high = REFERENCE_RANGES[feature_name]
        range_size = high - low
        
        if range_size == 0:
            return 0.5
        
        # Distance from middle of range
        mid = (low + high) / 2
        distance_from_mid = abs(value - mid)
        
        # Normalize to 0-1 (cap at 2x range)
        strength = min(distance_from_mid / (range_size * 2), 1.0)
        
        return strength
    
    def analyze_features(self, feature_vector: np.ndarray,
                        feature_names: List[str]) -> ExplanationResult:
        """
        Analyze feature vector and generate evidence.
        
        Args:
            feature_vector: Extracted features
            feature_names: Names corresponding to each feature
            
        Returns:
            ExplanationResult with evidence items
        """
        explanation = ExplanationResult()
        all_evidence = []
        
        for i, (value, name) in enumerate(zip(feature_vector, feature_names)):
            if np.isnan(value):
                continue
            
            direction = self._get_direction(name, value)
            strength = self._calculate_strength(name, value)
            
            ref_range = REFERENCE_RANGES.get(name, (np.nan, np.nan))
            
            evidence = FeatureEvidence(
                feature_name=name,
                direction=direction,
                strength=strength,
                value=value,
                reference_range=ref_range
            )
            
            all_evidence.append(evidence)
            
            if direction == "AI-like":
                explanation.ai_indicating_features += 1
            elif direction == "human-like":
                explanation.human_indicating_features += 1
            else:
                explanation.neutral_features += 1
        
        # Sort by strength and take top K
        sorted_evidence = sorted(all_evidence, key=lambda e: e.strength, reverse=True)
        explanation.evidence = sorted_evidence[:self.top_k]
        
        # Generate summary
        explanation.summary = self._generate_summary(explanation)
        
        return explanation
    
    def _generate_summary(self, explanation: ExplanationResult) -> str:
        """Generate human-readable summary of evidence."""
        ai_count = explanation.ai_indicating_features
        human_count = explanation.human_indicating_features
        
        if ai_count > human_count * 2:
            summary = "Multiple features show patterns commonly associated with AI-generated text."
        elif human_count > ai_count * 2:
            summary = "Multiple features show patterns commonly associated with human-written text."
        elif ai_count > human_count:
            summary = "More features lean toward AI-like patterns, but evidence is mixed."
        elif human_count > ai_count:
            summary = "More features lean toward human-like patterns, but evidence is mixed."
        else:
            summary = "Features show mixed signals; classification uncertainty is expected."
        
        return summary
    
    def create_chunk_explanations(self, texts: List[str],
                                  all_features: List[np.ndarray],
                                  feature_names: List[str]) -> List[ExplanationResult]:
        """Generate explanations for each chunk of a long document."""
        explanations = []
        
        for features in all_features:
            chunk_exp = self.analyze_features(features, feature_names)
            explanations.append(chunk_exp)
        
        return explanations
    
    def explain_prediction(self, feature_vector: np.ndarray,
                          feature_names: List[str],
                          ai_probability: float,
                          chunks: Optional[List[str]] = None,
                          chunk_features: Optional[List[np.ndarray]] = None) -> ExplanationResult:
        """
        Generate complete explanation for a prediction.
        
        Args:
            feature_vector: Main feature vector
            feature_names: Feature names
            ai_probability: Predicted probability of AI
            chunks: Optional text chunks for long documents
            chunk_features: Optional features for each chunk
            
        Returns:
            Complete ExplanationResult
        """
        # Main explanation
        explanation = self.analyze_features(feature_vector, feature_names)
        
        # Add chunk-level analysis if available
        if chunk_features is not None and len(chunk_features) > 1:
            explanation.chunk_explanations = self.create_chunk_explanations(
                chunks if chunks else [],
                chunk_features,
                feature_names
            )
        
        # Enhance summary with probability context
        if ai_probability > 0.8:
            explanation.summary += f" High AI probability ({ai_probability:.2f})."
        elif ai_probability < 0.2:
            explanation.summary += f" Low AI probability ({ai_probability:.2f})."
        else:
            explanation.summary += f" Moderate AI probability ({ai_probability:.2f})."
        
        return explanation


def get_feature_names() -> List[str]:
    """Return list of feature names in standard order."""
    return [
        "mean_perplexity", "perplexity_variance", "sentence_perplexity_std",
        "burstiness", "type_token_ratio", "hapax_legomena_ratio", "yules_k",
        "bigram_repetition_rate", "trigram_repetition_rate",
        "mean_sentence_length", "sentence_length_std",
        "punctuation_ratio", "comma_to_period_ratio", "function_word_ratio",
        "pos_noun_ratio", "pos_verb_ratio", "pos_adj_ratio", "pos_adv_ratio",
        "pos_pronoun_ratio", "pos_preposition_ratio", "pos_conjunction_ratio",
        "mean_clause_length", "clauses_per_sentence", "subordinate_clause_ratio",
        "discourse_marker_frequency", "sentence_opening_variety", "passive_voice_ratio"
    ]
