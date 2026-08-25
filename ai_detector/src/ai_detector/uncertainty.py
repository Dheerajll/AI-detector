"""
Uncertainty estimation module for AI text detection.

Provides methods to quantify prediction uncertainty:
- Ensemble disagreement
- Out-of-distribution detection
- Confidence scoring
- Length-based reliability warnings

Why uncertainty matters:
- Prevents overconfident wrong predictions
- Guides users on when to trust outputs
- Enables human-in-the-loop workflows
- Ethically responsible design
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""
    enable_ood_detection: bool = True
    ood_method: str = "mahalanobis"  # mahalanobis, entropy, ensemble_disagreement
    min_confidence: float = 0.3
    max_entropy_threshold: float = 0.95
    
    # For Mahalanobis distance
    contamination: float = 0.1
    
    # Length-based warnings
    min_tokens_for_high_confidence: int = 100
    min_tokens_for_any_prediction: int = 20


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty metrics."""
    # Overall confidence (0-1, higher = more confident)
    confidence: float = np.nan
    
    # Entropy of prediction (higher = more uncertain)
    entropy: float = np.nan
    
    # OOD score (higher = more out-of-distribution)
    ood_score: float = np.nan
    
    # Ensemble disagreement (if applicable)
    ensemble_std: float = np.nan
    
    # Reliability classification
    reliability: str = "medium"  # high, medium, low
    
    # Warnings
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class UncertaintyEstimator:
    """
    Estimates uncertainty for AI detection predictions.
    
    Combines multiple signals:
    1. Prediction entropy
    2. Ensemble disagreement
    3. OOD detection
    4. Text length heuristics
    """
    
    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.ood_detector = None
        self.feature_mean = None
        self.feature_cov = None
        self.is_fitted = False
    
    def _calculate_entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Calculate entropy of probability distribution.
        
        Higher entropy = more uncertain prediction
        
        Max entropy for binary classification: log(2) ≈ 0.693
        """
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # Binary entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs), axis=1)
        
        return entropy
    
    def _calculate_ensemble_disagreement(self, 
                                         all_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate disagreement among ensemble members.
        
        Args:
            all_predictions: List of probability predictions from different models
            
        Returns:
            Standard deviation across predictions
        """
        if len(all_predictions) < 2:
            return np.array([0.0])
        
        stacked = np.stack(all_predictions, axis=0)
        std = np.std(stacked, axis=0)
        
        return std[:, 1]  # Std for AI class
    
    def fit_ood_detector(self, feature_vectors: np.ndarray):
        """
        Fit OOD detector on training features.
        
        Uses Mahalanobis distance to detect out-of-distribution samples.
        """
        if not self.config.enable_ood_detection:
            return
        
        try:
            from sklearn.covariance import EmpiricalCovariance
            
            # Fit covariance matrix on training data
            self.feature_mean = feature_vectors.mean(axis=0)
            self.ood_detector = EmpiricalCovariance().fit(feature_vectors)
            
            self.is_fitted = True
            logger.info("OOD detector fitted using Mahalanobis distance")
            
        except Exception as e:
            logger.warning(f"Could not fit OOD detector: {e}")
            self.config.enable_ood_detection = False
    
    def calculate_mahalanobis_distance(self, feature_vector: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance from training distribution.
        
        Higher distance = more out-of-distribution
        """
        if not self.is_fitted or self.ood_detector is None:
            return 0.0
        
        try:
            distance = self.ood_detector.mahalanobis([feature_vector])[0]
            return distance
        except Exception:
            return 0.0
    
    def estimate_uncertainty(self, probs: np.ndarray,
                            feature_vectors: Optional[np.ndarray] = None,
                            all_predictions: Optional[List[np.ndarray]] = None,
                            num_tokens: int = 100) -> UncertaintyEstimate:
        """
        Estimate uncertainty for a prediction.
        
        Args:
            probs: Calibrated probability predictions (n_samples, 2)
            feature_vectors: Statistical/linguistic features for OOD detection
            all_predictions: All ensemble member predictions for disagreement
            num_tokens: Number of tokens in input text
            
        Returns:
            UncertaintyEstimate with confidence, entropy, OOD score, etc.
        """
        estimate = UncertaintyEstimate()
        
        # 1. Calculate entropy
        entropy = self._calculate_entropy(probs)
        estimate.entropy = float(entropy[0]) if len(entropy) > 0 else np.nan
        
        # Normalize entropy to 0-1 scale (max binary entropy = log(2))
        max_entropy = np.log(2)
        normalized_entropy = estimate.entropy / max_entropy if estimate.entropy == estimate.entropy else 0
        
        # 2. Calculate ensemble disagreement
        if all_predictions is not None and len(all_predictions) > 1:
            disagreement = self._calculate_ensemble_disagreement(all_predictions)
            estimate.ensemble_std = float(disagreement[0]) if len(disagreement) > 0 else 0.0
        else:
            estimate.ensemble_std = 0.0
        
        # 3. Calculate OOD score
        if feature_vectors is not None and self.config.enable_ood_detection:
            ood_score = self.calculate_mahalanobis_distance(feature_vectors[0])
            estimate.ood_score = float(ood_score)
        else:
            estimate.ood_score = 0.0
        
        # 4. Calculate overall confidence
        # Combine: prediction confidence, low entropy, low OOD, low disagreement
        prediction_confidence = np.max(probs[0])
        
        confidence_factors = [prediction_confidence]
        
        # Entropy factor (low entropy = high confidence)
        confidence_factors.append(1 - normalized_entropy)
        
        # Ensemble agreement (low std = high confidence)
        if estimate.ensemble_std > 0:
            agreement_factor = 1 - min(estimate.ensemble_std * 2, 1.0)
            confidence_factors.append(agreement_factor)
        
        # OOD factor (low distance = high confidence)
        if estimate.ood_score > 0:
            # Normalize OOD score (assume distances > 10 are very OOD)
            ood_factor = 1 - min(estimate.ood_score / 10.0, 1.0)
            confidence_factors.append(ood_factor)
        
        # Combined confidence (geometric mean)
        estimate.confidence = float(np.prod(confidence_factors) ** (1/len(confidence_factors)))
        
        # 5. Length-based adjustments
        if num_tokens < self.config.min_tokens_for_any_prediction:
            estimate.warnings.append(
                f"Text has only {num_tokens} tokens. "
                f"Minimum recommended: {self.config.min_tokens_for_any_prediction}"
            )
            estimate.confidence *= 0.5  # Reduce confidence significantly
            
        elif num_tokens < self.config.min_tokens_for_high_confidence:
            estimate.warnings.append(
                f"Text has {num_tokens} tokens. "
                f"For higher confidence, provide at least {self.config.min_tokens_for_high_confidence} tokens."
            )
            estimate.confidence *= 0.75
        
        # 6. Classify reliability
        if estimate.confidence >= 0.7:
            estimate.reliability = "high"
        elif estimate.confidence >= 0.4:
            estimate.reliability = "medium"
        else:
            estimate.reliability = "low"
            estimate.warnings.append(
                "Low confidence prediction. Interpret results with caution."
            )
        
        # 7. Check for extreme OOD
        if estimate.ood_score > 20:  # Threshold for extreme OOD
            estimate.warnings.append(
                "Input appears significantly different from training data. "
                "Results may be unreliable."
            )
            estimate.reliability = "low"
        
        return estimate
    
    def should_classify_as_uncertain(self, probs: np.ndarray,
                                     uncertainty: UncertaintyEstimate) -> bool:
        """
        Determine if prediction should be classified as 'uncertain'.
        
        This prevents forcing binary classification when evidence is weak.
        """
        ai_prob = probs[0, 1] if len(probs.shape) > 1 else probs[1]
        
        # If probability is in uncertain range
        if (ai_prob > self.config.min_confidence and 
            ai_prob < (1 - self.config.min_confidence)):
            return True
        
        # If confidence is too low
        if uncertainty.confidence < self.config.min_confidence:
            return True
        
        # If reliability is low
        if uncertainty.reliability == "low":
            return True
        
        return False


def create_uncertainty_estimator(config: Optional[UncertaintyConfig] = None) -> UncertaintyEstimator:
    """Factory function for uncertainty estimator."""
    return UncertaintyEstimator(config)
