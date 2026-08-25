"""
Probability calibration module for AI text detection.

Raw classifier outputs are often poorly calibrated (overconfident or underconfident).
This module implements calibration techniques to produce reliable probabilities.

Why calibration matters:
- Uncalibrated probabilities mislead users about actual confidence
- Decision thresholds become arbitrary without calibration
- Risk assessment requires accurate probability estimates

Techniques implemented:
- Temperature scaling (for neural models)
- Isotonic regression (non-parametric, flexible)
- Platt scaling (logistic calibration)

Evaluation metrics:
- Expected Calibration Error (ECE)
- Brier score
- Reliability diagrams
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for probability calibration."""
    method: str = "temperature_scaling"  # temperature_scaling, isotonic, platt
    validation_split: float = 0.15
    target_false_positive_rate: float = 0.01
    
    # For temperature scaling
    initial_temperature: float = 1.0
    max_iterations: int = 1000
    
    # Threshold settings
    ai_threshold: float = 0.5
    uncertainty_low: float = 0.3
    uncertainty_high: float = 0.7


class TemperatureScaling:
    """
    Temperature scaling calibration for neural network outputs.
    
    Divides logits by a learned temperature parameter T:
    calibrated_prob = softmax(logit / T)
    
    T > 1: Softens probabilities (reduces overconfidence)
    T < 1: Sharpens probabilities
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.temperature = config.initial_temperature
        self.is_fitted = False
    
    def _nll_loss(self, logits: np.ndarray, labels: np.ndarray, 
                  temperature: float) -> float:
        """Calculate negative log-likelihood loss with temperature scaling."""
        import torch
        
        scaled_logits = logits / temperature
        probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
        
        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        # NLL loss
        n = len(labels)
        nll = -np.sum(np.log(probs[np.arange(n), labels])) / n
        
        return nll
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: Raw model logits (n_samples, n_classes)
            labels: True labels
            
        Returns:
            Optimal temperature value
        """
        try:
            import torch
            from scipy.optimize import minimize_scalar
        except ImportError as e:
            logger.warning(f"Required imports not available: {e}")
            self.temperature = 1.0
            self.is_fitted = True
            return self.temperature
        
        # Define objective function
        def objective(T):
            if T <= 0:
                return float('inf')
            return self._nll_loss(logits, labels, T)
        
        # Optimize temperature
        result = minimize_scalar(
            objective,
            bounds=(0.1, 10.0),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        self.temperature = result.x
        self.is_fitted = True
        
        logger.info(f"Temperature scaling fitted: T={self.temperature:.4f}")
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        import torch
        
        if not self.is_fitted:
            logger.warning("Temperature scaling not fitted, using T=1.0")
            temperature = 1.0
        else:
            temperature = self.temperature
        
        scaled_logits = logits / temperature
        probs = torch.softmax(torch.tensor(scaled_logits), dim=1).numpy()
        
        return probs


class IsotonicCalibration:
    """
    Isotonic regression calibration.
    
    Non-parametric method that fits a monotonically increasing function
    to map raw scores to calibrated probabilities.
    
    More flexible than temperature scaling but can overfit on small datasets.
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibration':
        """
        Fit isotonic regression on validation scores.
        
        Args:
            scores: Raw probability scores for positive class (AI)
            labels: True labels
        """
        from sklearn.isotonic import IsotonicRegression
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(scores, labels)
        self.is_fitted = True
        
        logger.info("Isotonic calibration fitted")
        return self
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to scores."""
        if not self.is_fitted:
            logger.warning("Isotonic calibrator not fitted, returning original scores")
            return scores
        
        calibrated = self.calibrator.transform(scores)
        # Ensure valid probability range
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        return calibrated


class PlattScaling:
    """
    Platt scaling (logistic calibration).
    
    Fits a logistic regression to map raw scores to calibrated probabilities.
    
    P(y=1|s) = 1 / (1 + exp(A*s + B))
    
    where s is the raw score and A, B are learned parameters.
    """
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """Fit Platt scaling on validation scores."""
        from sklearn.linear_model import LogisticRegression
        
        # Reshape for sklearn
        scores_reshaped = scores.reshape(-1, 1)
        
        self.calibrator = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000
        )
        self.calibrator.fit(scores_reshaped, labels)
        self.is_fitted = True
        
        logger.info("Platt scaling fitted")
        return self
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to scores."""
        if not self.is_fitted:
            logger.warning("Platt calibrator not fitted, returning original scores")
            return scores
        
        scores_reshaped = scores.reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(scores_reshaped)[:, 1]
        
        return calibrated


def calculate_ece(probs: np.ndarray, labels: np.ndarray, 
                  n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the average gap between predicted confidence and actual accuracy
    across probability bins.
    
    Perfect calibration: ECE = 0
    
    Args:
        probs: Predicted probabilities for positive class
        labels: True binary labels
        n_bins: Number of bins for grouping predictions
        
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average confidence in bin
            avg_confidence = probs[in_bin].mean()
            # Average accuracy in bin
            avg_accuracy = labels[in_bin].mean()
            # Weighted absolute difference
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error of probabilities).
    
    Brier score = mean((prob - label)^2)
    
    Perfect prediction: Brier = 0
    Worst prediction: Brier = 1
    
    Lower is better.
    """
    return np.mean((probs - labels) ** 2)


def create_reliability_diagram_data(probs: np.ndarray, labels: np.ndarray,
                                    n_bins: int = 10) -> Dict[str, np.ndarray]:
    """
    Create data for plotting a reliability diagram.
    
    Returns bin centers, actual accuracies, and predicted confidences
    for visualization.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    confidences = []
    counts = []
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            accuracies.append(labels[in_bin].mean())
            confidences.append(probs[in_bin].mean())
            counts.append(in_bin.sum())
        else:
            accuracies.append(np.nan)
            confidences.append(np.nan)
            counts.append(0)
    
    return {
        'bin_centers': bin_centers,
        'accuracies': np.array(accuracies),
        'confidences': np.array(confidences),
        'counts': np.array(counts)
    }


class ProbabilityCalibrator:
    """
    Main calibrator class that wraps different calibration methods.
    
    Provides unified interface for:
    - Fitting calibration on validation data
    - Applying calibration to new predictions
    - Evaluating calibration quality
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.calibrator = None
        self.is_fitted = False
    
    def _create_calibrator(self):
        """Create appropriate calibrator based on config."""
        if self.config.method == "temperature_scaling":
            return TemperatureScaling(self.config)
        elif self.config.method == "isotonic":
            return IsotonicCalibration(self.config)
        elif self.config.method == "platt":
            return PlattScaling(self.config)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
    
    def fit(self, logits_or_scores: np.ndarray, 
            labels: np.ndarray,
            is_logits: bool = False) -> Dict[str, float]:
        """
        Fit calibrator on validation data.
        
        Args:
            logits_or_scores: Either raw logits (for temperature scaling)
                             or probability scores (for isotonic/platt)
            labels: True binary labels
            is_logits: Whether input is logits (True) or probabilities (False)
            
        Returns:
            Calibration metrics (ECE, Brier score before and after)
        """
        self.calibrator = self._create_calibrator()
        
        # Convert logits to probabilities if needed
        if is_logits:
            import torch
            probs_before = torch.softmax(torch.tensor(logits_or_scores), dim=1).numpy()[:, 1]
        else:
            probs_before = logits_or_scores
        
        # Calculate pre-calibration metrics
        ece_before = calculate_ece(probs_before, labels)
        brier_before = calculate_brier_score(probs_before, labels)
        
        logger.info(f"Pre-calibration - ECE: {ece_before:.4f}, Brier: {brier_before:.4f}")
        
        # Fit calibrator
        if isinstance(self.calibrator, TemperatureScaling) and is_logits:
            self.calibrator.fit(logits_or_scores, labels)
        else:
            self.calibrator.fit(probs_before, labels)
        
        # Apply calibration
        probs_after = self.calibrate(probs_before)
        
        # Calculate post-calibration metrics
        ece_after = calculate_ece(probs_after, labels)
        brier_after = calculate_brier_score(probs_after, labels)
        
        logger.info(f"Post-calibration - ECE: {ece_after:.4f}, Brier: {brier_after:.4f}")
        
        self.is_fitted = True
        
        return {
            'ece_before': ece_before,
            'ece_after': ece_after,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'improvement_ece': ece_before - ece_after,
            'improvement_brier': brier_before - brier_after
        }
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probability scores.
        
        Args:
            probs: Uncalibrated probability scores
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning uncalibrated probabilities")
            return probs
        
        return self.calibrator.calibrate(probs)
    
    def save(self, path: Path):
        """Save calibrator to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'calibrator': self.calibrator,
                'config': self.config,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Calibrator saved to {path}")
    
    def load(self, path: Path):
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.calibrator = data['calibrator']
            self.config = data.get('config', self.config)
            self.is_fitted = data.get('is_fitted', False)
        
        logger.info(f"Calibrator loaded from {path}")
    
    def get_optimal_threshold(self, probs: np.ndarray, labels: np.ndarray,
                              target_fpr: Optional[float] = None) -> float:
        """
        Find optimal classification threshold.
        
        Args:
            probs: Calibrated probabilities
            labels: True labels
            target_fpr: Target false positive rate (if specified, finds threshold achieving it)
            
        Returns:
            Optimal threshold value
        """
        if target_fpr is not None:
            # Find threshold that achieves target FPR
            # Sort by probability descending
            sorted_indices = np.argsort(-probs)
            sorted_labels = labels[sorted_indices]
            
            n_negatives = (labels == 0).sum()
            target_fp = target_fpr * n_negatives
            
            cumulative_fp = 0
            threshold = 0.5
            
            for i, label in enumerate(sorted_labels):
                if label == 0:
                    cumulative_fp += 1
                if cumulative_fp >= target_fp:
                    threshold = probs[sorted_indices[i]]
                    break
            
            logger.info(f"Threshold for FPR={target_fpr}: {threshold:.4f}")
            return threshold
        else:
            # Default: use configured threshold
            return self.config.ai_threshold
