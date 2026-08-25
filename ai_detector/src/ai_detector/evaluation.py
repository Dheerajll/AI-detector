"""
Evaluation module for AI text detection.

Provides comprehensive evaluation metrics and analysis:
- Standard classification metrics (accuracy, precision, recall, F1)
- Ranking metrics (ROC-AUC, PR-AUC)
- Calibration metrics (Brier score, ECE)
- Stratified analysis by domain, length, model type
- Adversarial robustness testing

Important: Accuracy alone is insufficient - we especially track false positive rates.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Basic metrics
    accuracy: float = np.nan
    precision: float = np.nan
    recall: float = np.nan
    f1: float = np.nan
    
    # Ranking metrics
    roc_auc: float = np.nan
    pr_auc: float = np.nan
    
    # Calibration metrics
    brier_score: float = np.nan
    expected_calibration_error: float = np.nan
    
    # Confusion matrix values
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Rates
    false_positive_rate: float = np.nan
    false_negative_rate: float = np.nan
    true_positive_rate: float = np.nan
    true_negative_rate: float = np.nan
    
    # Threshold info
    threshold_used: float = np.nan
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier_score": round(self.brier_score, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives
            },
            "rates": {
                "false_positive_rate": round(self.false_positive_rate, 4),
                "false_negative_rate": round(self.false_negative_rate, 4),
                "true_positive_rate": round(self.true_positive_rate, 4),
                "true_negative_rate": round(self.true_negative_rate, 4)
            },
            "threshold_used": round(self.threshold_used, 4) if self.threshold_used == self.threshold_used else None
        }


@dataclass
class StratifiedResults:
    """Results broken down by different strata."""
    by_domain: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    by_text_length: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    by_ai_model: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    by_editing_level: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.by_domain:
            result["by_domain"] = {k: v.to_dict() for k, v in self.by_domain.items()}
        if self.by_text_length:
            result["by_text_length"] = {k: v.to_dict() for k, v in self.by_text_length.items()}
        if self.by_ai_model:
            result["by_ai_model"] = {k: v.to_dict() for k, v in self.by_ai_model.items()}
        if self.by_editing_level:
            result["by_editing_level"] = {k: v.to_dict() for k, v in self.by_editing_level.items()}
        return result


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: np.ndarray, threshold: float = 0.5) -> EvaluationMetrics:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels (0=human, 1=AI)
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for AI class
        threshold: Classification threshold
        
    Returns:
        EvaluationMetrics with all calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    metrics = EvaluationMetrics()
    metrics.threshold_used = threshold
    
    # Basic metrics
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.precision = precision_score(y_true, y_pred, zero_division=0)
    metrics.recall = recall_score(y_true, y_pred, zero_division=0)
    metrics.f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Ranking metrics
    if len(np.unique(y_true)) > 1:
        metrics.roc_auc = roc_auc_score(y_true, y_proba)
        metrics.pr_auc = average_precision_score(y_true, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.true_positives = int(tp)
    metrics.true_negatives = int(tn)
    metrics.false_positives = int(fp)
    metrics.false_negatives = int(fn)
    
    # Rates
    if tn + fp > 0:
        metrics.false_positive_rate = fp / (tn + fp)
    if fn + tp > 0:
        metrics.false_negative_rate = fn / (fn + tp)
    metrics.true_positive_rate = metrics.recall
    if tn + fp > 0:
        metrics.true_negative_rate = tn / (tn + fp)
    
    # Calibration metrics
    metrics.brier_score = np.mean((y_proba - y_true) ** 2)
    metrics.expected_calibration_error = calculate_ece(y_proba, y_true)
    
    return metrics


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = probs[in_bin].mean()
            avg_accuracy = labels[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece


def find_threshold_for_fpr(y_true: np.ndarray, y_proba: np.ndarray,
                          target_fpr: float) -> float:
    """
    Find classification threshold that achieves target false positive rate.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_fpr: Target false positive rate (e.g., 0.01 for 1%)
        
    Returns:
        Threshold value
    """
    # Get negative samples (human)
    neg_mask = y_true == 0
    neg_probas = y_proba[neg_mask]
    
    if len(neg_probas) == 0:
        logger.warning("No negative samples for threshold calculation")
        return 0.5
    
    # Sort negative probabilities
    sorted_probas = np.sort(neg_probas)
    
    # Find threshold where X% of negatives are below it
    target_idx = int(target_fpr * len(neg_probas))
    target_idx = min(target_idx, len(sorted_probas) - 1)
    
    threshold = sorted_probas[target_idx]
    
    logger.info(f"Threshold for FPR={target_fpr}: {threshold:.4f}")
    return threshold


def evaluate_with_multiple_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[float, EvaluationMetrics]:
    """
    Evaluate at multiple thresholds to show trade-offs.
    
    Returns dict mapping threshold to metrics.
    """
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        results[thresh] = calculate_metrics(y_true, y_pred, y_proba, thresh)
    
    return results


def stratified_evaluation(predictions: np.ndarray, probas: np.ndarray,
                         labels: np.ndarray, metadata: Dict[str, np.ndarray]) -> StratifiedResults:
    """
    Evaluate performance stratified by different factors.
    
    Args:
        predictions: Binary predictions
        probas: Probability predictions
        labels: True labels
        metadata: Dict mapping stratum name to array of stratum values
        
    Returns:
        StratifiedResults with metrics for each stratum
    """
    results = StratifiedResults()
    
    for stratum_name, stratum_values in metadata.items():
        unique_values = np.unique(stratum_values)
        stratum_metrics = {}
        
        for value in unique_values:
            mask = stratum_values == value
            if mask.sum() < 5:  # Skip small groups
                continue
            
            y_true_stratum = labels[mask]
            y_pred_stratum = predictions[mask]
            y_proba_stratum = probas[mask]
            
            metrics = calculate_metrics(y_true_stratum, y_pred_stratum, y_proba_stratum)
            stratum_metrics[str(value)] = metrics
        
        # Store in appropriate attribute
        if stratum_name == "domain":
            results.by_domain = stratum_metrics
        elif stratum_name == "text_length_bucket":
            results.by_text_length = stratum_metrics
        elif stratum_name == "ai_model":
            results.by_ai_model = stratum_metrics
        elif stratum_name == "editing_level":
            results.by_editing_level = stratum_metrics
    
    return results


def run_robustness_tests(detector, test_texts: List[str], 
                        original_labels: List[int]) -> Dict[str, Any]:
    """
    Run adversarial robustness tests.
    
    Tests:
    - Synonym replacement
    - Sentence reordering
    - Punctuation changes
    - Spelling variations
    
    Returns dict with robustness metrics.
    """
    import random
    
    original_results = detector.predict_batch(test_texts)
    original_preds = [1 if r.classification == "likely_ai" else 0 for r in original_results]
    
    robustness_results = {
        "original_accuracy": np.mean([p == l for p, l in zip(original_preds, original_labels)])
    }
    
    # Test 1: Punctuation variation
    punct_variants = []
    for text in test_texts[:50]:  # Limit for speed
        variant = text.replace('.', '. ').replace('  ', ' ')
        if random.random() > 0.5:
            variant = variant.replace(',', '')
        punct_variants.append(variant)
    
    punct_results = detector.predict_batch(punct_variants)
    punct_preds = [1 if r.classification == "likely_ai" else 0 for r in punct_results]
    robustness_results["punctuation_robustness"] = np.mean([p == l for p, l in zip(punct_preds, original_labels[:50])])
    
    # Note: Full robustness testing would require more sophisticated text manipulation
    # This is a simplified version
    
    return robustness_results


def evaluate_cross_model_generalization(train_models: List[str], test_models: List[str],
                                       train_metrics: EvaluationMetrics,
                                       test_metrics: EvaluationMetrics) -> Dict[str, Any]:
    """
    Evaluate how well detector generalizes to unseen AI models.
    
    Args:
        train_models: Models used in training
        test_models: Models held out from training
        train_metrics: Performance on seen models
        test_metrics: Performance on unseen models
        
    Returns:
        Dict with generalization analysis
    """
    performance_drop = train_metrics.f1 - test_metrics.f1
    relative_drop = performance_drop / train_metrics.f1 if train_metrics.f1 > 0 else 0
    
    return {
        "train_models": train_models,
        "test_models": test_models,
        "train_f1": train_metrics.f1,
        "test_f1": test_metrics.f1,
        "absolute_drop": performance_drop,
        "relative_drop": relative_drop,
        "generalization_quality": "good" if relative_drop < 0.1 else "moderate" if relative_drop < 0.2 else "poor"
    }


def save_evaluation_results(results: Dict[str, Any], path: Path):
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {path}")


def load_evaluation_results(path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(Path(path), 'r') as f:
        return json.load(f)
