"""
Tests for evaluation module.
"""

import pytest
import numpy as np
from src.ai_detector.evaluation import (
    calculate_metrics, EvaluationMetrics, calculate_ece,
    find_threshold_for_fpr, evaluate_with_multiple_thresholds
)


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_basic_metrics(self):
        """Test basic classification metrics."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.1, 0.6, 0.8, 0.9, 0.4])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix calculation."""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.true_positives == 2
        assert metrics.true_negatives == 2
    
    def test_false_positive_rate(self):
        """Test FPR calculation."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        # All negatives predicted positive -> FPR = 1.0
        y_pred_all_fp = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_proba = np.array([0.6] * 8)
        
        metrics = calculate_metrics(y_true, y_pred_all_fp, y_proba)
        assert metrics.false_positive_rate == 1.0
        
        # No false positives -> FPR = 0.0
        y_pred_no_fp = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        metrics = calculate_metrics(y_true, y_pred_no_fp, y_proba)
        assert metrics.false_positive_rate == 0.0
    
    def test_roc_auc(self):
        """Test ROC-AUC calculation."""
        # Perfect separation
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        assert metrics.roc_auc == 1.0
    
    def test_calibration_metrics(self):
        """Test calibration metrics."""
        # Well-calibrated predictions
        y_true = np.array([0, 0, 1, 1])
        y_proba_well = np.array([0.1, 0.2, 0.8, 0.9])
        
        brier = np.mean((y_proba_well - y_true) ** 2)
        ece = calculate_ece(y_proba_well, y_true)
        
        assert brier >= 0
        assert ece >= 0
    
    def test_threshold_selection(self):
        """Test threshold selection for target FPR."""
        y_true = np.array([0] * 100 + [1] * 100)
        y_proba = np.random.rand(200)
        
        # Find threshold for 5% FPR
        threshold = find_threshold_for_fpr(y_true, y_proba, target_fpr=0.05)
        
        assert 0 <= threshold <= 1
        
        # Verify achieved FPR
        y_pred = (y_proba >= threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        achieved_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Should be approximately at or below target
        assert achieved_fpr <= 0.1  # Allow some tolerance
    
    def test_multiple_thresholds(self):
        """Test evaluation at multiple thresholds."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
        
        results = evaluate_with_multiple_thresholds(y_true, y_proba)
        
        assert len(results) > 0
        for threshold, metrics in results.items():
            assert isinstance(metrics, EvaluationMetrics)
            assert metrics.threshold_used == threshold
    
    def test_edge_cases(self):
        """Test edge cases in metric calculation."""
        # All same class
        y_true_all_zero = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.3])
        
        metrics = calculate_metrics(y_true_all_zero, y_pred, y_proba)
        # Should not crash, may have NaN for some metrics
    
    def test_brier_score_range(self):
        """Test Brier score is in valid range."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.9, 0.6])
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 0 <= metrics.brier_score <= 1
    
    def test_ece_perfect_calibration(self):
        """Test ECE for perfectly calibrated predictions."""
        # Perfectly calibrated: confidence matches accuracy
        n_samples = 1000
        y_proba = np.random.rand(n_samples)
        # Create labels that match probabilities
        y_true = (np.random.rand(n_samples) < y_proba).astype(int)
        
        ece = calculate_ece(y_proba, y_true)
        
        # Should be low for well-calibrated predictions
        assert ece < 0.2  # Reasonable threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
