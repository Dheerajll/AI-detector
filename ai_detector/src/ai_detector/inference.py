"""
Main inference module for AI text detection.

Provides the AIDetector class with a clean API:
- Load trained models
- Predict on single texts or batches
- Handle chunking for long documents
- Return calibrated probabilities with uncertainty estimates
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import json

from .preprocessing import TextPreprocessor, PreprocessingConfig, PreprocessedText
from .features import FeatureExtractor, FeatureConfig, AllFeatures
from .classifiers import HybridClassifier, ClassifierConfig, create_classifier
from .calibration import ProbabilityCalibrator, CalibrationConfig
from .uncertainty import UncertaintyEstimator, UncertaintyConfig, UncertaintyEstimate
from .explainability import ExplainabilityEngine, get_feature_names

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Complete prediction result."""
    # Classification
    classification: str  # "likely_ai", "likely_human", "uncertain"
    
    # Probabilities
    ai_probability: float
    human_probability: float
    
    # Confidence and reliability
    confidence: float
    reliability: str  # "high", "medium", "low"
    
    # Warnings
    warnings: List[str]
    
    # Evidence/explanation
    evidence: List[Dict[str, Any]]
    
    # Chunk-level results (for long documents)
    chunk_results: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    num_tokens: int = 0
    num_chunks: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "classification": self.classification,
            "ai_probability": round(self.ai_probability, 4),
            "human_probability": round(self.human_probability, 4),
            "confidence": round(self.confidence, 4),
            "reliability": self.reliability,
            "warnings": self.warnings,
            "evidence": self.evidence,
            "num_tokens": self.num_tokens,
            "num_chunks": self.num_chunks
        }
        
        if self.chunk_results:
            result["chunk_results"] = self.chunk_results
        
        return result


@dataclass 
class DetectorConfig:
    """Configuration for AIDetector."""
    preprocessing: PreprocessingConfig = None
    features: FeatureConfig = None
    classifier: ClassifierConfig = None
    calibration: CalibrationConfig = None
    uncertainty: UncertaintyConfig = None
    
    # Thresholds
    ai_threshold: float = 0.5
    uncertainty_low: float = 0.3
    uncertainty_high: float = 0.7
    
    # Paths
    models_dir: str = "models"
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.classifier is None:
            self.classifier = ClassifierConfig()
        if self.calibration is None:
            self.calibration = CalibrationConfig()
        if self.uncertainty is None:
            self.uncertainty = UncertaintyConfig()


class AIDetector:
    """
    Main AI text detector class.
    
    Provides a unified interface for:
    - Loading trained models
    - Making predictions with calibration and uncertainty
    - Explaining predictions
    
    Example usage:
        detector = AIDetector.load("models/final")
        result = detector.predict(text)
        print(result.to_dict())
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        
        # Initialize components
        self.preprocessor = TextPreprocessor(self.config.preprocessing)
        self.feature_extractor = FeatureExtractor(self.config.features)
        self.classifier = None
        self.calibrator = ProbabilityCalibrator(self.config.calibration)
        self.uncertainty_estimator = UncertaintyEstimator(self.config.uncertainty)
        self.explainer = ExplainabilityEngine(top_k=5)
        
        self.is_loaded = False
    
    @classmethod
    def load(cls, model_path: Union[str, Path], 
             config_path: Optional[Union[str, Path]] = None) -> 'AIDetector':
        """
        Load a trained detector from disk.
        
        Args:
            model_path: Path to saved model directory
            config_path: Optional path to config file (uses model_path/config.json if not specified)
            
        Returns:
            Loaded AIDetector instance
        """
        model_path = Path(model_path)
        
        # Load config
        if config_path is None:
            config_path = model_path / "detector_config.json"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            # Convert dict back to DetectorConfig (simplified)
            config = DetectorConfig()
        else:
            config = DetectorConfig()
            logger.warning(f"Config not found at {config_path}, using defaults")
        
        detector = cls(config)
        
        # Load classifier
        classifier_path = model_path / "classifier"
        if classifier_path.exists():
            detector.classifier = HybridClassifier(ClassifierConfig())
            detector.classifier.load(classifier_path)
        else:
            logger.warning(f"No classifier found at {classifier_path}")
        
        # Load calibrator
        calibrator_path = model_path / "calibrator.pkl"
        if calibrator_path.exists():
            detector.calibrator.load(calibrator_path)
        
        # Load feature statistics for explainability
        stats_path = model_path / "feature_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            detector.explainer.set_reference_statistics(stats)
        
        detector.is_loaded = True
        logger.info(f"Detector loaded from {model_path}")
        
        return detector
    
    def save(self, model_path: Union[str, Path]):
        """
        Save detector to disk.
        
        Args:
            model_path: Directory to save model components
        """
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        if self.classifier is not None:
            classifier_path = model_path / "classifier"
            self.classifier.save(classifier_path)
        
        # Save calibrator
        calibrator_path = model_path / "calibrator.pkl"
        self.calibrator.save(calibrator_path)
        
        # Save config
        config_path = model_path / "detector_config.json"
        # Simplified config saving
        with open(config_path, 'w') as f:
            json.dump({
                "ai_threshold": self.config.ai_threshold,
                "uncertainty_low": self.config.uncertainty_low,
                "uncertainty_high": self.config.uncertainty_high
            }, f, indent=2)
        
        logger.info(f"Detector saved to {model_path}")
    
    def predict(self, text: str, return_explanation: bool = True) -> PredictionResult:
        """
        Make prediction on a single text.
        
        Args:
            text: Input text to analyze
            return_explanation: Whether to include explanation/evidence
            
        Returns:
            PredictionResult with classification, probabilities, and metadata
        """
        if not self.is_loaded and self.classifier is None:
            raise RuntimeError("Detector must be loaded or trained before prediction")
        
        # Preprocess
        preprocessed = self.preprocessor.preprocess(text)
        
        # Collect warnings from preprocessing
        warnings = list(preprocessed.warnings)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            preprocessed.cleaned_text,
            preprocessed.sentences,
            preprocessed.tokens
        )
        feature_vector = features.to_vector().reshape(1, -1)
        
        # Get prediction from classifier
        if self.classifier is not None:
            # For hybrid classifier
            if hasattr(self.classifier, 'predict_proba'):
                raw_proba = self.classifier.predict_proba([preprocessed.cleaned_text], feature_vector)
            else:
                # Fallback for individual classifiers
                raw_proba = self.classifier.transformer_clf.predict_proba([preprocessed.cleaned_text])
        else:
            # No classifier available - return uncertain
            raw_proba = np.array([[0.5, 0.5]])
            warnings.append("No classifier loaded, returning uncertain prediction")
        
        # Apply calibration
        calibrated_proba = self.calibrator.calibrate(raw_proba)[0]
        
        ai_prob = float(calibrated_proba[1])
        human_prob = float(calibrated_proba[0])
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(
            calibrated_proba.reshape(1, -1),
            feature_vectors=feature_vector,
            num_tokens=preprocessed.num_tokens
        )
        
        # Determine classification
        classification, final_ai_prob = self._determine_classification(
            ai_prob, uncertainty, warnings
        )
        
        # Build evidence
        evidence = []
        if return_explanation and self.classifier is not None:
            feature_names = get_feature_names()
            # Pad or truncate feature vector to match names
            n_features = min(len(feature_names), feature_vector.shape[1])
            explanation = self.explainer.explain_prediction(
                feature_vector[0, :n_features],
                feature_names[:n_features],
                ai_prob
            )
            evidence = [e.to_dict() for e in explanation.evidence]
        
        # Build result
        result = PredictionResult(
            classification=classification,
            ai_probability=final_ai_prob,
            human_probability=1 - final_ai_prob if classification != "uncertain" else human_prob,
            confidence=uncertainty.confidence,
            reliability=uncertainty.reliability,
            warnings=warnings,
            evidence=evidence,
            num_tokens=preprocessed.num_tokens,
            num_chunks=len(preprocessed.chunks)
        )
        
        return result
    
    def _determine_classification(self, ai_prob: float, 
                                  uncertainty: UncertaintyEstimate,
                                  warnings: List[str]) -> Tuple[str, float]:
        """
        Determine final classification based on probability and uncertainty.
        
        Returns:
            (classification, adjusted_probability)
        """
        # Check if should be uncertain
        if self.uncertainty_estimator.should_classify_as_uncertain(
            np.array([[1 - ai_prob, ai_prob]]), uncertainty
        ):
            return "uncertain", ai_prob
        
        # Apply thresholds
        if ai_prob >= (1 - self.config.uncertainty_low):
            return "likely_ai", ai_prob
        elif ai_prob <= self.config.uncertainty_low:
            return "likely_human", ai_prob
        else:
            return "uncertain", ai_prob
    
    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Make predictions on multiple texts."""
        return [self.predict(text) for text in texts]
    
    def predict_with_chunks(self, text: str) -> PredictionResult:
        """
        Predict on long document with chunk-level analysis.
        
        Splits long documents into chunks, analyzes each, and aggregates.
        """
        preprocessed = self.preprocessor.preprocess(text)
        
        if len(preprocessed.chunks) <= 1:
            return self.predict(text)
        
        # Analyze each chunk
        chunk_results = []
        chunk_probs = []
        
        for chunk in preprocessed.chunks:
            chunk_prep = self.preprocessor.preprocess(chunk)
            chunk_features = self.feature_extractor.extract_all_features(
                chunk_prep.cleaned_text,
                chunk_prep.sentences,
                chunk_prep.tokens
            )
            feature_vec = chunk_features.to_vector().reshape(1, -1)
            
            raw_proba = self.classifier.predict_proba([chunk_prep.cleaned_text], feature_vec)
            calibrated = self.calibrator.calibrate(raw_proba)[0]
            
            chunk_probs.append(calibrated[1])
            
            chunk_results.append({
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "ai_probability": round(float(calibrated[1]), 4),
                "classification": "likely_ai" if calibrated[1] > 0.5 else "likely_human"
            })
        
        # Aggregate chunk probabilities (weighted mean by chunk length)
        # Using simple mean for now; could use more sophisticated aggregation
        aggregated_ai_prob = np.mean(chunk_probs)
        
        # Create overall result
        overall_result = self.predict(text)
        overall_result.chunk_results = chunk_results
        overall_result.ai_probability = round(aggregated_ai_prob, 4)
        overall_result.human_probability = round(1 - aggregated_ai_prob, 4)
        
        return overall_result


def load_detector(model_path: Union[str, Path]) -> AIDetector:
    """Convenience function to load a detector."""
    return AIDetector.load(model_path)
