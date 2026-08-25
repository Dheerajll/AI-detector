#!/usr/bin/env python3
"""
Training Script for AI Text Detector

This script trains the hybrid AI detection model using:
1. Statistical features (perplexity, burstiness, etc.)
2. Linguistic features (POS patterns, syntax)
3. Neural representations (transformer embeddings)

The trained model is calibrated for reliable probability estimates.

Usage:
    python scripts/train.py --data-dir data/processed --output models/final
    python scripts/train.py --model-type hybrid --epochs 5
    python scripts/train.py --use-pretrained-embeddings
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_detector.preprocessing import TextPreprocessor, PreprocessingConfig
from ai_detector.features import FeatureExtractor, FeatureConfig
from ai_detector.classifiers import HybridClassifier, TransformerClassifier
from ai_detector.calibration import ProbabilityCalibrator


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Load training dataset from parquet files."""
    
    data_path = Path(data_dir)
    train_file = data_path / "train.parquet"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")
    
    logger.info(f"Loading training data from {train_file}")
    df = pd.read_parquet(train_file)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"  Human samples: {len(df[df['label'] == 0])}")
    logger.info(f"  AI samples: {len(df[df['label'] == 1])}")
    
    return df


def prepare_features(
    texts: List[str],
    labels: List[int],
    preprocessor: TextPreprocessor,
    extractor: FeatureExtractor,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from texts."""
    
    X = []
    y = []
    
    logger.info(f"Extracting features from {len(texts)} samples...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        for text, label in zip(batch_texts, batch_labels):
            try:
                # Preprocess
                processed = preprocessor.preprocess(text)
                
                # Extract features
                features = extractor.extract_all_features(
                    processed.cleaned_text,
                    processed.sentences,
                    processed.tokens
                )
                
                feature_vec = features.to_vector()
                
                # Handle NaN/Inf values
                if not np.all(np.isfinite(feature_vec)):
                    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=-1.0)
                
                X.append(feature_vec)
                y.append(label)
                
            except Exception as e:
                logger.warning(f"Feature extraction failed for sample: {e}")
                continue
    
    return np.array(X), np.array(y)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "hybrid",
    output_dir: Path = None,
    **kwargs
) -> Tuple[object, Dict]:
    """Train the detection model."""
    
    logger.info(f"Training {model_type} model...")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Feature dimensions: {X_train.shape[1]}")
    
    if model_type == "hybrid":
        model = HybridClassifier(
            use_transformer=kwargs.get("use_transformer", True),
            transformer_model=kwargs.get("transformer_model", "roberta-base"),
            use_statistical_features=kwargs.get("use_statistical", True),
            use_linguistic_features=kwargs.get("use_linguistic", True),
            device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=kwargs.get("epochs", 3),
            batch_size=kwargs.get("batch_size", 32),
            learning_rate=kwargs.get("learning_rate", 2e-5),
        )
    
    elif model_type == "transformer":
        model = TransformerClassifier(
            model_name=kwargs.get("transformer_model", "roberta-base"),
            device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # For transformer-only, we need raw texts
        # This is a simplified version - full implementation would handle this differently
        logger.warning("Transformer-only mode requires text inputs, not features")
        return None, {}
    
    elif model_type == "sklearn":
        from sklearn.ensemble import GradientBoostingClassifier
        
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 5),
            random_state=42,
        )
        model.fit(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "brier_score": brier_score_loss(y_val, y_pred_proba),
        "accuracy": (y_pred == y_val).mean(),
    }
    
    logger.info(f"\nValidation Results:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
    
    return model, metrics


def calibrate_model(
    model: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> Tuple[object, ProbabilityCalibrator]:
    """Calibrate model probabilities."""
    
    logger.info(f"Calibrating probabilities using {method} regression...")
    
    # Get uncalibrated probabilities
    proba = model.predict_proba(X_cal)[:, 1]
    
    # Fit calibrator
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(proba, y_cal)
    
    # Evaluate calibration
    calibrated_proba = calibrator.transform(proba)
    
    from sklearn.metrics import brier_score_loss
    
    original_brier = brier_score_loss(y_cal, proba)
    calibrated_brier = brier_score_loss(y_cal, calibrated_proba)
    
    logger.info(f"  Original Brier score: {original_brier:.4f}")
    logger.info(f"  Calibrated Brier score: {calibrated_brier:.4f}")
    logger.info(f"  Improvement: {original_brier - calibrated_brier:.4f}")
    
    return model, calibrator


def save_model(
    model: object,
    calibrator: ProbabilityCalibrator,
    feature_extractor: FeatureExtractor,
    preprocessor: TextPreprocessor,
    config: Dict,
    metrics: Dict,
    output_dir: Path,
):
    """Save trained model and artifacts."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    
    # Save model weights
    model_path = output_dir / "model.pt"
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        import joblib
        joblib.dump(model, model_path)
    
    # Save calibrator
    calibrator.save(output_dir / "calibration_params.json")
    
    # Save feature extractor config
    feature_extractor.save_config(output_dir / "feature_config.json")
    
    # Save preprocessor config
    preprocessor.config.save(output_dir / "preprocessing_config.json")
    
    # Save metadata
    metadata = {
        "model_type": config.get("model_type", "hybrid"),
        "training_date": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
        "config": config,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Model saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Train AI text detector")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/final",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hybrid", "transformer", "sklearn"],
        default="hybrid",
        help="Type of model to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for transformer models",
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="roberta-base",
        help="Transformer model name",
    )
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        default=True,
        help="Use transformer embeddings",
    )
    parser.add_argument(
        "--use-statistical",
        action="store_true",
        default=True,
        help="Use statistical features",
    )
    parser.add_argument(
        "--use-linguistic",
        action="store_true",
        default=True,
        help="Use linguistic features",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Validation set split ratio",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        choices=["isotonic", "platt", "temperature"],
        default="isotonic",
        help="Probability calibration method",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger(__name__)
    
    # Set seeds
    set_seed(args.seed)
    
    # Detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n=== Loading Data ===")
    df = load_dataset(args.data_dir)
    
    # Split into train/val/calibration
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=args.seed, stratify=df["label"]
    )
    val_df, cal_df = train_test_split(
        temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["label"]
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Cal: {len(cal_df)}")
    
    # Initialize preprocessor and feature extractor
    logger.info("\n=== Initializing Preprocessor and Feature Extractor ===")
    
    prep_config = PreprocessingConfig()
    preprocessor = TextPreprocessor(prep_config)
    
    feat_config = FeatureConfig(
        use_perplexity=args.use_statistical,
        use_burstiness=args.use_statistical,
        use_pos_features=args.use_linguistic,
        use_dependency_features=args.use_linguistic,
        use_transformer_embeddings=args.use_transformer,
        transformer_model=args.transformer_model,
    )
    extractor = FeatureExtractor(feat_config, device=device)
    
    # Prepare features
    logger.info("\n=== Preparing Training Features ===")
    
    X_train, y_train = prepare_features(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        preprocessor,
        extractor,
        batch_size=args.batch_size,
    )
    
    X_val, y_val = prepare_features(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        preprocessor,
        extractor,
        batch_size=args.batch_size,
    )
    
    X_cal, y_cal = prepare_features(
        cal_df["text"].tolist(),
        cal_df["label"].tolist(),
        preprocessor,
        extractor,
        batch_size=args.batch_size,
    )
    
    # Train model
    logger.info("\n=== Training Model ===")
    
    config = {
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "transformer_model": args.transformer_model,
        "use_transformer": args.use_transformer,
        "use_statistical": args.use_statistical,
        "use_linguistic": args.use_linguistic,
    }
    
    model, metrics = train_model(
        X_train, y_train,
        X_val, y_val,
        model_type=args.model_type,
        output_dir=Path(args.output_dir),
        device=device,
        **config,
    )
    
    if model is None:
        logger.error("Training failed!")
        return
    
    # Calibrate model
    logger.info("\n=== Calibrating Probabilities ===")
    
    model, calibrator = calibrate_model(
        model, X_cal, y_cal, method=args.calibration_method
    )
    
    # Save model
    logger.info("\n=== Saving Model ===")
    
    save_model(
        model=model,
        calibrator=calibrator,
        feature_extractor=extractor,
        preprocessor=preprocessor,
        config=config,
        metrics=metrics,
        output_dir=Path(args.output_dir),
    )
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"\nFinal Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
