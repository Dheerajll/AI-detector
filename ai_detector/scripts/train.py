"""
Training script for AI text detector.

Usage:
    python scripts/train.py --data-dir data/processed --output models/final
    
This script:
1. Loads preprocessed data
2. Extracts features
3. Trains hybrid classifier
4. Calibrates probabilities
5. Saves trained model
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load preprocessed data from JSONL files."""
    data = {}
    
    for split in ['train', 'val', 'test_in_dist']:
        split_path = data_dir / f"{split}.jsonl"
        if split_path.exists():
            samples = []
            with open(split_path, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))
            data[split] = samples
            logger.info(f"Loaded {len(samples)} samples from {split}")
        else:
            logger.warning(f"No data found for split: {split}")
            data[split] = []
    
    return data


def extract_features(texts: List[str], config: Dict) -> tuple:
    """Extract features from texts."""
    from src.ai_detector.preprocessing import TextPreprocessor, PreprocessingConfig
    from src.ai_detector.features import FeatureExtractor, FeatureConfig
    
    # Create preprocessors and extractors
    prep_config = PreprocessingConfig(**config.get('preprocessing', {}))
    feat_config = FeatureConfig(**config.get('features', {}))
    
    preprocessor = TextPreprocessor(prep_config)
    extractor = FeatureExtractor(feat_config)
    
    feature_vectors = []
    valid_texts = []
    
    for text in texts:
        try:
            preprocessed = preprocessor.preprocess(text)
            features = extractor.extract_all_features(
                preprocessed.cleaned_text,
                preprocessed.sentences,
                preprocessed.tokens
            )
            feature_vec = features.to_vector()
            
            if np.all(np.isfinite(feature_vec)):
                feature_vectors.append(feature_vec)
                valid_texts.append(preprocessed.cleaned_text)
            else:
                # Replace NaN with 0
                feature_vec = np.nan_to_num(feature_vec, nan=0.0)
                feature_vectors.append(feature_vec)
                valid_texts.append(preprocessed.cleaned_text)
                
        except Exception as e:
            logger.warning(f"Feature extraction failed for a sample: {e}")
            continue
    
    return np.array(feature_vectors), valid_texts


def train(config: Dict, args: argparse.Namespace):
    """Main training function."""
    logger.info("Starting training...")
    
    # Load data
    data = load_data(Path(args.data_dir))
    
    if not data.get('train'):
        logger.error("No training data found!")
        return
    
    # Prepare training data
    train_texts = [s['text'] for s in data['train']]
    train_labels = [s['label'] for s in data['train']]
    
    val_texts = [s['text'] for s in data.get('val', [])]
    val_labels = [s['label'] for s in data.get('val', [])]
    
    logger.info(f"Training samples: {len(train_texts)} (AI={sum(train_labels)}, Human={len(train_labels)-sum(train_labels)})")
    if val_texts:
        logger.info(f"Validation samples: {len(val_texts)}")
    
    # Extract features
    logger.info("Extracting features...")
    train_features, train_texts_clean = extract_features(train_texts, config)
    val_features, val_texts_clean = extract_features(val_texts, config) if val_texts else (None, None)
    
    logger.info(f"Feature vector shape: {train_features.shape}")
    
    # Train classifier
    logger.info("Training classifier...")
    from src.ai_detector.classifiers import HybridClassifier, ClassifierConfig
    
    clf_config = ClassifierConfig(**config.get('classifier', {}))
    classifier = HybridClassifier(clf_config)
    
    training_metrics = classifier.fit(
        train_texts_clean,
        train_features,
        train_labels,
        val_texts=val_texts_clean,
        val_features=val_features,
        val_labels=val_labels
    )
    
    logger.info(f"Training complete. Metrics: {training_metrics}")
    
    # Calibrate
    logger.info("Calibrating probabilities...")
    from src.ai_detector.calibration import ProbabilityCalibrator, CalibrationConfig
    
    cal_config = CalibrationConfig(**config.get('calibration', {}))
    calibrator = ProbabilityCalibrator(cal_config)
    
    # Get validation predictions for calibration
    if val_features is not None:
        val_proba = classifier.predict_proba(val_texts_clean, val_features)
        cal_metrics = calibrator.fit(val_proba, np.array(val_labels), is_logits=False)
        logger.info(f"Calibration metrics: {cal_metrics}")
    
    # Save model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_path}...")
    classifier.save(output_path / "classifier")
    calibrator.save(output_path / "calibrator.pkl")
    
    # Save config
    with open(output_path / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Compute feature statistics for explainability
    feature_stats = {
        "mean": train_features.mean(axis=0).tolist(),
        "std": train_features.std(axis=0).tolist(),
        "min": train_features.min(axis=0).tolist(),
        "max": train_features.max(axis=0).tolist()
    }
    with open(output_path / "feature_stats.json", 'w') as f:
        json.dump(feature_stats, f, indent=2)
    
    logger.info("Training complete!")
    return classifier, calibrator


def main():
    parser = argparse.ArgumentParser(description="Train AI text detector")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing preprocessed data")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for trained model")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (uses defaults if not specified)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "preprocessing": {
                "min_tokens": 50,
                "max_tokens": 4096,
                "chunk_size": 512
            },
            "features": {
                "use_perplexity": True,
                "use_burstiness": True,
                "use_vocabulary_diversity": True,
                "use_neural_features": True,
                "encoder_name": "roberta-base"
            },
            "classifier": {
                "type": "hybrid",
                "classical_type": "gradient_boosting",
                "n_estimators": 200,
                "encoder_name": "roberta-base",
                "num_epochs": 3,
                "batch_size": 16
            },
            "calibration": {
                "method": "temperature_scaling"
            }
        }
    
    train(config, args)


if __name__ == "__main__":
    main()
