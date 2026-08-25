"""
Evaluation script for AI text detector.

Usage:
    python scripts/evaluate.py --model models/final --data-dir data/processed --output eval_results.json

This script:
1. Loads trained model
2. Evaluates on test sets
3. Computes comprehensive metrics
4. Analyzes performance by domain, length, model type
5. Tests cross-model generalization
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


def load_test_data(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load test data from JSONL files."""
    data = {}
    
    test_splits = [
        'test_in_dist',
        'test_unseen_model', 
        'test_unseen_topics',
        'test_paraphrased',
        'test_edited',
        'test_short',
        'test_long'
    ]
    
    for split in test_splits:
        split_path = data_dir / f"{split}.jsonl"
        if split_path.exists():
            samples = []
            with open(split_path, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))
            data[split] = samples
            logger.info(f"Loaded {len(samples)} samples from {split}")
    
    return data


def evaluate_split(detector, samples: List[Dict]) -> Dict[str, Any]:
    """Evaluate detector on a single test split."""
    from src.ai_detector.evaluation import calculate_metrics
    
    texts = [s['text'] for s in samples]
    labels = np.array([s['label'] for s in samples])
    
    # Get predictions
    predictions = []
    probas = []
    
    for text in texts:
        try:
            result = detector.predict(text, return_explanation=False)
            probas.append(result.ai_probability)
            
            if result.classification == "likely_ai":
                predictions.append(1)
            elif result.classification == "likely_human":
                predictions.append(0)
            else:  # uncertain - use probability threshold
                predictions.append(1 if result.ai_probability >= 0.5 else 0)
                
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            predictions.append(0)
            probas.append(0.5)
    
    predictions = np.array(predictions)
    probas = np.array(probas)
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions, probas)
    
    return metrics.to_dict()


def evaluate_by_attribute(detector, samples: List[Dict], attribute: str) -> Dict[str, Any]:
    """Evaluate performance broken down by an attribute."""
    results = {}
    
    # Group by attribute
    by_attr = {}
    for sample in samples:
        value = sample.get(attribute, "unknown")
        if value not in by_attr:
            by_attr[value] = []
        by_attr[value].append(sample)
    
    # Evaluate each group
    for value, group_samples in by_attr.items():
        if len(group_samples) < 10:  # Skip small groups
            continue
        
        metrics = evaluate_split(detector, group_samples)
        results[value] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI text detector")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing test data")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # Load detector
    logger.info(f"Loading model from {args.model}...")
    from src.ai_detector.inference import AIDetector
    
    try:
        detector = AIDetector.load(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Load test data
    test_data = load_test_data(Path(args.data_dir))
    
    if not test_data:
        logger.error("No test data found!")
        return
    
    # Evaluate on each split
    all_results = {
        "model_path": args.model,
        "split_results": {}
    }
    
    for split_name, samples in test_data.items():
        logger.info(f"Evaluating on {split_name}...")
        metrics = evaluate_split(detector, samples)
        all_results["split_results"][split_name] = metrics
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['rates']['false_positive_rate']:.4f}")
    
    # Stratified analysis on in-distribution test set
    if 'test_in_dist' in test_data:
        logger.info("Running stratified analysis...")
        
        # By domain
        domain_results = evaluate_by_attribute(
            detector, test_data['test_in_dist'], 'domain'
        )
        all_results["by_domain"] = domain_results
        
        # By AI model (for AI samples)
        ai_samples = [s for s in test_data['test_in_dist'] if s['label'] == 1]
        if ai_samples:
            model_results = evaluate_by_attribute(detector, ai_samples, 'ai_model')
            all_results["by_ai_model"] = model_results
        
        # Cross-model generalization analysis
        train_models = set()
        test_models = set()
        
        # This would require knowing which models were in training
        # For now, just report per-model results
        all_results["cross_model_analysis"] = {
            "note": "Full cross-model analysis requires knowledge of training models"
        }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for split_name, metrics in all_results["split_results"].items():
        print(f"\n{split_name}:")
        print(f"  Samples: {len(test_data[split_name])}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  False Positive Rate: {metrics['rates']['false_positive_rate']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("IMPORTANT REMINDERS:")
    print("- These metrics are probabilistic, not definitive")
    print("- False positive rate is critical - high FPR harms innocent users")
    print("- Performance on unseen models may be lower than reported")
    print("- Short texts and edited AI text are harder to detect")
    print("=" * 60)


if __name__ == "__main__":
    main()
