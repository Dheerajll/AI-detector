"""
Prediction script for quick inference.

Usage:
    python scripts/predict.py --model models/final --text "Some text"
    python scripts/predict.py --model models/final --file document.txt
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Make predictions with AI detector")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--text", type=str, default=None,
                       help="Text to analyze")
    parser.add_argument("--file", type=str, default=None,
                       help="File containing text to analyze")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (prints to stdout if not specified)")
    parser.add_argument("--chunk-analysis", action="store_true",
                       help="Analyze long documents by chunks")
    
    args = parser.parse_args()
    
    # Load detector
    from src.ai_detector.inference import AIDetector
    
    logger.info(f"Loading model from {args.model}...")
    detector = AIDetector.load(args.model)
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        logger.error("Must provide either --text or --file")
        return 1
    
    # Make prediction
    logger.info("Making prediction...")
    
    if args.chunk_analysis and len(text.split()) > 500:
        result = detector.predict_with_chunks(text)
    else:
        result = detector.predict(text)
    
    # Output results
    output_data = result.to_dict()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())
