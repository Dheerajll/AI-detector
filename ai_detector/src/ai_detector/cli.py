"""
Command-line interface for AI Text Detector.

Usage:
    python -m ai_detector predict --text "Some text to analyze"
    python -m ai_detector predict --file essay.txt
    python -m ai_detector batch --input-dir ./texts --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_detector(model_path: Optional[str] = None):
    """Load detector from specified or default model path."""
    from .inference import AIDetector
    
    if model_path is None:
        # Try default locations
        default_paths = [
            Path(__file__).parent.parent.parent / "models" / "final",
            Path("models") / "final",
            Path("models") / "default"
        ]
        
        for path in default_paths:
            if path.exists():
                model_path = str(path)
                break
        
        if model_path is None:
            logger.warning("No trained model found. Using untrained detector.")
            return AIDetector()
    
    return AIDetector.load(model_path)


def cmd_predict(args):
    """Handle predict command."""
    from .inference import AIDetector
    
    # Load detector
    try:
        detector = load_detector(args.model)
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        print(json.dumps({
            "error": f"Failed to load detector: {e}",
            "classification": "uncertain",
            "ai_probability": 0.5,
            "human_probability": 0.5,
            "confidence": 0.0,
            "reliability": "low",
            "warnings": ["Model not loaded"]
        }, indent=2))
        return 1
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(json.dumps({"error": f"File not found: {args.file}"}), indent=2)
            return 1
        
        # Check if PDF
        if file_path.suffix.lower() == '.pdf':
            # Use PDF analyzer
            if not hasattr(detector, 'pdf_analyzer') or detector.pdf_analyzer is None:
                try:
                    from .pdf_analyzer import PDFAnalyzer
                    detector.pdf_analyzer = PDFAnalyzer(ai_detector=detector)
                except ImportError:
                    print(json.dumps({
                        "error": "PDF support not available. Install PyMuPDF, pdfplumber, or PyPDF2.",
                        "classification": "unknown"
                    }), indent=2)
                    return 1
            
            result = detector.pdf_analyzer.analyze(file_path, run_ai_detection=True)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Read from stdin
        text = sys.stdin.read()
    
    # Make prediction
    try:
        result = detector.predict(text, return_explanation=not args.no_explanation)
        
        if args.chunk_analysis and len(text.split()) > 500:
            result = detector.predict_with_chunks(text)
        
        print(json.dumps(result.to_dict(), indent=2))
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(json.dumps({
            "error": str(e),
            "classification": "uncertain",
            "ai_probability": 0.5,
            "human_probability": 0.5,
            "confidence": 0.0,
            "reliability": "low",
            "warnings": [f"Prediction error: {e}"]
        }, indent=2))
        return 1
    
    return 0


def cmd_batch(args):
    """Handle batch processing command."""
    from .inference import AIDetector
    from tqdm import tqdm
    
    # Load detector
    try:
        detector = load_detector(args.model)
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return 1
    
    # Initialize PDF analyzer if needed
    pdf_analyzer = None
    if args.include_pdf:
        try:
            from .pdf_analyzer import PDFAnalyzer
            pdf_analyzer = PDFAnalyzer(ai_detector=detector)
        except ImportError:
            logger.warning("PDF support not available. Install PyMuPDF, pdfplumber, or PyPDF2.")
            args.include_pdf = False
    
    # Find input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Get all text files
    extensions = ['*.txt', '*.md', '*.json']
    if args.include_pdf:
        extensions.append('*.pdf')
    
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
    
    if not files:
        logger.warning(f"No text files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process files
    results = {}
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Handle PDFs
            if file_path.suffix.lower() == '.pdf' and pdf_analyzer is not None:
                result = pdf_analyzer.analyze(file_path, run_ai_detection=True)
                results[str(file_path)] = result.to_dict()
                continue
            
            # Handle text files
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = detector.predict(text, return_explanation=False)
            results[str(file_path)] = result.to_dict()
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            results[str(file_path)] = {"error": str(e)}
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    return 0


def cmd_evaluate(args):
    """Handle evaluation command."""
    # This would require a labeled dataset
    logger.info("Evaluation command not fully implemented.")
    logger.info("Please use scripts/evaluate.py for full evaluation.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AI Text Detector - Detect AI-generated text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s predict --text "This is some text to analyze"
  %(prog)s predict --file essay.txt
  %(prog)s batch --input-dir ./documents --output results.json
  
Note: This detector provides probabilistic assessments, not definitive proof.
      Always interpret results with caution, especially for short texts.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on single text')
    predict_parser.add_argument('--text', '-t', type=str, help='Text to analyze')
    predict_parser.add_argument('--file', '-f', type=str, help='File containing text')
    predict_parser.add_argument('--model', '-m', type=str, help='Path to model directory')
    predict_parser.add_argument('--no-explanation', action='store_true', 
                               help='Skip explanation generation')
    predict_parser.add_argument('--chunk-analysis', action='store_true',
                               help='Analyze long documents by chunks')
    predict_parser.set_defaults(func=cmd_predict)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process files')
    batch_parser.add_argument('--input-dir', type=str, required=True,
                             help='Directory containing text files')
    batch_parser.add_argument('--output', type=str, required=True,
                             help='Output JSON file path')
    batch_parser.add_argument('--model', '-m', type=str, help='Path to model directory')
    batch_parser.add_argument('--include-pdf', action='store_true',
                             help='Include PDF files in batch processing')
    batch_parser.set_defaults(func=cmd_batch)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--dataset', type=str, help='Path to test dataset')
    eval_parser.add_argument('--model', '-m', type=str, help='Path to model directory')
    eval_parser.add_argument('--output', type=str, help='Output results file')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
