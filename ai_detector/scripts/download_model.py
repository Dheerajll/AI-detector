#!/usr/bin/env python3
"""
Download Pre-trained AI Detector Model

This script downloads pre-trained model weights from HuggingFace Hub.
The models are trained on the full dataset described in data/README.md.

Usage:
    python scripts/download_model.py --model ai-detector-base
    python scripts/download_model.py --model ai-detector-large
    python scripts/download_model.py --all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


# Available pre-trained models on HuggingFace
AVAILABLE_MODELS = {
    "ai-detector-base": {
        "repo_id": "your-org/ai-detector-base",  # Replace with actual repo
        "description": "Base model (RoBERTa-base + statistical features)",
        "size_mb": 450,
        "files": [
            "model.pt",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "calibration_params.json",
            "feature_scalers.pkl",
            "metadata.json",
        ],
    },
    "ai-detector-large": {
        "repo_id": "your-org/ai-detector-large",  # Replace with actual repo
        "description": "Large model (DeBERTa-v3 + hybrid features)",
        "size_mb": 1200,
        "files": [
            "model.pt",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "calibration_params.json",
            "feature_scalers.pkl",
            "metadata.json",
        ],
    },
    "ai-detector-multilingual": {
        "repo_id": "your-org/ai-detector-multilingual",  # Replace with actual repo
        "description": "Multilingual model (XLM-RoBERTa + features)",
        "size_mb": 1100,
        "files": [
            "model.pt",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "calibration_params.json",
            "feature_scalers.pkl",
            "metadata.json",
        ],
    },
}


def download_model(
    model_name: str,
    output_dir: str = "models/pretrained",
    force_download: bool = False,
) -> Path:
    """
    Download a pre-trained model from HuggingFace Hub.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
        force_download: Force re-download even if cached
    
    Returns:
        Path to downloaded model directory
    """
    
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' not found.")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    model_info = AVAILABLE_MODELS[model_name]
    repo_id = model_info["repo_id"]
    
    print(f"\n=== Downloading {model_name} ===")
    print(f"Description: {model_info['description']}")
    print(f"Size: ~{model_info['size_mb']} MB")
    print(f"Repository: {repo_id}")
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download entire repository
        print(f"\nDownloading model files...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            force_download=force_download,
            ignore_patterns=["*.git*", "README.md"],
        )
        
        print(f"\n✓ Model downloaded successfully to: {output_path}")
        
        # Verify files
        expected_files = model_info["files"]
        missing_files = []
        
        for file in expected_files:
            if not (output_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠ Warning: Some files may be missing:")
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"\n✓ All expected files present")
        
        return output_path
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error downloading model: {error_msg}")
        
        if "404" in error_msg or "Repository Not Found" in error_msg or "401" in error_msg:
            print("\n" + "=" * 60)
            print("⚠️  TROUBLESHOOTING:")
            print("   The model repository does not exist or is private.")
            print("   This is EXPECTED - the placeholder repositories are not real.")
            print("\n👉 NEXT STEPS TO TRAIN LOCALLY:")
            print("   Step 1: Prepare dataset")
            print("     python scripts/prepare_dataset.py --download-all")
            print("\n   Step 2: Train model")
            print("     python scripts/train.py --data-dir data/processed --output-dir models/final")
            print("\n   Step 3: Use local model")
            print("     python -m ai_detector predict --text '...' --model-dir models/final")
            print("\n   Alternatively:")
            print("   - Find a valid public model on huggingface.co")
            print("   - Edit scripts/download_model.py to add the real repo_id")
            print("=" * 60)
        elif "authentication" in error_msg.lower() or "Invalid username" in error_msg:
            print("\n🔐 Authentication Required:")
            print("   Run: huggingface-cli login")
            print("   Then retry the download.")
        else:
            print("\n💡 Possible solutions:")
            print("   1. Check your internet connection")
            print("   2. Verify the model repository exists: https://huggingface.co/" + repo_id)
            print("   3. Run: huggingface-cli login (if private)")
            
        sys.exit(1)


def list_available_models():
    """List all available pre-trained models."""
    print("\n=== Available Pre-trained Models ===\n")
    
    for name, info in AVAILABLE_MODELS.items():
        print(f"Model: {name}")
        print(f"  Description: {info['description']}")
        print(f"  Size: ~{info['size_mb']} MB")
        print(f"  Repository: {info['repo_id']}")
        print()
    
    print("=" * 60)
    print("⚠️  IMPORTANT NOTICE:")
    print("    The repositories listed above are PLACEHOLDERS.")
    print("    Official pre-trained models are NOT yet published.")
    print("\n👉 HOW TO USE THIS SYSTEM:")
    print("\n    OPTION A: Train Your Own Model (Recommended)")
    print("      Step 1: Download datasets")
    print("        python scripts/prepare_dataset.py --download-all")
    print("\n      Step 2: Train the model")
    print("        python scripts/train.py --data-dir data/processed --output-dir models/final")
    print("\n      Step 3: Run inference")
    print("        python -m ai_detector predict --text 'Your text here...' --model-dir models/final")
    print("\n    OPTION B: Use a Community Model from HuggingFace")
    print("      1. Find a compatible model at: https://huggingface.co/models")
    print("      2. Note the full repo_id (e.g., 'username/model-name')")
    print("      3. Edit this script (scripts/download_model.py) to add it to AVAILABLE_MODELS")
    print("      4. Run: python scripts/download_model.py --model your-model-name")
    print("=" * 60)


def verify_model(model_path: Path) -> bool:
    """Verify that a downloaded model has all required files."""
    
    required_files = [
        "model.pt",
        "config.json",
        "calibration_params.json",
    ]
    
    missing = []
    for file in required_files:
        if not (model_path / file).exists():
            missing.append(file)
    
    if missing:
        print(f"✗ Model verification failed. Missing files: {missing}")
        return False
    
    print(f"✓ Model verification passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained AI detector models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Specific model to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/pretrained",
        help="Output directory for models",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify an existing model at the given path",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if args.verify:
        model_path = Path(args.verify)
        if model_path.exists():
            verify_model(model_path)
        else:
            print(f"Error: Path does not exist: {model_path}")
        return
    
    if args.all:
        print("Downloading all models...\n")
        for model_name in AVAILABLE_MODELS.keys():
            download_model(
                model_name,
                output_dir=args.output_dir,
                force_download=args.force_download,
            )
            print("\n" + "="*60 + "\n")
    elif args.model:
        download_model(
            args.model,
            output_dir=args.output_dir,
            force_download=args.force_download,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/download_model.py --model ai-detector-base")
        print("  python scripts/download_model.py --all")
        print("  python scripts/download_model.py --list")


if __name__ == "__main__":
    main()
