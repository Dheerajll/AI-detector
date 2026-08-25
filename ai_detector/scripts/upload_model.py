#!/usr/bin/env python3
"""
Upload Trained Model to HuggingFace Hub

This script uploads a trained model to HuggingFace Hub for sharing.

Usage:
    python scripts/upload_model.py --model-dir models/final --repo-id your-org/ai-detector-base
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, create_repo, upload_folder


def get_model_files(model_dir: Path) -> List[str]:
    """List all files in model directory."""
    files = []
    for pattern in ["*.pt", "*.json", "*.pkl", "*.txt", "*.model", "README.md"]:
        files.extend([f.name for f in model_dir.glob(pattern)])
    return sorted(files)


def validate_model(model_dir: Path) -> bool:
    """Validate that model directory has required files."""
    
    required_files = [
        "model.pt",
        "config.json",
        "calibration_params.json",
        "metadata.json",
    ]
    
    missing = []
    for file in required_files:
        if not (model_dir / file).exists():
            missing.append(file)
    
    if missing:
        print(f"✗ Missing required files: {missing}")
        return False
    
    print(f"✓ All required files present")
    return True


def create_model_card(
    model_dir: Path,
    repo_id: str,
    model_type: str = "hybrid",
    language: str = "en",
    license: str = "mit",
) -> str:
    """Create a model card README.md."""
    
    # Try to load metadata
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metrics = metadata.get("metrics", {})
    else:
        metrics = {}
    
    model_card = f"""---
language:
- {language}
license: {license}
tags:
- ai-detection
- text-classification
- transformer
- hybrid-model
pipeline_tag: text-classification
---

# AI Text Detection Model

This model detects AI-generated text using a hybrid approach combining statistical features, linguistic analysis, and neural representations.

## Model Details

- **Model Type**: {model_type}
- **Base Architecture**: Transformer + Statistical Features
- **Language**: {language.upper()}
- **License**: {license.upper()}

## Intended Use

This model is designed to:
- Provide probabilistic assessments of whether text is AI-generated
- Support multiple domains (essays, news, academic, forums, etc.)
- Offer calibrated probability estimates with uncertainty quantification

## Limitations

⚠️ **Important Disclaimers:**

- This model provides **probabilistic assessments**, not proof of authorship
- **False positives occur**, especially on:
  - Short texts (<100 words)
  - Non-native English writing
  - Highly formal or technical prose
  - Heavily edited text
- Performance may degrade on AI models not seen during training
- **Never use alone** for high-stakes decisions

## Training Data

The model was trained on balanced datasets including:
- **Human texts**: OpenWebText, RealNews, WritingPrompts, StackExchange, arXiv abstracts, student essays
- **AI texts**: HC3 (ChatGPT), M4GT (GPT-J), MALD, AIGC Detection datasets

Data was split by source/author to prevent train/test contamination.

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | {metrics.get('roc_auc', 'N/A')} |
| Brier Score | {metrics.get('brier_score', 'N/A')} |
| Accuracy | {metrics.get('accuracy', 'N/A')} |

*Note: Metrics are from validation set. Actual performance may vary.*

## How to Use

### Python API

```python
from ai_detector import AIDetector

# Load model
detector = AIDetector.from_pretrained("{repo_id}")

# Analyze text
result = detector.predict("Your text here...")
print(f"AI Probability: {result.ai_probability:.2%}")
```

### Command Line

```bash
# Install
pip install ai-detector

# Run inference
python -m ai_detector predict --text "Your text here..."

# Analyze file
python -m ai_detector predict --file document.txt
```

## Framework

This model was created using the [AI Detector Framework](https://github.com/your-org/ai-detector).

## Citation

If you use this model in research, please cite:

```bibtex
@software{{ai_detector_2024,
  title = {{AI Text Detection Framework}},
  year = {{2024}},
  url = {{https://github.com/your-org/ai-detector}},
}}
```

## Ethical Considerations

- Use responsibly and transparently
- Always acknowledge uncertainty in predictions
- Avoid bias against specific writing styles or non-native speakers
- Combine with human review for important decisions
"""
    
    return model_card


def upload_model(
    model_dir: Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
):
    """Upload model to HuggingFace Hub."""
    
    print(f"\n=== Uploading Model to HuggingFace ===")
    print(f"Model directory: {model_dir}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
    # Validate model
    if not validate_model(model_dir):
        print("\n✗ Cannot upload incomplete model")
        return False
    
    # Initialize API
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        print(f"\nCreating repository '{repo_id}'...")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✓ Repository ready")
        
        # Create model card
        print(f"\nCreating model card...")
        model_card = create_model_card(model_dir, repo_id)
        model_card_path = model_dir / "README.md"
        
        with open(model_card_path, "w") as f:
            f.write(model_card)
        print(f"✓ Model card created")
        
        # List files to upload
        files = get_model_files(model_dir)
        print(f"\nFiles to upload:")
        for f in files:
            print(f"  - {f}")
        
        # Upload folder
        print(f"\nUploading files...")
        upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            token=token,
            repo_type="model",
            create_pr=create_pr,
            ignore_patterns=["*.git*", "__pycache__/", "*.pyc"],
        )
        
        print(f"\n✓ Model uploaded successfully!")
        print(f"\nView your model at:")
        print(f"  https://huggingface.co/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error uploading model: {e}")
        print("\nPossible solutions:")
        print("1. Login: huggingface-cli login")
        print("2. Check repository name format: username/repo-name")
        print("3. Verify you have write permissions")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (username/repo-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or use huggingface-cli login)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of direct commit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        return
    
    if args.dry_run:
        print(f"\n=== Dry Run ===")
        print(f"Would upload {model_dir} to {args.repo_id}")
        files = get_model_files(model_dir)
        print(f"\nFiles ({len(files)}):")
        for f in files:
            print(f"  - {f}")
        return
    
    success = upload_model(
        model_dir=model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        create_pr=args.create_pr,
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
