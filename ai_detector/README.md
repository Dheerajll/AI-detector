# AI-Generated Text Detector

A production-grade, research-quality framework for detecting AI-generated text with calibrated probabilities, uncertainty quantification, and explainability.

## ⚠️ Important Disclaimers

**This detector is probabilistic, not definitive:**
- Results indicate likelihood, NOT proof of AI authorship
- False positives CAN occur, especially on non-native English, formal writing, or highly edited text
- Never use this as sole evidence for academic misconduct, hiring decisions, or legal matters
- Short texts (<50 words) have high uncertainty
- The model may not generalize to AI models released after training

**Ethical Use:**
- Use as one input among many in decision-making
- Always allow human appeal/review processes
- Do not deploy without understanding false positive rates in your specific domain
- Respect privacy: all processing happens locally by default

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Option A: Using Pre-trained Models (Recommended)](#option-a-using-pre-trained-models)
4. [Option B: Training Your Own Model](#option-b-training-your-own-model)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [PDF/Document Analysis](#pdfdocument-analysis)
8. [Understanding Output](#understanding-output)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)
12. [Contributing](#contributing)

---

## Quick Start

### Option A: Using Pre-trained Models (Coming Soon)

**⚠️ IMPORTANT:** Official pre-trained models are NOT yet published. The download script contains placeholder repositories that will fail with 401/404 errors. This is expected behavior.

You have two options:

1. **Train your own model locally** (Recommended - see detailed steps below)
2. **Find a community model** on HuggingFace and configure it yourself

Once you have a valid model (trained or from HuggingFace):

```bash
# Install the package
pip install -e .

# Run detection with your trained model
python -m ai_detector predict --text "Your text here..." --model-dir models/final

# Or use Python API
from ai_detector import AIDetector
detector = AIDetector.load("models/final")  # Your trained model
result = detector.predict("Your text here...")
print(result)
```

### Option B: Train Your Own Model (Recommended)

Complete step-by-step workflow:

#### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai_detector.git
cd ai_detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

#### Step 2: Prepare Dataset

```bash
# Download all datasets (requires ~50GB disk space)
python scripts/prepare_dataset.py --download-all

# Or download specific sources
python scripts/prepare_dataset.py --human-sources openwebtext,realnews
python scripts/prepare_dataset.py --ai-sources hc3,m4gt

# Inspect the prepared data
python scripts/prepare_dataset.py --inspect
```

This creates:
- Human writing from multiple domains (essays, news, academic, forums, technical)
- AI-generated text from multiple models (GPT-3.5, GPT-4, GPT-J, etc.)
- Proper train/validation/test splits with no contamination
- Specialized test sets for robustness evaluation

#### Step 3: Configure Training

Edit `configs/config.yaml` to customize your training:

```yaml
model:
  type: hybrid  # Options: hybrid, transformer, sklearn
  
training:
  batch_size: 32
  num_epochs: 10
  learning_rate: 2e-5
  max_false_positive_rate: 0.01  # Critical for reducing false alarms
  
calibration:
  method: isotonic  # Options: isotonic, platt, temperature
  
features:
  use_statistical: true
  use_linguistic: true
  use_neural: true
```

#### Step 4: Train the Model

```bash
# Train with GPU (recommended)
python scripts/train.py \
  --data-dir data/processed \
  --output-dir models/final \
  --config configs/config.yaml \
  --gpu

# Or train on CPU
python scripts/train.py \
  --data-dir data/processed \
  --output-dir models/final
```

Training typically takes:
- **Hybrid model**: 2-6 hours on GPU, 12-24 hours on CPU
- **Transformer-only**: 1-3 hours on GPU, 6-12 hours on CPU
- **Sklearn baseline**: 10-30 minutes on CPU

#### Step 5: Evaluate the Model

```bash
# Comprehensive evaluation
python scripts/evaluate.py \
  --model-dir models/final \
  --test-dir data/processed \
  --output-dir results/

# View results
cat results/metrics.json
```

Key metrics to check:
- **ROC-AUC**: Overall discrimination ability (>0.90 is good)
- **False Positive Rate**: Should be <1% at your chosen threshold
- **Calibration Error**: Lower is better (<0.05 is well-calibrated)
- **Cross-model generalization**: Performance on unseen AI generators

#### Step 6: Run Inference

```bash
# Single text prediction
python -m ai_detector predict \
  --text "Your text to analyze..." \
  --model-dir models/final

# From file
python -m ai_detector predict \
  --file essay.txt \
  --model-dir models/final

# PDF analysis (theses, reports)
python -m ai_detector predict \
  --file thesis.pdf \
  --model-dir models/final \
  --chunk-analysis

# Batch processing
python -m ai_detector batch \
  --input-dir documents/ \
  --output-dir predictions/ \
  --model-dir models/final
```

---

## Installation

### Prerequisites

- Python 3.9+ 
- pip or conda
- 4GB+ RAM (8GB+ recommended for transformer models)
- Optional: CUDA GPU for faster inference

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/ai-detector.git
cd ai-detector
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n ai-detector python=3.10
conda activate ai-detector
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -e .

# Or install with optional features
pip install -e ".[dev]"      # Development tools
pip install -e ".[pdf]"      # PDF analysis support
pip install -e ".[all]"      # All optional features
```

### Step 4: Verify Installation

```bash
python -c "from ai_detector import AIDetector; print('✓ Installation successful')"
```

---

## Pre-trained Models (Coming Soon)

**⚠️ NOTICE:** Official pre-trained models are NOT yet published. The table below shows planned models. For now, you must train your own model using the workflow above.

### Planned Model Releases

| Model | Size | Languages | Best For | Status |
|-------|------|-----------|----------|--------|
| `ai-detector-base` | ~450MB | English | General purpose, fast inference | 🔜 Coming Soon |
| `ai-detector-large` | ~1.2GB | English | Higher accuracy, critical applications | 🔜 Coming Soon |
| `ai-detector-multilingual` | ~1.1GB | 10+ languages | Non-English text | 🔜 Coming Soon |

### When Models Are Available

Once pre-trained models are published, you will be able to:

```bash
# Download specific model
python scripts/download_model.py --model ai-detector-base

# List available models
python scripts/download_model.py --list

# Verify download
python scripts/download_model.py --verify models/pretrained/ai-detector-base
```

### Using Community Models

You can also find community-trained models on HuggingFace:

1. Visit https://huggingface.co/models and search for "ai-detector" or "text-classification"
2. Find a compatible model (look for RoBERTa/DeBERTa-based classifiers)
3. Edit `scripts/download_model.py` to add the repository:
   ```python
   AVAILABLE_MODELS["community-model"] = {
       "repo_id": "username/model-name",
       "description": "Community model description",
       "size_mb": 500,
       "files": ["model.pt", "config.json", ...]
   }
   ```
4. Run: `python scripts/download_model.py --model community-model`

---

## Training Your Own Model (Recommended)

### When to Train Your Own Model

- You have domain-specific data (medical, legal, technical)
- You need to detect new AI models not in our training set
- You want to optimize for specific false-positive rates
- You have labeled data from your organization

### Complete Training Workflow

#### Step 1: Prepare Dataset

```bash
# Download all datasets automatically
python scripts/prepare_dataset.py --download-all

# Or download specific datasets
python scripts/prepare_dataset.py \
  --human-sources openwebtext,realnews,stackexchange \
  --ai-sources hc3,m4gt,mald

# Preview dataset statistics
python scripts/prepare_dataset.py --stats
```

**Dataset Sources:**

**Human Text:**
- OpenWebText (Reddit links)
- RealNews (news articles)
- WritingPrompts (creative writing)
- StackExchange (Q&A forums)
- arXiv (academic abstracts)
- Student Essays (educational writing)

**AI Text:**
- HC3 (ChatGPT responses)
- M4GT (GPT-J generations)
- MALD (multiple AI models)
- AIGC Detection Dataset

#### Step 2: Inspect Prepared Data

```bash
# View dataset splits
python scripts/prepare_dataset.py --inspect

# Check class balance
python scripts/prepare_dataset.py --balance-report
```

Expected directory structure:
```
data/
├── raw/              # Downloaded raw datasets
├── processed/        # Cleaned and split data
│   ├── train.parquet
│   ├── val.parquet
│   ├── test_in_dist.parquet
│   ├── test_unseen_models.parquet
│   ├── test_by_domain.parquet
│   └── test_short_long.parquet
└── README.md         # Dataset documentation
```

#### Step 3: Configure Training

Create or modify `configs/config.yaml`:

```yaml
training:
  model_type: "hybrid"  # hybrid, transformer, sklearn
  random_seed: 42
  
  # Data paths
  train_file: "data/processed/train.parquet"
  val_file: "data/processed/val.parquet"
  
  # Model configuration
  transformer_model: "roberta-base"
  max_length: 512
  batch_size: 16
  num_epochs: 10
  learning_rate: 2e-5
  
  # Calibration
  calibration_method: "isotonic"  # isotonic, platt, temperature
  
  # Threshold optimization
  target_false_positive_rate: 0.01
  
  # Output
  output_dir: "models/final"
  save_best_only: true
```

#### Step 4: Train Model

```bash
# Basic training
python scripts/train.py

# With custom config
python scripts/train.py --config configs/my_config.yaml

# With GPU acceleration
CUDA_VISIBLE_DEVICES=0 python scripts/train.py

# Monitor training progress
tensorboard --logdir models/final/logs
```

Training output:
```
Epoch 1/10: loss=0.623, val_loss=0.589, val_auc=0.847
Epoch 2/10: loss=0.512, val_loss=0.501, val_auc=0.891
...
Epoch 10/10: loss=0.312, val_loss=0.298, val_auc=0.943

✓ Training complete
✓ Model saved to: models/final
✓ Calibration applied: isotonic
✓ Optimal threshold: 0.67 (FPR=0.01)
```

#### Step 5: Evaluate Model

```bash
# Full evaluation suite
python scripts/evaluate.py --model-dir models/final

# Evaluate on specific test set
python scripts/evaluate.py \
  --model-dir models/final \
  --test-file data/processed/test_unseen_models.parquet

# Generate detailed report
python scripts/evaluate.py --model-dir models/final --report full
```

Evaluation metrics include:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- False Positive Rate, False Negative Rate
- Brier Score, Expected Calibration Error
- Confusion Matrix
- Performance by domain, length, AI model

#### Step 6: Upload Model (Optional)

```bash
# Upload to HuggingFace Hub
python scripts/upload_model.py \
  --model-dir models/final \
  --repo-id your-username/ai-detector-custom \
  --token hf_xxxxx

# Dry run first
python scripts/upload_model.py --model-dir models/final --dry-run
```

---

## Basic Usage

### Command Line Interface

#### Single Text Prediction

```bash
# From command line argument
python -m ai_detector predict --text "The quick brown fox jumps over the lazy dog."

# From file
python -m ai_detector predict --file essay.txt

# From stdin
cat essay.txt | python -m ai_detector predict --stdin

# Custom output format
python -m ai_detector predict --file essay.txt --format json
python -m ai_detector predict --file essay.txt --format table
```

#### Batch Prediction

```bash
# Process multiple files
python -m ai_detector batch --input-dir documents/ --output-dir results/

# Process CSV file
python -m ai_detector batch --input-csv texts.csv --output-csv predictions.csv

# With progress bar
python -m ai_detector batch --input-dir documents/ --show-progress
```

### Python API

#### Basic Prediction

```python
from ai_detector import AIDetector

# Load model
detector = AIDetector.load("models/pretrained/ai-detector-base")

# Simple prediction
result = detector.predict("Your text here...")

print(f"AI Probability: {result['ai_probability']:.2%}")
print(f"Human Probability: {result['human_probability']:.2%}")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Batch Prediction

```python
texts = [
    "First text to analyze...",
    "Second text to analyze...",
    "Third text to analyze..."
]

results = detector.predict_batch(texts)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['classification']} ({result['ai_probability']:.2%})")
```

#### With Custom Options

```python
result = detector.predict(
    text,
    min_length=50,           # Minimum tokens for reliable prediction
    max_length=10000,        # Maximum tokens before chunking
    return_evidence=True,    # Include feature explanations
    return_chunks=True       # Include chunk-level predictions
)
```

---

## Advanced Usage

### Understanding Classifications

The detector returns one of five classifications:

| Classification | Meaning | Action |
|---------------|---------|--------|
| `likely_ai` | High probability AI-generated | Review carefully |
| `likely_human` | High probability human-written | Low concern |
| `uncertain` | Model cannot decide reliably | Request more text or human review |
| `too_short` | Insufficient text for analysis | Provide longer sample |
| `out_of_distribution` | Text differs significantly from training data | Use with extreme caution |

### Threshold Configuration

```python
from ai_detector import AIDetector

# Load with custom threshold
detector = AIDetector.load(
    "models/final",
    threshold=0.67,  # Custom threshold
    fpr_target=0.01  # Target 1% false positive rate
)

# Or adjust after loading
detector.set_threshold(0.75)  # More conservative
detector.set_threshold(0.50)  # More sensitive
```

### Chunking Long Documents

```python
result = detector.predict(
    long_text,
    chunk_strategy="sliding_window",  # or "sentence", "paragraph"
    chunk_size=512,
    chunk_overlap=50,
    aggregation="weighted_average"
)

# Access chunk-level predictions
for i, chunk_pred in enumerate(result['chunk_predictions']):
    print(f"Chunk {i}: {chunk_pred['ai_probability']:.2%}")
```

### Uncertainty Quantification

```python
result = detector.predict(text, return_uncertainty=True)

print(f"Epistemic Uncertainty: {result['uncertainty']['epistemic']:.3f}")
print(f"Aleatoric Uncertainty: {result['uncertainty']['aleatoric']:.3f}")
print(f"OOD Score: {result['uncertainty']['ood_score']:.3f}")

if result['uncertainty']['total'] > 0.5:
    print("⚠️ High uncertainty - treat prediction with caution")
```

### Explainability

```python
result = detector.predict(text, return_evidence=True)

print("Evidence for prediction:")
for evidence in result['evidence']:
    print(f"  • {evidence['feature']}: {evidence['direction']} "
          f"(strength: {evidence['strength']:.2f})")
```

Example output:
```
Evidence for prediction:
  • token_predictability: AI-like (strength: 0.72)
  • perplexity_variance: AI-like (strength: 0.65)
  • syntactic_complexity: Human-like (strength: 0.41)
  • lexical_diversity: Neutral (strength: 0.15)
```

### Domain Adaptation

```python
# Evaluate on specific domain
from ai_detector import Evaluator

evaluator = Evaluator(detector)
report = evaluator.evaluate_by_domain(
    test_data,
    domains=['academic', 'creative', 'technical', 'casual']
)

print(report)
```

---

## PDF/Document Analysis

For analyzing educational reports, theses, and academic documents:

### Installation

```bash
pip install ai-detector[pdf]
```

### Basic PDF Analysis

```python
from ai_detector import PDFAnalyzer, AIDetector

# Initialize
detector = AIDetector.load("models/pretrained/ai-detector-base")
analyzer = PDFAnalyzer(ai_detector=detector)

# Analyze PDF
result = analyzer.analyze("thesis.pdf", run_ai_detection=True)

print(f"Overall AI Probability: {result.overall_ai_probability:.2%}")
print(f"Document Structure Score: {result.structure_score:.2f}")
print(f"Flow Coherence: {result.flow_coherence:.2f}")
```

### Section-Level Analysis

```python
# Get section-by-section breakdown
for section in result.section_classifications:
    print(f"\n{section['section_title']}")
    print(f"  AI Probability: {section['ai_probability']:.2%}")
    print(f"  Word Count: {section['word_count']}")
    print(f"  Citation Density: {section.get('citation_density', 'N/A')}")
```

### CLI for PDF

```bash
# Analyze single PDF
python -m ai_detector predict --file thesis.pdf

# Analyze with detailed report
python -m ai_detector predict --file thesis.pdf --format json --output report.json

# Batch analyze PDFs
python -m ai_detector batch --input-dir theses/ --pattern "*.pdf" --output-dir results/
```

### Understanding PDF Results

```python
# Structural evidence
print("Structural Evidence:")
print(f"  Academic Completeness: {result.structure_evidence['academic_completeness']:.2f}")
print(f"  Citation Consistency: {result.structure_evidence['citation_consistency']:.2f}")
print(f"  Technical Depth: {result.structure_evidence['technical_depth']:.2f}")

# Flow evidence
print("\nFlow Evidence:")
print(f"  Topic Coherence: {result.flow_evidence['topic_coherence']:.2f}")
print(f"  Generic Phrases: {result.flow_evidence['generic_phrase_density']:.2f}")
print(f"  Cross-section Repetition: {result.flow_evidence['repetition_score']:.2f}")
```

---

## Understanding Output

### Complete Output Schema

```json
{
  "classification": "likely_ai",
  "ai_probability": 0.91,
  "human_probability": 0.09,
  "confidence": 0.84,
  "reliability": "medium",
  "threshold_used": 0.67,
  "warnings": [],
  "evidence": [
    {
      "feature": "token_predictability",
      "direction": "AI-like",
      "strength": 0.72,
      "description": "Tokens show higher predictability than typical human writing"
    },
    {
      "feature": "perplexity_variance",
      "direction": "AI-like",
      "strength": 0.65,
      "description": "Low variance in sentence-level perplexity"
    }
  ],
  "uncertainty": {
    "epistemic": 0.12,
    "aleatoric": 0.08,
    "ood_score": 0.15,
    "total": 0.23
  },
  "metadata": {
    "text_length": 523,
    "num_tokens": 412,
    "processing_time_ms": 234,
    "model_version": "ai-detector-base-v1.0"
  }
}
```

### Interpreting Probabilities

**NOT a measure of certainty:**
- `ai_probability: 0.91` does NOT mean "91% chance this is AI"
- It means "this text shares 91% of patterns with AI training data"

**Proper interpretation:**
- Out of 100 similar texts, ~91 were AI-generated in our validation set
- Always consider confidence and uncertainty scores
- Never use as sole decision criterion

### Confidence Levels

| Confidence Range | Reliability | Recommended Action |
|-----------------|-------------|-------------------|
| 0.80 - 1.00 | High | Can inform decisions (with other evidence) |
| 0.60 - 0.80 | Medium | Use as one input among many |
| 0.40 - 0.60 | Low | Require human review |
| 0.00 - 0.40 | Very Low | Treat as uncertain |

---

## Evaluation & Metrics

### Running Evaluation

```bash
# Complete evaluation suite
python scripts/evaluate.py --model-dir models/final --output-dir eval_results

# Specific metrics
python scripts/evaluate.py --model-dir models/final --metrics roc_auc,pr_auc,brier

# By domain
python scripts/evaluate.py --model-dir models/final --breakdown domain

# By text length
python scripts/evaluate.py --model-dir models/final --breakdown length
```

### Key Metrics Explained

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **ROC-AUC** | Overall discrimination ability | Higher = better at separating AI/human |
| **PR-AUC** | Performance on imbalanced data | Better than ROC-AUC when classes uneven |
| **False Positive Rate** | % of human text flagged as AI | Critical for avoiding harm to humans |
| **False Negative Rate** | % of AI text missed | Important for security applications |
| **Brier Score** | Probability calibration quality | Lower = better calibrated probabilities |
| **ECE** | Expected Calibration Error | Measures probability reliability |

### Evaluating Cross-Model Generalization

```python
from ai_detector import Evaluator

evaluator = Evaluator(detector)

# Test on unseen AI models
unseen_results = evaluator.evaluate_on_unseen_models(
    test_file="data/processed/test_unseen_models.parquet"
)

print(f"AUC on seen models: {unseen_results['seen_auc']:.3f}")
print(f"AUC on unseen models: {unseen_results['unseen_auc']:.3f}")
print(f"Generalization gap: {unseen_results['gap']:.3f}")
```

### Adversarial Robustness Testing

```bash
# Test robustness to perturbations
python scripts/evaluate.py \
  --model-dir models/final \
  --robustness-tests synonym,reorder,paraphrase,spelling

# Generate robustness report
python scripts/evaluate.py --model-dir models/final --report robustness
```

---

## Troubleshooting

### Common Issues

#### Issue: "Model not found"

```bash
# Solution: Download model first
python scripts/download_model.py --model ai-detector-base

# Or specify correct path
detector = AIDetector.load("/full/path/to/model")
```

#### Issue: "CUDA out of memory"

```python
# Solution: Use CPU or smaller model
detector = AIDetector.load("models/pretrained/ai-detector-base", device="cpu")

# Or reduce batch size
detector = AIDetector.load("models/pretrained/ai-detector-base", batch_size=4)
```

#### Issue: "Text too short" warning

```python
# Solution: Provide longer text or adjust minimum
result = detector.predict(short_text, min_length=20)  # Default is 50

# Or combine multiple samples
combined_text = " ".join([text1, text2, text3])
result = detector.predict(combined_text)
```

#### Issue: High false positives on non-native English

```python
# Solution: Use multilingual model
detector = AIDetector.load("models/pretrained/ai-detector-multilingual")

# Or adjust threshold for your domain
detector.set_threshold(0.75)  # More conservative
```

#### Issue: Slow inference

```python
# Solution: Enable batching
results = detector.predict_batch(texts, batch_size=32)

# Or use smaller model
detector = AIDetector.load("models/pretrained/ai-detector-base")

# Or enable GPU
detector = AIDetector.load("models/pretrained/ai-detector-large", device="cuda")
```

### Getting Help

```bash
# Show help for any command
python -m ai_detector --help
python -m ai_detector predict --help
python scripts/train.py --help

# Check system info
python -c "from ai_detector import get_system_info; print(get_system_info())"

# Report issues
# Visit: https://github.com/your-org/ai-detector/issues
```

---

## API Reference

### AIDetector Class

```python
class AIDetector:
    @classmethod
    def load(cls, model_path: str, **kwargs) -> AIDetector
    def predict(self, text: str, **options) -> Dict
    def predict_batch(self, texts: List[str], **options) -> List[Dict]
    def set_threshold(self, threshold: float) -> None
    def get_threshold(self) -> float
    def save(self, output_path: str) -> None
```

### Prediction Options

```python
detector.predict(
    text,
    min_length: int = 50,
    max_length: int = 10000,
    chunk_strategy: str = "sliding_window",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    aggregation: str = "weighted_average",
    return_evidence: bool = True,
    return_uncertainty: bool = True,
    return_chunks: bool = False,
    device: str = None
)
```

### PDFAnalyzer Class

```python
class PDFAnalyzer:
    def __init__(self, ai_detector: AIDetector = None)
    def analyze(self, pdf_path: str, run_ai_detection: bool = True) -> PDFAnalysisResult
    def extract_text(self, pdf_path: str) -> ExtractedText
    def analyze_structure(self, text: str) -> StructureAnalysis
    def analyze_flow(self, text: str) -> FlowAnalysis
```

---

## Contributing

### Setting Up Development Environment

```bash
# Clone and install in dev mode
git clone https://github.com/your-org/ai-detector.git
cd ai-detector
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/ --check
mypy src/

# Build documentation
mkdocs serve
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=ai_detector --cov-report=html

# Integration tests
pytest tests/integration/ -v
```

### Code Style

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ scripts/
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{ai_detector_2024,
  title = {AI-Generated Text Detector},
  author = {Your Name and Contributors},
  year = {2024},
  url = {https://github.com/your-org/ai-detector}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

This project builds upon research and datasets from:
- HC3 Dataset (Hello ChatGPT)
- M4GT Dataset (Massive Multi-Model GPT Text)
- MALD Dataset (Machine-Authored Literature Detection)
- OpenWebText, RealNews, and other open datasets
- HuggingFace Transformers library
- Scikit-learn calibration methods

---

## Contact

- GitHub Issues: https://github.com/your-org/ai-detector/issues
- Email: your-email@example.com
- Documentation: https://your-org.github.io/ai-detector
