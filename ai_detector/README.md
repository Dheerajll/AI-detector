# AI Text Detector

A robust, research-quality framework for detecting AI-generated text using a hybrid approach combining statistical features, linguistic analysis, and neural representations.

## ⚠️ Important Disclaimers

**This is NOT a definitive authorship tool.** This detector provides **probabilistic assessments**, not proof of authorship. Key limitations:

- **False positives occur**, especially on short texts, non-native English writing, or highly formal prose
- **Performance degrades** on paraphrased or edited AI text
- **Unseen AI models** may evade detection
- **Never use alone** for high-stakes decisions (academic integrity, hiring, etc.)

Always interpret results with caution and consider uncertainty estimates.

---

## Features

### Core Capabilities

- **Hybrid Detection**: Combines statistical, linguistic, and neural features
- **Probability Calibration**: Produces reliable probability estimates via temperature scaling/isotonic regression
- **Uncertainty Estimation**: Identifies when predictions are unreliable
- **Explainability**: Provides feature-level evidence for each prediction
- **Chunk Analysis**: Handles long documents with section-by-section analysis
- **Low False-Positive Focus**: Optimized to minimize wrongful accusations

### What Makes This Different

| Simple Detectors | This Framework |
|-----------------|----------------|
| Single feature (perplexity) | Multiple complementary feature families |
| Raw scores as probabilities | Properly calibrated probabilities |
| Binary output | Three-way classification + uncertainty |
| No explanations | Feature-level evidence provided |
| Fails on long docs | Chunked analysis with aggregation |
| Untuned thresholds | Threshold selection for target FPR |

---

## Installation

```bash
# Clone repository
cd ai_detector

# Install dependencies
pip install -e .

# For development/testing
pip install -e ".[dev]"

# Download spaCy model (optional, for linguistic features)
python -m spacy download en_core_web_sm
```

### Requirements

- Python 3.9+
- PyTorch 1.9+
- Transformers 4.15+
- scikit-learn 1.0+
- spaCy 3.2+ (optional)

---

## Quick Start

### Python API

```python
from ai_detector import AIDetector

# Load trained model
detector = AIDetector.load("models/final")

# Analyze text
result = detector.predict("""
    Artificial intelligence has transformed how we interact with technology.
    Machine learning algorithms can now recognize patterns in data that were
    previously invisible to traditional computing methods.
""")

print(result.to_dict())
```

### Example Output

```json
{
    "classification": "likely_human",
    "ai_probability": 0.23,
    "human_probability": 0.77,
    "confidence": 0.81,
    "reliability": "high",
    "warnings": [],
    "evidence": [
        {
            "feature": "burstiness",
            "direction": "human-like",
            "strength": 0.72,
            "value": 0.65
        },
        {
            "feature": "type_token_ratio",
            "direction": "human-like", 
            "strength": 0.58,
            "value": 0.52
        }
    ],
    "num_tokens": 142,
    "num_chunks": 1
}
```

### CLI Usage

```bash
# Analyze text directly
python -m ai_detector predict --text "Your text here"

# Analyze file
python -m ai_detector predict --file essay.txt

# Batch process directory
python -m ai_detector batch --input-dir ./documents --output results.json

# With chunk analysis for long documents
python -m ai_detector predict --file long_document.txt --chunk-analysis
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Input     │────▶│ Preprocessing │────▶│   Feature   │
│    Text     │     │  & Chunking   │     │ Extraction  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
         ┌──────────────────────────────────────┤
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│ Statistical     │                   │ Neural          │
│ Features        │                   │ Representations │
│ - Perplexity    │                   │ - Transformer   │
│ - Burstiness    │                   │   Embeddings    │
│ - Vocabulary    │                   │                 │
└────────┬────────┘                   └────────┬────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│              Hybrid Classifier                       │
│   (Classical ML + Transformer Ensemble)             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Calibration Layer                       │
│   (Temperature Scaling / Isotonic Regression)       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Uncertainty Estimation                     │
│   - Entropy, OOD Detection, Ensemble Disagreement   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Explainability Module                   │
│   - Feature Attribution, Evidence Summary           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Final Output                        │
│   Classification + Probabilities + Confidence        │
└─────────────────────────────────────────────────────┘
```

---

## Training Your Own Model

### Step 1: Prepare Dataset

```bash
# You need to provide:
# - Human-written texts from diverse domains
# - AI-generated texts from multiple model families
# - Proper metadata (source, domain, AI model used)

python scripts/prepare_dataset.py
```

See `scripts/prepare_dataset.py` for detailed data schema.

### Step 2: Train Model

```bash
python scripts/train.py \
    --data-dir data/processed \
    --output models/final \
    --config configs/config.yaml
```

### Step 3: Evaluate

```bash
python scripts/evaluate.py \
    --model models/final \
    --data-dir data/processed \
    --output eval_results.json
```

### Step 4: Predict

```bash
python scripts/predict.py \
    --model models/final \
    --text "Your text here"
```

---

## Dataset Design

### Required Splits

The training pipeline creates these evaluation sets:

| Split | Purpose | Why It Matters |
|-------|---------|----------------|
| `train` | Model training | Main training data |
| `val` | Calibration, threshold tuning | Prevents overfitting |
| `test_in_dist` | In-distribution evaluation | Baseline performance |
| `test_unseen_models` | Cross-model generalization | Tests if detector recognizes AI generally vs. memorizing specific models |
| `test_unseen_topics` | Topic independence | Ensures detector doesn't learn topic markers |
| `test_paraphrased` | Robustness to evasion | Tests resilience to paraphrasing attacks |
| `test_edited` | Edited AI text | Real-world scenario where AI text is lightly edited |
| `test_short` | Short text handling | Documents behavior on <100 token texts |
| `test_long` | Long document handling | Tests chunking and aggregation |

### Data Requirements

**Human Data** (multiple domains):
- Essays (student, professional)
- News articles
- Academic writing
- Forum posts (Reddit, StackExchange)
- Personal writing (blogs)
- Technical documentation
- Creative writing

**AI Data** (multiple model families):
- GPT-3.5/GPT-4 family
- Claude family
- PaLM/Gemini family
- LLaMA family
- Other open-source models

### Critical: Proper Splitting

**DO NOT randomly split!** Split by:
- Source/document
- Author (where available)
- Generation model
- Prompt/topic

This prevents train/test contamination and gives realistic performance estimates.

---

## Evaluation Metrics

### Why Accuracy Alone Is Insufficient

| Scenario | Accuracy | FPR | Problem |
|----------|----------|-----|---------|
| Always predict "human" on balanced test | 50% | 0% | Useless detector |
| High accuracy but 10% FPR | 90% | 10% | Wrongfully accuses 1 in 10 innocent people |

**We prioritize low false-positive rate (FPR)** because false accusations cause real harm.

### Reported Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of those flagged AI, how many actually are?
- **Recall**: Of actual AI texts, how many detected?
- **F1 Score**: Harmonic mean of precision/recall
- **ROC-AUC**: Ranking quality across all thresholds
- **PR-AUC**: Precision-recall tradeoff
- **Brier Score**: Probability calibration quality
- **Expected Calibration Error (ECE)**: How well confidence matches accuracy
- **False Positive Rate**: % of human texts wrongly flagged as AI **(critical!)**
- **False Negative Rate**: % of AI texts missed

### Performance at Multiple Thresholds

The system reports metrics at thresholds [0.1, 0.2, ..., 0.9] so you can choose based on your tolerance for false positives vs. false negatives.

---

## Configuration

### Key Settings (`configs/config.yaml`)

```yaml
# Minimum tokens for reliable detection
preprocessing:
  min_tokens: 50        # Below this: warn user
  max_tokens: 4096      # Above this: chunk

# Target false positive rate
calibration:
  target_false_positive_rate: 0.01  # 1% FPR target

# Classification thresholds
thresholds:
  ai_threshold: 0.5           # Default decision boundary
  uncertainty_low: 0.3        # Below: uncertain (human-leaning)
  uncertainty_high: 0.7       # Above: uncertain (AI-leaning)
  confidence_high: 0.8        # High confidence threshold
```

---

## Understanding Results

### Classification Values

| Value | Meaning | Action |
|-------|---------|--------|
| `likely_ai` | AI probability > 0.7 | Consider AI origin possible |
| `likely_human` | AI probability < 0.3 | Probably human-written |
| `uncertain` | AI probability 0.3-0.7 OR low confidence | Cannot reliably classify |

### Reliability Levels

| Level | Confidence Range | Interpretation |
|-------|-----------------|----------------|
| `high` | ≥ 0.7 | Model is confident; results more trustworthy |
| `medium` | 0.4-0.7 | Moderate confidence; interpret with caution |
| `low` | < 0.4 | Low confidence; do not rely on classification |

### Warnings

Common warnings:
- "Text has only X tokens" - Too short for reliable detection
- "Input appears significantly different from training data" - OOD detection triggered
- "Low confidence prediction" - Model uncertain

---

## Limitations

### Known Weaknesses

1. **Short Texts**: <50 tokens provides insufficient signal
2. **Paraphrased AI**: Sophisticated paraphrasing can evade detection
3. **Unseen Models**: New AI models may have different signatures
4. **Mixed Authorship**: Documents with both human and AI sections challenge the classifier
5. **Non-Native English**: May increase false positives due to unusual patterns
6. **Highly Formal Writing**: Can resemble AI patterns

### What This Cannot Do

- ❌ Prove authorship definitively
- ❌ Detect all AI-generated text
- ❌ Work reliably on very short texts
- ❌ Handle code, math, or specialized notation well
- ❌ Replace human judgment in high-stakes scenarios

---

## Ethical Considerations

### Appropriate Uses

- ✅ Educational tool for understanding AI text characteristics
- ✅ First-pass screening with human review
- ✅ Research on AI text detection
- ✅ Personal curiosity about text origins

### Inappropriate Uses

- ❌ Sole evidence for academic misconduct
- ❌ Automated rejection of applications/submissions
- ❌ Surveillance without consent
- ❌ Discriminatory screening

### Best Practices

1. **Always include uncertainty estimates**
2. **Set conservative thresholds for high-stakes decisions**
3. **Require human review for any consequential action**
4. **Be transparent about limitations**
5. **Monitor false positive rates across demographics**

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/ai_detector --cov-report=html

# Specific test file
pytest tests/test_preprocessing.py -v
```

---

## Project Structure

```
ai_detector/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── requirements.txt        # Dependencies
├── configs/
│   └── config.yaml         # Default configuration
├── data/
│   ├── raw/                # Raw datasets (not included)
│   ├── processed/          # Processed datasets (not included)
│   └── README.md           # Data documentation
├── models/                 # Trained models (created after training)
├── src/ai_detector/
│   ├── __init__.py
│   ├── preprocessing.py    # Text preprocessing
│   ├── features.py         # Feature extraction
│   ├── classifiers.py      # ML models
│   ├── calibration.py      # Probability calibration
│   ├── inference.py        # Main detector class
│   ├── uncertainty.py      # Uncertainty estimation
│   ├── explainability.py   # Explanations
│   ├── evaluation.py       # Metrics
│   └── cli.py              # Command-line interface
├── scripts/
│   ├── prepare_dataset.py  # Data preparation
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── predict.py          # Inference script
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_inference.py
    └── test_evaluation.py
```

---

## Commands Summary

```bash
# 1. Install dependencies
pip install -e .

# 2. Prepare data (after collecting your datasets)
python scripts/prepare_dataset.py

# 3. Train model
python scripts/train.py --data-dir data/processed --output models/final

# 4. Evaluate
python scripts/evaluate.py --model models/final --data-dir data/processed

# 5. Predict
python -m ai_detector predict --text "Your text"
# or
python scripts/predict.py --model models/final --text "Your text"

# 6. Run tests
pytest tests/ -v
```

---

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Citation

If you use this in research, please cite appropriately.

---

**Remember**: This tool provides probabilistic assessments, not definitive answers. Use responsibly.
