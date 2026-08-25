# AI Detector - Complete Usage Guide

This guide explains how to use the AI text detection framework, from quick start with pre-trained models to training your own custom model.

---

## Table of Contents

1. [Quick Start with Pre-trained Models](#quick-start-with-pre-trained-models)
2. [Training Your Own Model](#training-your-own-model)
3. [Dataset Details](#dataset-details)
4. [Model Architecture](#model-architecture)
5. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
6. [PDF Analysis](#pdf-analysis)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start with Pre-trained Models

### Option A: Download Pre-trained Model (Recommended)

For most users, downloading a pre-trained model is the fastest way to get started:

```bash
# 1. Install the package
cd ai_detector
pip install -e .

# 2. Download spaCy language model
python -m spacy download en_core_web_sm

# 3. List available pre-trained models
python scripts/download_model.py --list

# 4. Download the base model
python scripts/download_model.py --model ai-detector-base

# 5. Use the model
python -m ai_detector predict --text "This is my sample text to analyze."
```

**Available Pre-trained Models:**

| Model | Size | Best For |
|-------|------|----------|
| `ai-detector-base` | ~450 MB | General use, fast inference |
| `ai-detector-large` | ~1.2 GB | Maximum accuracy, research |
| `ai-detector-multilingual` | ~1.1 GB | Non-English text detection |

### Python API Example

```python
from ai_detector import AIDetector

# Load pre-trained model
detector = AIDetector.load("models/pretrained/ai-detector-base")

# Analyze single text
result = detector.predict("""
    The rapid advancement of artificial intelligence has transformed numerous industries.
    Machine learning algorithms now power everything from recommendation systems to 
    autonomous vehicles, fundamentally changing how we interact with technology.
""")

print(f"AI Probability: {result.ai_probability:.2%}")
print(f"Classification: {result.classification}")
print(f"Confidence: {result.confidence:.2%}")

# Analyze PDF document
from ai_detector import PDFAnalyzer

pdf_analyzer = PDFAnalyzer(ai_detector=detector)
pdf_result = pdf_analyzer.analyze("documents/thesis.pdf", run_ai_detection=True)

print(f"\nOverall AI Probability: {pdf_result.overall_ai_probability:.2%}")
for section in pdf_result.section_classifications[:5]:
    print(f"  {section['section_title']}: {section['ai_probability']:.2%}")
```

---

## Training Your Own Model

### Step 1: Prepare Your Dataset

The framework automatically downloads and processes multiple datasets:

```bash
# Download all datasets (requires internet, ~10-50 GB depending on sources)
python scripts/prepare_dataset.py --download-all

# This downloads:
# - Human texts: OpenWebText, RealNews, WritingPrompts, StackExchange, arXiv, student essays
# - AI texts: HC3 (ChatGPT), M4GT (GPT-J), MALD, AIGC Detection

# Verify no train/test contamination
python scripts/prepare_dataset.py --verify-splits

# View dataset statistics
python scripts/analyze_dataset.py --input data/processed/train.parquet
```

**Expected Output:**
```
=== Dataset Statistics ===
Training set: 150000 samples
  - Human: 75000
  - AI: 75000
Validation set: 30000 samples
Test set (in-dist): 35000 samples

Average word count: 285.3
Domains: ['web', 'news', 'creative', 'technical', 'academic', 'essays']
Model families: ['gpt3.5', 'gptj', 'gpt4', 'claude']
```

### Step 2: Train the Model

```bash
# Train hybrid model (recommended configuration)
python scripts/train.py \
    --data-dir data/processed \
    --output-dir models/final \
    --model-type hybrid \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --transformer-model roberta-base \
    --use-transformer \
    --use-statistical \
    --use-linguistic \
    --calibration-method isotonic

# Training time: ~2-6 hours on GPU, ~12-24 hours on CPU
```

**Training Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-type` | hybrid | `hybrid`, `transformer`, or `sklearn` |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 32 | Batch size (reduce if OOM) |
| `--transformer-model` | roberta-base | Base transformer model |
| `--calibration-method` | isotonic | `isotonic`, `platt`, or `temperature` |

### Step 3: Evaluate the Model

```bash
# Run comprehensive evaluation
python scripts/evaluate.py \
    --model-dir models/final \
    --test-dir data/processed \
    --output results/evaluation.json \
    --detailed

# Generate visualization reports
python scripts/evaluate.py \
    --model-dir models/final \
    --test-dir data/processed \
    --output-dir results/ \
    --plots
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Brier Score (calibration quality)
- Expected Calibration Error (ECE)
- False Positive Rate at various thresholds
- Performance by domain, model family, text length

### Step 4: Share Your Model (Optional)

```bash
# Login to HuggingFace
huggingface-cli login

# Upload your trained model
python scripts/upload_model.py \
    --model-dir models/final \
    --repo-id your-username/ai-detector-custom \
    --public

# Your model is now available at:
# https://huggingface.co/your-username/ai-detector-custom
```

---

## Dataset Details

### Data Sources

#### Human-Written Text
| Source | Domain | Samples | License |
|--------|--------|---------|---------|
| OpenWebText | Web articles | ~50K | MIT |
| RealNews | News | ~30K | CC-BY |
| WritingPrompts | Creative | ~25K | Public domain |
| StackExchange | Technical Q&A | ~25K | CC-BY |
| arXiv Abstracts | Academic | ~20K | MIT |
| Student Essays | Essays | ~25K | Educational |

#### AI-Generated Text
| Source | Model | Samples | License |
|--------|-------|---------|---------|
| HC3 | GPT-3.5 | ~40K | CC-BY-NC |
| M4GT | GPT-J | ~35K | MIT |
| MALD | Multiple | ~40K | MIT |
| AIGC Detection | Various | ~35K | CC-BY |

### Data Splits

The dataset uses **group-based splitting** to prevent contamination:

- Documents from the same source/author stay together
- No overlapping content between train/val/test
- Specialized test sets for generalization evaluation

| Split | Purpose | Samples |
|-------|---------|---------|
| Train | Model training | 150K |
| Validation | Hyperparameter tuning | 30K |
| Calibration | Probability calibration | 30K |
| Test (in-dist) | Baseline evaluation | 35K |
| Test (unseen models) | Cross-model generalization | 15K |
| Test (by domain) | Domain-specific performance | varies |
| Test (short/long) | Length sensitivity | varies |

---

## Model Architecture

### Hybrid Model (Default)

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Text                              │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Statistical   │ │   Linguistic    │ │    Neural       │
│    Features     │ │    Features     │ │  Representations│
│                 │ │                 │ │                 │
│ • Perplexity    │ │ • POS patterns  │ │ • RoBERTa/      │
│ • Burstiness    │ │ • Syntax trees  │ │   DeBERTa       │
│ • Sentence len  │ │ • Dependencies  │ │   embeddings    │
│ • Vocabulary    │ │ • Discourse     │ │                 │
│ • Repetition    │ │ • Markers       │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                  ┌───────────────────┐
                  │  Fusion Layer     │
                  │  (Concat + MLP)   │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Classification    │
                  │ Head               │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  Calibration      │
                  │  (Isotonic/Platt) │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  AI Probability   │
                  │  + Uncertainty    │
                  └───────────────────┘
```

### Feature Families

**Statistical Features (32 dimensions):**
- Token-level log probability
- Perplexity (overall and sentence-level)
- Perplexity variance
- Burstiness measures
- Sentence length distribution
- Token length distribution
- Vocabulary diversity (TTR, MTLD)
- Repeated n-grams
- Punctuation distribution
- Function word frequencies

**Linguistic Features (48 dimensions):**
- POS tag distributions
- Syntactic complexity metrics
- Dependency tree statistics
- Clause structure patterns
- Discourse marker usage
- Sentence opening patterns
- Paragraph structure

**Neural Representations (768 dimensions for RoBERTa-base):**
- [CLS] token embedding
- Mean pooling of token embeddings
- Attention-weighted representations

---

## Evaluation and Benchmarking

### Running Benchmarks

```bash
# Full evaluation suite
python scripts/evaluate.py \
    --model-dir models/final \
    --test-dir data/processed \
    --output results/full_eval.json \
    --detailed \
    --plots

# Specific test set
python scripts/evaluate.py \
    --model-dir models/final \
    --test-file data/processed/test_unseen_models.parquet \
    --output results/unseen_models.json
```

### Key Metrics Explained

| Metric | What It Measures | Target |
|--------|------------------|--------|
| ROC-AUC | Overall discriminative ability | >0.90 |
| PR-AUC | Performance on imbalanced data | >0.85 |
| Brier Score | Probability calibration | <0.15 |
| ECE | Expected calibration error | <0.05 |
| FPR@1%TPR | False positives at low threshold | <0.10 |

### Important Considerations

**Why Accuracy Is Misleading:**
- A model can have 90% accuracy but still produce many false accusations
- False positives are more harmful than false negatives in this context
- Calibration matters more than raw accuracy

**Focus on False Positive Rate:**
- Wrongful accusations damage credibility
- Configure thresholds based on acceptable FPR
- Use uncertainty estimates for borderline cases

---

## PDF Analysis

### Analyzing Academic Documents

```python
from ai_detector import PDFAnalyzer, AIDetector

# Initialize
detector = AIDetector.load("models/final")
analyzer = PDFAnalyzer(ai_detector=detector)

# Analyze a thesis or report
result = analyzer.analyze(
    "documents/phd_thesis.pdf",
    run_ai_detection=True,
    min_chunk_words=100
)

# Results
print(f"Document: {result.metadata.get('filename', 'Unknown')}")
print(f"Total pages: {result.metadata.get('page_count', 0)}")
print(f"Word count: {result.word_count}")
print(f"Overall AI probability: {result.overall_ai_probability:.2%}")

# Section-by-section analysis
print("\nSection Classifications:")
for section in result.section_classifications:
    flag = "⚠️" if section['ai_probability'] > 0.7 else ""
    print(f"  {section['section_title']} ({section['word_count']} words): "
          f"{section['ai_probability']:.2%} {flag}")

# Structural evidence
print("\nStructural Evidence:")
for evidence in result.evidence:
    print(f"  • {evidence['feature']}: {evidence['direction']} "
          f"(strength: {evidence['strength']:.2f})")
```

### PDF-Specific Features

The PDF analyzer examines:

1. **Document Structure**
   - Section/chapter detection
   - Heading consistency
   - Academic structure completeness

2. **Citation Analysis**
   - Citation density
   - Format consistency
   - Reference list presence

3. **Technical Depth**
   - Specific numbers and statistics
   - Method descriptions
   - Figure/table mentions

4. **Flow Coherence**
   - Topic transitions between sections
   - Generic phrase detection
   - Cross-section repetition

---

## API Reference

### AIDetector Class

```python
from ai_detector import AIDetector, PredictionResult

# Load model
detector = AIDetector.load(path: str)

# Single prediction
result: PredictionResult = detector.predict(
    text: str,
    return_evidence: bool = True,
    chunk_long_texts: bool = True
)

# Batch prediction
results: List[PredictionResult] = detector.predict_batch(
    texts: List[str],
    batch_size: int = 32
)

# PredictionResult attributes
result.classification      # "likely_ai", "likely_human", "uncertain"
result.ai_probability      # float (0.0-1.0)
result.human_probability   # float (0.0-1.0)
result.confidence          # float (0.0-1.0)
result.reliability         # "high", "medium", "low"
result.warnings            # List[str]
result.evidence            # List[Dict]
result.chunk_results       # List[Dict] (for long texts)
```

### PDFAnalyzer Class

```python
from ai_detector import PDFAnalyzer

analyzer = PDFAnalyzer(
    ai_detector: AIDetector,
    extractor_backend: str = "pymupdf"  # or "pdfplumber", "pypdf2"
)

result = analyzer.analyze(
    pdf_path: str,
    run_ai_detection: bool = True,
    min_chunk_words: int = 100,
    detect_sections: bool = True
)

# PDFAnalysisResult attributes
result.overall_ai_probability
result.section_classifications
result.structural_evidence
result.flow_analysis
result.metadata
```

---

## Troubleshooting

### Common Issues

**1. "No module named 'ai_detector'"**
```bash
# Ensure you're in the project directory and installed correctly
cd ai_detector
pip install -e .
```

**2. "spaCy model not found"**
```bash
python -m spacy download en_core_web_sm
```

**3. CUDA out of memory during training**
```bash
# Reduce batch size
python scripts/train.py --batch-size 16 ...

# Or use CPU-only
python scripts/train.py --device cpu ...
```

**4. Dataset download fails**
```bash
# Check internet connection
# Some datasets may require HuggingFace login
huggingface-cli login

# Try downloading individual datasets
python scripts/prepare_dataset.py --dataset hc3
```

**5. Model produces all uncertain predictions**
- Check if text is too short (<50 words)
- Verify model was trained properly
- Adjust uncertainty thresholds in config

### Getting Help

- Documentation: See README.md
- Issues: GitHub Issues
- Model questions: Check data/README.md for dataset details

---

## Best Practices

1. **Always check uncertainty**: High uncertainty means the prediction is unreliable
2. **Use for screening, not decisions**: Combine with human review
3. **Consider text length**: Very short texts are hard to classify reliably
4. **Watch for false positives**: Formal writing and non-native English may trigger false alarms
5. **Update regularly**: New AI models may evade older detectors

---

## Ethical Guidelines

⚠️ **Important Reminders:**

- This tool provides probabilistic assessments, NOT proof
- Never use alone for academic integrity decisions
- Acknowledge limitations when reporting results
- Be aware of potential bias against non-native speakers
- Consider privacy implications of text analysis

---

*Last updated: 2024*
