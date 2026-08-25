# AI Text Detection Dataset

This directory contains datasets for training and evaluating the AI text detector.

## Overview

The dataset is designed to train a robust AI-generated text detector that generalizes across:
- Multiple AI model families (GPT, Claude, Llama, etc.)
- Multiple domains (essays, news, academic, forums, technical, creative)
- Various editing levels (raw AI, lightly edited, heavily edited, mixed human/AI)

## Directory Structure

```
data/
├── raw/                    # Raw downloaded datasets
│   ├── human/              # Human-written text sources
│   │   ├── essays/
│   │   ├── news/
│   │   ├── academic/
│   │   ├── forums/
│   │   ├── technical/
│   │   └── creative/
│   └── ai/                 # AI-generated text sources
│       ├── gpt3/
│       ├── gpt4/
│       ├── claude/
│       ├── llama/
│       └── other_models/
├── processed/              # Processed and tokenized datasets
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test_in_dist.parquet
│   ├── test_unseen_models.parquet
│   ├── test_unseen_topics.parquet
│   ├── test_paraphrased.parquet
│   └── test_mixed.parquet
└── README.md               # This file
```

## Dataset Sources

### Human-Written Text

| Source | Domain | License | Notes |
|--------|--------|---------|-------|
| [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) | Web articles | MIT | Reddit-shared URLs |
| [RealNews](https://github.com/yhcc/RealNewsDataset) | News | CC-BY | High-quality news |
| [WritingPrompts](https://www.kaggle.com/datasets/writingprompts) | Creative | Public domain | Reddit writing prompts |
| [StackExchange](https://huggingface.co/datasets/stackexchange) | Q&A/Technical | CC-BY | Technical discussions |
| [arXiv Abstracts](https://huggingface.co/datasets/arxiv) | Academic | MIT | Scientific abstracts |
| [Student Essays](https://huggingface.co/datasets/student_essays) | Essays | Educational | Student writing samples |
| [Enron Emails](https://huggingface.co/datasets/trec_enron) | Personal/Business | Public domain | Email corpus |

### AI-Generated Text

| Source | Model Family | License | Notes |
|--------|--------------|---------|-------|
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | GPT-3 | CC-BY-NC | Human-ChatGPT Comparison |
| [M4GT](https://huggingface.co/datasets/skrishna/gpt4chan) | GPT-J/GPT-Neo | MIT | 4chan-style generations |
| [OpenAI Detections](https://huggingface.co/datasets/cais/mald) | Various | MIT | Multi-model AI detections |
| [Self-Detection](https://huggingface.co/datasets/ufal/aigc_detection) | Multiple | CC-BY | Self-generated detection data |
| Custom generations | GPT-4, Claude, Llama | - | See generation scripts |

## Data Splits

The dataset is split by **source/document/author** to prevent contamination:

### Training Set
- Human text from known domains
- AI text from GPT-3, GPT-J, early models
- Balanced across domains

### Validation Set
- Same distribution as training
- Used for hyperparameter tuning and calibration

### Test Sets

1. **In-Distribution (test_in_dist.parquet)**
   - Same domains and AI models as training
   - Measures baseline performance

2. **Unseen AI Models (test_unseen_models.parquet)**
   - AI text from models NOT in training (e.g., GPT-4, Claude, newer Llama)
   - Tests cross-model generalization

3. **Unseen Topics (test_unseen_topics.parquet)**
   - Topics/domains not seen during training
   - Tests topic generalization

4. **Unseen Human Authors (test_unseen_authors.parquet)**
   - Human text from authors not in training
   - Tests author generalization

5. **Paraphrased AI (test_paraphrased.parquet)**
   - AI text that has been paraphrased or edited
   - Tests robustness to editing

6. **Mixed Authorship (test_mixed.parquet)**
   - Documents with both human and AI sections
   - Tests chunk-level detection

7. **Short Text (test_short.parquet)**
   - Text under 100 tokens
   - Tests length sensitivity

8. **Long Text (test_long.parquet)**
   - Text over 2000 tokens
   - Tests document-level aggregation

## Download Instructions

### Option 1: Automatic Download (Recommended)

Run the preparation script which will download datasets automatically:

```bash
cd ai_detector
pip install -r requirements.txt
python scripts/prepare_dataset.py --download-all
```

### Option 2: Manual Download

Download each dataset manually and place in `data/raw/`:

```bash
# Example for HC3 dataset
cd data/raw
git lfs install
git clone https://huggingface.co/datasets/Hello-SimpleAI/HC3
mv HC3 ai/

# Example for RealNews
git clone https://huggingface.co/datasets/yhcc/RealNews
mv RealNews human/news/
```

### Option 3: Use Pre-processed HuggingFace Datasets

Some datasets are available pre-processed on HuggingFace:

```python
from datasets import load_dataset

# Load combined dataset
dataset = load_dataset("your-org/ai-detection-corpus")
```

## Dataset Schema

Each processed sample contains:

```json
{
    "text": "The full text content...",
    "label": 0,  // 0=human, 1=AI
    "source_type": "human",  // or "ai"
    "domain": "essays",  // essays, news, academic, forums, technical, creative
    "model_family": null,  // For AI: gpt3, gpt4, claude, llama, etc.
    "author_id": "author_123",  // Anonymous author identifier
    "document_id": "doc_456",  // Document identifier for splitting
    "topic": "technology",  // Topic category
    "edit_level": "raw",  // raw, light_edit, heavy_edit, mixed
    "word_count": 250,
    "split": "train"  // train, validation, test_*
}
```

## Generating Custom AI Data

To generate custom AI training data:

```bash
python scripts/generate_ai_data.py \
    --models gpt-3.5-turbo,gpt-4,claude-2,llama-2-70b \
    --prompts-file prompts.jsonl \
    --output-dir data/raw/ai/custom \
    --samples-per-prompt 3
```

Requires API keys for commercial models.

## Data Statistics

After processing, run:

```bash
python scripts/analyze_dataset.py --input data/processed/train.parquet
```

Expected statistics:
- Total samples: ~500,000
- Human/AI balance: 50/50
- Domain distribution: balanced
- Average word count: 200-500 words

## Important Notes

1. **No Train/Test Contamination**: Documents from the same source/author are kept together in one split.

2. **Cross-Model Evaluation**: Always evaluate on AI models not seen during training.

3. **False Positive Focus**: Pay special attention to false positives on human writing, especially:
   - Non-native English
   - Formal/academic writing
   - Technical writing
   - Highly edited text

4. **Ethical Considerations**: 
   - Do not use for high-stakes decisions without human review
   - Acknowledge uncertainty in predictions
   - Avoid bias against specific writing styles

## Citation

If using this dataset for research, please cite the original sources appropriately.

## License

Individual datasets retain their original licenses. The combined processed dataset is provided under CC-BY-NC-SA 4.0 for research purposes.
