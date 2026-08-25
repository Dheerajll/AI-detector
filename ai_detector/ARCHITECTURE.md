# AI Text Detector: Architecture and Scientific Foundations

## Scientific Requirements and Challenges

### 1. Why AI Detection is Fundamentally Difficult

AI text detection faces several inherent challenges:

- **Convergent Evolution**: Both humans and AIs learn from similar corpora, leading to overlapping statistical patterns
- **Rapid Model Improvement**: Newer models produce increasingly human-like text, making detectors trained on older models obsolete
- **Style Mimicry**: AIs can be prompted to imitate specific writing styles, including informal, error-prone, or domain-specific writing
- **Human Variability**: Human writing spans enormous diversity—from highly formal academic prose to casual chat—making a single "human signature" impossible
- **Distribution Shift**: The distribution of AI text changes as models improve, while human writing remains relatively stable but diverse

**Implication**: Detection must be treated as a moving target requiring continuous updates and validation.

### 2. Why Perplexity Alone is Insufficient

Perplexity measures how "surprising" text is to a language model, but:

- **Model Dependency**: Perplexity values depend entirely on the reference LM used
- **Domain Sensitivity**: Technical writing naturally has lower perplexity (more predictable terminology)
- **Non-Native Writing**: ESL writers may produce text with unusual perplexity patterns unrelated to AI generation
- **Editing Effects**: Light human editing can significantly alter perplexity without changing authorship
- **Calibration Issues**: Raw perplexity scores don't map cleanly to probabilities

**Our Approach**: Use perplexity as ONE feature among many, never as a sole determinant.

### 3. Why Classifiers Learn Dataset/Model Artifacts

Classifiers can exploit spurious correlations:

- **Prompt Artifacts**: AI text often contains patterns from common prompts ("In conclusion", "Let's explore")
- **Model-Specific Tokens**: Different models have characteristic token distributions
- **Training Contamination**: If train/test splits aren't carefully designed, classifiers memorize rather than generalize
- **Topic Confounding**: If AI text covers different topics than human text, the classifier learns topic markers

**Our Mitigation**: 
- Split by source/author/model, not randomly
- Include diverse domains in both classes
- Test on unseen models and topics

### 4. Why False Positives are Especially Important

False positives (labeling human text as AI) have severe consequences:

- **Academic Integrity**: Wrongful accusations against students
- **Professional Reputation**: Mislabeling professional writers
- **Bias**: Non-native speakers, neurodivergent writers, and certain demographics may be disproportionately affected
- **Trust Erosion**: High false-positive rates destroy confidence in the system

**Our Priority**: Optimize for low false-positive rates even at the cost of some false negatives. Provide uncertainty estimates rather than forcing binary decisions.

### 5. Why Paraphrasing Degrades Detection

Paraphrasing attacks work because:

- **Surface Pattern Disruption**: Many features (n-grams, punctuation patterns) change with paraphrasing
- **Semantic Preservation**: The underlying meaning stays the same while surface statistics shift
- **Tool Availability**: Easy-to-use paraphrasing tools can systematically evade simple detectors

**Our Defense**: Focus on deeper linguistic patterns (syntax, discourse structure) that are harder to paraphrase away. Use ensemble methods combining multiple feature types.

### 6. Why Unseen-Model Evaluation is Necessary

Testing only on training-distribution AI models leads to:

- **Overfitting**: Detector learns specific model artifacts rather than general AI patterns
- **False Confidence**: High accuracy on known models doesn't predict real-world performance
- **Obsolescence**: New models immediately evade the detector

**Our Protocol**: 
- Train on models A, B, C
- Test on models D, E (held out)
- Report separate metrics for seen vs. unseen models

### 7. Why Probability Calibration Matters

Uncalibrated probabilities are misleading:

- **Overconfidence**: Neural networks often output 0.99 for wrong predictions
- **Decision Thresholds**: Without calibration, threshold selection is arbitrary
- **Risk Assessment**: Users need accurate probability estimates to make informed decisions

**Our Solution**: 
- Temperature scaling for neural models
- Isotonic regression for classical models
- Validation via reliability diagrams and Expected Calibration Error (ECE)

### 8. Why No Detector Should Claim Absolute Certainty

Fundamental limitations:

- **Probabilistic Nature**: Authorship attribution is inherently uncertain
- **Evidence Quality**: Short texts provide insufficient evidence
- **Adversarial Adaptation**: Bad actors will find ways to evade detection
- **Ethical Responsibility**: False certainty leads to harmful decisions

**Our Stance**: Always report uncertainty, provide confidence intervals, and explicitly warn when evidence is weak.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                        │
│  - Unicode normalization                                        │
│  - Language detection                                           │
│  - Length validation                                            │
│  - Chunking for long documents                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                             │
│  ┌─────────────┬──────────────┬─────────────────────────────┐   │
│  │ Statistical │ Linguistic   │ Neural Representations      │   │
│  │ - Perplexity│ - POS tags   │ - Transformer embeddings    │   │
│  │ - Burstiness│ - Dependencies│ - [CLS] representations    │   │
│  │ - Vocabulary│ - Clause complexity│ - Attention patterns   │   │
│  │ - N-grams   │ - Discourse  │                             │   │
│  └─────────────┴──────────────┴─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID CLASSIFIER                             │
│  ┌─────────────┬──────────────┬─────────────────────────────┐   │
│  │ Classical   │ Transformer  │ Ensemble Combination        │   │
│  │ (RF/GBM)    │ Encoder      │ (Stacking/Voting)           │   │
│  └─────────────┴──────────────┴─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CALIBRATION LAYER                              │
│  - Temperature scaling / Isotonic regression                     │
│  - Reliability validation                                        │
│  - Calibrated probability outputs                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   UNCERTAINTY ESTIMATION                         │
│  - Ensemble disagreement                                        │
│  - OOD detection                                                │
│  - Length-based warnings                                        │
│  - Confidence scoring                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY MODULE                         │
│  - SHAP values for feature importance                           │
│  - Chunk-level analysis                                         │
│  - Evidence summary                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                      │
│  {                                                               │
│    "classification": "likely_ai" | "likely_human" | "uncertain",│
│    "ai_probability": 0.0-1.0,                                    │
│    "human_probability": 0.0-1.0,                                 │
│    "confidence": 0.0-1.0,                                        │
│    "reliability": "high" | "medium" | "low",                     │
│    "warnings": [...],                                            │
│    "evidence": [...]                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Overview

### Preprocessing (`preprocessing.py`)
- Text normalization (Unicode, whitespace)
- Language detection (filter non-English)
- Length validation and chunking strategy
- Sentence segmentation

### Feature Extraction (`features.py`)
- **Statistical**: Perplexity, burstiness, vocabulary diversity, n-gram repetition
- **Linguistic**: POS distributions, dependency patterns, clause complexity (using spaCy)
- **Neural**: Transformer encoder representations (RoBERTa/DeBERTa)

### Classifiers (`classifiers.py`)
- **Baseline**: Logistic Regression, Random Forest, Gradient Boosting
- **Neural**: Fine-tuned transformer classifier
- **Hybrid**: Stacked ensemble combining all approaches

### Calibration (`calibration.py`)
- Temperature scaling for neural models
- Isotonic regression for classical models
- Reliability diagram generation
- Expected Calibration Error computation

### Uncertainty (`uncertainty.py`)
- Ensemble disagreement measurement
- Out-of-distribution detection
- Confidence interval estimation
- Length-based reliability warnings

### Explainability (`explainability.py`)
- SHAP integration for feature attribution
- Chunk-level probability visualization
- Evidence aggregation and summarization

### Inference (`inference.py`)
- Unified prediction API
- Batch processing support
- CPU/GPU mode selection
- Caching for efficiency

### Evaluation (`evaluation.py`)
- Comprehensive metrics (precision, recall, F1, ROC-AUC, PR-AUC)
- Calibration metrics (Brier score, ECE)
- Stratified analysis by domain, length, model type
- Adversarial robustness testing

## Data Flow

1. **Training Phase**:
   ```
   Raw Data → Preprocessing → Feature Extraction → Train/Val/Test Split
       → Model Training → Calibration → Evaluation → Saved Model
   ```

2. **Inference Phase**:
   ```
   Input Text → Preprocessing → Feature Extraction → Classification
       → Calibration → Uncertainty Estimation → Explanation → Output
   ```

## Design Decisions

### Why Hybrid Architecture?
- Statistical features capture surface patterns efficiently
- Linguistic features provide deeper structural analysis
- Neural features capture semantic and contextual patterns
- Ensemble reduces over-reliance on any single signal

### Why Multiple Evaluation Splits?
- In-distribution: Baseline performance
- Unseen models: Generalization capability
- Unseen topics: Topic independence
- Paraphrased text: Robustness to evasion
- Mixed authorship: Real-world applicability

### Why Explicit Uncertainty?
- Prevents overconfident wrong predictions
- Guides users on when to trust the output
- Enables human-in-the-loop workflows
- Ethically responsible design

### Why Local Processing?
- Privacy preservation
- No API costs or rate limits
- Reproducibility
- Offline capability
