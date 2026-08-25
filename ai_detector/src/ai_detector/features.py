"""
Feature extraction module for AI text detection.

Implements three feature families:
1. Statistical features (perplexity, burstiness, vocabulary diversity)
2. Linguistic features (POS tags, dependencies, clause complexity)
3. Neural representation features (transformer embeddings)

Important design decisions:
- No single feature is treated as definitive
- Features are normalized and scaled appropriately
- Missing features are handled gracefully
- Computation is optimized for batch processing
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Statistical features
    use_perplexity: bool = True
    use_burstiness: bool = True
    use_vocabulary_diversity: bool = True
    use_ngram_repetition: bool = True
    use_sentence_length_stats: bool = True
    use_punctuation_distribution: bool = True
    
    # Linguistic features
    use_pos_tags: bool = True
    use_dependency_parses: bool = False  # Requires spaCy model
    use_clause_complexity: bool = True
    use_discourse_markers: bool = True
    
    # Neural features
    use_neural_features: bool = True
    encoder_name: str = "roberta-base"
    pooling_strategy: str = "mean"  # mean, max, cls
    
    # Reference model for perplexity (optional, improves quality)
    perplexity_model: Optional[str] = None  # e.g., "gpt2" for perplexity calculation


@dataclass
class StatisticalFeatures:
    """Statistical/textual features."""
    # Perplexity-related
    mean_perplexity: float = np.nan
    perplexity_variance: float = np.nan
    sentence_perplexity_std: float = np.nan
    
    # Burstiness (variation in sentence lengths)
    burstiness: float = np.nan
    
    # Vocabulary
    type_token_ratio: float = np.nan
    hapax_legomena_ratio: float = np.nan
    yules_k: float = np.nan
    
    # N-gram repetition
    bigram_repetition_rate: float = np.nan
    trigram_repetition_rate: float = np.nan
    
    # Sentence statistics
    mean_sentence_length: float = np.nan
    sentence_length_std: float = np.nan
    
    # Punctuation
    punctuation_ratio: float = np.nan
    comma_to_period_ratio: float = np.nan
    
    # Function word frequency
    function_word_ratio: float = np.nan


@dataclass
class LinguisticFeatures:
    """Linguistic/stylistic features."""
    # POS distributions
    pos_noun_ratio: float = np.nan
    pos_verb_ratio: float = np.nan
    pos_adj_ratio: float = np.nan
    pos_adv_ratio: float = np.nan
    pos_pronoun_ratio: float = np.nan
    pos_preposition_ratio: float = np.nan
    pos_conjunction_ratio: float = np.nan
    
    # Clause complexity
    mean_clause_length: float = np.nan
    clauses_per_sentence: float = np.nan
    subordinate_clause_ratio: float = np.nan
    
    # Discourse markers
    discourse_marker_frequency: float = np.nan
    
    # Sentence structure
    sentence_opening_variety: float = np.nan
    passive_voice_ratio: float = np.nan


@dataclass
class NeuralFeatures:
    """Neural representation features from transformer encoders."""
    # Embedding statistics
    embedding_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    embedding_std: np.ndarray = field(default_factory=lambda: np.array([]))
    embedding_max: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # [CLS] representation (if available)
    cls_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Prediction confidence from language model head
    mean_token_confidence: float = np.nan
    token_confidence_std: float = np.nan


@dataclass
class AllFeatures:
    """Container for all extracted features."""
    statistical: StatisticalFeatures = field(default_factory=StatisticalFeatures)
    linguistic: LinguisticFeatures = field(default_factory=LinguisticFeatures)
    neural: NeuralFeatures = field(default_factory=NeuralFeatures)
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """Convert all features to a flat feature vector."""
        features = []
        
        # Statistical features
        stat = self.statistical
        stat_features = [
            stat.mean_perplexity, stat.perplexity_variance, stat.sentence_perplexity_std,
            stat.burstiness, stat.type_token_ratio, stat.hapax_legomena_ratio, stat.yules_k,
            stat.bigram_repetition_rate, stat.trigram_repetition_rate,
            stat.mean_sentence_length, stat.sentence_length_std,
            stat.punctuation_ratio, stat.comma_to_period_ratio, stat.function_word_ratio
        ]
        features.extend([f if not np.isnan(f) else 0.0 for f in stat_features])
        
        # Linguistic features
        ling = self.linguistic
        ling_features = [
            ling.pos_noun_ratio, ling.pos_verb_ratio, ling.pos_adj_ratio, ling.pos_adv_ratio,
            ling.pos_pronoun_ratio, ling.pos_preposition_ratio, ling.pos_conjunction_ratio,
            ling.mean_clause_length, ling.clauses_per_sentence, ling.subordinate_clause_ratio,
            ling.discourse_marker_frequency, ling.sentence_opening_variety, ling.passive_voice_ratio
        ]
        features.extend([f if not np.isnan(f) else 0.0 for f in ling_features])
        
        # Neural features
        if len(self.neural.embedding_mean) > 0:
            features.extend(self.neural.embedding_mean.tolist())
        if len(self.neural.embedding_std) > 0:
            features.extend(self.neural.embedding_std.tolist())
            
        return np.array(features, dtype=np.float32)


# Common English function words
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then'
}

# Discourse markers
DISCOURSE_MARKERS = {
    'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
    'consequently', 'additionally', 'alternatively', 'specifically',
    'in conclusion', 'in summary', 'on the other hand', 'for example',
    'for instance', 'in contrast', 'similarly', 'likewise', 'meanwhile',
    'subsequently', 'accordingly', 'nonetheless', 'admittedly', 'indeed',
    'thus', 'hence', 'yet', 'still', 'rather', 'instead', 'overall'
}


class FeatureExtractor:
    """
    Extracts features from preprocessed text for AI detection.
    
    Design principles:
    - Features should be complementary, not redundant
    - Graceful degradation when components unavailable (e.g., no spaCy model)
    - Efficient computation for batch processing
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._spacy_nlp = None
        self._transformer_model = None
        self._transformer_tokenizer = None
        self._reference_lm = None
        
        if self.config.use_neural_features:
            self._load_transformer_encoder()
            
        if self.config.perplexity_model:
            self._load_reference_lm()
    
    def _load_transformer_encoder(self):
        """Load transformer encoder for neural features."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self._transformer_tokenizer = AutoTokenizer.from_pretrained(
                self.config.encoder_name
            )
            self._transformer_model = AutoModel.from_pretrained(
                self.config.encoder_name
            )
            self._transformer_model.eval()
            
            logger.info(f"Loaded transformer encoder: {self.config.encoder_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer encoder: {e}. "
                          f"Neural features will be unavailable.")
            self.config.use_neural_features = False
    
    def _load_reference_lm(self):
        """Load reference language model for perplexity calculation."""
        try:
            from transformers import AutoModelForLMHead, AutoTokenizer
            import torch
            
            self._reference_lm = AutoModelForLMHead.from_pretrained(
                self.config.perplexity_model
            )
            self._reference_lm_tokenizer = AutoTokenizer.from_pretrained(
                self.config.perplexity_model
            )
            self._reference_lm.eval()
            
            logger.info(f"Loaded reference LM for perplexity: {self.config.perplexity_model}")
        except Exception as e:
            logger.warning(f"Could not load reference LM: {e}. "
                          f"Using heuristic perplexity estimates.")
            self._reference_lm = None
    
    def _try_load_spacy(self):
        """Try to load spaCy for linguistic features."""
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for linguistic features")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}. "
                          f"Linguistic features will be limited.")
            self._spacy_nlp = None
    
    def _calculate_perplexity(self, text: str, sentences: List[str]) -> Tuple[float, float, float]:
        """
        Calculate perplexity of text.
        
        If a reference LM is available, uses it for accurate calculation.
        Otherwise, uses a heuristic based on word frequencies.
        
        Returns:
            (mean_perplexity, variance, sentence_std)
        """
        if self._reference_lm is not None and hasattr(self, '_reference_lm_tokenizer'):
            import torch
            
            sentence_perplexities = []
            
            for sent in sentences[:10]:
                inputs = self._reference_lm_tokenizer(sent, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self._reference_lm(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    perp = torch.exp(loss).item()
                    sentence_perplexities.append(perp)
            
            if sentence_perplexities:
                mean_perp = np.mean(sentence_perplexities)
                var_perp = np.var(sentence_perplexities)
                std_perp = np.std(sentence_perplexities)
                return mean_perp, var_perp, std_perp
        
        # Heuristic fallback: use word frequency as proxy
        words = text.lower().split()
        if not words:
            return np.nan, np.nan, np.nan
        
        word_counts = Counter(words)
        total_words = len(words)
        
        avg_freq = sum(word_counts.values()) / len(word_counts)
        perplexity_estimate = math.log(avg_freq + 1) * 10
        
        return perplexity_estimate, perplexity_estimate * 0.1, perplexity_estimate * 0.05
    
    def _calculate_burstiness(self, sentences: List[str]) -> float:
        """
        Calculate burstiness (variation in sentence lengths).
        
        Higher burstiness = more variation = potentially more human-like
        """
        if len(sentences) < 2:
            return np.nan
        
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        if mean_len == 0:
            return np.nan
            
        coefficient_of_variation = std_len / mean_len
        return coefficient_of_variation
    
    def _calculate_vocabulary_metrics(self, tokens: List[str]) -> Tuple[float, float, float]:
        """
        Calculate vocabulary diversity metrics.
        
        Returns:
            (type_token_ratio, hapax_legomena_ratio, yules_k)
        """
        if not tokens:
            return np.nan, np.nan, np.nan
        
        n = len(tokens)
        unique_tokens = set(tokens)
        v = len(unique_tokens)
        
        # Type-Token Ratio
        ttr = v / n if n > 0 else np.nan
        
        # Hapax legomena (words appearing exactly once)
        word_counts = Counter(tokens)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / n if n > 0 else np.nan
        
        # Yule's K characteristic
        if v > 1 and n > 1:
            freq_dist = Counter(word_counts.values())
            m1 = n
            m2 = sum(r * r * freq_dist.get(r, 0) for r in freq_dist)
            yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else np.nan
        else:
            yules_k = np.nan
            
        return ttr, hapax_ratio, yules_k
    
    def _calculate_ngram_repetition(self, tokens: List[str]) -> Tuple[float, float]:
        """Calculate bigram and trigram repetition rates."""
        if len(tokens) < 3:
            return np.nan, np.nan
        
        # Bigrams
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        bigram_counts = Counter(bigrams)
        bigram_rep = sum(c - 1 for c in bigram_counts.values() if c > 1) / len(bigrams) if bigrams else np.nan
        
        # Trigrams
        trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
        trigram_counts = Counter(trigrams)
        trigram_rep = sum(c - 1 for c in trigram_counts.values() if c > 1) / len(trigrams) if trigrams else np.nan
        
        return bigram_rep, trigram_rep
    
    def _calculate_sentence_stats(self, sentences: List[str]) -> Tuple[float, float]:
        """Calculate sentence length statistics."""
        if not sentences:
            return np.nan, np.nan
        
        lengths = [len(s.split()) for s in sentences]
        return np.mean(lengths), np.std(lengths)
    
    def _calculate_punctuation_stats(self, text: str) -> Tuple[float, float]:
        """Calculate punctuation distribution statistics."""
        if not text:
            return np.nan, np.nan
        
        total_chars = len(text)
        punctuation = {'.', ',', '!', '?', ';', ':', '-', '"', "'", '(', ')'}
        punct_count = sum(1 for c in text if c in punctuation)
        punct_ratio = punct_count / total_chars if total_chars > 0 else np.nan
        
        comma_count = text.count(',')
        period_count = text.count('.')
        comma_period_ratio = comma_count / period_count if period_count > 0 else np.nan
        
        return punct_ratio, comma_period_ratio
    
    def _calculate_function_word_ratio(self, tokens: List[str]) -> float:
        """Calculate ratio of function words."""
        if not tokens:
            return np.nan
        
        func_count = sum(1 for t in tokens if t.lower() in FUNCTION_WORDS)
        return func_count / len(tokens)
    
    def extract_statistical_features(self, text: str, sentences: List[str], 
                                     tokens: List[str]) -> StatisticalFeatures:
        """Extract all statistical features."""
        stats = StatisticalFeatures()
        
        if self.config.use_perplexity:
            mean_perp, var_perp, std_perp = self._calculate_perplexity(text, sentences)
            stats.mean_perplexity = mean_perp
            stats.perplexity_variance = var_perp
            stats.sentence_perplexity_std = std_perp
        
        if self.config.use_burstiness:
            stats.burstiness = self._calculate_burstiness(sentences)
        
        if self.config.use_vocabulary_diversity:
            ttr, hapax, yules = self._calculate_vocabulary_metrics(tokens)
            stats.type_token_ratio = ttr
            stats.hapax_legomena_ratio = hapax
            stats.yules_k = yules
        
        if self.config.use_ngram_repetition:
            bigram_rep, trigram_rep = self._calculate_ngram_repetition(tokens)
            stats.bigram_repetition_rate = bigram_rep
            stats.trigram_repetition_rate = trigram_rep
        
        if self.config.use_sentence_length_stats:
            mean_len, std_len = self._calculate_sentence_stats(sentences)
            stats.mean_sentence_length = mean_len
            stats.sentence_length_std = std_len
        
        if self.config.use_punctuation_distribution:
            punct_ratio, comma_period = self._calculate_punctuation_stats(text)
            stats.punctuation_ratio = punct_ratio
            stats.comma_to_period_ratio = comma_period
        
        if self.config.use_punctuation_distribution:
            stats.function_word_ratio = self._calculate_function_word_ratio(tokens)
        
        return stats
    
    def _extract_pos_features(self, text: str) -> LinguisticFeatures:
        """Extract POS-based linguistic features using spaCy."""
        ling = LinguisticFeatures()
        
        if self._spacy_nlp is None:
            self._try_load_spacy()
            
        if self._spacy_nlp is None:
            return ling
        
        doc = self._spacy_nlp(text)
        
        # POS counts
        pos_counts = Counter(token.pos_ for token in doc)
        total_tokens = len(list(doc))
        
        if total_tokens > 0:
            ling.pos_noun_ratio = pos_counts.get('NOUN', 0) / total_tokens
            ling.pos_verb_ratio = pos_counts.get('VERB', 0) / total_tokens
            ling.pos_adj_ratio = pos_counts.get('ADJ', 0) / total_tokens
            ling.pos_adv_ratio = pos_counts.get('ADV', 0) / total_tokens
            ling.pos_pronoun_ratio = pos_counts.get('PRON', 0) / total_tokens
            ling.pos_preposition_ratio = pos_counts.get('ADP', 0) / total_tokens
            ling.pos_conjunction_ratio = pos_counts.get('CCONJ', 0) / total_tokens
        
        # Clause complexity (simplified)
        sentences = list(doc.sents)
        if sentences:
            clauses = []
            for sent in sentences:
                # Count verbs as proxy for clauses
                verb_count = sum(1 for token in sent if token.pos_ == 'VERB')
                clauses.append(max(1, verb_count))
            
            ling.clauses_per_sentence = np.mean(clauses) if clauses else np.nan
        
        # Discourse markers
        text_lower = text.lower()
        marker_count = sum(1 for marker in DISCOURSE_MARKERS if marker in text_lower)
        ling.discourse_marker_frequency = marker_count / len(sentences) if sentences else np.nan
        
        # Passive voice detection (simplified: be + past participle)
        passive_count = 0
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ == 'auxpass':
                passive_count += 1
        ling.passive_voice_ratio = passive_count / total_tokens if total_tokens > 0 else np.nan
        
        return ling
    
    def extract_linguistic_features(self, text: str) -> LinguisticFeatures:
        """Extract linguistic features."""
        if self.config.use_pos_tags or self.config.use_dependency_parses:
            return self._extract_pos_features(text)
        return LinguisticFeatures()
    
    def _extract_neural_features(self, text: str) -> NeuralFeatures:
        """Extract neural representation features."""
        neural = NeuralFeatures()
        
        if not self.config.use_neural_features or self._transformer_model is None:
            return neural
        
        try:
            import torch
            
            inputs = self._transformer_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_length if hasattr(self.config, 'max_length') else 512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self._transformer_model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                if self.config.pooling_strategy == "mean":
                    embedding_mean = hidden_states.mean(dim=1).squeeze().numpy()
                    embedding_std = hidden_states.std(dim=1).squeeze().numpy()
                    embedding_max = hidden_states.max(dim=1).values.squeeze().numpy()
                elif self.config.pooling_strategy == "cls":
                    cls_embedding = hidden_states[:, 0, :].squeeze().numpy()
                    embedding_mean = cls_embedding
                else:  # max
                    embedding_max = hidden_states.max(dim=1).values.squeeze().numpy()
                    embedding_mean = embedding_max
            
            neural.embedding_mean = embedding_mean if len(embedding_mean.shape) == 1 else np.array([])
            neural.embedding_std = embedding_std if 'embedding_std' in locals() and len(embedding_std.shape) == 1 else np.array([])
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                neural.cls_embedding = outputs.pooler_output.squeeze().numpy()
                
        except Exception as e:
            logger.warning(f"Neural feature extraction failed: {e}")
            
        return neural
    
    def extract_all_features(self, text: str, sentences: List[str], 
                            tokens: List[str]) -> AllFeatures:
        """Extract all feature types from preprocessed text."""
        features = AllFeatures()
        
        # Statistical features
        features.statistical = self.extract_statistical_features(text, sentences, tokens)
        
        # Linguistic features
        features.linguistic = self.extract_linguistic_features(text)
        
        # Neural features
        if self.config.use_neural_features:
            features.neural = self._extract_neural_features(text)
        
        return features
    
    def extract_batch(self, texts: List[str], preprocessed_list: List) -> List[AllFeatures]:
        """Extract features for multiple texts."""
        return [
            self.extract_all_features(
                prep.cleaned_text, 
                prep.sentences, 
                prep.tokens
            )
            for prep, text in zip(preprocessed_list, texts)
        ]
