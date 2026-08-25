"""
Text preprocessing module for AI text detection.

Handles:
- Unicode normalization
- Text cleaning
- Sentence segmentation
- Tokenization
- Length validation
- Document chunking
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    min_tokens: int = 50
    max_tokens: int = 4096
    chunk_size: int = 512
    chunk_overlap: int = 50
    language: str = "en"
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    

@dataclass
class PreprocessedText:
    """Result of text preprocessing."""
    original_text: str
    cleaned_text: str
    sentences: List[str]
    tokens: List[str]
    num_tokens: int
    num_sentences: int
    is_valid: bool
    chunks: List[str]
    warnings: List[str]


class TextPreprocessor:
    """
    Preprocesses text for AI detection.
    
    Important design decisions:
    - We preserve as much original text as possible (minimal cleaning)
    - We track warnings for edge cases (short text, unusual characters)
    - We support chunking for long documents
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self._sentence_pattern = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s',
            re.UNICODE
        )
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize Unicode and clean text while preserving meaning.
        
        We avoid aggressive cleaning that might remove informative signals.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
            
        if self.config.normalize_unicode:
            # Normalize to NFC form (canonical decomposition followed by canonical composition)
            text = unicodedata.normalize('NFC', text)
            
        if self.config.remove_extra_whitespace:
            # Replace multiple whitespace with single space, preserve newlines
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple heuristic.
        
        For more accurate counts, use the actual tokenizer from the transformer model.
        This is a fast approximation for preprocessing decisions.
        """
        # Simple word-based tokenization for estimation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return len(tokens)
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses regex-based segmentation. For production, consider using
        a more sophisticated segmenter like spaCy's sentencizer.
        """
        # Handle common abbreviations that shouldn't end sentences
        abbreviations = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 
                        'vs', 'etc', 'e.g', 'i.e', 'cf', 'al', 'fig'}
        
        # Split on sentence boundaries
        raw_sentences = self._sentence_pattern.split(text)
        
        # Clean and filter empty sentences
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Filter very short fragments
                sentences.append(sent)
                
        return sentences
    
    def create_chunks(self, text: str, strategy: str = "sentence") -> List[str]:
        """
        Split long documents into overlapping chunks.
        
        Args:
            text: Input text
            strategy: Chunking strategy - "sentence" or "token"
            
        Returns:
            List of text chunks
            
        Strategy explanation:
        - Sentence-based: Keeps sentences intact, better for coherence
        - Token-based: More precise size control, may split sentences
        """
        token_count = self.count_tokens(text)
        
        if token_count <= self.config.max_tokens:
            return [text]
            
        chunks = []
        
        if strategy == "sentence":
            sentences = self.segment_sentences(text)
            current_chunk = []
            current_tokens = 0
            
            for sent in sentences:
                sent_tokens = self.count_tokens(sent)
                
                if current_tokens + sent_tokens > self.config.chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
        else:  # token-based (simplified)
            words = text.split()
            chunk_words = []
            
            for i, word in enumerate(words):
                chunk_words.append(word)
                
                if len(chunk_words) >= self.config.chunk_size:
                    chunks.append(' '.join(chunk_words))
                    # Overlap: keep last portion for next chunk
                    overlap_size = min(self.config.chunk_overlap, len(chunk_words) // 4)
                    chunk_words = chunk_words[-overlap_size:]
                    
            if chunk_words:
                chunks.append(' '.join(chunk_words))
                
        return chunks
    
    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text meets minimum requirements for reliable detection.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True
        
        if not text or not text.strip():
            return False, ["Empty text provided"]
            
        token_count = self.count_tokens(text)
        
        if token_count < self.config.min_tokens:
            warnings.append(
                f"Text has only {token_count} tokens (minimum recommended: {self.config.min_tokens}). "
                "Detection reliability will be low."
            )
            # Don't mark as invalid, just warn
            
        if token_count > self.config.max_tokens:
            warnings.append(
                f"Text has {token_count} tokens, exceeding maximum ({self.config.max_tokens}). "
                "Will be processed in chunks."
            )
            
        # Check for unusual character distributions
        char_counts = {}
        for char in text:
            category = unicodedata.category(char)
            char_counts[category] = char_counts.get(category, 0) + 1
            
        total_chars = sum(char_counts.values())
        if total_chars > 0:
            # High proportion of non-letter characters might indicate code, math, etc.
            letter_chars = char_counts.get('Ll', 0) + char_counts.get('Lu', 0)
            if letter_chars / total_chars < 0.5:
                warnings.append(
                    "Text contains high proportion of non-letter characters. "
                    "May be code, mathematical notation, or other special content."
                )
                
        return is_valid, warnings
    
    def preprocess(self, text: str) -> PreprocessedText:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            PreprocessedText object with all preprocessing results
        """
        warnings = []
        
        # Validate first
        is_valid, validation_warnings = self.validate_text(text)
        warnings.extend(validation_warnings)
        
        # Normalize
        try:
            cleaned = self.normalize_text(text)
        except Exception as e:
            logger.error(f"Unicode normalization failed: {e}")
            cleaned = text
            warnings.append("Unicode normalization failed, using original text")
            
        # Segment sentences
        sentences = self.segment_sentences(cleaned)
        
        # Tokenize (simple word tokenization)
        tokens = re.findall(r'\b\w+\b', cleaned.lower())
        
        # Create chunks if needed
        chunks = self.create_chunks(cleaned)
        
        return PreprocessedText(
            original_text=text,
            cleaned_text=cleaned,
            sentences=sentences,
            tokens=tokens,
            num_tokens=len(tokens),
            num_sentences=len(sentences),
            is_valid=is_valid,
            chunks=chunks,
            warnings=warnings
        )
    
    def preprocess_batch(self, texts: List[str]) -> List[PreprocessedText]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]
