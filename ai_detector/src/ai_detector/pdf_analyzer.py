"""
PDF Document Analyzer for AI Detection.

This module provides capabilities to:
1. Extract text from PDF documents (educational reports, theses, etc.)
2. Analyze document structure (sections, chapters, headings)
3. Detect document flow and coherence patterns
4. Identify structural patterns common in AI-generated academic writing
5. Integrate with the main AI detector for document-level analysis

Important notes:
- PDF extraction quality depends on the PDF format (text-based vs scanned)
- Structural analysis helps detect AI-generated academic writing patterns
- Section-level analysis can identify mixed authorship within documents
- This is not proof of AI authorship, only probabilistic indicators
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    raw_text: str
    metadata: Dict[str, Any]
    pages: List[str]
    is_extractable: bool
    warnings: List[str]
    file_path: str
    

@dataclass
class DocumentSection:
    """Represents a section/chapter in a document."""
    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection, etc.
    content: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    word_count: int = 0
    ai_probability: Optional[float] = None
    classification: Optional[str] = None


@dataclass
class StructuralFeatures:
    """Structural/document-level features for AI detection."""
    # Document organization
    num_sections: int = 0
    num_subsections: int = 0
    section_depth_max: int = 0
    section_depth_avg: float = np.nan
    
    # Section balance
    section_length_variance: float = np.nan
    section_length_coefficient_variation: float = np.nan
    
    # Heading patterns
    heading_style_consistency: float = np.nan
    numbering_consistency: float = np.nan
    
    # Flow metrics
    transition_quality: float = np.nan
    topic_coherence_between_sections: float = np.nan
    
    # Academic structure indicators
    has_abstract: bool = False
    has_introduction: bool = False
    has_methodology: bool = False
    has_results: bool = False
    has_discussion: bool = False
    has_conclusion: bool = False
    has_references: bool = False
    
    # Reference patterns
    citation_density: float = np.nan
    reference_format_consistency: float = np.nan
    
    # Figure/table mentions
    figure_mention_density: float = np.nan
    table_mention_density: float = np.nan


@dataclass
class PDFAnalysisResult:
    """Complete PDF analysis result."""
    # Basic info
    file_path: str
    total_pages: int
    total_words: int
    
    # Extraction quality
    extraction_quality: str  # "high", "medium", "low"
    warnings: List[str]
    
    # Structure
    sections: List[DocumentSection]
    structural_features: StructuralFeatures
    
    # AI detection results
    overall_ai_probability: float
    overall_classification: str
    section_classifications: List[Dict[str, Any]]
    
    # Evidence
    structural_evidence: List[Dict[str, Any]]
    flow_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "extraction_quality": self.extraction_quality,
            "warnings": self.warnings,
            "num_sections": len(self.sections),
            "structural_features": {
                "num_sections": self.structural_features.num_sections,
                "num_subsections": self.structural_features.num_subsections,
                "section_depth_max": self.structural_features.section_depth_max,
                "has_abstract": self.structural_features.has_abstract,
                "has_introduction": self.structural_features.has_introduction,
                "has_methodology": self.structural_features.has_methodology,
                "has_results": self.structural_features.has_results,
                "has_discussion": self.structural_features.has_discussion,
                "has_conclusion": self.structural_features.has_conclusion,
                "has_references": self.structural_features.has_references,
                "citation_density": self.structural_features.citation_density,
                "section_length_variance": self.structural_features.section_length_variance,
                "heading_style_consistency": self.structural_features.heading_style_consistency,
                "transition_quality": self.structural_features.transition_quality,
                "topic_coherence_between_sections": self.structural_features.topic_coherence_between_sections
            },
            "overall_ai_probability": round(self.overall_ai_probability, 4),
            "overall_classification": self.overall_classification,
            "section_classifications": self.section_classifications,
            "structural_evidence": self.structural_evidence,
            "flow_analysis": self.flow_analysis
        }


class PDFTextExtractor:
    """
    Extracts text from PDF files.
    
    Supports multiple backends:
    - PyPDF2 (pure Python, widely compatible)
    - pdfplumber (better layout preservation)
    - pymupdf/fitz (fastest, best quality)
    """
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize PDF extractor.
        
        Args:
            backend: Extraction backend - "auto", "pypdf2", "pdfplumber", or "pymupdf"
        """
        self.backend = backend
        self._pypdf2_available = False
        self._pdfplumber_available = False
        self._pymupdf_available = False
        
        self._check_availability()
        
    def _check_availability(self):
        """Check which PDF libraries are available."""
        try:
            import PyPDF2
            self._pypdf2_available = True
            logger.debug("PyPDF2 available")
        except ImportError:
            logger.debug("PyPDF2 not available")
            
        try:
            import pdfplumber
            self._pdfplumber_available = True
            logger.debug("pdfplumber available")
        except ImportError:
            logger.debug("pdfplumber not available")
            
        try:
            import fitz  # PyMuPDF
            self._pymupdf_available = True
            logger.debug("PyMuPDF available")
        except ImportError:
            logger.debug("PyMuPDF not available")
    
    def _get_best_backend(self) -> str:
        """Determine best available backend."""
        if self.backend != "auto":
            return self.backend
            
        # Prefer PyMuPDF for quality and speed
        if self._pymupdf_available:
            return "pymupdf"
        elif self._pdfplumber_available:
            return "pdfplumber"
        elif self._pypdf2_available:
            return "pypdf2"
        else:
            return "none"
    
    def extract(self, file_path: Union[str, Path]) -> PDFExtractionResult:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        file_path = Path(file_path)
        warnings = []
        
        if not file_path.exists():
            return PDFExtractionResult(
                raw_text="",
                metadata={},
                pages=[],
                is_extractable=False,
                warnings=[f"File not found: {file_path}"],
                file_path=str(file_path)
            )
        
        if file_path.suffix.lower() != '.pdf':
            warnings.append(f"File extension is not .pdf: {file_path}")
        
        backend = self._get_best_backend()
        
        if backend == "none":
            return PDFExtractionResult(
                raw_text="",
                metadata={},
                pages=[],
                is_extractable=False,
                warnings=[
                    "No PDF extraction library available. "
                    "Install one of: PyMuPDF (recommended), pdfplumber, or PyPDF2"
                ],
                file_path=str(file_path)
            )
        
        try:
            if backend == "pymupdf":
                return self._extract_pymupdf(file_path, warnings)
            elif backend == "pdfplumber":
                return self._extract_pdfplumber(file_path, warnings)
            elif backend == "pypdf2":
                return self._extract_pypdf2(file_path, warnings)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return PDFExtractionResult(
                raw_text="",
                metadata={},
                pages=[],
                is_extractable=False,
                warnings=[f"Extraction failed: {str(e)}"],
                file_path=str(file_path)
            )
    
    def _extract_pymupdf(self, file_path: Path, warnings: List[str]) -> PDFExtractionResult:
        """Extract using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF
        
        metadata = {}
        pages = []
        
        try:
            doc = fitz.open(file_path)
            
            # Extract metadata
            meta = doc.metadata
            metadata = {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "creation_date": meta.get("creationDate", ""),
                "modification_date": meta.get("modDate", ""),
                "num_pages": len(doc)
            }
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages.append(text)
            
            doc.close()
            
            raw_text = "\n\n".join(pages)
            
            # Check for potential issues
            if len(raw_text.strip()) < 100:
                warnings.append("Very little text extracted - may be a scanned PDF")
                
            return PDFExtractionResult(
                raw_text=raw_text,
                metadata=metadata,
                pages=pages,
                is_extractable=True,
                warnings=warnings,
                file_path=str(file_path)
            )
            
        except Exception as e:
            raise RuntimeError(f"PyMuPDF extraction failed: {e}")
    
    def _extract_pdfplumber(self, file_path: Path, warnings: List[str]) -> PDFExtractionResult:
        """Extract using pdfplumber."""
        import pdfplumber
        
        metadata = {}
        pages = []
        
        with pdfplumber.open(file_path) as pdf:
            # Basic metadata
            metadata = {
                "num_pages": len(pdf.pages),
                "title": "",
                "author": ""
            }
            
            # Extract text from each page
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        
        raw_text = "\n\n".join(pages)
        
        if len(raw_text.strip()) < 100:
            warnings.append("Very little text extracted - may be a scanned PDF")
        
        return PDFExtractionResult(
            raw_text=raw_text,
            metadata=metadata,
            pages=pages,
            is_extractable=True,
            warnings=warnings,
            file_path=str(file_path)
        )
    
    def _extract_pypdf2(self, file_path: Path, warnings: List[str]) -> PDFExtractionResult:
        """Extract using PyPDF2."""
        import PyPDF2
        
        metadata = {}
        pages = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Extract metadata
            if reader.metadata:
                metadata = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                    "producer": reader.metadata.get("/Producer", ""),
                    "num_pages": len(reader.pages)
                }
            
            # Extract text from each page
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
        
        raw_text = "\n\n".join(pages)
        
        if len(raw_text.strip()) < 100:
            warnings.append("Very little text extracted - may be a scanned PDF")
        
        return PDFExtractionResult(
            raw_text=raw_text,
            metadata=metadata,
            pages=pages,
            is_extractable=True,
            warnings=warnings,
            file_path=str(file_path)
        )


class DocumentStructureAnalyzer:
    """
    Analyzes document structure for AI detection.
    
    Identifies:
    - Sections and subsections
    - Academic structure elements (abstract, intro, methodology, etc.)
    - Heading patterns and consistency
    - Section length distributions
    - Transition quality between sections
    """
    
    # Common academic section patterns
    SECTION_PATTERNS = {
        'abstract': [r'^abstract', r'^summary', r'^executive summary'],
        'introduction': [r'^introduction', r'^intro', r'^background', r'^motivation'],
        'methodology': [r'^methodology', r'^methods', r'^materials? and methods?', 
                       r'^approach', r'^design', r'^procedure', r'^experimental'],
        'results': [r'^results', r'^findings', r'^outcomes', r'^data analysis'],
        'discussion': [r'^discussion', r'^analysis', r'^interpretation'],
        'conclusion': [r'^conclusion', r'^conclusions?', r'^summary', r'^final remarks'],
        'references': [r'^references', r'^bibliography', r'^works cited', r'^literature cited'],
        'appendix': [r'^appendix', r'^appendices', r'^supplementary']
    }
    
    # Heading patterns
    HEADING_PATTERNS = [
        # Numbered headings: "1. Introduction", "1.1 Background", "2.3.1 Details"
        (r'^(\d+\.?\s*[A-Z][a-z])', 1),  # Level 1
        (r'^(\d+\.\d+\.?\s*[A-Z][a-z])', 2),  # Level 2
        (r'^(\d+\.\d+\.\d+\.?\s*[A-Z][a-z])', 3),  # Level 3
        
        # Unnumbered but capitalized headings
        (r'^([A-Z][A-Z\s]{3,})$', 1),  # ALL CAPS
        
        # Markdown-style headings
        (r'^(#{1,6}\s+[A-Z])', 1),  # Will adjust based on # count
    ]
    
    def __init__(self):
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns."""
        for section_type, patterns in self.SECTION_PATTERNS.items():
            self._compiled_patterns[section_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
            ]
    
    def detect_sections(self, text: str, pages: Optional[List[str]] = None) -> List[DocumentSection]:
        """
        Detect sections in document.
        
        Args:
            text: Full document text
            pages: Optional list of page texts for page number estimation
            
        Returns:
            List of detected sections
        """
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        current_level = 1
        current_start_line = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this is a heading
            heading_match = self._detect_heading(line_stripped)
            
            if heading_match:
                # Save previous section
                if current_section is not None:
                    sections.append(DocumentSection(
                        title=current_section,
                        level=current_level,
                        content='\n'.join(current_content),
                        word_count=len(' '.join(current_content).split())
                    ))
                
                # Start new section
                current_section = heading_match['title']
                current_level = heading_match['level']
                current_content = []
                current_start_line = i
                
            elif current_section is not None:
                current_content.append(line)
        
        # Don't forget last section
        if current_section is not None:
            sections.append(DocumentSection(
                title=current_section,
                level=current_level,
                content='\n'.join(current_content),
                word_count=len(' '.join(current_content).split())
            ))
        
        # If no sections detected, treat whole document as one section
        if not sections:
            sections.append(DocumentSection(
                title="Full Document",
                level=1,
                content=text,
                word_count=len(text.split())
            ))
        
        return sections
    
    def _detect_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """Detect if a line is a heading and return its properties."""
        if not line or len(line) < 3:
            return None
        
        # Skip very long lines (probably not headings)
        if len(line) > 200:
            return None
        
        # Check numbered patterns
        for pattern_str, base_level in self.HEADING_PATTERNS:
            match = re.match(pattern_str, line)
            if match:
                # Adjust level for markdown headings
                if '#' in pattern_str:
                    level = line.count('#')
                else:
                    level = base_level
                
                # Extract clean title
                title = re.sub(r'^[\d#.\s]+', '', line).strip()
                
                if title and len(title) > 2:
                    return {'title': title, 'level': level}
        
        # Check if it matches a known section type
        for section_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(line):
                    return {'title': line, 'level': 1, 'type': section_type}
        
        return None
    
    def identify_academic_structure(self, sections: List[DocumentSection]) -> StructuralFeatures:
        """
        Identify academic structure elements in sections.
        
        Returns:
            StructuralFeatures with identified elements
        """
        features = StructuralFeatures()
        
        features.num_sections = len(sections)
        features.num_subsections = sum(1 for s in sections if s.level > 1)
        features.section_depth_max = max((s.level for s in sections), default=0)
        
        # Calculate average depth
        if sections:
            features.section_depth_avg = np.mean([s.level for s in sections])
        
        # Identify standard academic sections
        section_titles_lower = [s.title.lower() for s in sections]
        
        features.has_abstract = any('abstract' in t or 'summary' in t for t in section_titles_lower)
        features.has_introduction = any('introduction' in t or 'intro' in t for t in section_titles_lower)
        features.has_methodology = any(t in ['methodology', 'methods', 'approach', 'design'] 
                                       for t in section_titles_lower)
        features.has_results = any('result' in t or 'finding' in t for t in section_titles_lower)
        features.has_discussion = any('discussion' in t or 'analysis' in t for t in section_titles_lower)
        features.has_conclusion = any('conclusion' in t for t in section_titles_lower)
        features.has_references = any('reference' in t or 'bibliography' in t for t in section_titles_lower)
        
        # Section length statistics
        if sections:
            lengths = [s.word_count for s in sections]
            features.section_length_variance = float(np.var(lengths))
            mean_len = np.mean(lengths)
            if mean_len > 0:
                features.section_length_coefficient_variation = float(np.std(lengths) / mean_len)
        
        return features
    
    def analyze_heading_consistency(self, sections: List[DocumentSection]) -> float:
        """
        Analyze consistency of heading styles.
        
        Returns:
            Consistency score (0-1, higher = more consistent)
        """
        if len(sections) < 2:
            return np.nan
        
        # Check numbering consistency
        numbered = sum(1 for s in sections if re.match(r'^\d+', s.title))
        numbering_consistency = numbered / len(sections) if sections else 0
        
        # Check capitalization consistency
        title_case = sum(1 for s in sections if s.title.istitle() or s.title.isupper())
        capitalization_consistency = title_case / len(sections) if sections else 0
        
        return (numbering_consistency + capitalization_consistency) / 2
    
    def analyze_transitions(self, sections: List[DocumentSection]) -> float:
        """
        Analyze transition quality between sections.
        
        AI-generated text often has weaker transitions between major sections.
        
        Returns:
            Transition quality score (0-1, higher = better transitions)
        """
        if len(sections) < 2:
            return np.nan
        
        transition_markers = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'in addition', 'on the other hand', 'similarly', 'likewise',
            'building on', 'extending', 'related to', 'in contrast',
            'as discussed', 'as shown', 'as mentioned', 'previously',
            'next', 'subsequently', 'following', 'turning to', 'now we'
        }
        
        transition_scores = []
        
        for i in range(len(sections) - 1):
            # Check end of current section and beginning of next
            prev_end = sections[i].content[-500:] if len(sections[i].content) > 500 else sections[i].content
            next_start = sections[i+1].content[:500] if len(sections[i+1].content) > 500 else sections[i+1].content
            
            # Count transition markers
            prev_lower = prev_end.lower()
            next_lower = next_start.lower()
            
            has_transition = any(marker in prev_lower or marker in next_lower 
                                for marker in transition_markers)
            
            transition_scores.append(1.0 if has_transition else 0.3)
        
        return float(np.mean(transition_scores)) if transition_scores else np.nan
    
    def analyze_citations(self, text: str) -> Tuple[float, float]:
        """
        Analyze citation patterns.
        
        Returns:
            (citation_density, format_consistency)
        """
        # Common citation patterns
        citation_patterns = [
            r'\[\d+\]',  # [1], [2, 3]
            r'\(\w+, \d{4}\)',  # (Smith, 2020)
            r'\w+ et al\., \d{4}',  # Smith et al., 2020
            r'\w+ and \w+, \d{4}',  # Smith and Jones, 2020
            r'\^\d+',  # ^1 (footnote style)
        ]
        
        total_citations = 0
        citation_types = Counter()
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            total_citations += len(matches)
            citation_types[pattern] = len(matches)
        
        # Citation density (per 1000 words)
        word_count = len(text.split())
        citation_density = (total_citations / word_count * 1000) if word_count > 0 else 0
        
        # Format consistency (how many citations use the dominant format)
        if citation_types:
            most_common_count = citation_types.most_common(1)[0][1]
            format_consistency = most_common_count / total_citations if total_citations > 0 else 0
        else:
            format_consistency = np.nan
        
        return citation_density, format_consistency
    
    def analyze_figure_table_mentions(self, text: str) -> Tuple[float, float]:
        """Analyze mentions of figures and tables."""
        figure_pattern = r'\bfigure\s*\d+|fig\.\s*\d+|fig\s*\d+'
        table_pattern = r'\btable\s*\d+'
        
        figure_matches = len(re.findall(figure_pattern, text, re.IGNORECASE))
        table_matches = len(re.findall(table_pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        
        figure_density = (figure_matches / word_count * 1000) if word_count > 0 else 0
        table_density = (table_matches / word_count * 1000) if word_count > 0 else 0
        
        return figure_density, table_density


class PDFFlowAnalyzer:
    """
    Analyzes document flow and coherence patterns.
    
    Detects patterns that may indicate AI generation:
    - Abrupt topic shifts
    - Repetitive phrasing across sections
    - Inconsistent terminology
    - Lack of deep technical detail
    - Generic transitional phrases
    """
    
    def __init__(self):
        self._generic_phrases = {
            'it is important to note',
            'in conclusion',
            'in summary',
            'this paper discusses',
            'this study examines',
            'the results show',
            'further research is needed',
            'it is worth noting',
            'as mentioned earlier',
            'as previously discussed',
            'in recent years',
            'have gained significant attention',
            'plays a crucial role',
            'has been widely studied',
            'remains a challenge',
            'offers promising results'
        }
    
    def analyze_topic_coherence(self, sections: List[DocumentSection]) -> float:
        """
        Analyze topic coherence between sections.
        
        Uses simple keyword overlap as proxy for coherence.
        
        Returns:
            Coherence score (0-1)
        """
        if len(sections) < 2:
            return np.nan
        
        def extract_keywords(text: str, top_n: int = 20) -> set:
            """Extract top keywords from text."""
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            # Filter common words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                         'would', 'could', 'should', 'which', 'their', 'there', 
                         'where', 'when', 'what', 'into', 'upon', 'about'}
            filtered = [w for w in words if w not in stop_words]
            
            from collections import Counter
            counts = Counter(filtered)
            return set(word for word, _ in counts.most_common(top_n))
        
        coherence_scores = []
        
        for i in range(len(sections) - 1):
            keywords_prev = extract_keywords(sections[i].content)
            keywords_next = extract_keywords(sections[i+1].content)
            
            if keywords_prev and keywords_next:
                overlap = len(keywords_prev & keywords_next)
                avg_size = (len(keywords_prev) + len(keywords_next)) / 2
                coherence = overlap / avg_size if avg_size > 0 else 0
                coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else np.nan
    
    def detect_generic_phrases(self, text: str) -> Dict[str, Any]:
        """
        Detect overuse of generic academic phrases.
        
        AI-generated academic writing often relies heavily on stock phrases.
        
        Returns:
            Dictionary with phrase counts and density
        """
        text_lower = text.lower()
        
        found_phrases = []
        for phrase in self._generic_phrases:
            count = text_lower.count(phrase)
            if count > 0:
                found_phrases.append({'phrase': phrase, 'count': count})
        
        word_count = len(text.split())
        total_generic = sum(p['count'] for p in found_phrases)
        density = (total_generic / word_count * 1000) if word_count > 0 else 0
        
        return {
            'generic_phrases': found_phrases,
            'total_count': total_generic,
            'density_per_1000_words': density,
            'is_high': density > 15  # Threshold for concern
        }
    
    def analyze_repetition_across_sections(self, sections: List[DocumentSection]) -> float:
        """
        Analyze repetition patterns across sections.
        
        High repetition of exact phrases across sections can indicate AI generation.
        
        Returns:
            Repetition score (higher = more repetition)
        """
        if len(sections) < 2:
            return np.nan
        
        # Extract n-grams from each section
        def get_ngrams(text: str, n: int = 4) -> set:
            words = text.lower().split()
            return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}
        
        all_ngrams = []
        for section in sections:
            ngrams = get_ngrams(section.content, n=4)
            all_ngrams.append(ngrams)
        
        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(all_ngrams)):
            for j in range(i+1, len(all_ngrams)):
                if all_ngrams[i] and all_ngrams[j]:
                    overlap = len(all_ngrams[i] & all_ngrams[j])
                    avg_size = (len(all_ngrams[i]) + len(all_ngrams[j])) / 2
                    overlaps.append(overlap / avg_size if avg_size > 0 else 0)
        
        return float(np.mean(overlaps)) if overlaps else np.nan
    
    def analyze_technical_depth(self, sections: List[DocumentSection]) -> float:
        """
        Analyze technical depth of content.
        
        AI-generated text often lacks specific technical details.
        
        Returns:
            Technical depth score (higher = more technical)
        """
        # Indicators of technical depth
        technical_indicators = {
            'numbers_with_units': r'\d+\s*(kg|mm|cm|m|s|ms|hz|ghz|mb|gb|tb|°c|°f|%)',
            'equations': r'[=+\-*/^]\s*\d+',
            'technical_terms': r'(algorithm|neural|network|optimization|parameter|variable|function|model|dataset|experiment|measurement|accuracy|precision|recall|f1|p-value|confidence interval)',
            'specific_citations': r'\[\d{1,3}\]|\(\w+,\s*\d{4}\)',
            'methodology_details': r'(using|employed|applied|implemented|configured)\s+(the|a|an)?\s*\w+'
        }
        
        depth_scores = []
        
        for section in sections:
            section_score = 0
            text = section.content.lower()
            
            for indicator_name, pattern in technical_indicators.items():
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                # Normalize by section length
                normalized = matches / (len(section.content.split()) / 1000) if section.content else 0
                section_score += min(normalized, 10)  # Cap contribution
            
            depth_scores.append(section_score)
        
        return float(np.mean(depth_scores)) if depth_scores else np.nan


class PDFAnalyzer:
    """
    Main PDF analyzer combining all components.
    
    Provides complete PDF analysis for AI detection including:
    - Text extraction
    - Structure analysis
    - Flow analysis
    - Integration with main AI detector
    """
    
    def __init__(self, ai_detector=None):
        """
        Initialize PDF analyzer.
        
        Args:
            ai_detector: Optional AIDetector instance for AI probability estimation
        """
        self.extractor = PDFTextExtractor()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.flow_analyzer = PDFFlowAnalyzer()
        self.ai_detector = ai_detector
    
    def analyze(self, file_path: Union[str, Path], 
                run_ai_detection: bool = True) -> PDFAnalysisResult:
        """
        Perform complete PDF analysis.
        
        Args:
            file_path: Path to PDF file
            run_ai_detection: Whether to run AI detection on sections
            
        Returns:
            PDFAnalysisResult with complete analysis
        """
        file_path = Path(file_path)
        
        # Step 1: Extract text
        extraction_result = self.extractor.extract(file_path)
        
        if not extraction_result.is_extractable:
            return PDFAnalysisResult(
                file_path=str(file_path),
                total_pages=0,
                total_words=0,
                extraction_quality="failed",
                warnings=extraction_result.warnings,
                sections=[],
                structural_features=StructuralFeatures(),
                overall_ai_probability=np.nan,
                overall_classification="unknown",
                section_classifications=[],
                structural_evidence=[],
                flow_analysis={}
            )
        
        # Determine extraction quality
        word_count = len(extraction_result.raw_text.split())
        if word_count > 1000:
            extraction_quality = "high"
        elif word_count > 100:
            extraction_quality = "medium"
        else:
            extraction_quality = "low"
        
        # Step 2: Analyze structure
        sections = self.structure_analyzer.detect_sections(
            extraction_result.raw_text,
            extraction_result.pages
        )
        
        structural_features = self.structure_analyzer.identify_academic_structure(sections)
        structural_features.heading_style_consistency = self.structure_analyzer.analyze_heading_consistency(sections)
        structural_features.transition_quality = self.structure_analyzer.analyze_transitions(sections)
        
        citation_density, citation_consistency = self.structure_analyzer.analyze_citations(extraction_result.raw_text)
        structural_features.citation_density = citation_density
        structural_features.reference_format_consistency = citation_consistency
        
        fig_density, table_density = self.structure_analyzer.analyze_figure_table_mentions(extraction_result.raw_text)
        structural_features.figure_mention_density = fig_density
        structural_features.table_mention_density = table_density
        
        # Step 3: Analyze flow
        topic_coherence = self.flow_analyzer.analyze_topic_coherence(sections)
        structural_features.topic_coherence_between_sections = topic_coherence
        
        generic_analysis = self.flow_analyzer.detect_generic_phrases(extraction_result.raw_text)
        repetition_score = self.flow_analyzer.analyze_repetition_across_sections(sections)
        technical_depth = self.flow_analyzer.analyze_technical_depth(sections)
        
        flow_analysis = {
            'topic_coherence': topic_coherence,
            'generic_phrase_density': generic_analysis['density_per_1000_words'],
            'generic_phrase_is_high': generic_analysis['is_high'],
            'cross_section_repetition': repetition_score,
            'technical_depth': technical_depth,
            'generic_phrases_found': generic_analysis['generic_phrases'][:10]  # Top 10
        }
        
        # Step 4: Run AI detection on sections if requested
        section_classifications = []
        ai_probabilities = []
        
        if run_ai_detection and self.ai_detector is not None:
            for section in sections:
                if len(section.content.split()) >= 50:  # Minimum length
                    try:
                        result = self.ai_detector.predict(section.content)
                        section.ai_probability = result.ai_probability
                        section.classification = result.classification
                        ai_probabilities.append(result.ai_probability)
                        
                        section_classifications.append({
                            'section_title': section.title,
                            'word_count': section.word_count,
                            'ai_probability': round(result.ai_probability, 4),
                            'classification': result.classification
                        })
                    except Exception as e:
                        logger.warning(f"AI detection failed for section '{section.title}': {e}")
                        section_classifications.append({
                            'section_title': section.title,
                            'word_count': section.word_count,
                            'ai_probability': np.nan,
                            'classification': 'error',
                            'error': str(e)
                        })
        
        # Calculate overall AI probability
        if ai_probabilities:
            # Weight by section length
            weights = [s.word_count for s in sections if s.ai_probability is not None]
            if sum(weights) > 0:
                overall_ai_prob = np.average(ai_probabilities, weights=weights)
            else:
                overall_ai_prob = np.mean(ai_probabilities)
        else:
            overall_ai_prob = np.nan
        
        # Determine overall classification
        if np.isnan(overall_ai_prob):
            overall_classification = "unknown"
        elif overall_ai_prob >= 0.7:
            overall_classification = "likely_ai"
        elif overall_ai_prob <= 0.3:
            overall_classification = "likely_human"
        else:
            overall_classification = "uncertain"
        
        # Build structural evidence
        structural_evidence = self._build_structural_evidence(structural_features, flow_analysis)
        
        return PDFAnalysisResult(
            file_path=str(file_path),
            total_pages=extraction_result.metadata.get('num_pages', len(extraction_result.pages)),
            total_words=word_count,
            extraction_quality=extraction_quality,
            warnings=extraction_result.warnings,
            sections=sections,
            structural_features=structural_features,
            overall_ai_probability=overall_ai_prob,
            overall_classification=overall_classification,
            section_classifications=section_classifications,
            structural_evidence=structural_evidence,
            flow_analysis=flow_analysis
        )
    
    def _build_structural_evidence(self, features: StructuralFeatures, 
                                   flow_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build evidence list from structural and flow analysis."""
        evidence = []
        
        # Section structure evidence
        if features.num_sections > 0:
            evidence.append({
                'feature': 'document_structure',
                'observation': f'{features.num_sections} sections detected',
                'direction': 'neutral',
                'strength': 0.1
            })
        
        # Academic structure completeness
        academic_elements = sum([
            features.has_abstract, features.has_introduction, features.has_methodology,
            features.has_results, features.has_discussion, features.has_conclusion
        ])
        if academic_elements >= 5:
            evidence.append({
                'feature': 'academic_structure',
                'observation': f'Complete academic structure ({academic_elements}/6 elements)',
                'direction': 'human-like',
                'strength': 0.3
            })
        elif academic_elements <= 2:
            evidence.append({
                'feature': 'academic_structure',
                'observation': f'Incomplete academic structure ({academic_elements}/6 elements)',
                'direction': 'ai-like',
                'strength': 0.4
            })
        
        # Citation density
        if features.citation_density > 0:
            if features.citation_density > 20:
                evidence.append({
                    'feature': 'citation_density',
                    'observation': f'High citation density ({features.citation_density:.1f}/1000 words)',
                    'direction': 'human-like',
                    'strength': 0.4
                })
            elif features.citation_density < 5:
                evidence.append({
                    'feature': 'citation_density',
                    'observation': f'Low citation density ({features.citation_density:.1f}/1000 words)',
                    'direction': 'ai-like',
                    'strength': 0.3
                })
        
        # Generic phrases
        if flow_analysis.get('generic_phrase_is_high'):
            evidence.append({
                'feature': 'generic_phrases',
                'observation': f'High density of generic academic phrases ({flow_analysis["generic_phrase_density"]:.1f}/1000 words)',
                'direction': 'ai-like',
                'strength': 0.5
            })
        
        # Topic coherence
        if not np.isnan(features.topic_coherence_between_sections):
            if features.topic_coherence_between_sections < 0.1:
                evidence.append({
                    'feature': 'topic_coherence',
                    'observation': 'Low topic coherence between sections',
                    'direction': 'ai-like',
                    'strength': 0.4
                })
            elif features.topic_coherence_between_sections > 0.3:
                evidence.append({
                    'feature': 'topic_coherence',
                    'observation': 'Good topic coherence between sections',
                    'direction': 'human-like',
                    'strength': 0.3
                })
        
        # Technical depth
        if not np.isnan(flow_analysis.get('technical_depth', np.nan)):
            if flow_analysis['technical_depth'] < 5:
                evidence.append({
                    'feature': 'technical_depth',
                    'observation': 'Low technical depth/specificity',
                    'direction': 'ai-like',
                    'strength': 0.4
                })
            elif flow_analysis['technical_depth'] > 15:
                evidence.append({
                    'feature': 'technical_depth',
                    'observation': 'High technical depth/specificity',
                    'direction': 'human-like',
                    'strength': 0.3
                })
        
        return evidence


def analyze_pdf(file_path: Union[str, Path], ai_detector=None) -> PDFAnalysisResult:
    """Convenience function to analyze a PDF file."""
    analyzer = PDFAnalyzer(ai_detector=ai_detector)
    return analyzer.analyze(file_path)
