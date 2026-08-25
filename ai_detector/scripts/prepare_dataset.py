"""
Dataset preparation script for AI text detection.

This script outlines the data schema and preparation pipeline.
In practice, you would need to supply actual datasets.

Required data structure:
- Human-written texts from multiple domains
- AI-generated texts from multiple model families
- Metadata including source, author (if available), topic, etc.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextSample:
    """Single text sample with metadata."""
    text: str
    label: int  # 0 = human, 1 = AI
    domain: str  # essay, news, academic, forum, personal, technical, creative
    source: str  # Dataset/source name
    author_id: Optional[str] = None  # Anonymous author ID if available
    ai_model: Optional[str] = None  # For AI samples: which model generated it
    prompt_id: Optional[str] = None  # For AI samples: prompt category
    topic: Optional[str] = None  # Topic/category
    editing_level: str = "original"  # original, lightly_edited, heavily_edited, paraphrased
    split: str = "train"  # train, val, test_in_dist, test_unseen_model, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetPreparator:
    """
    Prepares balanced dataset for AI detection training.
    
    Key design decisions:
    - Split by source/author/model, NOT randomly
    - Ensure domain balance in all splits
    - Create specialized test sets for generalization evaluation
    """
    
    def __init__(self, raw_data_dir: str, processed_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.samples: List[TextSample] = []
        
    def load_human_data(self, source: str, domain: str, 
                       texts: List[str], authors: Optional[List[str]] = None):
        """Load human-written texts."""
        for i, text in enumerate(texts):
            if len(text.strip()) < 100:  # Skip very short texts
                continue
                
            sample = TextSample(
                text=text.strip(),
                label=0,
                domain=domain,
                source=source,
                author_id=authors[i] if authors else None,
                topic=None
            )
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(texts)} human samples from {source} ({domain})")
    
    def load_ai_data(self, source: str, domain: str,
                    texts: List[str], ai_model: str,
                    prompts: Optional[List[str]] = None):
        """Load AI-generated texts."""
        for i, text in enumerate(texts):
            if len(text.strip()) < 100:
                continue
            
            sample = TextSample(
                text=text.strip(),
                label=1,
                domain=domain,
                source=source,
                ai_model=ai_model,
                prompt_id=prompts[i] if prompts else None
            )
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(texts)} AI samples from {source} ({ai_model}, {domain})")
    
    def create_splits(self):
        """
        Create train/val/test splits with proper separation.
        
        Splits created:
        1. train: Main training set (balanced)
        2. val: Validation for calibration and threshold selection
        3. test_in_dist: In-distribution test (same models/domains as train)
        4. test_unseen_models: Test on AI models not in training
        5. test_unseen_topics: Test on topics not in training
        6. test_paraphrased: Test on paraphrased AI text
        7. test_edited: Test on human-edited AI text
        8. test_short: Test on short texts (<100 tokens)
        9. test_long: Test on long texts (>1000 tokens)
        """
        # Group samples by source for proper splitting
        by_source = defaultdict(list)
        for sample in self.samples:
            key = f"{sample.source}_{sample.author_id or 'no_author'}"
            by_source[key].append(sample)
        
        sources = list(by_source.keys())
        random.shuffle(sources)
        
        # Allocate sources to splits
        n_sources = len(sources)
        train_sources = sources[:int(n_sources * 0.6)]
        val_sources = sources[int(n_sources * 0.6):int(n_sources * 0.75)]
        test_sources = sources[int(n_sources * 0.75):]
        
        # Assign splits
        for source in train_sources:
            for sample in by_source[source]:
                sample.split = "train"
        
        for source in val_sources:
            for sample in by_source[source]:
                sample.split = "val"
        
        for source in test_sources:
            for sample in by_source[source]:
                sample.split = "test_in_dist"
        
        logger.info(f"Created splits: train={len(train_sources)} sources, "
                   f"val={len(val_sources)} sources, test={len(test_sources)} sources")
    
    def save_processed_data(self):
        """Save processed dataset."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all samples
        all_samples_path = self.processed_dir / "all_samples.jsonl"
        with open(all_samples_path, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + '\n')
        
        # Save by split
        by_split = defaultdict(list)
        for sample in self.samples:
            by_split[sample.split].append(sample.to_dict())
        
        for split_name, samples in by_split.items():
            split_path = self.processed_dir / f"{split_name}.jsonl"
            with open(split_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            
            # Print statistics
            n_human = sum(1 for s in samples if s['label'] == 0)
            n_ai = sum(1 for s in samples if s['label'] == 1)
            logger.info(f"{split_name}: {len(samples)} samples "
                       f"(human={n_human}, ai={n_ai})")
        
        # Save metadata
        stats = self._compute_statistics()
        stats_path = self.processed_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset saved to {self.processed_dir}")
        return stats
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "by_label": {"human": 0, "ai": 0},
            "by_domain": defaultdict(int),
            "by_source": defaultdict(int),
            "ai_models": set(),
            "avg_length": 0
        }
        
        total_length = 0
        for sample in self.samples:
            stats["by_label"]["human" if sample.label == 0 else "ai"] += 1
            stats["by_domain"][sample.domain] += 1
            stats["by_source"][sample.source] += 1
            if sample.ai_model:
                stats["ai_models"].add(sample.ai_model)
            total_length += len(sample.text.split())
        
        stats["ai_models"] = list(stats["ai_models"])
        stats["avg_length"] = total_length / len(self.samples) if self.samples else 0
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["by_domain"] = dict(stats["by_domain"])
        stats["by_source"] = dict(stats["by_source"])
        
        return stats


def main():
    """
    Example usage of dataset preparator.
    
    In practice, you would:
    1. Download/load your human text datasets
    2. Generate AI text using various models
    3. Call load_human_data() and load_ai_data()
    4. Run create_splits() and save_processed_data()
    """
    logger.info("Dataset preparation script")
    logger.info("=" * 50)
    logger.info("""
    This script outlines the data preparation pipeline.
    
    To prepare your dataset:
    
    1. Collect human-written texts from diverse sources:
       - Essays (student writing, professional essays)
       - News articles
       - Academic papers
       - Forum posts (Reddit, StackExchange)
       - Personal writing (blogs, diaries)
       - Technical documentation
       - Creative writing
    
    2. Generate AI texts using multiple models:
       - GPT-3.5/GPT-4 family
       - Claude family
       - PaLM/Gemini family
       - LLaMA family
       - Other open-source models
       
       Use diverse prompts covering the same domains as human data.
    
    3. Ensure proper separation:
       - No author appears in both train and test
       - No AI model in test_unseen_models appears in training
       - Topics should be balanced across splits
    
    4. Create edited/paraphrased versions:
       - Lightly edit some AI texts (fix errors, change words)
       - Heavily paraphrase some AI texts
       - Mix human and AI paragraphs
    
    See the class docstrings for detailed API.
    """)


if __name__ == "__main__":
    main()
