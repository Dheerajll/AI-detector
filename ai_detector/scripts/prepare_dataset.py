#!/usr/bin/env python3
"""
Dataset Preparation Script for AI Text Detection

This script automatically downloads, processes, and splits datasets for training
the AI text detector. It handles multiple data sources and ensures proper
train/validation/test splits without contamination.

Usage:
    python scripts/prepare_dataset.py --download-all
    python scripts/prepare_dataset.py --dataset hc3
    python scripts/prepare_dataset.py --verify-splits
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DatasetConfig:
    """Configuration for each dataset source."""
    
    HUMAN_DATASETS = {
        "openwebtext": {
            "hf_name": "skylion007/openwebtext",
            "domain": "web",
            "text_column": "text",
            "author_column": None,
            "min_length": 100,
            "max_length": 2000,
        },
        "realnews": {
            "hf_name": "yhcc/RealNews",
            "domain": "news",
            "text_column": "body",
            "author_column": None,
            "min_length": 150,
            "max_length": 2000,
        },
        "writingprompts": {
            "hf_name": "Frostinomial/writingprompts",
            "domain": "creative",
            "text_column": "story",
            "author_column": None,
            "min_length": 200,
            "max_length": 1500,
        },
        "stackexchange": {
            "hf_name": "StackExchange/posts",
            "domain": "technical",
            "text_column": "Body",
            "author_column": "OwnerUserId",
            "min_length": 100,
            "max_length": 2000,
        },
        "arxiv": {
            "hf_name": "Cohere/arxiv",
            "domain": "academic",
            "text_column": "abstract",
            "author_column": "authors",
            "min_length": 50,
            "max_length": 500,
        },
        "student_essays": {
            "hf_name": "setu/student_essays",
            "domain": "essays",
            "text_column": "essay",
            "author_column": None,
            "min_length": 100,
            "max_length": 1000,
        },
    }
    
    AI_DATASETS = {
        "hc3": {
            "hf_name": "Hello-SimpleAI/HC3-Chinese",
            "hf_name_en": "Hello-SimpleAI/HC3-English",
            "domain": "mixed",
            "text_column": "human_answers",
            "ai_column": "chatgpt_answers",
            "model_family": "gpt3.5",
            "min_length": 50,
            "max_length": 1000,
        },
        "m4gt": {
            "hf_name": "skrishna/gpt4chan",
            "domain": "forums",
            "text_column": "text",
            "ai_column": None,
            "model_family": "gptj",
            "min_length": 100,
            "max_length": 1500,
        },
        "malicious_ai": {
            "hf_name": "cais/mald",
            "domain": "mixed",
            "text_column": "text",
            "label_column": "label",
            "model_column": "model",
            "min_length": 50,
            "max_length": 1000,
        },
        "aigc_detection": {
            "hf_name": "ufal/aigc_detection",
            "domain": "mixed",
            "text_column": "text",
            "label_column": "label",
            "model_column": "model",
            "min_length": 50,
            "max_length": 1000,
        },
    }


class DatasetDownloader:
    """Handles downloading datasets from HuggingFace."""
    
    def __init__(self, cache_dir: str = "~/.cache/huggingface"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_human_dataset(self, name: str, config: dict) -> Dataset:
        """Download a human-written dataset."""
        print(f"Downloading human dataset: {name}")
        
        try:
            dataset = load_dataset(
                config["hf_name"],
                split="train",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )
            return dataset
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            return None
    
    def download_ai_dataset(self, name: str, config: dict) -> Tuple[Dataset, Dataset]:
        """Download an AI dataset, returning both human and AI samples."""
        print(f"Downloading AI dataset: {name}")
        
        try:
            if "hc_name_en" in config:
                dataset = load_dataset(
                    config["hf_name_en"],
                    split="train",
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
                
                human_data = []
                ai_data = []
                
                for item in dataset:
                    human_answers = item.get(config["text_column"], [])
                    ai_answers = item.get(config["ai_column"], [])
                    
                    for ans in human_answers:
                        if isinstance(ans, str) and len(ans.split()) >= config["min_length"]:
                            human_data.append({
                                "text": ans,
                                "label": 0,
                                "source": name,
                            })
                    
                    for ans in ai_answers:
                        if isinstance(ans, str) and len(ans.split()) >= config["min_length"]:
                            ai_data.append({
                                "text": ans,
                                "label": 1,
                                "source": name,
                                "model_family": config["model_family"],
                            })
                
                return (
                    Dataset.from_list(human_data) if human_data else None,
                    Dataset.from_list(ai_data) if ai_data else None,
                )
            
            elif config.get("label_column"):
                dataset = load_dataset(
                    config["hf_name"],
                    split="train",
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
                
                human_data = []
                ai_data = []
                
                for item in dataset:
                    text = item.get(config["text_column"], "")
                    label = item.get(config["label_column"], 0)
                    model = item.get(config.get("model_column", "model"), "unknown")
                    
                    if not isinstance(text, str) or len(text.split()) < config["min_length"]:
                        continue
                    
                    if label == 1:
                        ai_data.append({
                            "text": text,
                            "label": 1,
                            "source": name,
                            "model_family": model,
                        })
                    else:
                        human_data.append({
                            "text": text,
                            "label": 0,
                            "source": name,
                        })
                
                return (
                    Dataset.from_list(human_data) if human_data else None,
                    Dataset.from_list(ai_data) if ai_data else None,
                )
            
            else:
                dataset = load_dataset(
                    config["hf_name"],
                    split="train",
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
                
                ai_data = []
                for item in dataset:
                    text = item.get(config["text_column"], "")
                    if isinstance(text, str) and len(text.split()) >= config["min_length"]:
                        ai_data.append({
                            "text": text,
                            "label": 1,
                            "source": name,
                            "model_family": config["model_family"],
                        })
                
                return None, Dataset.from_list(ai_data) if ai_data else None
        
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            return None, None


class DatasetProcessor:
    """Processes and cleans text data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        text = " ".join(text.split())
        text = text.replace("[deleted]", "").replace("[removed]", "")
        text = text.replace("&#x200B;", "")
        
        return text.strip()
    
    @staticmethod
    def filter_sample(sample: dict, min_length: int, max_length: int) -> bool:
        """Filter samples based on length criteria."""
        text = sample.get("text", "")
        if not isinstance(text, str):
            return False
        
        word_count = len(text.split())
        return min_length <= word_count <= max_length
    
    @staticmethod
    def add_metadata(sample: dict, domain: str, source_type: str) -> dict:
        """Add metadata to samples."""
        text = sample.get("text", "")
        
        return {
            "text": text,
            "label": sample.get("label", 0 if source_type == "human" else 1),
            "source_type": source_type,
            "domain": domain,
            "source": sample.get("source", "unknown"),
            "model_family": sample.get("model_family", None),
            "word_count": len(text.split()),
            "char_count": len(text),
        }


class DatasetSplitter:
    """Creates train/validation/test splits without contamination."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def create_splits(
        self,
        dataset: pd.DataFrame,
        group_column: str = "doc_id",
    ) -> Dict[str, pd.DataFrame]:
        """Create splits ensuring no group appears in multiple splits."""
        
        import random
        random.seed(self.seed)
        
        groups = dataset[group_column].unique()
        random.shuffle(groups)
        
        n_groups = len(groups)
        train_groups = groups[:int(0.7 * n_groups)]
        val_groups = groups[int(0.7 * n_groups):int(0.85 * n_groups)]
        test_groups = groups[int(0.85 * n_groups):]
        
        train_df = dataset[dataset[group_column].isin(train_groups)].reset_index(drop=True)
        val_df = dataset[dataset[group_column].isin(val_groups)].reset_index(drop=True)
        test_df = dataset[dataset[group_column].isin(test_groups)].reset_index(drop=True)
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }


def prepare_all_datasets(output_dir: str, force_download: bool = False):
    """Main function to prepare all datasets."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    downloader = DatasetDownloader()
    processor = DatasetProcessor()
    
    all_human_data = []
    all_ai_data = []
    
    print("\n=== Downloading Human Datasets ===")
    for name, config in DatasetConfig.HUMAN_DATASETS.items():
        dataset = downloader.download_human_dataset(name, config)
        
        if dataset is not None:
            processed = []
            for item in tqdm(dataset, desc=f"Processing {name}"):
                text = processor.clean_text(item.get(config["text_column"], ""))
                sample = {"text": text, "label": 0, "source": name}
                
                if processor.filter_sample(sample, config["min_length"], config["max_length"]):
                    processed.append(processor.add_metadata(
                        sample,
                        domain=config["domain"],
                        source_type="human"
                    ))
            
            all_human_data.extend(processed)
            print(f"  {name}: {len(processed)} samples")
    
    print("\n=== Downloading AI Datasets ===")
    for name, config in DatasetConfig.AI_DATASETS.items():
        human_ds, ai_ds = downloader.download_ai_dataset(name, config)
        
        if human_ds is not None:
            for item in tqdm(human_ds, desc=f"Processing {name} (human)"):
                text = processor.clean_text(item.get("text", ""))
                sample = {"text": text, "label": 0, "source": name}
                
                if processor.filter_sample(sample, config["min_length"], config["max_length"]):
                    all_human_data.append(processor.add_metadata(
                        sample,
                        domain=config["domain"],
                        source_type="human"
                    ))
        
        if ai_ds is not None:
            for item in tqdm(ai_ds, desc=f"Processing {name} (AI)"):
                text = processor.clean_text(item.get("text", ""))
                sample = {
                    "text": text,
                    "label": 1,
                    "source": name,
                    "model_family": item.get("model_family", "unknown"),
                }
                
                if processor.filter_sample(sample, config["min_length"], config["max_length"]):
                    all_ai_data.append(processor.add_metadata(
                        sample,
                        domain=config["domain"],
                        source_type="ai"
                    ))
    
    print(f"\nTotal human samples: {len(all_human_data)}")
    print(f"Total AI samples: {len(all_ai_data)}")
    
    # Balance datasets
    min_samples = min(len(all_human_data), len(all_ai_data))
    if min_samples == 0:
        print("ERROR: No samples downloaded. Check your internet connection and dataset availability.")
        return None
    
    print(f"\nBalancing to {min_samples} samples per class...")
    
    import random
    random.seed(42)
    
    balanced_human = random.sample(all_human_data, min_samples)
    balanced_ai = random.sample(all_ai_data, min_samples)
    
    all_data = balanced_human + balanced_ai
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add document IDs for splitting
    df["doc_id"] = df["text"].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:10])
    
    print("\n=== Creating Train/Validation/Test Splits ===")
    
    splitter = DatasetSplitter()
    splits = splitter.create_splits(df, group_column="doc_id")
    
    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]
    
    print(f"\nSaving datasets to {output_path}/")
    
    train_df.to_parquet(output_path / "train.parquet", index=False)
    val_df.to_parquet(output_path / "validation.parquet", index=False)
    test_df.to_parquet(output_path / "test_in_dist.parquet", index=False)
    
    # Create specialized test sets
    print("\n=== Creating Specialized Test Sets ===")
    
    held_out_models = ["gpt4", "claude", "llama2", "gpt-4", "claude-2", "llama-2"]
    if "model_family" in test_df.columns:
        unseen_models_df = test_df[test_df["model_family"].isin(held_out_models)]
        if len(unseen_models_df) > 0:
            unseen_models_df.to_parquet(output_path / "test_unseen_models.parquet", index=False)
            print(f"  Unseen models: {len(unseen_models_df)} samples")
        else:
            ai_test = test_df[test_df["label"] == 1]
            if len(ai_test) > 0:
                unseen_models_df = ai_test.sample(min(500, len(ai_test)), random_state=42)
                unseen_models_df.to_parquet(output_path / "test_unseen_models.parquet", index=False)
                print(f"  Unseen models (synthetic): {len(unseen_models_df)} samples")
    
    for domain in df["domain"].unique():
        if domain is not None:
            domain_df = df[df["domain"] == domain]
            if len(domain_df) > 200:
                domain_test = domain_df.sample(min(500, len(domain_df)), random_state=42)
                domain_test.to_parquet(output_path / f"test_{domain}.parquet", index=False)
                print(f"  {domain}: {len(domain_test)} samples")
    
    short_df = df[df["word_count"] < 100]
    if len(short_df) > 100:
        short_df.to_parquet(output_path / "test_short.parquet", index=False)
        print(f"  Short text: {len(short_df)} samples")
    
    long_df = df[df["word_count"] > 1000]
    if len(long_df) > 100:
        long_df.to_parquet(output_path / "test_long.parquet", index=False)
        print(f"  Long text: {len(long_df)} samples")
    
    print("\n=== Dataset Statistics ===")
    print(f"Training set: {len(train_df)} samples")
    print(f"  - Human: {len(train_df[train_df['label'] == 0])}")
    print(f"  - AI: {len(train_df[train_df['label'] == 1])}")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set (in-dist): {len(test_df)} samples")
    print(f"\nAverage word count: {df['word_count'].mean():.1f}")
    print(f"Domains: {df['domain'].unique().tolist()}")
    if "model_family" in df.columns:
        print(f"Model families: {df['model_family'].dropna().unique().tolist()}")
    
    print(f"\n✓ Datasets saved to {output_path}/")
    return output_path


def verify_splits(output_dir: str):
    """Verify that splits don't have contamination."""
    
    output_path = Path(output_dir)
    
    if not (output_path / "train.parquet").exists():
        print("ERROR: Datasets not found. Run --download-all first.")
        return False
    
    train_df = pd.read_parquet(output_path / "train.parquet")
    test_df = pd.read_parquet(output_path / "test_in_dist.parquet")
    
    train_docs = set(train_df["doc_id"])
    test_docs = set(test_df["doc_id"])
    
    overlap = train_docs & test_docs
    
    if overlap:
        print(f"⚠ WARNING: {len(overlap)} documents appear in both train and test!")
        return False
    else:
        print("✓ No document contamination detected between train and test sets")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare AI detection datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download and process all datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DatasetConfig.HUMAN_DATASETS.keys()) + list(DatasetConfig.AI_DATASETS.keys()),
        help="Download specific dataset only",
    )
    parser.add_argument(
        "--verify-splits",
        action="store_true",
        help="Verify train/test splits for contamination",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of datasets",
    )
    
    args = parser.parse_args()
    
    if args.verify_splits:
        verify_splits(args.output_dir)
        return
    
    if args.download_all or args.dataset:
        prepare_all_datasets(args.output_dir, args.force_download)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
