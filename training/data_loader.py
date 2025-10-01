#!/usr/bin/env python3
"""
Dataset-agnostic data loaders for Humigence training
Supports Wikitext, JSONL SFT datasets, and generic Hugging Face datasets
"""

import json
import os
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datasets import Dataset, load_dataset
from rich.console import Console
from abc import ABC, abstractmethod
import huggingface_hub

console = Console()


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, path_or_name: str, text_field: Optional[str] = None, **kwargs):
        self.path_or_name = path_or_name
        self.text_field = text_field
        self.kwargs = kwargs
        self.metadata = {}
    
    @abstractmethod
    def load(self, split: str = "train") -> Tuple[Dataset, Dataset]:
        """Load dataset and return train/eval splits"""
        pass
    
    @abstractmethod
    def preprocess(self, tokenizer, max_len: int = 1024) -> Tuple[Dataset, Dataset]:
        """Preprocess dataset for training"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not compute hash for {file_path}: {e}[/yellow]")
            return "unknown"
    
    def _get_dataset_commit_hash(self, dataset_name: str, config_name: Optional[str] = None) -> Optional[str]:
        """Get commit hash for a Hugging Face dataset"""
        try:
            # Try to get commit hash from dataset info
            from datasets import get_dataset_infos
            infos = get_dataset_infos(dataset_name)
            if infos and config_name in infos:
                info = infos[config_name]
                if hasattr(info, 'download_checksums') and info.download_checksums:
                    # Extract commit hash from download checksums
                    for checksum in info.download_checksums.values():
                        if 'commit_hash' in checksum:
                            return checksum['commit_hash']
            
            # Fallback: try to get from huggingface_hub
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                dataset_info = api.dataset_info(dataset_name)
                if hasattr(dataset_info, 'sha') and dataset_info.sha:
                    return dataset_info.sha
            except Exception:
                pass
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not get commit hash for {dataset_name}: {e}[/yellow]")
        
        return None
    
    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Split dataset into train/eval with smart handling for small datasets"""
        return self._split_dataset_with_fraction(dataset, 0.1)
    
    def _split_dataset_with_fraction(self, dataset: Dataset, eval_split: float) -> Tuple[Dataset, Dataset]:
        """Split dataset into train/eval using specified fraction"""
        if len(dataset) < 10:
            # For very small datasets, use all data for training and create a small eval set
            if len(dataset) == 1:
                # Single sample: use it for training, create a copy for eval
                train_dataset = dataset
                eval_dataset = dataset
            else:
                # Small dataset: use most for training, rest for eval
                split_ratio = max(eval_split, 1.0 / len(dataset))
                split_dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
        else:
            split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        return train_dataset, eval_dataset
    
    def _load_eval_file(self, eval_file: str) -> Dataset:
        """Load separate evaluation file"""
        data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️  Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
        
        if not data:
            raise ValueError(f"No valid data found in eval file {eval_file}")
        
        # Process data for SFT format
        processed_data = []
        for item in data:
            processed_item = self._process_sft_item(item)
            if processed_item:
                processed_data.append(processed_item)
        
        if not processed_data:
            raise ValueError(f"No valid data after processing eval file with SFT schema")
        
        return Dataset.from_list(processed_data)


class WikitextLoader(DatasetLoader):
    """Loader for Wikitext datasets"""
    
    def __init__(self, path_or_name: str, text_field: Optional[str] = None, **kwargs):
        super().__init__(path_or_name, text_field, **kwargs)
        self.dataset_config = kwargs.get('dataset_config', 'wikitext-2-raw-v1')
        self.metadata = {
            "dataset_type": "wikitext",
            "text_field": "text",
            "schema": "plain"
        }
    
    def load(self, split: str = "train") -> Tuple[Dataset, Dataset]:
        """Load Wikitext dataset"""
        console.print(f"[blue]📊 Loading Wikitext dataset: {self.dataset_config}[/blue]")
        
        # Load the dataset
        raw_dataset = load_dataset("wikitext", self.dataset_config)
        
        # Get commit hash for reproducibility
        commit_hash = self._get_dataset_commit_hash("wikitext", self.dataset_config)
        
        # Split into train/eval (typically 90/10 split)
        split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        console.print(f"[green]✅ Loaded Wikitext: {len(train_dataset)} train, {len(eval_dataset)} eval samples[/green]")
        if commit_hash:
            console.print(f"[blue]📦 Dataset pinned: wikitext@{self.dataset_config}@commit={commit_hash[:12]}...[/blue]")
        
        # Update metadata
        self.metadata.update({
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "total_size": len(train_dataset) + len(eval_dataset),
            "sha256": f"hf:{self.dataset_config}",  # HF datasets use name + config as identifier
            "dataset_name": self.dataset_config,
            "dataset_revision": "main",  # Default revision for HF datasets
            "commit_hash": commit_hash
        })
        
        return train_dataset, eval_dataset
    
    def preprocess(self, tokenizer, max_len: int = 1024) -> Tuple[Dataset, Dataset]:
        """Preprocess Wikitext dataset"""
        train_dataset, eval_dataset = self.load()
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        # Set format for PyTorch
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        return tokenized_train, tokenized_eval


class JsonlSFTLoader(DatasetLoader):
    """Loader for JSONL SFT datasets"""
    
    def __init__(self, path_or_name: str, text_field: Optional[str] = None, **kwargs):
        super().__init__(path_or_name, text_field, **kwargs)
        self.schema = kwargs.get('schema', 'sft')
        self.metadata = {
            "dataset_type": "jsonl",
            "text_field": "text",
            "schema": "sft"
        }
    
    def load(self, split: str = "train") -> Tuple[Dataset, Dataset]:
        """Load JSONL SFT dataset"""
        console.print(f"[blue]📊 Loading JSONL SFT dataset: {self.path_or_name}[/blue]")
        
        if not os.path.exists(self.path_or_name):
            raise FileNotFoundError(f"JSONL file not found: {self.path_or_name}")
        
        # Read and parse JSONL
        data = []
        with open(self.path_or_name, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️  Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
        
        if not data:
            raise ValueError(f"No valid data found in {self.path_or_name}")
        
        # Process data for SFT format
        processed_data = []
        for item in data:
            processed_item = self._process_sft_item(item)
            if processed_item:
                processed_data.append(processed_item)
        
        if not processed_data:
            raise ValueError(f"No valid data after processing with SFT schema")
        
        # Convert to Dataset
        dataset = Dataset.from_list(processed_data)
        
        # Check for separate eval file
        eval_file = self.kwargs.get('eval_file')
        if eval_file and os.path.exists(eval_file):
            console.print(f"[blue]📊 Loading separate eval file: {eval_file}[/blue]")
            eval_dataset = self._load_eval_file(eval_file)
        else:
            # Split into train/eval using fraction
            eval_split = self.kwargs.get('eval_split', 0.1)
            train_dataset, eval_dataset = self._split_dataset_with_fraction(dataset, eval_split)
        
        console.print(f"[green]✅ Loaded JSONL SFT: {len(train_dataset)} train, {len(eval_dataset)} eval samples[/green]")
        
        # Update metadata
        self.metadata.update({
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "total_size": len(train_dataset) + len(eval_dataset),
            "file_path": self.path_or_name,
            "sha256": self._compute_file_hash(self.path_or_name),
            "eval_split": self.kwargs.get('eval_split', 0.1),
            "eval_file": eval_file
        })
        
        return train_dataset, eval_dataset
    
    def _process_sft_item(self, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Process SFT item"""
        try:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if not instruction or not output:
                return None
            
            # Format as prompt/response
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            return {
                "text": prompt + output,
                "prompt": prompt,
                "response": output
            }
        except Exception as e:
            console.print(f"[yellow]⚠️  Error processing SFT item: {e}[/yellow]")
            return None
    
    def preprocess(self, tokenizer, max_len: int = 1024) -> Tuple[Dataset, Dataset]:
        """Preprocess JSONL SFT dataset"""
        train_dataset, eval_dataset = self.load()
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        # Set format for PyTorch
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        return tokenized_train, tokenized_eval


class JsonlDialogueLoader(DatasetLoader):
    """Loader for JSONL Dialogue datasets with role-aware tokenization"""
    
    def __init__(self, path_or_name: str, text_field: Optional[str] = None, **kwargs):
        super().__init__(path_or_name, text_field, **kwargs)
        self.schema = kwargs.get('schema', 'dialogue')
        self.role_markers = kwargs.get('role_markers', True)
        self.user_marker = kwargs.get('user_marker', '<user>')
        self.assistant_marker = kwargs.get('assistant_marker', '<assistant>')
        self.metadata = {
            "dataset_type": "jsonl",
            "text_field": "text",
            "schema": "dialogue",
            "role_markers": self.role_markers
        }
    
    def load(self, split: str = "train") -> Tuple[Dataset, Dataset]:
        """Load JSONL Dialogue dataset"""
        console.print(f"[blue]📊 Loading JSONL Dialogue dataset: {self.path_or_name}[/blue]")
        
        if not os.path.exists(self.path_or_name):
            raise FileNotFoundError(f"JSONL file not found: {self.path_or_name}")
        
        # Read and parse JSONL
        data = []
        with open(self.path_or_name, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️  Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                    continue
        
        if not data:
            raise ValueError(f"No valid data found in {self.path_or_name}")
        
        # Process data for dialogue format
        processed_data = []
        for item in data:
            processed_item = self._process_dialogue_item(item)
            if processed_item:
                processed_data.append(processed_item)
        
        if not processed_data:
            raise ValueError(f"No valid data after processing with dialogue schema")
        
        # Convert to Dataset
        dataset = Dataset.from_list(processed_data)
        
        # Split into train/eval
        train_dataset, eval_dataset = self._split_dataset(dataset)
        
        console.print(f"[green]✅ Loaded JSONL Dialogue: {len(train_dataset)} train, {len(eval_dataset)} eval samples[/green]")
        console.print(f"[blue]📋 Role markers: {self.user_marker} / {self.assistant_marker}[/blue]")
        
        # Update metadata
        self.metadata.update({
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "total_size": len(train_dataset) + len(eval_dataset),
            "file_path": self.path_or_name,
            "sha256": self._compute_file_hash(self.path_or_name),
            "user_marker": self.user_marker,
            "assistant_marker": self.assistant_marker
        })
        
        return train_dataset, eval_dataset
    
    def _process_dialogue_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process dialogue item with role-aware tokenization"""
        try:
            messages = item.get("messages", [])
            if not messages or not isinstance(messages, list):
                return None
            
            # Flatten dialogue with role markers
            dialogue_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    if self.role_markers:
                        if role == "user":
                            dialogue_text += f"{self.user_marker}: {content}\n"
                        elif role == "assistant":
                            dialogue_text += f"{self.assistant_marker}: {content}\n"
                        else:
                            dialogue_text += f"{role}: {content}\n"
                    else:
                        dialogue_text += f"{role}: {content}\n"
            
            if not dialogue_text.strip():
                return None
            
            return {
                "text": dialogue_text.strip(),
                "messages": messages,
                "role_markers": self.role_markers
            }
        except Exception as e:
            console.print(f"[yellow]⚠️  Error processing dialogue item: {e}[/yellow]")
            return None
    
    def preprocess(self, tokenizer, max_len: int = 1024) -> Tuple[Dataset, Dataset]:
        """Preprocess JSONL Dialogue dataset with special token support"""
        train_dataset, eval_dataset = self.load()
        
        # Add special tokens if role markers are enabled
        if self.role_markers:
            special_tokens = [self.user_marker, self.assistant_marker]
            # Check if tokens are already in the tokenizer
            existing_tokens = set(tokenizer.special_tokens_map.get("additional_special_tokens", []))
            new_tokens = [token for token in special_tokens if token not in existing_tokens]
            
            if new_tokens:
                console.print(f"[blue]🔧 Adding special tokens: {new_tokens}[/blue]")
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                
                # Resize model embeddings if tokenizer has a model
                if hasattr(tokenizer, 'model') and tokenizer.model is not None:
                    # This will be handled by the training code
                    console.print(f"[blue]📏 Tokenizer vocabulary size: {len(tokenizer)}[/blue]")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        # Set format for PyTorch
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        return tokenized_train, tokenized_eval


class HFTextLoader(DatasetLoader):
    """Loader for generic Hugging Face text datasets"""
    
    def __init__(self, path_or_name: str, text_field: Optional[str] = None, **kwargs):
        super().__init__(path_or_name, text_field, **kwargs)
        self.text_field = text_field or "text"
        self.split = kwargs.get('split', None)
        self.metadata = {
            "dataset_type": "hf",
            "text_field": self.text_field,
            "schema": "plain"
        }
    
    def load(self, split: str = "train") -> Tuple[Dataset, Dataset]:
        """Load Hugging Face dataset"""
        console.print(f"[blue]📊 Loading HF dataset: {self.path_or_name}[/blue]")
        
        # Get commit hash for reproducibility
        commit_hash = self._get_dataset_commit_hash(self.path_or_name)
        
        # Load dataset
        if self.split:
            raw_dataset = load_dataset(self.path_or_name, split=self.split)
            # If single split, split it
            if isinstance(raw_dataset, Dataset):
                split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
            else:
                # Multiple splits provided
                train_dataset = raw_dataset.get("train")
                eval_dataset = raw_dataset.get("validation") or raw_dataset.get("test")
                if eval_dataset is None:
                    # Create eval split from train
                    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                    train_dataset = split_dataset["train"]
                    eval_dataset = split_dataset["test"]
        else:
            raw_dataset = load_dataset(self.path_or_name)
            
            # Try to get train and eval splits
            train_dataset = raw_dataset.get("train")
            eval_dataset = raw_dataset.get("validation") or raw_dataset.get("test")
            
            if train_dataset is None:
                raise ValueError(f"No 'train' split found in dataset {self.path_or_name}")
            
            if eval_dataset is None:
                # Create eval split from train
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
        
        # Validate text field exists
        if self.text_field not in train_dataset.features:
            available_fields = list(train_dataset.features.keys())
            raise ValueError(f"Text field '{self.text_field}' not found. Available fields: {available_fields}")
        
        console.print(f"[green]✅ Loaded HF dataset: {len(train_dataset)} train, {len(eval_dataset)} eval samples[/green]")
        console.print(f"[blue]📋 Text field: {self.text_field}[/blue]")
        if commit_hash:
            console.print(f"[blue]📦 Dataset pinned: {self.path_or_name}@commit={commit_hash[:12]}...[/blue]")
        
        # Update metadata
        self.metadata.update({
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "total_size": len(train_dataset) + len(eval_dataset),
            "dataset_name": self.path_or_name,
            "sha256": f"hf:{self.path_or_name}",  # HF datasets use name as identifier
            "dataset_revision": "main",  # Default revision for HF datasets
            "commit_hash": commit_hash
        })
        
        return train_dataset, eval_dataset
    
    def preprocess(self, tokenizer, max_len: int = 1024) -> Tuple[Dataset, Dataset]:
        """Preprocess HF dataset"""
        train_dataset, eval_dataset = self.load()
        
        def tokenize_function(examples):
            return tokenizer(
                examples[self.text_field],
                truncation=True,
                padding="max_length",
                max_length=max_len
            )
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        # Set format for PyTorch
        tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
        tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        return tokenized_train, tokenized_eval


# Schema Registry
SCHEMAS = {
    "wikitext": WikitextLoader,
    "sft": JsonlSFTLoader,
    "dialogue": JsonlDialogueLoader,
    "plain": HFTextLoader,
}


def detect_dataset_schema(dataset_spec: str) -> str:
    """
    Detect dataset schema from specification.
    
    Args:
        dataset_spec: Dataset specification
        
    Returns:
        Detected schema type
    """
    if dataset_spec == "wikitext":
        return "wikitext"
    elif dataset_spec.startswith("jsonl:"):
        return "jsonl"
    elif dataset_spec.startswith("hf:"):
        return "hf"
    else:
        # Assume it's a direct HF dataset name
        return "hf"


def detect_jsonl_schema(file_path: str) -> str:
    """
    Auto-detect JSONL schema from a sample item.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Detected schema type
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                sample_item = json.loads(first_line)
                keys = set(sample_item.keys())
                
                # Check for SFT schema
                if "instruction" in keys and "output" in keys:
                    return "sft"
                
                # Check for dialogue schema
                if "messages" in keys and isinstance(sample_item["messages"], list):
                    return "dialogue"
                
                # Check for simple text schema
                if "text" in keys:
                    return "plain"
    except Exception as e:
        console.print(f"[yellow]⚠️  Error detecting schema: {e}[/yellow]")
    
    # Default to plain if we can't detect
    console.print("[yellow]⚠️  Could not auto-detect schema, defaulting to 'plain'[/yellow]")
    return "plain"


def create_dataset_loader(dataset_spec: str, text_field: Optional[str] = None, schema: Optional[str] = None, 
                         eval_split: Optional[float] = None, eval_file: Optional[str] = None, **kwargs) -> DatasetLoader:
    """
    Create appropriate dataset loader based on specification.
    
    Args:
        dataset_spec: Dataset specification
        text_field: Text field for HF datasets
        schema: Schema for JSONL datasets
        eval_split: Fraction of data to use for evaluation (0.0-1.0)
        eval_file: Path to separate evaluation file (for JSONL)
        **kwargs: Additional loader parameters
        
    Returns:
        DatasetLoader instance
    """
    detected_schema = detect_dataset_schema(dataset_spec)
    
    # Add eval parameters to kwargs
    if eval_split is not None:
        kwargs['eval_split'] = eval_split
    if eval_file is not None:
        kwargs['eval_file'] = eval_file
    
    if detected_schema == "wikitext":
        return WikitextLoader(dataset_spec, text_field, **kwargs)
    
    elif detected_schema == "jsonl":
        file_path = dataset_spec[6:]  # Remove "jsonl:" prefix
        if not schema or schema == "auto":
            schema = detect_jsonl_schema(file_path)
            console.print(f"[blue]🔍 Auto-detected schema: {schema}[/blue]")
        
        if schema == "sft":
            return JsonlSFTLoader(file_path, text_field, schema=schema, **kwargs)
        elif schema == "dialogue":
            return JsonlDialogueLoader(file_path, text_field, schema=schema, **kwargs)
        else:
            # Default to plain text processing
            return JsonlDialogueLoader(file_path, text_field, schema="plain", **kwargs)
    
    elif detected_schema == "hf":
        dataset_name = dataset_spec[3:] if dataset_spec.startswith("hf:") else dataset_spec
        return HFTextLoader(dataset_name, text_field, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset specification: {dataset_spec}")


def auto_load_dataset(dataset_spec: str, text_field: Optional[str] = None, schema: Optional[str] = None, 
                     eval_split: Optional[float] = None, eval_file: Optional[str] = None, **kwargs) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
    """
    Automatically load dataset using schema registry.
    
    Args:
        dataset_spec: Dataset specification (wikitext, jsonl:path, hf:name)
        text_field: Text field for HF datasets
        schema: Schema for JSONL datasets
        eval_split: Fraction of data to use for evaluation (0.0-1.0)
        eval_file: Path to separate evaluation file (for JSONL)
        **kwargs: Additional loader parameters
        
    Returns:
        Tuple of (train_dataset, eval_dataset, metadata)
    """
    # Create loader
    loader = create_dataset_loader(dataset_spec, text_field, schema, eval_split, eval_file, **kwargs)
    
    # Load dataset
    train_dataset, eval_dataset = loader.load()
    
    # Get metadata
    metadata = loader.get_metadata()
    metadata["dataset_spec"] = dataset_spec
    
    return train_dataset, eval_dataset, metadata


# Legacy functions for backward compatibility
def load_wikitext(dataset_config: str = "wikitext-2-raw-v1") -> Tuple[Dataset, Dataset]:
    """Legacy function - use auto_load_dataset instead"""
    loader = WikitextLoader("wikitext", dataset_config=dataset_config)
    return loader.load()


def load_jsonl(file_path: str, schema: str = "auto") -> Tuple[Dataset, Dataset]:
    """Legacy function - use auto_load_dataset instead"""
    loader = create_dataset_loader(f"jsonl:{file_path}", schema=schema)
    return loader.load()


def load_hf_dataset(dataset_name: str, text_field: str = "text", split: Optional[str] = None) -> Tuple[Dataset, Dataset]:
    """Legacy function - use auto_load_dataset instead"""
    loader = HFTextLoader(dataset_name, text_field, split=split)
    return loader.load()
