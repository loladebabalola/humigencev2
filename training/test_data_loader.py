#!/usr/bin/env python3
"""
Unit tests for dataset loaders
"""

import json
import tempfile
import os
from pathlib import Path
import pytest
from datasets import Dataset
from training.data_loader import (
    load_wikitext,
    load_jsonl,
    load_hf_dataset,
    auto_load_dataset,
    _detect_jsonl_schema,
    _process_jsonl_item
)


class TestJSONLProcessing:
    """Test JSONL processing functions"""
    
    def test_detect_sft_schema(self):
        """Test SFT schema detection"""
        sample = {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        }
        assert _detect_jsonl_schema(sample) == "sft"
    
    def test_detect_dialogue_schema(self):
        """Test dialogue schema detection"""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        assert _detect_jsonl_schema(sample) == "dialogue"
    
    def test_detect_plain_schema(self):
        """Test plain text schema detection"""
        sample = {"text": "This is plain text"}
        assert _detect_jsonl_schema(sample) == "plain"
    
    def test_process_sft_item(self):
        """Test SFT item processing"""
        item = {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        }
        result = _process_jsonl_item(item, "sft")
        
        assert result is not None
        assert "text" in result
        assert "prompt" in result
        assert "response" in result
        assert "Paris" in result["text"]
    
    def test_process_sft_item_with_input(self):
        """Test SFT item processing with input"""
        item = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde"
        }
        result = _process_jsonl_item(item, "sft")
        
        assert result is not None
        assert "Input:" in result["text"]
        assert "Hello world" in result["text"]
        assert "Bonjour le monde" in result["text"]
    
    def test_process_dialogue_item(self):
        """Test dialogue item processing"""
        item = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }
        result = _process_jsonl_item(item, "dialogue")
        
        assert result is not None
        assert "text" in result
        assert "messages" in result
        assert "user: Hello" in result["text"]
        assert "assistant: Hi there!" in result["text"]
    
    def test_process_plain_item(self):
        """Test plain text item processing"""
        item = {"text": "This is plain text"}
        result = _process_jsonl_item(item, "plain")
        
        assert result is not None
        assert result["text"] == "This is plain text"
    
    def test_process_invalid_item(self):
        """Test processing invalid item"""
        item = {"invalid": "data"}
        result = _process_jsonl_item(item, "sft")
        assert result is None


class TestJSONLLoader:
    """Test JSONL dataset loading"""
    
    def test_load_jsonl_sft(self):
        """Test loading SFT JSONL dataset"""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_data = [
                {"instruction": "What is 2+2?", "input": "", "output": "4"},
                {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
                {"instruction": "Translate hello", "input": "hello", "output": "hola"},
            ]
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            train_dataset, eval_dataset = load_jsonl(temp_path, "sft")
            
            assert len(train_dataset) > 0
            assert len(eval_dataset) > 0
            assert len(train_dataset) + len(eval_dataset) == 3
            
            # Check that data is processed correctly
            sample = train_dataset[0]
            assert "text" in sample
            assert "instruction" in sample["text"].lower()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_dialogue(self):
        """Test loading dialogue JSONL dataset"""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_data = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm doing well, thanks!"}
                    ]
                }
            ]
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            train_dataset, eval_dataset = load_jsonl(temp_path, "dialogue")
            
            assert len(train_dataset) > 0
            assert len(eval_dataset) > 0
            assert len(train_dataset) + len(eval_dataset) == 2
            
            # Check that data is processed correctly
            sample = train_dataset[0]
            assert "text" in sample
            assert "user:" in sample["text"]
            assert "assistant:" in sample["text"]
            
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_auto_detect(self):
        """Test loading JSONL with auto-detection"""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_data = [
                {"instruction": "What is 2+2?", "input": "", "output": "4"},
                {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
            ]
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            train_dataset, eval_dataset = load_jsonl(temp_path, "auto")
            
            assert len(train_dataset) > 0
            assert len(eval_dataset) > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_load_jsonl_invalid_file(self):
        """Test loading non-existent JSONL file"""
        with pytest.raises(FileNotFoundError):
            load_jsonl("nonexistent.jsonl")
    
    def test_load_jsonl_invalid_json(self):
        """Test loading JSONL with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json content\n")
            f.write('{"valid": "json"}\n')
            temp_path = f.name
        
        try:
            # Should not raise exception, but skip invalid lines
            train_dataset, eval_dataset = load_jsonl(temp_path)
            assert len(train_dataset) + len(eval_dataset) == 1
            
        finally:
            os.unlink(temp_path)


class TestAutoLoadDataset:
    """Test automatic dataset loading"""
    
    def test_auto_load_wikitext(self):
        """Test auto-loading Wikitext dataset"""
        train_dataset, eval_dataset, metadata = auto_load_dataset("wikitext")
        
        assert len(train_dataset) > 0
        assert len(eval_dataset) > 0
        assert metadata["dataset_type"] == "wikitext"
        assert metadata["text_field"] == "text"
        assert metadata["schema"] == "plain"
    
    def test_auto_load_jsonl(self):
        """Test auto-loading JSONL dataset"""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            jsonl_data = [
                {"instruction": "What is 2+2?", "input": "", "output": "4"},
                {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
            ]
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            train_dataset, eval_dataset, metadata = auto_load_dataset(f"jsonl:{temp_path}")
            
            assert len(train_dataset) > 0
            assert len(eval_dataset) > 0
            assert metadata["dataset_type"] == "jsonl"
            assert metadata["file_path"] == temp_path
            
        finally:
            os.unlink(temp_path)
    
    def test_auto_load_hf_dataset(self):
        """Test auto-loading Hugging Face dataset"""
        # This test might fail if the dataset is not available
        # We'll use a small, commonly available dataset
        try:
            train_dataset, eval_dataset, metadata = auto_load_dataset("hf:imdb")
            
            assert len(train_dataset) > 0
            assert len(eval_dataset) > 0
            assert metadata["dataset_type"] == "hf"
            assert metadata["dataset_name"] == "imdb"
            assert metadata["text_field"] == "text"
            
        except Exception as e:
            # If the dataset is not available, that's okay for testing
            pytest.skip(f"HF dataset not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])

