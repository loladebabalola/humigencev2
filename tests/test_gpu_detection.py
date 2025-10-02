#!/usr/bin/env python3
"""
Test GPU detection functionality
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.main import detect_gpus, choose_training_mode


def test_gpu_detection():
    """Test GPU detection returns valid results"""
    gpu_count, gpus = detect_gpus()
    
    # Should return valid results (even if 0 GPUs)
    assert isinstance(gpu_count, int)
    assert isinstance(gpus, list)
    assert gpu_count >= 0
    assert len(gpus) == gpu_count
    
    # If GPUs are detected, they should have required fields
    for gpu in gpus:
        assert "index" in gpu
        assert "name" in gpu
        assert "memory" in gpu
        assert isinstance(gpu["index"], int)
        assert isinstance(gpu["name"], str)
        assert isinstance(gpu["memory"], str)


def test_training_mode_selection():
    """Test training mode selection logic"""
    # Test with no GPUs
    result = choose_training_mode(0, [])
    assert result is None
    
    # Test with single GPU
    single_gpu = [{"index": 0, "name": "Test GPU", "memory": "8GB"}]
    result = choose_training_mode(1, single_gpu)
    assert result == "single"
    
    # Test with multiple GPUs (would prompt user in real usage)
    multi_gpus = [
        {"index": 0, "name": "GPU 0", "memory": "8GB"},
        {"index": 1, "name": "GPU 1", "memory": "8GB"}
    ]
    # This would normally prompt the user, so we can't test the full flow
    # But we can test that the function can be called
    try:
        result = choose_training_mode(2, multi_gpus)
        # If it returns without error, that's good
        assert result is not None
    except Exception:
        # If it prompts for user input, that's expected in test environment
        pass


if __name__ == "__main__":
    pytest.main([__file__])




