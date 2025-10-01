"""
Custom error handling for Humigence training pipeline
"""
import torch
import torch.distributed as dist
from typing import Optional

class HumigenceError(Exception):
    """Base exception for Humigence training errors"""
    def __init__(self, message: str, suggested_fix: Optional[str] = None):
        super().__init__(message)
        self.suggested_fix = suggested_fix

class ConfigurationError(HumigenceError):
    """Configuration validation errors"""
    pass

class DatasetError(HumigenceError):
    """Dataset loading and processing errors"""
    pass

class ModelError(HumigenceError):
    """Model loading and setup errors"""
    pass

class TrainingError(HumigenceError):
    """Training process errors"""
    pass

class EvaluationError(HumigenceError):
    """Evaluation process errors"""
    pass

class DistributedError(HumigenceError):
    """Distributed training errors"""
    pass

def handle_cuda_error(error: Exception) -> HumigenceError:
    """Convert CUDA errors to HumigenceError with suggested fixes"""
    error_msg = str(error)
    
    if "out of memory" in error_msg.lower():
        return TrainingError(
            "CUDA out of memory",
            "Reduce batch size or use gradient checkpointing"
        )
    elif "illegal memory access" in error_msg.lower():
        return DistributedError(
            "NCCL illegal memory access",
            "Reduce batch size or retry single-GPU mode"
        )
    elif "device" in error_msg.lower() and "mismatch" in error_msg.lower():
        return TrainingError(
            "Device mismatch detected",
            "Ensure all tensors are on the same device"
        )
    else:
        return TrainingError(f"CUDA error: {error_msg}")

def handle_distributed_error(error: Exception) -> HumigenceError:
    """Convert distributed training errors to HumigenceError"""
    error_msg = str(error)
    
    if "nccl" in error_msg.lower():
        return DistributedError(
            "NCCL communication error",
            "Check network configuration or retry single-GPU mode"
        )
    elif "process group" in error_msg.lower():
        return DistributedError(
            "Process group initialization failed",
            "Check distributed setup or retry single-GPU mode"
        )
    else:
        return DistributedError(f"Distributed training error: {error_msg}")

def handle_model_error(error: Exception) -> HumigenceError:
    """Convert model-related errors to HumigenceError"""
    error_msg = str(error)
    
    if "out of memory" in error_msg.lower():
        return ModelError(
            "Model loading out of memory",
            "Use smaller model or enable model sharding"
        )
    elif "not found" in error_msg.lower():
        return ModelError(
            "Model not found",
            "Check model name or download the model first"
        )
    else:
        return ModelError(f"Model error: {error_msg}")

def handle_dataset_error(error: Exception) -> HumigenceError:
    """Convert dataset-related errors to HumigenceError"""
    error_msg = str(error)
    
    if "not found" in error_msg.lower():
        return DatasetError(
            "Dataset file not found",
            "Check dataset path and ensure file exists"
        )
    elif "column" in error_msg.lower() and "not in" in error_msg.lower():
        return DatasetError(
            "Dataset column mismatch",
            "Check dataset schema and column names"
        )
    else:
        return DatasetError(f"Dataset error: {error_msg}")

def clean_error_message(error: HumigenceError) -> str:
    """Create a clean error message with suggested fix"""
    message = f"❌ {error.__class__.__name__}: {error}"
    
    if error.suggested_fix:
        message += f"\n   Suggested fix: {error.suggested_fix}"
    
    return message
