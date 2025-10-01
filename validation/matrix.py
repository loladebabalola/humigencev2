"""
Validation matrix for model, precision, and GPU capabilities
"""

import importlib
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

PRECISIONS = {"fp32", "fp16", "bf16", "qlora4bit"}


@dataclass
class GpuInfo:
    available: bool
    name: str
    total_bytes: int
    free_bytes: int
    cc_major: int
    cc_minor: int
    bf16_supported: bool
    device_index: int = 0


@dataclass
class MultiGpuInfo:
    """Information about all available GPUs"""
    gpus: list[GpuInfo]
    count: int
    total_vram_gb: float
    
    def __post_init__(self):
        self.count = len(self.gpus)
        self.total_vram_gb = sum(gpu.total_bytes for gpu in self.gpus) / (1024**3)


def get_gpu_info(device: int = 0) -> GpuInfo:
    """Get GPU information and capabilities for a specific device"""
    if not torch.cuda.is_available():
        return GpuInfo(False, "cpu", 0, 0, 0, 0, False, device)
    
    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free = torch.cuda.mem_get_info(device)[0]
    major, minor = torch.cuda.get_device_capability(device)
    bf16_ok = torch.cuda.is_bf16_supported()
    
    return GpuInfo(True, name, total, free, major, minor, bf16_ok, device)


def get_all_gpu_info() -> MultiGpuInfo:
    """Get information about all available GPUs"""
    if not torch.cuda.is_available():
        return MultiGpuInfo([], 0, 0.0)
    
    gpus = []
    device_count = torch.cuda.device_count()
    
    for device_idx in range(device_count):
        gpu_info = get_gpu_info(device_idx)
        gpus.append(gpu_info)
    
    return MultiGpuInfo(gpus, device_count, sum(gpu.total_bytes for gpu in gpus) / (1024**3))


def has_bitsandbytes() -> bool:
    """Check if bitsandbytes is available"""
    try:
        importlib.import_module("bitsandbytes")
        return True
    except Exception:
        return False


def precision_supported(precision: str, gpu: GpuInfo) -> Tuple[bool, str]:
    """Check if precision is supported on the given GPU"""
    if precision == "bf16" and not gpu.bf16_supported:
        return False, "bf16 is not supported on this GPU. Try fp16."
    
    if precision == "qlora4bit":
        if not gpu.available:
            return False, "4-bit requires CUDA GPU. Try fp16 on CPU is not supported for 4-bit."
        if not has_bitsandbytes():
            return False, "bitsandbytes not installed. `pip install bitsandbytes` or use fp16."
        if (gpu.cc_major, gpu.cc_minor) < (7, 0):
            return False, f"Compute capability {gpu.cc_major}.{gpu.cc_minor} may be insufficient for 4-bit. Use fp16."
    
    # fp16/fp32 are generally ok if CUDA present (or CPU for fp32)
    return True, "ok"


def estimate_model_params(config) -> Optional[int]:
    """Estimate model parameters from config"""
    try:
        hs = int(getattr(config, "hidden_size", 0))
        nl = int(getattr(config, "num_hidden_layers", 0))
        vs = int(getattr(config, "vocab_size", 0))
        
        if hs == 0 or nl == 0 or vs == 0:
            return None
            
        # rough param estimate: attention/MLP per layer + embeddings
        per_layer = 12 * hs * hs  # coarse
        total = per_layer * nl + vs * hs
        return total
    except Exception:
        return None


def bytes_per_param(precision: str) -> float:
    """Get bytes per parameter for given precision"""
    return {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "qlora4bit": 0.5,  # model weights quantized; activations use higher precision at runtime
    }.get(precision, 2)


def estimate_memory_bytes(params: Optional[int], precision: str, adam: bool, lora: bool) -> Optional[int]:
    """Estimate memory usage in bytes"""
    if params is None:
        return None
    
    base = params * bytes_per_param(precision)
    # optimizer & gradients (very rough; LoRA reduces trainable params a lot)
    overhead = 2.0 if adam else 0.6
    if lora:
        overhead *= 0.3  # fewer trainable params
    
    return int(base * (1.0 + overhead))


def tokenizer_ok(model_id_or_path: str) -> Tuple[bool, str]:
    """Check if tokenizer can be loaded and is properly configured"""
    try:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None:
            # safe default: set pad_token to eos_token
            if getattr(tok, "eos_token", None):
                return True, "No pad_token; will use eos_token for padding."
            return False, "Tokenizer missing pad_token and eos_token."
        return True, "ok"
    except Exception as e:
        return False, f"Tokenizer load failed: {e}"



