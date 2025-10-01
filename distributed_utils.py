# distributed_utils.py
import os
import torch
import torch.distributed as dist
import logging
from typing import Tuple, Optional
from compatibility_utils import setup_timeout

def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    """
    First-principles DDP setup with single source of truth for device mapping.
    Returns: (is_ddp, rank, local_rank, world_size, device)
    """
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    if ddp:
        # Initialize process group with robust timeout
        if not dist.is_initialized():
            # Use compatibility-aware timeout
            timeout = setup_timeout()
            dist.init_process_group(
                backend="nccl",
                timeout=timeout
            )
        
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Critical: Set device BEFORE any CUDA operations
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        
        # Verify device mapping
        assert torch.cuda.current_device() == local_rank, \
            f"Device mapping error: current={torch.cuda.current_device()}, local_rank={local_rank}"
        
    else:
        local_rank, rank, world_size = 0, 0, 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return ddp, rank, local_rank, world_size, device

def setup_environment():
    """Set environment variables once at process start"""
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")  # Modern replacement
    os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand on single node
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Prevent tokenizer conflicts
    
    # Remove deprecated variables
    if "NCCL_ASYNC_ERROR_HANDLING" in os.environ:
        del os.environ["NCCL_ASYNC_ERROR_HANDLING"]
    
    # Do NOT set NCCL_P2P_DISABLE - allow peer-to-peer on single node

def cleanup_distributed():
    """Clean shutdown of process group"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception as e:
            logging.warning(f"Cleanup warning: {e}")

class RankZeroOnly:
    """Context manager for rank-0 only execution"""
    def __init__(self, is_main: bool):
        self.is_main = is_main
        self.original_level = None
    
    def __enter__(self):
        if not self.is_main:
            # Suppress logging for non-main ranks
            self.original_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.WARNING)
        return self
    
    def __exit__(self, *args):
        if not self.is_main and self.original_level is not None:
            logging.getLogger().setLevel(self.original_level)
    
    def print(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs)
