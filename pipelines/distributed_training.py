# pipelines/distributed_training.py

import torch
import torch.distributed as dist
import os
from typing import Optional, Tuple
from rich.console import Console

console = Console()

def setup_distributed_training(rank: int, world_size: int, local_rank: int) -> torch.device:
    """
    Setup distributed training environment and return the device for this process.
    
    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        local_rank: Local rank of this process
        
    Returns:
        Device to use for this process
    """
    # Set device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    console.print(f"[blue]🖥️ Process {rank}/{world_size-1} using device: {device}[/blue]")
    
    return device

def ensure_device_consistency(model, device: torch.device) -> None:
    """
    Ensure model is on the correct device for distributed training.
    
    Args:
        model: Model to move to device
        device: Target device
    """
    # Move model to device
    model = model.to(device)
    
    # Verify all parameters are on the correct device
    for name, param in model.named_parameters():
        if param.device != device:
            console.print(f"[red]❌ Parameter {name} on wrong device: {param.device} vs {device}[/red]")
            raise RuntimeError(f"Model parameter {name} not on correct device")
    
    console.print(f"[green]✅ Model moved to {device}[/green]")

def sync_distributed_training() -> None:
    """Synchronize all processes in distributed training"""
    if dist.is_initialized():
        dist.barrier()
        console.print("[blue]🔄 Distributed training synchronized[/blue]")

def cleanup_distributed_training() -> None:
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
        console.print("[blue]🧹 Distributed training cleaned up[/blue]")

def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return dist.is_initialized() and dist.get_world_size() > 1

def get_rank() -> int:
    """Get current process rank"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    """Get total number of processes"""
    return dist.get_world_size() if dist.is_initialized() else 1

def get_local_rank() -> int:
    """Get local rank of this process"""
    return int(os.environ.get("LOCAL_RANK", 0))

def should_save_checkpoint(rank: int) -> bool:
    """Check if this process should save checkpoints (rank 0 only)"""
    return rank == 0

def should_log(rank: int) -> bool:
    """Check if this process should log (rank 0 only)"""
    return rank == 0

def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensor from all processes
    """
    if not is_distributed():
        return tensor
    
    # Create list to hold gathered tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    
    # Gather tensors from all processes
    dist.all_gather(gathered_tensors, tensor)
    
    # Concatenate all tensors
    return torch.cat(gathered_tensors, dim=0)

def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    # Reduce tensor across all processes
    dist.all_reduce(tensor, op=op)
    
    return tensor
