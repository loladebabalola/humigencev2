# pipelines/memory_monitor.py

import torch
import gc
import psutil
import os
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

class MemoryMonitor:
    """Memory monitoring and error recovery for distributed training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)
        self.device = config.get("device", "cuda:0")
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_frequency = 10   # Cleanup every 10 steps
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_memory": 0.0, "cpu_memory": 0.0}
        
        # GPU memory
        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        gpu_memory = min(gpu_memory, 1.0)  # Cap at 100%
        
        # CPU memory
        cpu_memory = psutil.virtual_memory().percent / 100.0
        
        return {
            "gpu_memory": gpu_memory,
            "cpu_memory": cpu_memory
        }
    
    def should_cleanup(self, step: int) -> bool:
        """Check if memory cleanup is needed"""
        if step % self.cleanup_frequency != 0:
            return False
        
        memory_usage = self.check_memory_usage()
        return memory_usage["gpu_memory"] > self.memory_threshold
    
    def cleanup_memory(self) -> None:
        """Clean up GPU memory"""
        if not torch.cuda.is_available():
            return
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory usage
        memory_usage = self.check_memory_usage()
        if self.rank == 0:  # Only log from rank 0
            console.print(f"[blue]🧹 Memory cleanup: GPU {memory_usage['gpu_memory']:.1%}, CPU {memory_usage['cpu_memory']:.1%}[/blue]")
    
    def monitor_training_step(self, step: int, model, optimizer) -> bool:
        """Monitor training step and handle memory issues"""
        try:
            # Check if cleanup is needed
            if self.should_cleanup(step):
                self.cleanup_memory()
            
            # Check for OOM
            memory_usage = self.check_memory_usage()
            if memory_usage["gpu_memory"] > 0.95:  # 95% threshold for OOM
                console.print(f"[red]⚠️ High memory usage: {memory_usage['gpu_memory']:.1%}[/red]")
                self.cleanup_memory()
                
                # If still high, reduce batch size
                if memory_usage["gpu_memory"] > 0.90:
                    console.print("[yellow]⚠️ Reducing batch size due to memory pressure[/yellow]")
                    return False  # Signal to reduce batch size
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                console.print(f"[red]❌ OOM detected: {e}[/red]")
                self.cleanup_memory()
                return False  # Signal to reduce batch size
            else:
                raise e
    
    def handle_nccl_error(self, error: Exception) -> bool:
        """Handle NCCL errors with recovery"""
        error_str = str(error).lower()
        
        if "nccl" in error_str or "cuda error" in error_str:
            console.print(f"[red]❌ NCCL/CUDA error detected: {error}[/red]")
            
            # Clean up memory
            self.cleanup_memory()
            
            # Check if we can recover
            memory_usage = self.check_memory_usage()
            if memory_usage["gpu_memory"] < 0.80:  # If memory is low enough
                console.print("[yellow]🔄 Attempting recovery...[/yellow]")
                return True  # Try to recover
            else:
                console.print("[red]❌ Memory too high for recovery, falling back to single GPU[/red]")
                return False  # Fall back to single GPU
        
        return False  # Not an NCCL error
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        device = torch.device(self.device)
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_max = torch.cuda.max_memory_allocated(device)
        
        return {
            "gpu_available": True,
            "device": str(device),
            "memory_allocated": memory_allocated,
            "memory_reserved": memory_reserved,
            "memory_max": memory_max,
            "memory_allocated_gb": memory_allocated / 1024**3,
            "memory_reserved_gb": memory_reserved / 1024**3,
            "memory_max_gb": memory_max / 1024**3,
        }
    
    def log_memory_stats(self, step: int) -> None:
        """Log memory statistics"""
        if self.rank != 0:  # Only log from rank 0
            return
        
        stats = self.get_memory_stats()
        if stats["gpu_available"]:
            console.print(f"[blue]📊 Step {step}: GPU Memory - Allocated: {stats['memory_allocated_gb']:.2f}GB, "
                        f"Reserved: {stats['memory_reserved_gb']:.2f}GB, Max: {stats['memory_max_gb']:.2f}GB[/blue]")
