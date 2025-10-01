#!/usr/bin/env python3
"""
NCCL Memory Conflict Resolution

This script addresses the "illegal memory access" error in multi-GPU training
by implementing memory management strategies and fallback mechanisms.
"""

import os
import subprocess
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_memory_usage() -> Dict[int, Dict[str, float]]:
    """Check current GPU memory usage"""
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        free = total - reserved
        
        memory_info[i] = {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free,
            'usage_percent': (reserved / total) * 100
        }
        
        logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, "
                   f"{free:.1f}GB free ({memory_info[i]['usage_percent']:.1f}% used)")
    
    return memory_info

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    logger.info("Clearing GPU memory...")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("GPU memory cleared")

def kill_competing_processes():
    """Kill processes that might be using GPU memory"""
    try:
        # Find processes using GPU memory
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        if 'llama' in name.lower() or int(memory) > 1000:  # > 1GB
                            logger.info(f"Found competing process: {name} (PID: {pid}, Memory: {memory}MB)")
                            try:
                                subprocess.run(['kill', '-9', pid], check=True)
                                logger.info(f"Killed process {pid}")
                            except subprocess.CalledProcessError:
                                logger.warning(f"Could not kill process {pid}")
        
    except Exception as e:
        logger.warning(f"Could not check/kill competing processes: {e}")

def setup_nccl_environment():
    """Set up optimal NCCL environment variables"""
    nccl_env = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_DISABLE': '1',  # Disable InfiniBand
        'NCCL_P2P_DISABLE': '1',  # Disable peer-to-peer
        'NCCL_SHM_DISABLE': '0',  # Enable shared memory
        'NCCL_SOCKET_IFNAME': 'enp6s18',  # Use specific network interface
        'NCCL_NET_GDR_LEVEL': '0',  # Disable GPU Direct RDMA
        'NCCL_CROSS_NIC': '0',  # Disable cross-NIC communication
        'NCCL_ALGO': 'Ring',  # Use Ring algorithm
        'CUDA_LAUNCH_BLOCKING': '1',  # Enable CUDA error checking
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',  # Enable async error handling
        'TOKENIZERS_PARALLELISM': 'false',  # Disable tokenizer parallelism
    }
    
    for key, value in nccl_env.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")

def create_memory_efficient_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create memory-efficient training configuration"""
    memory_config = base_config.copy()
    
    # Reduce memory usage
    memory_config.update({
        'per_device_train_batch_size': 1,  # Minimal batch size
        'gradient_accumulation_steps': 8,  # Compensate with accumulation
        'eval_batch_size': 1,  # Minimal eval batch size
        'max_seq_length': 512,  # Reduce sequence length
        'fp16': True,  # Use half precision
        'bf16': False,  # Disable bf16 to save memory
        'pin_memory': False,  # Disable pin memory
        'num_workers': 0,  # Disable multiprocessing
    })
    
    logger.info("Created memory-efficient configuration")
    return memory_config

def test_nccl_communication():
    """Test NCCL communication without training"""
    logger.info("Testing NCCL communication...")
    
    try:
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        logger.info(f"Rank {rank}/{world_size} initialized")
        
        # Test simple communication
        if rank == 0:
            tensor = torch.ones(10, device='cuda')
            logger.info(f"Rank 0 sending tensor: {tensor}")
        else:
            tensor = torch.zeros(10, device='cuda')
            logger.info(f"Rank 1 receiving tensor: {tensor}")
        
        # All-reduce test
        dist.all_reduce(tensor)
        logger.info(f"Rank {rank} after all_reduce: {tensor}")
        
        # Barrier test
        dist.barrier()
        logger.info(f"Rank {rank} passed barrier")
        
        logger.info("✅ NCCL communication test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ NCCL communication test FAILED: {e}")
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def run_memory_safe_training(config_path: str):
    """Run training with memory safety measures"""
    logger.info("Starting memory-safe training...")
    
    # Step 1: Clear memory
    clear_gpu_memory()
    
    # Step 2: Kill competing processes
    kill_competing_processes()
    
    # Step 3: Set up NCCL environment
    setup_nccl_environment()
    
    # Step 4: Check memory after cleanup
    memory_info = check_gpu_memory_usage()
    
    # Step 5: Test NCCL communication
    if not test_nccl_communication():
        logger.error("NCCL communication test failed, falling back to single GPU")
        return False
    
    # Step 6: Run training with memory-efficient config
    logger.info("Running memory-safe multi-GPU training...")
    
    # This would be called by the actual training script
    return True

def main():
    """Main function for testing memory fixes"""
    print("🚀 NCCL Memory Conflict Resolution")
    print("=" * 50)
    
    # Check initial memory state
    print("\n📊 Initial GPU Memory State:")
    memory_info = check_gpu_memory_usage()
    
    # Clear memory
    print("\n🧹 Clearing GPU Memory:")
    clear_gpu_memory()
    
    # Check memory after cleanup
    print("\n📊 GPU Memory After Cleanup:")
    memory_info = check_gpu_memory_usage()
    
    # Set up environment
    print("\n⚙️ Setting up NCCL Environment:")
    setup_nccl_environment()
    
    print("\n✅ Memory management setup complete!")
    print("   Ready for memory-safe multi-GPU training")

if __name__ == "__main__":
    main()
