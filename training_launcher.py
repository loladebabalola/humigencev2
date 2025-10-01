# training_launcher.py
import os
import sys
import traceback
import json
import argparse
import torch
from pathlib import Path
from distributed_utils import setup_distributed, setup_environment, cleanup_distributed, RankZeroOnly

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Humigence Training Launcher")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--fallback_single_gpu", action="store_true", help="Force single GPU training")
    args = parser.parse_args()
    
    # Set default values for error handling
    ddp = False
    is_main = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set environment before ANY other imports
    setup_environment()
    
    try:
        # Initialize distributed training
        ddp, rank, local_rank, world_size, device = setup_distributed()
        is_main = (rank == 0)
        
        with RankZeroOnly(is_main) as rank_zero:
            rank_zero.print(f"Training Mode: {'DDP' if ddp else 'Single-GPU'} "
                          f"(world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device})")
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update config with distributed training info
        config.update({
            "device": str(device),
            "ddp": ddp,
            "rank": rank,
            "world_size": world_size,
            "is_main": is_main,
            "local_rank": local_rank,
        })
        
        # Import trainer after device setup to ensure proper CUDA initialization
        from pipelines.production_pipeline import ProductionPipeline
        
        # Create pipeline with distributed config
        pipeline = ProductionPipeline(config)
        
        # Run training
        results = pipeline.run()
        
        # Clean shutdown
        cleanup_distributed()
        
        return results
        
    except Exception as e:
        # Ensure cleanup even on error
        cleanup_distributed()
        
        # Enhanced error logging
        error_msg = f"Training error: {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        
        # Check if this is a DDP-related error that should trigger fallback
        if _should_fallback_to_single_gpu(e):
            if is_main:  # Now is_main is always defined
                print("DDP failed, falling back to single-GPU...")
            return _run_single_gpu_fallback(args.config)
        else:
            # Re-raise for actual errors
            raise

def _should_fallback_to_single_gpu(error: Exception) -> bool:
    """Determine if error warrants single-GPU fallback"""
    fallback_errors = (
        AttributeError,  # Missing methods like set_memory_monitor
        RuntimeError,    # NCCL errors, device mismatches
        ConnectionError, # Process group initialization failures
    )
    return isinstance(error, fallback_errors)

def _run_single_gpu_fallback(config_path: str):
    """Clean single-GPU fallback implementation"""
    # Force single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Clear any existing process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    # Load original config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update config for single GPU
    config.update({
        "device": "cuda:0",
        "ddp": False,
        "rank": 0,
        "world_size": 1,
        "is_main": True,
        "local_rank": 0,
        "multi_gpu": False,
        "use_distributed": False,
    })
    
    print("Running single-GPU fallback training...")
    
    try:
        from pipelines.production_pipeline import ProductionPipeline
        pipeline = ProductionPipeline(config)
        return pipeline.run()
    except Exception as e:
        print(f"Single-GPU fallback also failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    try:
        results = main()
        if results and results.get("status") == "success":
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
