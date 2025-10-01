"""
Dual-GPU LoRA Training Launcher for Humigence
Provides programmatic interface for launching dual-GPU training
"""

import os
import subprocess
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return system info"""
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": "PyTorch not installed"
        }
    
    if not torch.cuda.is_available():
        return {
            "available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": "CUDA not available"
        }
    
    gpu_count = torch.cuda.device_count()
    gpus = []
    
    for i in range(gpu_count):
        gpu_info = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB"
        }
        gpus.append(gpu_info)
    
    return {
        "available": True,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "error": None
    }

def setup_environment():
    """Setup environment variables for dual-GPU training"""
    # Set environment variables for optimal performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def launch_dual_gpu_training(
    model_name: str = "unsloth/Llama-3-8B-Instruct",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    output_dir: str = "./runs/humigence/out_lora_dual",
    max_steps: int = 1000,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    block_size: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    precision: str = "qlora_4bit",
    launch_method: str = "torchrun"
) -> Dict[str, Any]:
    """
    Launch dual-GPU LoRA training with Unsloth
    
    Args:
        model_name: Model to fine-tune (default: unsloth/Llama-3-8B-Instruct)
        dataset_name: Dataset name (default: wikitext)
        dataset_config: Dataset configuration (default: wikitext-2-raw-v1)
        output_dir: Output directory for results
        max_steps: Maximum training steps
        batch_size: Batch size per device
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        block_size: Sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        precision: Training precision ("qlora_4bit", "lora_fp16", "lora_bf16")
        launch_method: Launch method ("torchrun" or "accelerate")
    
    Returns:
        Dict with training results and status
    """
    
    console.print(Panel.fit(
        "[bold cyan]🚀 Humigence Dual-GPU LoRA Training[/bold cyan]\n"
        f"Model: {model_name}\n"
        f"Dataset: {dataset_name}/{dataset_config}\n"
        f"Output: {output_dir}\n"
        f"Method: {launch_method}",
        title="Training Configuration"
    ))
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    if not gpu_info["available"]:
        return {
            "status": "error",
            "error": gpu_info["error"],
            "gpu_info": gpu_info
        }
    
    if gpu_info["gpu_count"] < 2:
        console.print(f"[yellow]⚠️ Only {gpu_info['gpu_count']} GPU(s) detected. Dual-GPU training may not be optimal.[/yellow]")
    
    # Display GPU info
    console.print(f"[blue]🖥️ Available GPUs: {gpu_info['gpu_count']}[/blue]")
    for gpu in gpu_info["gpus"]:
        console.print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory']})")
    
    # Setup environment
    setup_environment()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_lora_dual.py"
    
    if not train_script.exists():
        return {
            "status": "error",
            "error": f"Training script not found: {train_script}"
        }
    
    # Prepare command arguments
    cmd_args = [
        "--model", model_name,
        "--dataset", dataset_name,
        "--dataset_config", dataset_config,
        "--out_dir", output_dir,
        "--max_steps", str(max_steps),
        "--per_device_batch", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--learning_rate", str(learning_rate),
        "--block_size", str(block_size),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--precision", precision
    ]
    
    try:
        if launch_method == "torchrun":
            # Use TorchRun for true data parallel training
            console.print("[blue]🚀 Launching with TorchRun (DDP)...[/blue]")
            console.print(f"[dim]Command: torchrun --nproc_per_node=2 {train_script} {' '.join(cmd_args)}[/dim]")
            
            result = subprocess.run([
                "torchrun", "--nproc_per_node=2", str(train_script)
            ] + cmd_args, 
            cwd=script_dir,
            capture_output=True, 
            text=True
            )
            
        elif launch_method == "accelerate":
            # Use Accelerate for model sharding
            console.print("[blue]🚀 Launching with Accelerate...[/blue]")
            console.print(f"[dim]Command: accelerate launch --gpu_ids all {train_script} {' '.join(cmd_args)}[/dim]")
            
            result = subprocess.run([
                "accelerate", "launch", "--gpu_ids", "all", str(train_script)
            ] + cmd_args,
            cwd=script_dir,
            capture_output=True,
            text=True
            )
            
        else:
            return {
                "status": "error",
                "error": f"Invalid launch method: {launch_method}. Use 'torchrun' or 'accelerate'"
            }
        
        # Process results
        if result.returncode == 0:
            console.print("[green]✅ Training completed successfully![/green]")
            return {
                "status": "success",
                "output_dir": output_dir,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "gpu_info": gpu_info
            }
        else:
            console.print(f"[red]❌ Training failed with return code: {result.returncode}[/red]")
            console.print(f"[red]Error output:[/red]")
            console.print(result.stderr)
            return {
                "status": "error",
                "error": f"Training failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "gpu_info": gpu_info
            }
            
    except FileNotFoundError as e:
        error_msg = f"Required command not found: {e}"
        if "torchrun" in str(e):
            error_msg += "\nInstall PyTorch with: pip install torch"
        elif "accelerate" in str(e):
            error_msg += "\nInstall Accelerate with: pip install accelerate"
        
        return {
            "status": "error",
            "error": error_msg,
            "gpu_info": gpu_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error: {e}",
            "gpu_info": gpu_info
        }

def train_lora_dual_gpu(
    model_name: str = "unsloth/Llama-3-8B-Instruct",
    dataset_name: str = "wikitext", 
    dataset_config: str = "wikitext-2-raw-v1",
    output_dir: str = "./runs/humigence/out_lora_dual",
    max_steps: int = 1000,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    block_size: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    precision: str = "qlora_4bit"
) -> Dict[str, Any]:
    """
    Convenience function for dual-GPU LoRA training
    Uses TorchRun by default for optimal dual-GPU performance
    """
    return launch_dual_gpu_training(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        block_size=block_size,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        precision=precision,
        launch_method="torchrun"
    )

if __name__ == "__main__":
    # Example usage
    result = train_lora_dual_gpu()
    print(f"Training result: {result}")