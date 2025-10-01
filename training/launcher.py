#!/usr/bin/env python3
"""
GPU-aware training launcher for Humigence

This module handles GPU selection and launches training in either single-GPU or multi-GPU mode
based on the configuration. It ensures proper CUDA_VISIBLE_DEVICES setup and Accelerate configuration.
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.schema import TrainingConfig
from validation.matrix import get_all_gpu_info, MultiGpuInfo, GpuInfo

console = Console()


def validate_gpu_selection(config: TrainingConfig) -> bool:
    """
    Validate that the selected GPU configuration is valid.
    
    Args:
        config: TrainingConfig with gpu_mode and gpu_ids
        
    Returns:
        True if valid, False otherwise
    """
    multi_gpu_info = get_all_gpu_info()
    
    if not multi_gpu_info.gpus:
        if config.gpu_mode == "multi":
            console.print("[red]❌ Multi-GPU mode requested but no GPUs available[/red]")
            return False
        return True
    
    # Check if all requested GPU IDs are valid
    available_indices = [gpu.device_index for gpu in multi_gpu_info.gpus]
    invalid_ids = [gpu_id for gpu_id in config.gpu_ids if gpu_id not in available_indices]
    
    if invalid_ids:
        console.print(f"[red]❌ Invalid GPU IDs: {invalid_ids}. Available: {available_indices}[/red]")
        return False
    
    # Enforce strict single GPU mode validation
    if config.gpu_mode == "single":
        if len(config.gpu_ids) != 1:
            console.print(f"[red]❌ Single GPU mode requires exactly 1 GPU ID, got {len(config.gpu_ids)}[/red]")
            return False
        if not config.gpu_ids:
            console.print("[red]❌ Single GPU mode requires at least 1 GPU ID[/red]")
            return False
    
    # Enforce strict multi-GPU mode validation
    if config.gpu_mode == "multi":
        if len(config.gpu_ids) < 2:
            console.print(f"[red]❌ Multi-GPU mode requires at least 2 GPU IDs, got {len(config.gpu_ids)}[/red]")
            return False
    
    return True


def print_gpu_selection_summary(config: TrainingConfig) -> None:
    """
    Print a summary of the selected GPU configuration.
    
    Args:
        config: TrainingConfig with gpu_mode and gpu_ids
    """
    multi_gpu_info = get_all_gpu_info()
    
    if not multi_gpu_info.gpus:
        console.print(Panel(
            "[bold blue]🖥️ Training Mode: CPU[/bold blue]\n"
            "[dim]No GPUs detected - training will run on CPU[/dim]",
            title="GPU Configuration",
            border_style="blue"
        ))
        return
    
    # Create GPU selection table
    table = Table(title="Selected GPU Configuration")
    table.add_column("GPU Index", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("VRAM", style="green")
    table.add_column("Selected", style="bold blue")
    
    selected_gpus = []
    for gpu in multi_gpu_info.gpus:
        is_selected = gpu.device_index in config.gpu_ids
        vram_gb = gpu.total_bytes / (1024**3)
        
        table.add_row(
            str(gpu.device_index),
            gpu.name,
            f"{vram_gb:.1f} GB",
            "✅" if is_selected else "❌"
        )
        
        if is_selected:
            selected_gpus.append(gpu)
    
    console.print(table)
    
    # Print training mode summary
    if config.gpu_mode == "single":
        selected_gpu = selected_gpus[0] if selected_gpus else None
        if selected_gpu:
            vram_gb = selected_gpu.total_bytes / (1024**3)
            mode_text = f"[bold blue]🖥️ Training Mode: Single GPU[/bold blue]\n"
            mode_text += f"[cyan]GPU: {selected_gpu.name} (id={selected_gpu.device_index}, {vram_gb:.1f} GB)[/cyan]"
        else:
            mode_text = "[bold blue]🖥️ Training Mode: Single GPU[/bold blue]\n[red]No valid GPU selected[/red]"
    else:
        if selected_gpus:
            total_vram = sum(gpu.total_bytes for gpu in selected_gpus) / (1024**3)
            gpu_names = [gpu.name for gpu in selected_gpus]
            mode_text = f"[bold blue]🖥️ Training Mode: Multi-GPU[/bold blue]\n"
            mode_text += f"[cyan]GPUs: {len(selected_gpus)}x {gpu_names[0] if len(set(gpu_names)) == 1 else 'Mixed'}[/cyan]\n"
            mode_text += f"[cyan]IDs: {config.gpu_ids} | Total VRAM: {total_vram:.1f} GB[/cyan]"
        else:
            mode_text = "[bold blue]🖥️ Training Mode: Multi-GPU[/bold blue]\n[red]No valid GPUs selected[/red]"
    
    console.print(Panel(
        mode_text,
        title="Training Configuration",
        border_style="green"
    ))


# setup_cuda_environment function removed - now handled directly in launch_training()


def create_accelerate_config(config: TrainingConfig) -> Optional[str]:
    """
    Create Accelerate configuration file for multi-GPU training.
    
    Args:
        config: TrainingConfig with gpu_mode and gpu_ids
        
    Returns:
        Path to the created config file, or None for single GPU
    """
    if config.gpu_mode == "single":
        return None
    
    # Create Accelerate config for multi-GPU
    accelerate_config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "downcast_bf16": "no",
        "gpu_ids": config.gpu_ids,
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16" if config.dtype == "bf16" else "fp16",
        "num_machines": 1,
        "num_processes": len(config.gpu_ids),
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False
    }
    
    # Write to temporary file
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    import yaml
    yaml.dump(accelerate_config, config_file, default_flow_style=False)
    config_file.close()
    
    console.print(f"[blue]🔧 Created Accelerate config: {config_file.name}[/blue]")
    return config_file.name


def launch_training(config: TrainingConfig) -> Dict[str, Any]:
    """
    Launch training with the specified GPU configuration.
    
    Args:
        config: TrainingConfig with gpu_mode and gpu_ids
        
    Returns:
        Dictionary with training results
    """
    # Validate GPU selection
    if not validate_gpu_selection(config):
        return {
            "status": "error",
            "error": "Invalid GPU selection",
            "output_dir": config.output_dir
        }
    
    # Print GPU selection summary
    print_gpu_selection_summary(config)
    
    # Enforce strict single vs multi-GPU paths
    if config.gpu_mode == "single":
        # Single GPU training - enforce single GPU path
        assert len(config.gpu_ids) == 1, f"Single GPU mode requires exactly 1 GPU ID, got {len(config.gpu_ids)}"
        
        # Set CUDA_VISIBLE_DEVICES to the single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids[0])
        console.print(f"[blue]🔧 Set CUDA_VISIBLE_DEVICES={config.gpu_ids[0]} for single GPU training[/blue]")
        
        # Run single GPU training (no Accelerate, no NCCL)
        from training.universal_trainer import run_universal_training
        return run_universal_training(config)
        
    elif config.gpu_mode == "multi":
        # Multi-GPU training - enforce multi-GPU path
        assert len(config.gpu_ids) >= 2, f"Multi-GPU mode requires at least 2 GPU IDs, got {len(config.gpu_ids)}"
        
        # Set CUDA_VISIBLE_DEVICES to all selected GPUs
        gpu_ids_str = ",".join(map(str, config.gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        console.print(f"[blue]🔧 Set CUDA_VISIBLE_DEVICES={gpu_ids_str} for multi-GPU training[/blue]")
        
        # Create Accelerate config and launch multi-GPU training
        accelerate_config_path = create_accelerate_config(config)
        try:
            return launch_multi_gpu_training(config, accelerate_config_path)
        finally:
            # Clean up temporary config file
            if accelerate_config_path and os.path.exists(accelerate_config_path):
                os.unlink(accelerate_config_path)
    
    else:
        return {
            "status": "error",
            "error": f"Invalid gpu_mode: {config.gpu_mode}. Must be 'single' or 'multi'",
            "output_dir": config.output_dir
        }


def launch_multi_gpu_training(config: TrainingConfig, accelerate_config_path: str) -> Dict[str, Any]:
    """
    Launch multi-GPU training using Accelerate.
    
    Args:
        config: TrainingConfig with multi-GPU settings
        accelerate_config_path: Path to Accelerate config file
        
    Returns:
        Dictionary with training results
    """
    try:
        # Import accelerate
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        
        # Initialize accelerator
        accelerator = Accelerator()
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Import training function after accelerator setup
        from training.universal_trainer import run_universal_training_with_accelerator
        
        # Run training with accelerator
        return run_universal_training_with_accelerator(config, accelerator)
        
    except ImportError:
        console.print("[red]❌ Accelerate not available. Falling back to single GPU training.[/red]")
        # Fallback to single GPU
        config.gpu_mode = "single"
        config.gpu_ids = [config.gpu_ids[0]]
        from training.universal_trainer import run_universal_training
        return run_universal_training(config)
    except Exception as e:
        console.print(f"[red]❌ Multi-GPU training failed: {str(e)}[/red]")
        # Fallback to single GPU
        console.print("[yellow]⚠️  Falling back to single GPU training...[/yellow]")
        config.gpu_mode = "single"
        config.gpu_ids = [config.gpu_ids[0]]
        from training.universal_trainer import run_universal_training
        return run_universal_training(config)


def main():
    """Main entry point for the launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Humigence GPU-aware Training Launcher")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    args = parser.parse_args()
    
    # Load configuration
    from config.schema import load_config
    config, metadata = load_config(args.config, TrainingConfig)
    
    # Launch training
    result = launch_training(config)
    
    # Print results
    if result["status"] == "success":
        console.print(Panel(
            f"""[bold green]✅ Training Completed Successfully![/bold green]
            
[cyan]Output Directory:[/cyan] {result['output_dir']}
[cyan]Model Path:[/cyan] {result['model_path']}

[bold blue]Final Metrics:[/bold blue]
[cyan]Train Loss:[/cyan] {result['metrics'].get('train_loss', 'N/A')}
[cyan]Eval Loss:[/cyan] {result['metrics'].get('eval_loss', 'N/A')}
[cyan]Total Steps:[/cyan] {result['metrics'].get('total_steps', 'N/A')}
[cyan]Epochs:[/cyan] {result['metrics'].get('epochs', 'N/A')}
[cyan]Train Runtime:[/cyan] {result['metrics'].get('train_runtime', 'N/A')}s
[cyan]Samples/Second:[/cyan] {result['metrics'].get('train_samples_per_second', 'N/A')}""",
            title="🎉 Training Results",
            border_style="green"
        ))
        sys.exit(0)
    else:
        console.print(Panel(
            f"""[bold red]❌ Training Failed[/bold red]
            
[red]Error:[/red] {result.get('error', 'Unknown error')}
[cyan]Output Directory:[/cyan] {result.get('output_dir', 'N/A')}""",
            title="💥 Training Error",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
