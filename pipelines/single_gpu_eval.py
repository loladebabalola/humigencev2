# pipelines/single_gpu_eval.py

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console

console = Console()

def _prepare_model_for_single_gpu_eval(model, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Prepare model for single GPU evaluation by removing multi-GPU wrappers and ensuring
    all components are on a single device (cuda:0 or first visible GPU).
    
    This fixes the cuda:0 vs cuda:1 device mismatch issue by ensuring the model
    is completely on one device before evaluation.
    
    Args:
        model: The trained model (may be wrapped with DDP/FSDP/DataParallel)
        config: Configuration dictionary
        
    Returns:
        Clean model on single GPU (cuda:0 if available, otherwise cpu)
    """
    console.print("[blue]🔄 Preparing model for evaluation on single GPU...[/blue]")
    
    # Determine target device
    if torch.cuda.is_available():
        # Use first visible GPU or cuda:0
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if visible_devices and visible_devices != "-1":
            target_device = "cuda:0"  # First visible device
        else:
            target_device = "cuda:0"
        console.print(f"[blue]   🎯 Target device: {target_device}[/blue]")
    else:
        target_device = "cpu"
        console.print("[blue]   🎯 Target device: cpu (no CUDA available)[/blue]")
    
    try:
        # Step 1: Remove any multi-GPU wrappers
        original_model = model
        if hasattr(model, 'module'):
            # Remove DataParallel/DistributedDataParallel wrapper
            model = model.module
            console.print("[blue]   ✅ Removed DDP/DataParallel wrapper[/blue]")
        
        # Step 2: Move model to CPU first to clear any device state
        model = model.cpu()
        console.print("[blue]   ✅ Model moved to CPU[/blue]")
        
        # Step 3: Clear any cached states or buffers
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
            console.print("[blue]   ✅ Model cache cleared[/blue]")
        
        # Step 4: Move to target device
        model = model.to(target_device)
        console.print(f"[blue]   ✅ Model moved to {target_device}[/blue]")
        
        # Step 5: Verify all components are on the same device
        _verify_model_device_consistency(model, target_device)
        
        # Step 6: Re-attach LoRA adapters if needed
        if config.get("training_recipe", "").lower() in ["lora", "qlora"]:
            model = _reattach_lora_adapters(model, target_device, config)
        
        console.print(f"[green]✅ Model prepared for single GPU evaluation on {target_device}[/green]")
        return model
        
    except Exception as e:
        console.print(f"[red]❌ Failed to prepare model for single GPU evaluation: {e}[/red]")
        # Fallback: return original model
        console.print("[yellow]⚠️ Falling back to original model[/yellow]")
        return original_model

def _verify_model_device_consistency(model: torch.nn.Module, target_device: str) -> None:
    """
    Verify that all model components are on the target device.
    
    Args:
        model: The model to verify
        target_device: Expected device (e.g., "cuda:0", "cpu")
    """
    target_device = torch.device(target_device)
    issues = []
    
    # Check parameters
    for name, param in model.named_parameters():
        if param.device != target_device:
            issues.append(f"Parameter {name} on {param.device}, expected {target_device}")
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer.device != target_device:
            issues.append(f"Buffer {name} on {buffer.device}, expected {target_device}")
    
    if issues:
        console.print(f"[red]❌ Device consistency issues found:[/red]")
        for issue in issues:
            console.print(f"[red]   - {issue}[/red]")
        raise RuntimeError(f"Model device consistency issues: {issues}")
    else:
        console.print(f"[green]   ✅ All model components on {target_device}[/green]")

def _reattach_lora_adapters(model: torch.nn.Module, target_device: str, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Re-attach LoRA adapters if they were used during training.
    
    Args:
        model: The base model
        target_device: Target device for the model
        config: Configuration dictionary
        
    Returns:
        Model with LoRA adapters re-attached
    """
    try:
        # Check if LoRA adapters exist
        adapter_path = Path(config.get("output_dir", "runs/humigence")) / "final_model"
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            console.print("[blue]   🔧 Re-attaching LoRA adapters...[/blue]")
            
            # Import PEFT here to avoid issues if not available
            try:
                from peft import PeftModel
                
                # Load the model with LoRA adapters
                model = PeftModel.from_pretrained(model, str(adapter_path))
                model = model.to(target_device)
                console.print("[blue]   ✅ LoRA adapters re-attached[/blue]")
                
            except ImportError:
                console.print("[yellow]   ⚠️ PEFT not available, skipping LoRA re-attachment[/yellow]")
            except Exception as e:
                console.print(f"[yellow]   ⚠️ Failed to re-attach LoRA adapters: {e}[/yellow]")
        
        return model
        
    except Exception as e:
        console.print(f"[yellow]   ⚠️ LoRA re-attachment failed: {e}[/yellow]")
        return model

def _move_batch_to_device(batch: Dict[str, torch.Tensor], target_device: str) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in a batch to the target device.
    
    Args:
        batch: Dictionary of tensors
        target_device: Target device (e.g., "cuda:0", "cpu")
        
    Returns:
        Batch with all tensors on target device
    """
    target_device = torch.device(target_device)
    moved_batch = {}
    
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved_batch[key] = value.to(target_device)
        else:
            moved_batch[key] = value
    
    return moved_batch

def _move_tensors_to_cpu(*tensors) -> tuple:
    """
    Move all tensors to CPU and detach them.
    
    Args:
        *tensors: Variable number of tensors
        
    Returns:
        Tuple of CPU tensors
    """
    cpu_tensors = []
    for tensor in tensors:
        if hasattr(tensor, "detach"):
            cpu_tensors.append(tensor.detach().cpu())
        else:
            cpu_tensors.append(tensor)
    return tuple(cpu_tensors)

