# pipelines/fresh_model_eval.py

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

def reload_fresh_model_for_evaluation(config: Dict[str, Any]) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Reload a fresh model instance for evaluation to avoid DDP/FSDP sharding issues.
    
    This is the correct fix for cuda:0 vs cuda:1 errors - instead of trying to move
    a distributed model, we reload a clean single-device model for evaluation.
    
    Args:
        config: Configuration dictionary containing model and training info
        
    Returns:
        Tuple of (fresh_model, tokenizer) both on single device
    """
    console.print("[blue]🔄 Reloading fresh model for evaluation on cuda:0[/blue]")
    
    # Determine target device
    if torch.cuda.is_available():
        target_device = "cuda:0"
        console.print("[blue]   🎯 Target device: cuda:0[/blue]")
    else:
        target_device = "cpu"
        console.print("[blue]   🎯 Target device: cpu (no CUDA available)[/blue]")
    
    try:
        # Step 1: Reload base model fresh from checkpoint
        base_model_name = config.get("base_model", "microsoft/DialoGPT-medium")
        console.print(f"[blue]   📥 Reloading base model: {base_model_name}[/blue]")
        
        # Reload tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        console.print("[blue]   ✅ Tokenizer reloaded[/blue]")
        
        # Reload base model with device_map=None to avoid auto-sharding
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=None,  # Critical: prevent auto-sharding
            trust_remote_code=True
        )
        console.print("[blue]   ✅ Base model reloaded[/blue]")
        
        # Step 2: Move to target device
        model = model.to(target_device)
        console.print(f"[blue]   ✅ Model moved to {target_device}[/blue]")
        
        # Step 3: Load LoRA adapters if used
        training_recipe = config.get("training_recipe", "").lower()
        if training_recipe in ["lora", "qlora"]:
            model = _load_lora_adapters(model, config, target_device)
        
        # Step 4: Verify all components are on the same device
        _verify_fresh_model_device_consistency(model, target_device)
        
        console.print(f"[green]✅ All weights reloaded to {target_device}[/green]")
        return model, tokenizer
        
    except Exception as e:
        console.print(f"[red]❌ Failed to reload fresh model: {e}[/red]")
        raise

def _load_lora_adapters(model: torch.nn.Module, config: Dict[str, Any], target_device: str) -> torch.nn.Module:
    """
    Load LoRA adapters on the fresh model.
    
    Args:
        model: Fresh base model
        config: Configuration dictionary
        target_device: Target device for the model
        
    Returns:
        Model with LoRA adapters loaded
    """
    try:
        # Check if LoRA adapters exist
        output_dir = Path(config.get("output_dir", "runs/humigence"))
        adapter_path = output_dir / "final_model"
        
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            console.print("[blue]   🔧 Loading LoRA adapters...[/blue]")
            
            # Import PEFT
            try:
                from peft import PeftModel
                
                # Load LoRA adapters
                model = PeftModel.from_pretrained(model, str(adapter_path))
                model = model.to(target_device)
                console.print("[blue]   ✅ LoRA adapters loaded[/blue]")
                
            except ImportError:
                console.print("[yellow]   ⚠️ PEFT not available, skipping LoRA adapters[/yellow]")
            except Exception as e:
                console.print(f"[yellow]   ⚠️ Failed to load LoRA adapters: {e}[/yellow]")
        else:
            console.print("[blue]   ℹ️ No LoRA adapters found, using base model[/blue]")
        
        return model
        
    except Exception as e:
        console.print(f"[yellow]   ⚠️ LoRA loading failed: {e}[/yellow]")
        return model

def _verify_fresh_model_device_consistency(model: torch.nn.Module, target_device: str) -> None:
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
        console.print(f"[red]❌ Fresh model device consistency issues:[/red]")
        for issue in issues:
            console.print(f"[red]   - {issue}[/red]")
        raise RuntimeError(f"Fresh model device consistency issues: {issues}")
    else:
        console.print(f"[green]   ✅ All fresh model components on {target_device}[/green]")

def ensure_model_saved_for_evaluation(config: Dict[str, Any]) -> bool:
    """
    Ensure the trained model is saved to disk for evaluation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if model is saved and ready for evaluation
    """
    try:
        output_dir = Path(config.get("output_dir", "runs/humigence"))
        model_path = output_dir / "final_model"
        
        if model_path.exists():
            console.print(f"[blue]✅ Model already saved at: {model_path}[/blue]")
            return True
        else:
            console.print(f"[yellow]⚠️ Model not found at: {model_path}[/yellow]")
            console.print("[yellow]⚠️ Make sure training completed and model was saved[/yellow]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Error checking model save status: {e}[/red]")
        return False
