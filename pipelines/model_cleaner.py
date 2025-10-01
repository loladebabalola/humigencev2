# pipelines/model_cleaner.py

import torch
import gc
from rich.console import Console

console = Console()

def deep_clean_model(model, target_device="cuda:0"):
    """
    Perform a deep clean of the model to ensure all components are on the same device
    """
    console.print(f"[blue]🧹 Deep cleaning model for device {target_device}...[/blue]")
    
    try:
        # 1. Move model to CPU first
        model = model.cpu()
        console.print("[blue]   ✅ Model moved to CPU[/blue]")
        
        # 2. Clear any cached states
        if hasattr(model, 'module'):
            # Handle wrapped models (DDP, etc.)
            model.module = model.module.cpu()
            console.print("[blue]   ✅ Unwrapped model moved to CPU[/blue]")
        
        # 3. Clear any cached computations
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
            console.print("[blue]   ✅ Model cache cleared[/blue]")
        
        # 4. Force garbage collection
        gc.collect()
        console.print("[blue]   ✅ Garbage collection completed[/blue]")
        
        # 5. Move to target device
        model = model.to(target_device)
        console.print(f"[blue]   ✅ Model moved to {target_device}[/blue]")
        
        # 6. Verify all components are on the correct device
        verify_model_device_consistency(model, target_device)
        
        return model
        
    except Exception as e:
        console.print(f"[red]   ❌ Deep model clean failed: {e}[/red]")
        raise

def verify_model_device_consistency(model, target_device="cuda:0"):
    """
    Verify that all model components are on the target device
    """
    console.print(f"[blue]🔍 Verifying model device consistency on {target_device}...[/blue]")
    
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
    
    # Check modules
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.device != target_device:
                issues.append(f"Module {name}.weight on {module.weight.device}, expected {target_device}")
    
    if issues:
        console.print(f"[red]   ❌ Device consistency issues found:[/red]")
        for issue in issues:
            console.print(f"[red]     - {issue}[/red]")
        raise RuntimeError(f"Model device consistency issues: {issues}")
    else:
        console.print(f"[green]   ✅ All model components on {target_device}[/green]")

def force_model_to_device(model, target_device="cuda:0"):
    """
    Force all model components to the target device, even if it means recreating some components
    """
    console.print(f"[blue]🔧 Force moving model to {target_device}...[/blue]")
    
    try:
        target_device = torch.device(target_device)
        
        # Move all parameters
        for param in model.parameters():
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)
        
        # Move all buffers
        for buffer in model.buffers():
            buffer.data = buffer.data.to(target_device)
        
        # Move model itself
        model = model.to(target_device)
        
        console.print(f"[blue]   ✅ Model force moved to {target_device}[/blue]")
        
        # Verify consistency
        verify_model_device_consistency(model, target_device)
        
        return model
        
    except Exception as e:
        console.print(f"[red]   ❌ Force move failed: {e}[/red]")
        raise

