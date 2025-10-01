# pipelines/device_cleaner.py

import torch
import gc
import os
from rich.console import Console

console = Console()

def clean_device_state():
    """
    Completely clean device state to prevent contamination from multi-GPU training
    """
    console.print("[blue]🧹 Cleaning device state...[/blue]")
    
    # 1. Set environment variables FIRST to limit visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only show GPU 0
    console.print("[blue]   ✅ CUDA_VISIBLE_DEVICES set to 0[/blue]")
    
    # 2. Set tokenizers parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    console.print("[blue]   ✅ TOKENIZERS_PARALLELISM set to false[/blue]")
    
    # 3. Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        console.print("[blue]   ✅ CUDA cache cleared[/blue]")
    
    # 4. Force garbage collection
    gc.collect()
    console.print("[blue]   ✅ Garbage collection completed[/blue]")
    
    # 5. Reset CUDA context if possible
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            console.print("[blue]   ✅ CUDA synchronized[/blue]")
    except Exception as e:
        console.print(f"[yellow]   ⚠️ CUDA sync failed: {e}[/yellow]")
    
    # 6. Set default device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        console.print("[blue]   ✅ Default CUDA device set to 0[/blue]")

def move_model_to_single_device(model, target_device="cuda:0"):
    """
    Safely move model to a single device, ensuring all parameters and buffers are on the same device
    """
    console.print(f"[blue]🔄 Moving model to {target_device}...[/blue]")
    
    try:
        # First, move all parameters to CPU to ensure clean state
        model = model.cpu()
        console.print("[blue]   ✅ Model moved to CPU first[/blue]")
        
        # Clear any cached states
        if hasattr(model, 'module'):
            # Handle wrapped models (DDP, etc.)
            model.module = model.module.cpu()
        
        # Now move to target device
        model = model.to(target_device)
        console.print(f"[blue]   ✅ Model moved to {target_device}[/blue]")
        
        # Verify all parameters are on the same device
        param_devices = set()
        for param in model.parameters():
            param_devices.add(param.device)
        
        # Verify all buffers are on the same device
        buffer_devices = set()
        for name, buffer in model.named_buffers():
            buffer_devices.add(buffer.device)
            if buffer.device != torch.device(target_device):
                console.print(f"[yellow]   ⚠️ Buffer {name} on wrong device: {buffer.device}[/yellow]")
                # Move buffer to correct device
                buffer.data = buffer.data.to(target_device)
        
        all_devices = param_devices.union(buffer_devices)
        
        if len(all_devices) == 1 and list(all_devices)[0] == torch.device(target_device):
            console.print(f"[green]   ✅ All parameters and buffers on device: {list(all_devices)[0]}[/green]")
        else:
            console.print(f"[red]   ❌ Parameters on devices: {param_devices}, Buffers on devices: {buffer_devices}[/red]")
            raise RuntimeError(f"Model components on multiple devices: {all_devices}")
        
        return model
        
    except Exception as e:
        console.print(f"[red]   ❌ Failed to move model to {target_device}: {e}[/red]")
        raise

def create_clean_evaluation_environment():
    """
    Create a completely clean environment for evaluation
    """
    console.print("[blue]🔧 Creating clean evaluation environment...[/blue]")
    
    # Clean device state
    clean_device_state()
    
    # Set up clean environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Set default device for PyTorch operations
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        console.print("[blue]   ✅ Default CUDA device set to 0[/blue]")
    
    console.print("[green]✅ Clean evaluation environment ready[/green]")

def verify_device_consistency(model, expected_device="cuda:0"):
    """
    Verify that all model parameters are on the expected device
    """
    console.print(f"[blue]🔍 Verifying device consistency on {expected_device}...[/blue]")
    
    device_check = set()
    for name, param in model.named_parameters():
        device_check.add(param.device)
        if param.device != torch.device(expected_device):
            console.print(f"[red]   ❌ Parameter {name} on wrong device: {param.device}[/red]")
            return False
    
    if len(device_check) == 1 and list(device_check)[0] == torch.device(expected_device):
        console.print(f"[green]   ✅ All parameters on {expected_device}[/green]")
        return True
    else:
        console.print(f"[red]   ❌ Device inconsistency: {device_check}[/red]")
        return False
