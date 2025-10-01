"""
Unsloth Training Module for Humigence
Dual-GPU LoRA fine-tuning with Unsloth + TorchRun
"""

try:
    from .train_lora_dual import train_lora_dual_gpu
    from .launcher import launch_dual_gpu_training
    __all__ = ["train_lora_dual_gpu", "launch_dual_gpu_training"]
except ImportError:
    # Define dummy functions when dependencies are missing
    def train_lora_dual_gpu(*args, **kwargs):
        return {"status": "error", "error": "Unsloth dependencies not installed. Run: python3 training/unsloth/setup_humigence_unsloth.py"}
    
    def launch_dual_gpu_training(*args, **kwargs):
        return {"status": "error", "error": "Unsloth dependencies not installed. Run: python3 training/unsloth/setup_humigence_unsloth.py"}
    
    __all__ = ["train_lora_dual_gpu", "launch_dual_gpu_training"]
