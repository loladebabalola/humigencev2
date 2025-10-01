"""
Fallback simulation loop for auto-healing validation failures
"""

import re
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console
from rich.table import Table

from .dryrun import dry_run, DryRunResult
from .matrix import get_gpu_info, precision_supported, has_bitsandbytes
from training.autodetect import suggested_lora_targets

console = Console()


@dataclass
class ConfigCandidate:
    """Represents a configuration candidate for testing"""
    model: str
    precision: str
    seq_len: int
    batch_size: int
    lora: bool
    lora_targets: Optional[List[str]] = None
    gradient_checkpointing: bool = False
    dataset: str = "wikitext"
    text_field: Optional[str] = None


@dataclass
class FallbackAttempt:
    """Represents a single fallback attempt"""
    attempt_num: int
    config: ConfigCandidate
    result: DryRunResult
    strategy: str
    notes: str


class FallbackSimulator:
    """Handles fallback simulation and auto-healing"""
    
    def __init__(self):
        try:
            self.gpu = get_gpu_info()
        except Exception:
            # If GPU info fails, create a fallback GPU info
            self.gpu = type('GpuInfo', (), {
                'available': True,
                'name': 'Unknown GPU',
                'total_bytes': 0,
                'free_bytes': 0,
                'cc_major': 7,
                'cc_minor': 0,
                'bf16_supported': True
            })()
        self.attempts: List[FallbackAttempt] = []
    
    def reset_gpu_state(self):
        """Reset GPU state to clear any CUDA errors"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass  # Ignore errors during reset
    
    def classify_error(self, error: str) -> str:
        """Classify error type from error message"""
        error_lower = error.lower()
        
        if "out of memory" in error_lower or "oom" in error_lower:
            return "oom"
        elif "bf16" in error_lower and "not supported" in error_lower:
            return "precision"
        elif "fp16" in error_lower and "not supported" in error_lower:
            return "precision"
        elif "4-bit" in error_lower and "not supported" in error_lower:
            return "precision"
        elif "bitsandbytes" in error_lower:
            return "precision"
        elif "seq_len" in error_lower and "model limit" in error_lower:
            return "seq_len"
        elif "position" in error_lower and "embedding" in error_lower:
            return "seq_len"
        elif "lora" in error_lower and "target" in error_lower:
            return "lora"
        elif "cuda error" in error_lower and "assert" in error_lower:
            return "seq_len"  # Often caused by seq_len overflow
        else:
            return "unknown"
    
    def apply_fallback_strategy(self, config: ConfigCandidate, error_type: str) -> Optional[ConfigCandidate]:
        """Apply fallback strategy based on error type"""
        new_config = ConfigCandidate(
            model=config.model,
            precision=config.precision,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            lora=config.lora,
            lora_targets=config.lora_targets,
            gradient_checkpointing=config.gradient_checkpointing,
            dataset=config.dataset,
            text_field=config.text_field
        )
        
        if error_type == "precision":
            # Precision fallback chain: bf16 -> fp16 -> qlora4bit -> fp16+grad_checkpoint
            if config.precision == "bf16" and not self.gpu.bf16_supported:
                new_config.precision = "fp16"
                return new_config
            elif config.precision == "qlora4bit" and not has_bitsandbytes():
                new_config.precision = "fp16"
                return new_config
            elif config.precision == "fp16" and not self.gpu.available:
                new_config.precision = "fp32"
                return new_config
            elif config.precision in ["bf16", "fp16"] and not self.gpu.available:
                new_config.precision = "fp32"
                return new_config
        
        elif error_type == "oom":
            # OOM fallback chain: reduce batch -> enable grad checkpoint -> reduce seq_len -> change precision
            if config.batch_size > 1:
                new_config.batch_size = max(1, config.batch_size // 2)
                return new_config
            elif not config.gradient_checkpointing:
                new_config.gradient_checkpointing = True
                return new_config
            elif config.seq_len > 512:
                new_config.seq_len = max(512, config.seq_len // 2)
                return new_config
            elif config.precision in ["bf16", "fp32"]:
                new_config.precision = "fp16"
                return new_config
            elif config.precision == "fp16" and has_bitsandbytes() and self.gpu.available:
                new_config.precision = "qlora4bit"
                return new_config
        
        elif error_type == "seq_len":
            # Sequence length fallback: reduce to model limit or reasonable default
            if config.seq_len > 1024:
                new_config.seq_len = 1024
                return new_config
            elif config.seq_len > 512:
                new_config.seq_len = 512
                return new_config
        
        elif error_type == "lora":
            # LoRA fallback: try default target modules
            if config.lora and config.lora_targets:
                new_config.lora_targets = ["q_proj", "v_proj"]
                return new_config
        
        return None  # No more fallbacks available
    
    def simulate_fallbacks(self, initial_config: ConfigCandidate, max_attempts: int = 10) -> Tuple[bool, Optional[ConfigCandidate]]:
        """Simulate fallback attempts until success or max attempts reached"""
        current_config = initial_config
        attempt_num = 0
        
        console.print(f"\n[bold blue]🔄 Starting Auto-Heal Simulation Loop[/bold blue]")
        console.print(f"[dim]Max attempts: {max_attempts}[/dim]\n")
        
        # Create attempts table
        attempts_table = Table(title="Fallback Simulation Attempts")
        attempts_table.add_column("Attempt", style="cyan", width=8)
        attempts_table.add_column("Precision", style="white", width=10)
        attempts_table.add_column("Seq Len", style="white", width=8)
        attempts_table.add_column("Batch", style="white", width=6)
        attempts_table.add_column("LoRA", style="white", width=6)
        attempts_table.add_column("Grad Check", style="white", width=10)
        attempts_table.add_column("Result", style="white", width=8)
        attempts_table.add_column("Strategy", style="yellow", width=20)
        
        while attempt_num < max_attempts:
            attempt_num += 1
            
            # Reset GPU state before each attempt
            self.reset_gpu_state()
            
            # Run dry-run test
            result = dry_run(
                model_id_or_path=current_config.model,
                precision=current_config.precision,
                seq_len=current_config.seq_len,
                batch_size=current_config.batch_size,
                lora=current_config.lora,
                lora_targets=current_config.lora_targets,
            )
            
            # Determine strategy name
            if attempt_num == 1:
                strategy = "Initial attempt"
            else:
                strategy = f"Fallback #{attempt_num-1}"
            
            # Create attempt record
            attempt = FallbackAttempt(
                attempt_num=attempt_num,
                config=current_config,
                result=result,
                strategy=strategy,
                notes=""
            )
            self.attempts.append(attempt)
            
            # Add to table
            result_text = "✅ PASS" if result.ok else "❌ FAIL"
            attempts_table.add_row(
                str(attempt_num),
                current_config.precision,
                str(current_config.seq_len),
                str(current_config.batch_size),
                "Yes" if current_config.lora else "No",
                "Yes" if current_config.gradient_checkpointing else "No",
                result_text,
                strategy
            )
            
            if result.ok:
                console.print(attempts_table)
                console.print(f"\n[bold green]✅ SUCCESS![/bold green] Auto-healing found working configuration at attempt {attempt_num}")
                return True, current_config
            
            # Classify error and get next fallback
            error_type = self.classify_error(result.error or "unknown")
            next_config = self.apply_fallback_strategy(current_config, error_type)
            
            if next_config is None:
                console.print(attempts_table)
                console.print(f"\n[bold red]❌ FAILED[/bold red] No more fallback strategies available")
                return False, None
            
            # Update notes for next attempt
            if error_type == "oom":
                attempt.notes = f"OOM detected, reducing batch size to {next_config.batch_size}"
            elif error_type == "precision":
                attempt.notes = f"Precision {current_config.precision} not supported, switching to {next_config.precision}"
            elif error_type == "seq_len":
                attempt.notes = f"Sequence length {current_config.seq_len} too long, reducing to {next_config.seq_len}"
            elif error_type == "lora":
                attempt.notes = f"LoRA target modules not found, using defaults"
            
            current_config = next_config
        
        console.print(attempts_table)
        console.print(f"\n[bold red]❌ FAILED[/bold red] Max attempts ({max_attempts}) reached")
        return False, None
    
    def generate_yaml_config(self, config: ConfigCandidate) -> str:
        """Generate YAML-style config block for the working configuration"""
        yaml_lines = [
            "# AUTO-HEALED CONFIG PATCH",
            f"model: {config.model}",
            f"precision: {config.precision}",
            f"seq_len: {config.seq_len}",
            f"batch_size: {config.batch_size}",
            f"lora: {str(config.lora).lower()}",
            f"gradient_checkpointing: {str(config.gradient_checkpointing).lower()}",
            f"dataset: {config.dataset}",
        ]
        
        if config.lora_targets:
            yaml_lines.append(f"lora_targets: {config.lora_targets}")
        
        if config.text_field:
            yaml_lines.append(f"text_field: {config.text_field}")
        
        return "\n".join(yaml_lines)
