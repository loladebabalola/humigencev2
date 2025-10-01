"""
Dry-run validation for model training setup
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


@dataclass
class DryRunResult:
    ok: bool
    error: Optional[str]
    oom: bool
    error_type: str = "unknown"
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


def _prep_inputs(tokenizer, seq_len: int, batch_size: int, device: str):
    """Prepare dummy inputs for dry run"""
    # minimal dummy text
    text = "The quick brown fox jumps over the lazy dog. " * 64
    toks = tokenizer(
        [text] * batch_size, 
        max_length=seq_len, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in toks.items()}


def dry_run(
    model_id_or_path: str,
    precision: str = "fp16",
    seq_len: int = 1024,
    batch_size: int = 2,
    lora: bool = True,
    lora_targets: Optional[list] = None,
) -> DryRunResult:
    """Perform a 1-batch forward+backward dry run"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        
        # Load model with selected precision
        kwargs = {"trust_remote_code": True}
        
        if precision == "fp16":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto" if device == "cuda" else None
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto" if device == "cuda" else None
        elif precision == "fp32":
            kwargs["torch_dtype"] = torch.float32
        elif precision == "qlora4bit":
            import bitsandbytes as bnb  # noqa
            kwargs.update(
                dict(
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                )
            )
        
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
        
        # Apply LoRA if requested
        if lora:
            if not lora_targets:
                lora_targets = ["q_proj", "v_proj"]
            lc = LoraConfig(
                r=16, 
                lora_alpha=32, 
                lora_dropout=0.05, 
                target_modules=lora_targets, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lc)
        
        model.train()
        if precision in ("fp32",) and device == "cuda":
            model.to(device)

        # Prepare and run forward+backward
        batch = _prep_inputs(tok, seq_len, batch_size, device)
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()  # ensure backward works
        
        return DryRunResult(True, None, False, "success", {"loss": float(loss.detach().cpu())})
        
    except RuntimeError as e:
        msg = str(e)
        error_type = _classify_error(msg)
        return DryRunResult(False, msg, "out of memory" in msg.lower(), error_type, {})
    except Exception as e:
        msg = str(e)
        error_type = _classify_error(msg)
        return DryRunResult(False, msg, False, error_type, {})


def _classify_error(error: str) -> str:
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
    elif "cuda error" in error_lower and "assert" in error_lower:
        return "seq_len"  # Often caused by seq_len overflow
    elif "lora" in error_lower and "target" in error_lower:
        return "lora"
    else:
        return "unknown"
