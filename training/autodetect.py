"""
Model family detection and configuration for Humigence
"""

from typing import Dict, List, Optional, Tuple
from transformers import AutoConfig


class ModelFamily:
    GPT2 = "gpt2"
    NEOX = "gpt_neox"
    LLAMA = "llama"
    MISTRAL = "mistral"
    PHI = "phi"
    GPTJ = "gptj"
    QWEN2 = "qwen2"
    UNKNOWN = "unknown"


FAMILY_HINTS = {
    "gpt2": ModelFamily.GPT2,
    "gpt_neox": ModelFamily.NEOX,
    "llama": ModelFamily.LLAMA,
    "mistral": ModelFamily.MISTRAL,
    "phi": ModelFamily.PHI,
    "gptj": ModelFamily.GPTJ,
    "qwen2": ModelFamily.QWEN2,
}

# Reasonable defaults per family
LORA_TARGETS: Dict[str, List[str]] = {
    ModelFamily.GPT2: ["c_attn"],
    ModelFamily.NEOX: ["query_key_value"],
    ModelFamily.LLAMA: ["q_proj", "v_proj"],
    ModelFamily.MISTRAL: ["q_proj", "v_proj"],
    ModelFamily.PHI: ["Wqkv"],
    ModelFamily.GPTJ: ["q_proj", "v_proj"],
    ModelFamily.QWEN2: ["q_proj", "v_proj"],
    ModelFamily.UNKNOWN: ["q_proj", "v_proj"],
}


def detect_family(model_id_or_path: str) -> Tuple[str, AutoConfig]:
    """Detect model family from model ID or path"""
    cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    mt = (getattr(cfg, "model_type", None) or "").lower()
    
    for k, fam in FAMILY_HINTS.items():
        if k in mt:
            return fam, cfg
    
    # Try architectures
    archs = [a.lower() for a in (getattr(cfg, "architectures", []) or [])]
    for k, fam in FAMILY_HINTS.items():
        if any(k in a for a in archs):
            return fam, cfg
    
    return ModelFamily.UNKNOWN, cfg


def suggested_lora_targets(family: str) -> List[str]:
    """Get suggested LoRA target modules for a model family"""
    return LORA_TARGETS.get(family, LORA_TARGETS[ModelFamily.UNKNOWN])



