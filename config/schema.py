"""
Pydantic schemas for Humigence configuration
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
from datetime import datetime
import os


class DatasetConfig(BaseModel):
    """Configuration for dataset loading"""
    type: str = Field(..., description="Dataset type: wikitext, jsonl, hf")
    path: Optional[str] = Field(None, description="Path to dataset file (for jsonl)")
    name: Optional[str] = Field(None, description="Dataset name (for hf/wikitext)")
    schema_type: Optional[str] = Field(None, alias="schema", description="Schema type: sft, dialogue, plain, auto")
    text_field: Optional[str] = Field(None, description="Text field for HF datasets")
    role_markers: bool = Field(default=True, description="Use role markers for dialogue")
    user_marker: str = Field(default="<user>", description="User role marker")
    assistant_marker: str = Field(default="<assistant>", description="Assistant role marker")
    eval_split: Optional[float] = Field(None, description="Fraction of data to use for evaluation (0.0-1.0)")
    eval_file: Optional[str] = Field(None, description="Path to separate evaluation file (for JSONL)")
    commit_hash: Optional[str] = Field(None, description="Dataset commit hash for reproducibility")
    
    class Config:
        extra = "allow"
        allow_population_by_field_name = True


class ConfigMetadata(BaseModel):
    """Metadata for configuration files"""
    created: str = Field(..., description="ISO timestamp when config was created")
    gpu: Optional[str] = Field(None, description="GPU name and capabilities")
    precision_supported: List[str] = Field(default_factory=list, description="Supported precisions")
    validator_version: str = Field(default="0.3", description="Humigence validator version")
    auto_heal: bool = Field(default=False, description="Whether auto-healing was used")
    fallback_chain: List[str] = Field(default_factory=list, description="Fallback chain applied")
    original_config: Optional[Dict[str, Any]] = Field(None, description="Original config before fallbacks")
    dataset: Optional[Dict[str, Any]] = Field(None, description="Dataset metadata")
    runtime: Optional[Dict[str, Any]] = Field(None, description="Runtime environment metadata")
    
    class Config:
        extra = "allow"


class ValidationConfig(BaseModel):
    """Configuration schema for validation"""
    model: str = Field(..., description="HF model id or local path")
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(type="wikitext", name="wikitext"), description="Dataset configuration")
    precision: str = Field(default="fp16", description="Precision: fp32|fp16|bf16|qlora4bit")
    seq_len: int = Field(default=1024, description="Sequence length")
    batch_size: int = Field(default=2, description="Batch size")
    lora: bool = Field(default=True, description="Enable LoRA")
    lora_targets: Optional[List[str]] = Field(default=None, description="LoRA target modules")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    flash_attn: bool = Field(default=False, description="Enable flash attention")
    dtype: str = Field(default="fp16", description="Data type: fp32|fp16|bf16")
    max_samples: int = Field(default=128, description="Max samples for schema sniff")
    
    # GPU selection fields
    gpu_mode: str = Field(default="single", description="GPU mode: single|multi")
    gpu_ids: List[int] = Field(default_factory=lambda: [0], description="List of GPU IDs to use")
    
    # Legacy fields for backward compatibility
    dataset_spec: Optional[str] = Field(default=None, description="Legacy dataset specification")
    text_field: Optional[str] = Field(default=None, description="Legacy text field")
    schema_legacy: Optional[str] = Field(default=None, alias="schema", description="Legacy schema")
    
    class Config:
        extra = "allow"  # Allow additional fields


class TrainingConfig(BaseModel):
    """Configuration schema for training"""
    model: str = Field(..., description="HF model id or local path")
    output_dir: str = Field(..., description="Directory where checkpoints will be saved")
    dataset: DatasetConfig = Field(default_factory=lambda: DatasetConfig(type="wikitext", name="wikitext"), description="Dataset configuration")
    precision: str = Field(default="fp16", description="Precision: fp32|fp16|bf16|qlora4bit")
    seq_len: int = Field(default=1024, description="Sequence length")
    batch_size: int = Field(default=2, description="Batch size")
    epochs: int = Field(default=1, description="Number of training epochs")
    learning_rate: float = Field(default=5e-5, description="Learning rate")
    max_steps: Optional[int] = Field(default=None, description="Maximum training steps")
    block_size: int = Field(default=1024, description="Maximum sequence length")
    grad_accum: int = Field(default=4, description="Gradient accumulation steps")
    warmup_steps: int = Field(default=100, description="Number of warmup steps")
    logging_steps: int = Field(default=10, description="Logging frequency in steps")
    save_steps: int = Field(default=200, description="Model saving frequency in steps")
    eval_steps: int = Field(default=200, description="Evaluation frequency in steps")
    lora: bool = Field(default=True, description="Enable LoRA")
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    lora_targets: Optional[List[str]] = Field(default=None, description="LoRA target modules")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    flash_attn: bool = Field(default=False, description="Enable flash attention")
    dtype: str = Field(default="fp16", description="Data type: fp32|fp16|bf16")
    
    # GPU selection fields
    gpu_mode: str = Field(default="single", description="GPU mode: single|multi")
    gpu_ids: List[int] = Field(default_factory=lambda: [0], description="List of GPU IDs to use")
    
    # Legacy fields for backward compatibility
    dataset_spec: Optional[str] = Field(default=None, description="Legacy dataset specification")
    dataset_config: str = Field(default="wikitext-2-raw-v1", description="Legacy dataset configuration")
    text_field: Optional[str] = Field(default=None, description="Legacy text field")
    schema_legacy: Optional[str] = Field(default=None, alias="schema", description="Legacy schema")
    
    class Config:
        extra = "allow"  # Allow additional fields


def save_config(
    config: BaseModel, 
    filepath: str, 
    metadata: Optional[ConfigMetadata] = None,
    overwrite: bool = False
) -> str:
    """Save configuration to YAML file with versioning and metadata"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate versioned filename if file exists and not overwriting
    if not overwrite and filepath.exists():
        filepath = _generate_versioned_filename(filepath)
    
    # Convert to dict
    config_dict = config.dict()
    
    # Add metadata if provided
    if metadata:
        config_dict["meta"] = metadata.dict()
    else:
        # Create default metadata
        config_dict["meta"] = ConfigMetadata(
            created=datetime.now().isoformat(),
            validator_version="0.3"
        ).dict()
    
    # Add legacy comments for backward compatibility
    config_dict["# Auto-generated by Humigence"] = None
    config_dict["# Generated from validation auto-healing"] = None
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    return str(filepath)


def _generate_versioned_filename(filepath: Path) -> Path:
    """Generate a versioned filename if the original exists"""
    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    
    # Try numbered versions first
    counter = 1
    while True:
        versioned_name = f"{stem}_{counter:03d}{suffix}"
        versioned_path = parent / versioned_name
        if not versioned_path.exists():
            return versioned_path
        counter += 1
        if counter > 999:  # Safety limit
            # Fall back to timestamp
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            timestamped_name = f"{stem}_{timestamp}{suffix}"
            return parent / timestamped_name


def load_config(filepath: str, config_type: type = TrainingConfig) -> tuple[BaseModel, Optional[ConfigMetadata]]:
    """Load configuration from YAML file and return config with metadata"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract metadata if present
    metadata = None
    if "meta" in config_dict:
        metadata_dict = config_dict.pop("meta")
        try:
            metadata = ConfigMetadata(**metadata_dict)
        except Exception:
            # If metadata parsing fails, continue without it
            pass
    
    # Remove legacy comments
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('#')}
    
    # Migrate legacy dataset config if needed
    config_dict = migrate_legacy_dataset_config(config_dict)
    
    # Migrate legacy GPU config if needed
    config_dict = migrate_legacy_gpu_config(config_dict)
    
    config = config_type(**config_dict)
    return config, metadata


def migrate_legacy_dataset_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy dataset configuration to new format"""
    # Check if we have legacy fields
    if "dataset_spec" in config_dict or "text_field" in config_dict or "schema" in config_dict:
        # Create dataset config from legacy fields
        dataset_spec = config_dict.get("dataset_spec", config_dict.get("dataset", "wikitext"))
        text_field = config_dict.get("text_field")
        schema = config_dict.get("schema") or config_dict.get("schema_legacy")
        
        # Detect dataset type and create appropriate config
        if dataset_spec == "wikitext":
            dataset_config = DatasetConfig(type="wikitext", name="wikitext")
        elif dataset_spec.startswith("jsonl:"):
            file_path = dataset_spec[6:]
            dataset_config = DatasetConfig(
                type="jsonl",
                path=file_path,
                schema_type=schema or "auto"
            )
        elif dataset_spec.startswith("hf:"):
            dataset_name = dataset_spec[3:]
            dataset_config = DatasetConfig(
                type="hf",
                name=dataset_name,
                text_field=text_field or "text"
            )
        else:
            # Assume it's a direct HF dataset name
            dataset_config = DatasetConfig(
                type="hf",
                name=dataset_spec,
                text_field=text_field or "text"
            )
        
        # Replace legacy fields with new dataset config
        config_dict["dataset"] = dataset_config.dict()
        
        # Remove legacy fields
        config_dict.pop("dataset_spec", None)
        config_dict.pop("text_field", None)
        config_dict.pop("schema", None)
    
    return config_dict


def migrate_legacy_gpu_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy GPU configuration to new format"""
    # Check if we have legacy GPU fields
    multi_gpu = config_dict.get("multi_gpu")
    use_distributed = config_dict.get("use_distributed")
    
    # If we have legacy fields, migrate them
    if multi_gpu is not None or use_distributed is not None:
        # Determine GPU mode and IDs based on legacy flags
        if multi_gpu and use_distributed:
            # Both true - use multi-GPU mode
            config_dict["gpu_mode"] = "multi"
            # Try to get available GPU count, default to all available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    config_dict["gpu_ids"] = list(range(gpu_count))
                else:
                    config_dict["gpu_ids"] = [0]
            except ImportError:
                config_dict["gpu_ids"] = [0]
        elif multi_gpu and not use_distributed:
            # Contradictory flags - prefer single GPU for safety
            config_dict["gpu_mode"] = "single"
            config_dict["gpu_ids"] = [0]
        else:
            # Single GPU mode
            config_dict["gpu_mode"] = "single"
            config_dict["gpu_ids"] = [0]
        
        # Remove legacy fields
        config_dict.pop("multi_gpu", None)
        config_dict.pop("use_distributed", None)
    
    # Ensure gpu_mode and gpu_ids are always present
    if "gpu_mode" not in config_dict:
        config_dict["gpu_mode"] = "single"
    if "gpu_ids" not in config_dict:
        config_dict["gpu_ids"] = [0]
    
    return config_dict


def validation_to_training_config(validation_config: ValidationConfig, output_dir: str, **overrides) -> TrainingConfig:
    """Convert ValidationConfig to TrainingConfig with training-specific defaults"""
    config_dict = validation_config.dict()
    
    # Migrate legacy dataset config if needed
    config_dict = migrate_legacy_dataset_config(config_dict)
    
    # Add training-specific defaults
    config_dict.update({
        "output_dir": output_dir,
        "dataset_config": "wikitext-2-raw-v1",  # Legacy field for backward compatibility
        "epochs": 1,
        "learning_rate": 5e-5,
        "max_steps": None,
        "block_size": validation_config.seq_len,
        "grad_accum": 4,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    })
    
    # Apply overrides
    config_dict.update(overrides)
    
    return TrainingConfig(**config_dict)
