"""
Pydantic configuration schema for Humigence training pipeline
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from pathlib import Path

class TrainConfig(BaseModel):
    """Strict configuration schema for Humigence training"""
    
    # Model configuration
    model_name: str = Field(..., description="Hugging Face model name")
    training_recipe: str = Field(default="LoRA (FP16)", description="Training recipe")
    
    # Training hyperparameters
    learning_rate: float = Field(..., ge=1e-6, le=1.0, description="Learning rate")
    num_train_epochs: int = Field(..., ge=1, le=100, description="Number of training epochs")
    per_device_train_batch_size: int = Field(..., ge=1, le=32, description="Batch size per device")
    gradient_accumulation_steps: int = Field(..., ge=1, le=32, description="Gradient accumulation steps")
    eval_batch_size: int = Field(..., ge=1, le=32, description="Evaluation batch size")
    
    # Precision settings
    fp16: bool = Field(default=True, description="Use FP16 precision")
    bf16: bool = Field(default=False, description="Use BF16 precision")
    
    # Multi-GPU settings
    multi_gpu: bool = Field(default=False, description="Enable multi-GPU training")
    selected_gpus: List[int] = Field(default=[0], description="Selected GPU indices")
    
    # Dataset configuration
    dataset_path: str = Field(..., description="Path to dataset file")
    data_schema: str = Field(default="instruction_output", description="Dataset schema")
    train_val_test_split: List[float] = Field(default=[0.8, 0.1, 0.1], description="Dataset split ratios")
    split_seed: int = Field(default=42, description="Random seed for dataset split")
    max_seq_length: int = Field(default=1024, ge=64, le=4096, description="Maximum sequence length")
    
    # LoRA configuration
    lora_r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout")
    
    # Logging and evaluation
    logging_steps: int = Field(default=10, ge=1, le=1000, description="Logging frequency")
    eval_steps: int = Field(default=100, ge=1, le=10000, description="Evaluation frequency")
    save_steps: int = Field(default=500, ge=1, le=10000, description="Save frequency")
    
    # Output configuration
    output_dir: str = Field(default="runs/humigence", description="Output directory")
    eval_single_gpu: bool = Field(default=True, description="Evaluate on single GPU")
    eval_gpu_index: int = Field(default=0, description="GPU index for evaluation")
    
    # System configuration
    num_workers: int = Field(default=4, ge=0, le=16, description="Number of data loader workers")
    pin_memory: bool = Field(default=True, description="Pin memory for data loading")
    
    @validator('train_val_test_split')
    def validate_split(cls, v):
        if len(v) != 3:
            raise ValueError("train_val_test_split must have exactly 3 values")
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("train_val_test_split values must sum to 1.0")
        return v
    
    @validator('fp16', 'bf16')
    def validate_precision(cls, v, values):
        if values.get('fp16') and values.get('bf16'):
            raise ValueError("Cannot use both fp16 and bf16 simultaneously")
        return v
    
    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Dataset file not found: {v}")
        if not path.suffix == '.jsonl':
            raise ValueError(f"Dataset must be a .jsonl file: {v}")
        return str(path)
    
    @validator('model_name')
    def validate_model_name(cls, v):
        # Basic validation for Hugging Face model names
        if not v or len(v.strip()) == 0:
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"  # Reject extra fields
        use_enum_values = True

def load_config(config_path: str) -> TrainConfig:
    """Load and validate configuration from JSON file"""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    try:
        return TrainConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

def save_config(config: TrainConfig, output_path: str) -> None:
    """Save configuration to JSON file (legacy function)"""
    import json
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config.dict(), f, indent=2)

def save_config_snapshot(config_dict: dict, output_path: str = "runs/humigence/config.snapshot.json") -> TrainConfig:
    """Save config with automatic migration and validation"""
    from config_migration import save_config_snapshot as _save_config_snapshot
    return _save_config_snapshot(config_dict, output_path)
