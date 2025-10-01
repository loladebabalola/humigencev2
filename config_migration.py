"""
Config Migration and Validation Utilities

This module provides robust config saving that ensures compatibility with the live TrainConfig schema.
It handles migration from old config formats, validates against the current schema, and provides
clear feedback about what changes were made.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from config_schema import TrainConfig

# Set up logging
logger = logging.getLogger(__name__)

# Legacy key mappings (old_key -> new_key)
LEGACY_KEY_MAPPINGS = {
    "base_model": "model_name",
    "model": "model_name", 
    "model_id": "model_name",
    "model_path": "model_name",
    "split_ratios": "train_val_test_split",
    "random_seed": "split_seed",
    "max_seq_len": "max_seq_length",
    "torch_dtype": "dtype",  # Handle deprecated torch_dtype
}

# Safe defaults for required fields that might be missing
SAFE_DEFAULTS = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.0002,
    "num_train_epochs": 1,
    "eval_batch_size": 8,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 100,
    "max_seq_length": 1024,
    "fp16": True,
    "bf16": False,
    "multi_gpu": False,
    "eval_single_gpu": True,
    "eval_gpu_index": 0,
    "num_workers": 4,
    "pin_memory": True,
    "split_seed": 42,
    "train_val_test_split": [0.8, 0.1, 0.1],
    "data_schema": "instruction_output",
    "training_recipe": "LoRA (FP16)",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "output_dir": "runs/humigence",
    "selected_gpus": [0],  # Default to single GPU
}

def migrate_config_dict(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Migrate a config dictionary to match the current TrainConfig schema.
    
    Args:
        config_dict: Raw config dictionary (potentially with legacy keys)
        
    Returns:
        Tuple of (migrated_config, dropped_keys, applied_defaults)
    """
    # Get the current schema fields (Pydantic v1/v2 compatibility)
    if hasattr(TrainConfig, 'model_fields'):
        # Pydantic v2
        schema_fields = set(TrainConfig.model_fields.keys())
    else:
        # Pydantic v1
        schema_fields = set(TrainConfig.__fields__.keys())
    
    migrated = {}
    dropped_keys = []
    applied_defaults = []
    
    # Step 1: Apply legacy key mappings
    for old_key, new_key in LEGACY_KEY_MAPPINGS.items():
        if old_key in config_dict and new_key not in config_dict:
            migrated[new_key] = config_dict[old_key]
            logger.info(f"Renamed '{old_key}' -> '{new_key}'")
        elif old_key in config_dict and new_key in config_dict:
            logger.warning(f"Both '{old_key}' and '{new_key}' present, using '{new_key}'")
    
    # Step 2: Copy valid keys from original config
    for key, value in config_dict.items():
        if key in schema_fields:
            migrated[key] = value
        elif key not in LEGACY_KEY_MAPPINGS:
            dropped_keys.append(key)
            logger.info(f"Dropped unsupported key: '{key}'")
    
    # Step 3: Apply safe defaults for missing required fields
    if hasattr(TrainConfig, 'model_fields'):
        # Pydantic v2
        fields = TrainConfig.model_fields
    else:
        # Pydantic v1
        fields = TrainConfig.__fields__
    
    for field_name, field_info in fields.items():
        if field_name not in migrated:
            if field_name in SAFE_DEFAULTS:
                migrated[field_name] = SAFE_DEFAULTS[field_name]
                applied_defaults.append(field_name)
                logger.info(f"Applied default for '{field_name}': {SAFE_DEFAULTS[field_name]}")
            else:
                # This is a required field with no safe default - validation will catch this
                logger.warning(f"Missing required field '{field_name}' with no safe default")
    
    return migrated, dropped_keys, applied_defaults

def validate_and_save_config(
    config_dict: Dict[str, Any], 
    output_path: str,
    context_info: Optional[Dict[str, Any]] = None
) -> TrainConfig:
    """
    Migrate, validate, and save a config dictionary to ensure it matches the current schema.
    
    Args:
        config_dict: Raw config dictionary to migrate and save
        output_path: Path where to save the validated config
        context_info: Optional runtime context to help fill missing values
        
    Returns:
        Validated TrainConfig instance
        
    Raises:
        ValueError: If config cannot be migrated to valid schema
    """
    logger.info("Starting config migration and validation...")
    
    # Step 1: Migrate the config
    migrated_config, dropped_keys, applied_defaults = migrate_config_dict(config_dict)
    
    # Step 2: Apply context info if provided
    if context_info:
        # Get schema fields for validation
        if hasattr(TrainConfig, 'model_fields'):
            schema_fields = set(TrainConfig.model_fields.keys())
        else:
            schema_fields = set(TrainConfig.__fields__.keys())
        
        for key, value in context_info.items():
            if key in schema_fields and key not in migrated_config:
                migrated_config[key] = value
                logger.info(f"Applied context value for '{key}': {value}")
    
    # Step 3: Validate against schema
    try:
        validated_config = TrainConfig(**migrated_config)
        logger.info("✅ Config validation successful")
    except Exception as e:
        logger.error(f"❌ Config validation failed: {e}")
        raise ValueError(f"Configuration validation failed after migration: {e}")
    
    # Step 4: Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(validated_config.dict(), f, indent=2)
    
    logger.info(f"✅ Config saved to {output_path}")
    
    # Step 5: Print summary
    print_config_migration_summary(dropped_keys, applied_defaults, output_path)
    
    return validated_config

def print_config_migration_summary(
    dropped_keys: List[str], 
    applied_defaults: List[str], 
    output_path: str
) -> None:
    """Print a summary of config migration changes."""
    print("\n" + "="*60)
    print("CONFIG MIGRATION SUMMARY")
    print("="*60)
    print(f"📁 Saved to: {output_path}")
    
    if dropped_keys:
        print(f"🗑️  Dropped keys ({len(dropped_keys)}): {', '.join(dropped_keys)}")
    else:
        print("✅ No keys dropped")
    
    if applied_defaults:
        print(f"⚙️  Applied defaults ({len(applied_defaults)}): {', '.join(applied_defaults)}")
    else:
        print("✅ No defaults applied")
    
    print("✅ Config is now compatible with current TrainConfig schema")
    print("="*60)

def save_config_snapshot(
    config_dict: Dict[str, Any], 
    output_path: str = "runs/humigence/config.snapshot.json",
    context_info: Optional[Dict[str, Any]] = None
) -> TrainConfig:
    """
    Save a config snapshot with automatic migration and validation.
    
    This is the main function that should be used throughout the codebase
    to ensure all saved configs are compatible with the current schema.
    
    Args:
        config_dict: Raw config dictionary to save
        output_path: Path where to save the config (default: runs/humigence/config.snapshot.json)
        context_info: Optional runtime context to help fill missing values
        
    Returns:
        Validated TrainConfig instance
    """
    return validate_and_save_config(config_dict, output_path, context_info)

def load_and_validate_config(config_path: str) -> TrainConfig:
    """
    Load and validate a config file against the current schema.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Validated TrainConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config cannot be validated
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Migrate and validate
    migrated_config, dropped_keys, applied_defaults = migrate_config_dict(config_dict)
    
    try:
        validated_config = TrainConfig(**migrated_config)
        return validated_config
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}")

# Backward compatibility function
def save_config(config: TrainConfig, output_path: str) -> None:
    """Legacy save_config function for backward compatibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config.dict(), f, indent=2)
    
    logger.info(f"Config saved to {output_path}")
