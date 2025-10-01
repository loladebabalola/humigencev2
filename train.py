#!/usr/bin/env python3
"""
Humigence Training Script with Hugging Face Accelerate
Clean DDP training with single-GPU evaluation
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig, get_linear_schedule_with_warmup
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Set environment variables for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

console = Console()

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    # Model config
    base_model: str = "microsoft/DialoGPT-small"
    training_recipe: str = "LoRA (FP16)"
    
    # Training config
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Data config
    dataset_path: str = ""
    train_val_test_split: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    split_seed: int = 42
    
    # Output config
    output_dir: str = "runs/humigence"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    # Evaluation config
    eval_gpu_index: int = 0  # Always use cuda:0 for evaluation

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Map config keys to dataclass fields
    config = TrainingConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def prepare_dataset(config: TrainingConfig, tokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """Prepare dataset splits with tokenization"""
    console.print("[blue]📊 Preparing dataset...[/blue]")
    
    # Load dataset
    with open(config.dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    console.print(f"[blue]   Loaded {len(data)} samples[/blue]")
    
    # Split dataset
    np.random.seed(config.split_seed)
    indices = np.random.permutation(len(data))
    
    train_size = int(len(data) * config.train_val_test_split[0])
    val_size = int(len(data) * config.train_val_test_split[1])
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    console.print(f"[blue]   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}[/blue]")
    
    # Simple tokenization function
    def tokenize_function(examples):
        # Handle different data schemas
        if "text" in examples:
            # Simple text schema
            texts = examples["text"]
        elif "instruction" in examples and "output" in examples:
            # Instruction-output schema
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""])[i] if examples.get("input") else ""
                output = examples["output"][i]
                
                # Format as conversation
                if input_text:
                    text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                else:
                    text = f"Instruction: {instruction}\nOutput: {output}"
                texts.append(text)
        else:
            # Fallback - use first available text column
            text_col = None
            for col in ["text", "instruction", "input", "output"]:
                if col in examples:
                    text_col = col
                    break
            
            if text_col:
                texts = examples[text_col]
            else:
                # Last resort - convert to string
                texts = [str(ex) for ex in examples[list(examples.keys())[0]]]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=config.max_seq_length,
            return_tensors=None
        )
        
        # Create labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Create datasets and tokenize
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Tokenize datasets - remove original columns after tokenization
    # First, get the original columns to remove
    original_columns = list(train_dataset.column_names)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=original_columns)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=original_columns)
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=original_columns)
    
    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    return train_dataset, val_dataset, test_dataset

def setup_model_and_tokenizer(config: TrainingConfig, accelerator: Accelerator):
    """Setup model and tokenizer with LoRA/QLoRA"""
    console.print(f"[blue]🤖 Loading model: {config.base_model}[/blue]")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if "QLoRA" in config.training_recipe:
        # QLoRA with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map=None,  # Let accelerate handle device placement
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Regular LoRA
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            device_map=None,  # Let accelerate handle device placement
            trust_remote_code=True,
            dtype=torch.bfloat16 if "BF16" in config.training_recipe else torch.float16
        )
    
    # Apply LoRA - use appropriate target modules for the model
    if "gpt" in config.base_model.lower() or "dialo" in config.base_model.lower():
        # For GPT-style models
        target_modules = ["c_attn", "c_proj"]
    elif "llama" in config.base_model.lower() or "mistral" in config.base_model.lower():
        # For LLaMA/Mistral models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        # Default fallback
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[blue]   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)[/blue]")
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, val_dataset, config: TrainingConfig, accelerator: Accelerator):
    """Train the model using Accelerate"""
    console.print("[blue]🚀 Starting training...[/blue]")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",  # Updated parameter name
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Save model
    if accelerator.is_main_process:
        trainer.save_model()
        console.print("[blue]💾 Model saved[/blue]")
    
    return trainer

def evaluate_model_on_single_gpu(model, tokenizer, test_dataset, config: TrainingConfig):
    """Evaluate model on single GPU (cuda:0) to avoid device mismatches"""
    console.print("[blue]🧪 Running evaluation on cuda:0...[/blue]")
    
    # Move model to cuda:0 for evaluation
    eval_device = torch.device("cuda:0")
    model = model.to(eval_device)
    model.eval()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create evaluation dataloader
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(
        test_dataset,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    # Evaluation metrics
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to cuda:0
            batch = {k: v.to(eval_device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate metrics
            total_loss += loss.item()
            num_samples += batch["input_ids"].size(0)
            
            # Token-level accuracy
            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            
            # Mask out ignored positions
            mask = labels != -100
            correct_tokens += (predictions[mask] == labels[mask]).sum().item()
            total_tokens += mask.sum().item()
    
    # Calculate final metrics
    avg_loss = total_loss / len(eval_dataloader)
    accuracy = correct_tokens / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "num_samples": num_samples
    }

def print_training_summary(config: TrainingConfig, train_dataset, val_dataset, test_dataset, eval_results):
    """Print structured training summary"""
    console.print("\n[bold cyan]=" * 80)
    console.print("[bold cyan]🎯 TRAINING SUMMARY[/bold cyan]")
    console.print("[bold cyan]=" * 80)
    
    # Dataset summary
    console.print(f"\n[bold green]📊 Dataset Summary[/bold green]")
    console.print(f"   Train: {len(train_dataset):,} samples")
    console.print(f"   Validation: {len(val_dataset):,} samples")
    console.print(f"   Test: {len(test_dataset):,} samples")
    
    # Model summary
    console.print(f"\n[bold blue]🤖 Model Summary[/bold blue]")
    console.print(f"   Base Model: {config.base_model}")
    console.print(f"   Training Recipe: {config.training_recipe}")
    console.print(f"   LoRA r: {config.lora_r}")
    console.print(f"   LoRA alpha: {config.lora_alpha}")
    
    # Training summary
    console.print(f"\n[bold yellow]🚀 Training Summary[/bold yellow]")
    console.print(f"   Epochs: {config.num_train_epochs}")
    console.print(f"   Learning Rate: {config.learning_rate}")
    console.print(f"   Batch Size: {config.per_device_train_batch_size}")
    console.print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    
    # Evaluation results
    console.print(f"\n[bold magenta]🧪 Evaluation Results (cuda:0)[/bold magenta]")
    console.print(f"   Loss: {eval_results['loss']:.4f}")
    console.print(f"   Accuracy: {eval_results['accuracy']:.4f}")
    console.print(f"   Perplexity: {eval_results['perplexity']:.2f}")
    console.print(f"   Correct Tokens: {eval_results['correct_tokens']:,}")
    console.print(f"   Total Tokens: {eval_results['total_tokens']:,}")
    console.print(f"   Samples: {eval_results['num_samples']:,}")
    
    console.print("\n[bold cyan]=" * 80)

def main():
    """Main training function"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Humigence Training with Accelerate")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    set_seed(42)
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Print accelerator info
    console.print(f"[blue]🚀 Accelerate Info:[/blue]")
    console.print(f"   Process index: {accelerator.process_index}")
    console.print(f"   Local process index: {accelerator.local_process_index}")
    console.print(f"   Device: {accelerator.device}")
    console.print(f"   Distributed: {accelerator.distributed_type}")
    console.print(f"   Mixed precision: {accelerator.mixed_precision}")
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config, accelerator)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = prepare_dataset(config, tokenizer)
        
        # Train model
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config, accelerator)
        
        # Wait for all processes to finish training
        accelerator.wait_for_everyone()
        
        # Evaluate on single GPU (main process only)
        if accelerator.is_main_process:
            eval_results = evaluate_model_on_single_gpu(model, tokenizer, test_dataset, config)
            print_training_summary(config, train_dataset, val_dataset, test_dataset, eval_results)
        else:
            eval_results = None
        
        # Wait for evaluation to complete
        accelerator.wait_for_everyone()
        
        return {"status": "success", "eval_results": eval_results}
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    results = main()
    if results["status"] == "success":
        console.print("[green]✅ Training completed successfully![/green]")
    else:
        console.print(f"[red]❌ Training failed: {results['message']}[/red]")
        exit(1)