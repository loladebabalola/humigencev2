#!/usr/bin/env python3
"""
Universal trainer for Humigence - supports all dataset types
"""

import os
import torch
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from rich.console import Console
from config.schema import TrainingConfig
from training.data_loader import auto_load_dataset, prepare_dataset_for_training

console = Console()


def run_universal_training(config: TrainingConfig, accelerator=None) -> Dict[str, Any]:
    """
    Run training with dataset-agnostic support.
    
    Args:
        config: TrainingConfig object with all training parameters
        accelerator: Optional Accelerate accelerator for multi-GPU training
        
    Returns:
        Dictionary with training results
    """
    try:
        console.print(f"[blue]🚀 Starting universal training with model: {config.model}[/blue]")
        console.print(f"[blue]📁 Output directory: {config.output_dir}[/blue]")
        console.print(f"[blue]📊 Dataset: {config.dataset}[/blue]")
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Load tokenizer
        console.print("[blue]📝 Loading tokenizer...[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        console.print("[blue]🤖 Loading model...[/blue]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Configure LoRA with model-specific target modules
        console.print("[blue]🔧 Configuring LoRA...[/blue]")
        
        # Determine target modules based on model type
        if "gpt" in config.model.lower() or "dialo" in config.model.lower():
            target_modules = ["c_attn", "c_proj"]
        elif "llama" in config.model.lower() or "mistral" in config.model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default to common transformer modules
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Use custom LoRA targets if provided
        if config.lora_targets:
            target_modules = config.lora_targets
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none"
        )
        
        model_obj = get_peft_model(model_obj, lora_config)
        
        # Enable input gradients for gradient checkpointing to work with LoRA
        model_obj.enable_input_require_grads()
        
        model_obj.print_trainable_parameters()
        
        # Load dataset using schema registry
        console.print(f"[blue]📊 Loading dataset: {config.dataset.type}[/blue]")
        
        # Create dataset specification from config
        if config.dataset.type == "wikitext":
            dataset_spec = "wikitext"
        elif config.dataset.type == "jsonl":
            dataset_spec = f"jsonl:{config.dataset.path}"
        elif config.dataset.type == "hf":
            dataset_spec = f"hf:{config.dataset.name}" if not config.dataset.name.startswith("hf:") else config.dataset.name
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset.type}")
        
        train_dataset, eval_dataset, dataset_metadata = auto_load_dataset(
            dataset_spec,
            text_field=config.dataset.text_field,
            schema=config.dataset.schema_type,
            role_markers=config.dataset.role_markers,
            user_marker=config.dataset.user_marker,
            assistant_marker=config.dataset.assistant_marker
        )
        
        # Determine text field for tokenization
        text_field = config.dataset.text_field or dataset_metadata.get("text_field", "text")
        
        # Prepare dataset for training (tokenize)
        tokenized_train, tokenized_eval = prepare_dataset_for_training(
            train_dataset,
            eval_dataset,
            tokenizer,
            text_field=text_field,
            max_length=config.block_size
        )
        
        console.print(f"[blue]📈 Train samples: {len(tokenized_train)}, Eval samples: {len(tokenized_eval)}[/blue]")
        
        # Calculate max_steps if not provided
        if config.max_steps is None:
            steps_per_epoch = len(tokenized_train) // (config.batch_size * config.grad_accum)
            config.max_steps = steps_per_epoch * config.epochs
            console.print(f"[blue]📊 Calculated max_steps: {config.max_steps} (steps_per_epoch: {steps_per_epoch}, epochs: {config.epochs})[/blue]")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            bf16=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            gradient_checkpointing=config.gradient_checkpointing,
            save_total_limit=2,
            do_eval=True,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=model_obj,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
        )
        
        # Start training
        console.print("[green]✅ Starting training...[/green]")
        training_result = trainer.train()
        
        # Save final model
        console.print("[blue]💾 Saving final model...[/blue]")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        # Get final metrics
        final_metrics = {
            "train_loss": training_result.training_loss,
            "train_runtime": training_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": training_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": training_result.metrics.get("train_steps_per_second", 0),
            "total_steps": training_result.global_step,
            "epochs": training_result.metrics.get("epoch", 0),
        }
        
        # Get evaluation metrics if available
        if tokenized_eval:
            eval_results = trainer.evaluate()
            final_metrics.update({
                "eval_loss": eval_results.get("eval_loss", 0),
                "eval_perplexity": eval_results.get("eval_perplexity", 0),
            })
        
        # Add dataset metadata to results
        final_metrics.update({
            "dataset_type": dataset_metadata.get("dataset_type", "unknown"),
            "dataset_spec": dataset_metadata.get("dataset_spec", config.dataset),
            "text_field": text_field,
            "schema": dataset_metadata.get("schema", "unknown"),
        })
        
        console.print("[green]✅ Training completed successfully![/green]")
        console.print(f"[blue]📊 Final metrics: {final_metrics}[/blue]")
        
        return {
            "status": "success",
            "metrics": final_metrics,
            "output_dir": config.output_dir,
            "model_path": config.output_dir,
            "dataset_metadata": dataset_metadata
        }
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        return {
            "status": "error",
            "error": str(e),
            "output_dir": config.output_dir
        }


def run_universal_training_with_accelerator(config: TrainingConfig, accelerator) -> Dict[str, Any]:
    """
    Run training with Accelerate for multi-GPU support.
    
    Args:
        config: TrainingConfig object with all training parameters
        accelerator: Accelerate accelerator instance
        
    Returns:
        Dictionary with training results
    """
    try:
        console.print(f"[blue]🚀 Starting multi-GPU training with model: {config.model}[/blue]")
        console.print(f"[blue]📁 Output directory: {config.output_dir}[/blue]")
        console.print(f"[blue]📊 Dataset: {config.dataset}[/blue]")
        console.print(f"[blue]🔧 Using {accelerator.num_processes} GPUs[/blue]")
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Load tokenizer
        console.print("[blue]📝 Loading tokenizer...[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        console.print("[blue]🤖 Loading model...[/blue]")
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.float16 if config.dtype == "fp16" else torch.bfloat16,
            device_map="auto" if config.gpu_mode == "multi" else None
        )
        
        # Apply LoRA if enabled
        if config.lora:
            console.print("[blue]🔧 Applying LoRA configuration...[/blue]")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        
        # Load dataset
        console.print("[blue]📊 Loading dataset...[/blue]")
        train_dataset, eval_dataset, dataset_metadata = auto_load_dataset(config)
        
        # Prepare datasets for training
        train_dataset = prepare_dataset_for_training(train_dataset, tokenizer, config)
        if eval_dataset:
            eval_dataset = prepare_dataset_for_training(eval_dataset, tokenizer, config)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=config.dtype == "fp16",
            bf16=config.dtype == "bf16",
            gradient_checkpointing=config.gradient_checkpointing,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"{config.output_dir}/logs",
            save_total_limit=3,
            seed=42,
            data_seed=42,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Prepare with accelerator
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
        )
        
        # Update trainer with prepared components
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.train_dataloader = lambda: train_dataloader
        trainer.eval_dataloader = lambda: eval_dataloader
        
        # Train
        console.print("[blue]🏃 Starting training...[/blue]")
        train_result = trainer.train()
        
        # Save model
        console.print("[blue]💾 Saving model...[/blue]")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        # Final metrics
        final_metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "total_steps": train_result.metrics.get("train_steps", 0),
            "epochs": train_result.metrics.get("train_epoch", 0),
        }
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            final_metrics.update({
                "eval_loss": eval_result.get("eval_loss", 0),
            })
        
        console.print("[green]✅ Multi-GPU training completed successfully![/green]")
        console.print(f"[blue]📊 Final metrics: {final_metrics}[/blue]")
        
        return {
            "status": "success",
            "metrics": final_metrics,
            "output_dir": config.output_dir,
            "model_path": config.output_dir,
            "dataset_metadata": dataset_metadata
        }
        
    except Exception as e:
        console.print(f"[red]❌ Multi-GPU training failed: {str(e)}[/red]")
        return {
            "status": "error",
            "error": str(e),
            "output_dir": config.output_dir
        }
