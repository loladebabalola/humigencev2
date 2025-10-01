#!/usr/bin/env python3
"""
Wikitext Training Module for Humigence
Refactored from standalone training script to be integrated into the Humigence workflow.
"""

import os
import torch
from typing import Optional, Dict, Any
from datasets import load_dataset
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

console = Console()


def run_training_from_config(config: TrainingConfig) -> Dict[str, Any]:
    """
    Run Wikitext training with LoRA fine-tuning using TrainingConfig.
    
    Args:
        config: TrainingConfig object with all training parameters
        
    Returns:
        Dictionary with training results
    """
    return run_training(
        model=config.model,
        output_dir=config.output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        dataset=config.dataset,
        dataset_config=config.dataset_config,
        max_steps=config.max_steps,
        block_size=config.block_size,
        grad_accum=config.grad_accum,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )


def run_training(
    model: str,
    output_dir: str,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    dataset: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_steps: Optional[int] = None,
    block_size: int = 1024,
    grad_accum: int = 4,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_steps: int = 200,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Dict[str, Any]:
    """
    Run Wikitext training with LoRA fine-tuning.
    
    Args:
        model: Path or Hugging Face model name
        output_dir: Where checkpoints are saved
        epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate for training
        dataset: Dataset name (default: wikitext)
        dataset_config: Dataset configuration (default: wikitext-2-raw-v1)
        max_steps: Maximum training steps (overrides epochs if set)
        block_size: Maximum sequence length
        grad_accum: Gradient accumulation steps
        warmup_steps: Number of warmup steps
        logging_steps: Logging frequency
        save_steps: Model saving frequency
        eval_steps: Evaluation frequency
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        Dictionary containing training results and metrics
    """
    try:
        console.print(f"[blue]🚀 Starting Wikitext training with model: {model}[/blue]")
        console.print(f"[blue]📁 Output directory: {output_dir}[/blue]")
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Load tokenizer
        console.print("[blue]📝 Loading tokenizer...[/blue]")
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        console.print("[blue]🤖 Loading model...[/blue]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Configure LoRA with model-specific target modules
        console.print("[blue]🔧 Configuring LoRA...[/blue]")
        
        # Determine target modules based on model type
        if "gpt" in model.lower() or "dialo" in model.lower():
            target_modules = ["c_attn", "c_proj"]
        elif "llama" in model.lower() or "mistral" in model.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default to common transformer modules
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none"
        )
        
        model_obj = get_peft_model(model_obj, lora_config)
        
        # Enable input gradients for gradient checkpointing to work with LoRA
        model_obj.enable_input_require_grads()
        
        model_obj.print_trainable_parameters()
        
        # Load dataset
        console.print(f"[blue]📊 Loading dataset: {dataset}/{dataset_config}[/blue]")
        raw_dataset = load_dataset(dataset, dataset_config)
        
        def tokenize_function(examples):
            """Tokenize the dataset"""
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=block_size
            )
        
        # Tokenize dataset
        console.print("[blue]🔄 Tokenizing dataset...[/blue]")
        tokenized_dataset = raw_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        tokenized_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask"]
        )
        
        # Split dataset
        split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        console.print(f"[blue]📈 Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}[/blue]")
        
        # Calculate max_steps if not provided
        if max_steps is None:
            steps_per_epoch = len(train_dataset) // (batch_size * grad_accum)
            max_steps = steps_per_epoch * epochs
            console.print(f"[blue]📊 Calculated max_steps: {max_steps} (steps_per_epoch: {steps_per_epoch}, epochs: {epochs})[/blue]")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            bf16=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,
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
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            processing_class=tokenizer,  # Use processing_class instead of tokenizer
        )
        
        # Start training
        console.print("[green]✅ Starting training...[/green]")
        training_result = trainer.train()
        
        # Save final model
        console.print("[blue]💾 Saving final model...[/blue]")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
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
        if eval_dataset:
            eval_results = trainer.evaluate()
            final_metrics.update({
                "eval_loss": eval_results.get("eval_loss", 0),
                "eval_perplexity": eval_results.get("eval_perplexity", 0),
            })
        
        console.print("[green]✅ Training completed successfully![/green]")
        console.print(f"[blue]📊 Final metrics: {final_metrics}[/blue]")
        
        return {
            "status": "success",
            "metrics": final_metrics,
            "output_dir": output_dir,
            "model_path": output_dir
        }
        
    except Exception as e:
        console.print(f"[red]❌ Training failed: {str(e)}[/red]")
        return {
            "status": "error",
            "error": str(e),
            "output_dir": output_dir
        }


def main():
    """
    Main function for standalone execution (for testing purposes)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Wikitext Training")
    parser.add_argument("--model", required=True, type=str, help="Model name or path")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save-steps", type=int, default=200, help="Save steps")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval steps")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    args = parser.parse_args()
    
    result = run_training(
        model=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        max_steps=args.max_steps,
        block_size=args.block_size,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    if result["status"] == "success":
        console.print("[green]✅ Training completed successfully![/green]")
        return 0
    else:
        console.print(f"[red]❌ Training failed: {result.get('error', 'Unknown error')}[/red]")
        return 1


if __name__ == "__main__":
    exit(main())
