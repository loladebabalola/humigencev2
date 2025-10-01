#!/usr/bin/env python3
"""
Dual-GPU LoRA Fine-tuning Script for Humigence
Optimized for dual NVIDIA RTX 5090 (Blackwell architecture)

This script implements efficient LoRA fine-tuning across two GPUs using Unsloth's
FastLanguageModel with automatic weight distribution and 4-bit quantization.

Usage:
    # With TorchRun (recommended for dual-GPU)
    torchrun --nproc_per_node=2 train_lora_dual.py
    
    # With Accelerate
    accelerate launch --gpu_ids all train_lora_dual.py
"""

import argparse
import os
import logging
import math
import time
from typing import Optional, Dict, Any
import warnings

# Check for required dependencies
try:
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from transformers import (
        TrainingArguments, 
        Trainer, 
        DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Please run: python3 training/unsloth/setup_humigence_unsloth.py")
    DEPENDENCIES_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize logger - will be configured later
logger = logging.getLogger(__name__)

def setup_logging(output_dir="./runs/humigence/out_lora_dual"):
    """Setup logging configuration"""
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / 'training.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dual-GPU LoRA Fine-tuning with Unsloth")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="unsloth/Llama-3-8B-Instruct",
                       help="Model name or path")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                       help="Dataset configuration")
    
    # Training arguments
    parser.add_argument("--block_size", type=int, default=1024,
                       help="Sequence length for training")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--per_device_batch", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout (keep 0.0 for Unsloth fast path)")
    
    # Precision arguments
    parser.add_argument("--precision", type=str, default="qlora_4bit",
                       choices=["qlora_4bit", "lora_fp16", "lora_bf16"],
                       help="Training precision method")
    
    # Output arguments
    parser.add_argument("--out_dir", type=str, default="./runs/humigence/out_lora_dual",
                       help="Output directory")
    
    return parser.parse_args()

def setup_environment():
    """Setup environment variables for RTX 5090/Blackwell compatibility"""
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    # Check for dual GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPU(s)")
    
    # Check if we're running in DDP mode
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if is_ddp:
        logger.info(f"Running in DDP mode: WORLD_SIZE={os.environ.get('WORLD_SIZE', 1)}, LOCAL_RANK={local_rank}")
        logger.info(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    else:
        logger.info("Running in single-process mode")
        if gpu_count < 2:
            logger.warning(f"Only {gpu_count} GPU(s) detected. For optimal performance, use 2 GPUs.")
    
    # Set environment variables for optimal performance and stability
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPUs
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"  # RTX 5090 architecture
    
    # RTX 5090 specific optimizations and stability fixes
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,roundup_power2_divisions:16"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for performance
    os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
    os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for stability
    os.environ["NCCL_SHM_DISABLE"] = "1"  # Disable shared memory
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback interface
    
    # Memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    
    logger.info("Environment setup completed")

def load_model_and_tokenizer(args):
    """Load Llama-3-8B model with 4-bit quantization and dual-GPU distribution"""
    logger.info(f"Loading model: {args.model}")
    
    try:
        # Check if we're in DDP mode
        is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if is_ddp:
            # In DDP mode, each process uses one GPU
            device_map = f"cuda:{local_rank}"
            logger.info(f"DDP mode: Using GPU {local_rank}")
        else:
            # In single-process mode, use single GPU to avoid device mismatch
            # Unsloth has issues with model sharding across multiple GPUs
            device_map = "cuda:0"
            logger.info("Single-process mode: Using device_map='cuda:0' (Unsloth compatibility)")
        
        # Determine precision settings based on user selection
        if args.precision == "qlora_4bit":
            # QLoRA uses 4-bit quantization
            load_in_4bit = True
            dtype = None  # Auto-detect for quantized models
            logger.info(f"Using QLoRA with 4-bit quantization (Unsloth default method)")
        else:
            # LoRA methods use full precision
            load_in_4bit = False
            if args.precision == "lora_bf16":
                dtype = torch.bfloat16
            elif args.precision == "lora_fp16":
                dtype = torch.float16
            else:
                dtype = None  # Auto-detect
            logger.info(f"Using LoRA with full precision: {dtype}")
        
        logger.info(f"Loading model with precision: {args.precision}")
        logger.info(f"Quantization: 4bit={load_in_4bit}, dtype={dtype}")
        
        # Load model with selected precision
        # Note: Unsloth uses its own 4-bit quantization method internally
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.block_size,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Apply LoRA adapters for efficient fine-tuning
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,  # Keep 0.0 for Unsloth fast path
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth" if not is_ddp else False,  # Disable for DDP
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Fix for DDP compatibility with Unsloth
        if is_ddp:
            # Disable gradient checkpointing for DDP as it conflicts with Unsloth
            model.gradient_checkpointing_disable()
            logger.info("Disabled gradient checkpointing for DDP compatibility")
        
        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def prepare_dataset(args, tokenizer):
    """Load and prepare the training dataset"""
    logger.info(f"Loading dataset: {args.dataset}/{args.dataset_config}")
    
    try:
        # Load dataset
        raw_dataset = load_dataset(args.dataset, args.dataset_config, split="train")
        
        # Filter out empty texts and limit size for demo
        raw_dataset = raw_dataset.filter(lambda x: len(x["text"]) > 0)
        raw_dataset = raw_dataset.select(range(min(1000, len(raw_dataset))))  # Limit to 1000 samples for demo
        
        # Use a simpler approach that works better with Unsloth
        # Just use the raw text and let the data collator handle tokenization
        def format_text(example):
            # Simple formatting for causal language modeling
            text = example["text"].strip()
            if len(text) < 10:  # Skip very short texts
                return {"text": ""}
            return {"text": text}
        
        # Format the dataset
        formatted_dataset = raw_dataset.map(format_text, remove_columns=raw_dataset.column_names)
        # Filter out empty texts
        formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 0)
        
        # Use the formatted dataset directly
        lm_datasets = formatted_dataset
        
        logger.info(f"Dataset prepared with {len(lm_datasets)} samples")
        return lm_datasets
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def create_training_args(args):
    """Create TrainingArguments for training with dual-GPU optimizations"""
    logger.info("Creating training configuration...")
    
    # Check if we're in DDP mode
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Calculate effective batch size
    effective_batch_size = args.per_device_batch * args.grad_accum * world_size
    logger.info(f"Batch size calculation:")
    logger.info(f"  Per device batch size: {args.per_device_batch}")
    logger.info(f"  Gradient accumulation steps: {args.grad_accum}")
    logger.info(f"  Number of processes/GPUs: {world_size}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    
    # Determine precision based on user selection
    if args.precision == "lora_bf16":
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = not bf16
        if not bf16:
            logger.warning("BF16 not supported on this hardware, falling back to FP16")
            fp16 = True
    elif args.precision == "lora_fp16":
        bf16 = False
        fp16 = True
    else:
        # For QLoRA methods, use mixed precision based on hardware
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = not bf16
    
    training_args = TrainingArguments(
        # Output configuration
        output_dir=args.out_dir,
        
        # Training parameters
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_steps=args.max_steps,
        
        # Optimization
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        
        # Logging and saving
        logging_steps=10,
        save_steps=200,
        save_strategy="steps",
        save_total_limit=2,
        
        # Memory optimization
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        gradient_checkpointing=not is_ddp,  # Disable for DDP (Unsloth compatibility)
        
        # DDP configuration will be handled by Accelerate
        
        # Mixed precision training
        bf16=bf16,
        fp16=fp16,
        
        # Other settings
        remove_unused_columns=False,  # Critical for proper data handling
        report_to="none",  # Disable wandb/tensorboard for simplicity
        seed=3407,
    )
    
    logger.info(f"Training configuration created (bf16={bf16}, fp16={fp16})")
    return training_args

def create_summary_display(args, trainer, start_time, end_time, eval_results=None, train_loss=None):
    """Create a visually pleasing ASCII summary of training results"""
    runtime = end_time - start_time
    runtime_str = f"{int(runtime // 60)}m {int(runtime % 60):02d}s"
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    for i in range(gpu_count):
        gpu_names.append(torch.cuda.get_device_name(i))
    
    # Determine if we're in DDP mode
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Create summary box
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("✅ Training Complete (Dual-GPU)" if is_ddp else "✅ Training Complete (Single-GPU)")
    summary_lines.append("")
    summary_lines.append(f"📁 Run Directory: {args.out_dir}")
    
    # Add training loss if available
    if train_loss is not None:
        summary_lines.append(f"📊 Final Train Loss:  {train_loss:.3f}")
    
    # Add evaluation results if available
    if eval_results is not None:
        eval_loss = eval_results.get('eval_loss', 'N/A')
        if eval_loss != 'N/A':
            summary_lines.append(f"📊 Final Eval Loss:   {eval_loss:.3f}")
            perplexity = math.exp(eval_loss)
            summary_lines.append(f"📊 Perplexity:        {perplexity:.3f}")
        else:
            summary_lines.append("📊 Final Eval Loss:   N/A")
            summary_lines.append("📊 Perplexity:        N/A")
    else:
        summary_lines.append("ℹ️  No validation split detected, skipping evaluation.")
    
    # Add GPU information
    if is_ddp:
        summary_lines.append(f"🖥️  GPUs Used:         {world_size} × {gpu_names[0] if gpu_names else 'Unknown'}")
    else:
        summary_lines.append(f"🖥️  GPU Used:          {gpu_names[0] if gpu_names else 'Unknown'}")
    
    # Add runtime
    summary_lines.append(f"⏱️  Runtime:           {runtime_str}")
    summary_lines.append("=" * 60)
    
    return "\n".join(summary_lines)

def main():
    """Main training function"""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Exiting.")
        return 1
    
    args = parse_args()
    
    # Setup logging now that we know dependencies are available
    setup_logging(args.out_dir)
    
    # Start timing
    start_time = time.time()
    
    logger.info("Starting dual-GPU LoRA fine-tuning...")
    logger.info(f"Arguments: {args}")
    
    try:
        # Setup environment
        setup_environment()
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Prepare dataset
        dataset = prepare_dataset(args, tokenizer)
        
        # Create training arguments
        training_args = create_training_args(args)
        
        # Create data collator that handles text tokenization
        class TextDataCollator:
            def __init__(self, tokenizer, max_length=1024):
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __call__(self, features):
                # Extract texts
                texts = [f["text"] for f in features]
                
                # Tokenize with padding and truncation
                batch = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # For causal language modeling, labels are the same as input_ids
                batch["labels"] = batch["input_ids"].clone()
                
                # Ensure all tensors are on the same device (let Accelerate handle device placement)
                # Don't manually move tensors to specific devices
                
                return batch
        
        data_collator = TextDataCollator(tokenizer, args.block_size)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Extract final training loss
        final_train_loss = None
        if hasattr(train_result, 'log_history') and train_result.log_history:
            # Get the last logged training loss
            for log_entry in reversed(train_result.log_history):
                if 'train_loss' in log_entry:
                    final_train_loss = log_entry['train_loss']
                    break
        
        # Run evaluation if validation dataset is available
        eval_results = None
        try:
            logger.info("Running evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation completed: {eval_results}")
        except Exception as e:
            logger.warning(f"Evaluation failed or skipped: {e}")
            logger.info("ℹ️  No validation split detected, skipping evaluation.")
        
        # Save LoRA adapters
        logger.info("Saving LoRA adapters...")
        model.save_pretrained(f"{args.out_dir}/lora_adapter")
        tokenizer.save_pretrained(f"{args.out_dir}/lora_adapter")
        
        # Try to save merged weights
        try:
            logger.info("Attempting to save merged weights...")
            FastLanguageModel.save_pretrained_merged(
                model, 
                tokenizer, 
                f"{args.out_dir}/merged", 
                save_method="merged_16bit"
            )
            logger.info("Merged weights saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save merged weights: {e}")
            logger.warning("This is acceptable - LoRA adapters are still saved")
        
        # End timing
        end_time = time.time()
        
        # Create and display summary
        summary = create_summary_display(args, trainer, start_time, end_time, eval_results, final_train_loss)
        
        # Print summary to console
        print("\n" + summary)
        
        # Log summary to file
        logger.info("Training Summary:")
        for line in summary.split('\n'):
            logger.info(line)
        
        logger.info("Training completed successfully!")
        
    except RuntimeError as e:
        if "CUDA error" in str(e) or "illegal memory access" in str(e):
            logger.error(f"CUDA memory error detected: {e}")
            logger.error("This is often caused by dual-GPU training issues.")
            logger.error("Consider using single-GPU training or checking GPU memory.")
            raise
        else:
            logger.error(f"Runtime error: {e}")
            raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()