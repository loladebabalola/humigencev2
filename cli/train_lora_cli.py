#!/usr/bin/env python3
"""
Command-line interface for Humigence LoRA training
"""

import argparse
import sys
from pathlib import Path

# Add the humigence directory to Python path
humigence_dir = Path(__file__).parent.parent
sys.path.insert(0, str(humigence_dir))

from cli.train_lora_single import train_lora_single_gpu, main as train_main


def main():
    """Main CLI entry point for LoRA training."""
    parser = argparse.ArgumentParser(
        description="Humigence LoRA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Llama-3 with default settings
  humigence train-lora --model meta-llama/Meta-Llama-3-8B-Instruct --output-dir ./out_lora

  # Train Mistral with custom parameters
  humigence train-lora --model mistralai/Mistral-7B-Instruct-v0.1 --output-dir ./out_mistral --max-steps 2000 --batch-size 2

  # Train Phi-2 with custom LoRA settings
  humigence train-lora --model microsoft/Phi-2 --output-dir ./out_phi --lora-r 32 --lora-alpha 64

  # Use accelerate for better memory management
  accelerate launch --num_processes=1 humigence train-lora --model meta-llama/Meta-Llama-3-8B-Instruct --output-dir ./out_lora
        """
    )
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    
    # Training arguments
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=512, help="Block size for text grouping")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Other arguments
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save-steps", type=int, default=200, help="Save frequency")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluation frequency")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    result = train_lora_single_gpu(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
    )
    
    # Exit with appropriate code
    if result["status"] == "success":
        print("✅ Training completed successfully!")
        sys.exit(0)
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()

