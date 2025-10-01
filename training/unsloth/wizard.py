"""
Unsloth Training Wizard for Humigence CLI
Interactive configuration for dual-GPU LoRA training
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich import print
import inquirer

console = Console()

def show_unsloth_banner():
    """Display Unsloth training banner"""
    console.print(Panel.fit(
        "[bold cyan]🚀 Unsloth Dual-GPU LoRA Training[/bold cyan]\n"
        "[dim]Fast, memory-efficient fine-tuning with Unsloth + TorchRun[/dim]",
        title="Humigence Training",
        border_style="cyan"
    ))

def choose_model():
    """Choose base model for fine-tuning"""
    models = [
        ("unsloth/Llama-3-8B-Instruct", "Llama-3-8B-Instruct (Unsloth optimized)"),
        ("unsloth/Llama-3-8B", "Llama-3-8B (Base model)"),
        ("unsloth/Llama-2-7B-Instruct", "Llama-2-7B-Instruct (Smaller, faster)"),
        ("unsloth/Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct (Alternative)"),
    ]
    
    questions = [
        inquirer.List('model',
                     message="Choose base model: (Use arrow keys)",
                     choices=[display_name for _, display_name in models],
                     default=models[0][1])
    ]
    
    answers = inquirer.prompt(questions)
    selected_display = answers['model']
    
    # Find the actual model name
    for model_name, display_name in models:
        if display_name == selected_display:
            return model_name
    
    return models[0][0]  # fallback

def choose_dataset():
    """Choose dataset for training"""
    datasets = [
        ("wikitext", "wikitext-2-raw-v1", "WikiText-2 (Wikipedia text)"),
        ("wikitext", "wikitext-103-raw-v1", "WikiText-103 (Large Wikipedia)"),
        ("squad", "plain_text", "SQuAD (Question answering)"),
        ("imdb", "plain_text", "IMDB (Movie reviews)"),
        ("custom", "custom", "Custom dataset (provide path)"),
    ]
    
    questions = [
        inquirer.List('dataset',
                     message="Choose dataset: (Use arrow keys)",
                     choices=[display_name for _, _, display_name in datasets],
                     default=datasets[0][2])
    ]
    
    answers = inquirer.prompt(questions)
    selected_display = answers['dataset']
    
    # Find the actual dataset
    for dataset_name, dataset_config, display_name in datasets:
        if display_name == selected_display:
            if dataset_name == "custom":
                custom_path = Prompt.ask("Enter path to your dataset file")
                return dataset_name, dataset_config, custom_path
            return dataset_name, dataset_config, None
    
    return datasets[0][0], datasets[0][1], None

def choose_training_config():
    """Choose training configuration"""
    console.print("\n[bold cyan]Training Configuration[/bold cyan]")
    
    # Training steps
    max_steps = int(Prompt.ask("Maximum training steps", default="1000"))
    
    # Batch size
    batch_size = int(Prompt.ask("Batch size per device", default="2"))
    
    # Gradient accumulation
    grad_accum = int(Prompt.ask("Gradient accumulation steps", default="4"))
    
    # Learning rate
    learning_rate = float(Prompt.ask("Learning rate", default="2e-4"))
    
    # Sequence length
    block_size = int(Prompt.ask("Sequence length", default="1024"))
    
    return {
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "block_size": block_size
    }

def choose_precision():
    """Choose training precision"""
    console.print("\n[bold cyan]Training Precision[/bold cyan]")
    console.print("[dim]Choose the precision method for training:[/dim]")
    
    questions = [
        inquirer.List('precision',
                     message="Choose training recipe: (Use arrow keys)",
                     choices=[
                         ("qlora_4bit", "QLoRA (4-bit) - Memory efficient"),
                         ("lora_fp16", "LoRA (FP16) - Full precision"),
                         ("lora_bf16", "LoRA (BF16) - Full precision with better stability")
                     ],
                     default="qlora_4bit")
    ]
    
    answers = inquirer.prompt(questions)
    return answers['precision']

def choose_lora_config():
    """Choose LoRA configuration"""
    console.print("\n[bold cyan]LoRA Configuration[/bold cyan]")
    
    # LoRA rank
    lora_r = int(Prompt.ask("LoRA rank", default="16"))
    
    # LoRA alpha
    lora_alpha = int(Prompt.ask("LoRA alpha", default="32"))
    
    # LoRA dropout
    lora_dropout = float(Prompt.ask("LoRA dropout", default="0.0"))
    
    return {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout
    }

def choose_launch_method():
    """Choose launch method"""
    console.print("\n[bold cyan]Launch Method[/bold cyan]")
    console.print("[dim]Choose how to launch dual-GPU training:[/dim]")
    
    questions = [
        inquirer.List('launch_method',
                     message="Choose launch method: (Use arrow keys)",
                     choices=[
                         ("torchrun", "TorchRun (DDP) - Recommended for dual-GPU"),
                         ("accelerate", "Accelerate (Model Sharding) - Alternative method")
                     ],
                     default="torchrun")
    ]
    
    answers = inquirer.prompt(questions)
    return answers['launch_method']

def show_config_summary(config):
    """Show configuration summary"""
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]")
    
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", width=25)
    table.add_column(style="white")
    
    table.add_row("Model:", config["model_name"])
    table.add_row("Dataset:", f"{config['dataset_name']}/{config['dataset_config']}")
    if config.get("dataset_path"):
        table.add_row("Dataset Path:", config["dataset_path"])
    table.add_row("Output Directory:", config["output_dir"])
    table.add_row("Max Steps:", str(config["max_steps"]))
    table.add_row("Batch Size:", str(config["batch_size"]))
    table.add_row("Grad Accum:", str(config["grad_accum"]))
    table.add_row("Learning Rate:", str(config["learning_rate"]))
    table.add_row("Block Size:", str(config["block_size"]))
    table.add_row("Precision:", config["precision"])
    table.add_row("LoRA Rank:", str(config["lora_r"]))
    table.add_row("LoRA Alpha:", str(config["lora_alpha"]))
    table.add_row("LoRA Dropout:", str(config["lora_dropout"]))
    table.add_row("Launch Method:", config["launch_method"])
    
    console.print(table)

def run_unsloth_wizard() -> Optional[Dict[str, Any]]:
    """Run the complete Unsloth training wizard"""
    try:
        # Show banner
        show_unsloth_banner()
        
        # Choose model
        model_name = choose_model()
        
        # Choose dataset
        dataset_name, dataset_config, dataset_path = choose_dataset()
        
        # Choose training config
        training_config = choose_training_config()
        
        # Choose precision
        precision = choose_precision()
        
        # Choose LoRA config
        lora_config = choose_lora_config()
        
        # Choose launch method
        launch_method = choose_launch_method()
        
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"./runs/humigence/out_lora_dual_{timestamp}"
        
        # Build configuration
        config = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "launch_method": launch_method,
            "precision": precision,
            **training_config,
            **lora_config
        }
        
        # Show summary
        show_config_summary(config)
        
        # Confirmation
        if not Confirm.ask("\nProceed with training?", default=True):
            console.print("[bold red]❌ Training cancelled.[/bold red]")
            return None
        
        return config
        
    except KeyboardInterrupt:
        console.print("\n[bold red]❌ Training cancelled.[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]❌ Error in wizard: {e}[/bold red]")
        return None

def save_config(config: Dict[str, Any], output_dir: str) -> str:
    """Save configuration to file"""
    config_path = Path(output_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)

if __name__ == "__main__":
    # Test the wizard
    config = run_unsloth_wizard()
    if config:
        print(f"Configuration: {config}")
        config_path = save_config(config, config["output_dir"])
        print(f"Config saved to: {config_path}")
    else:
        print("Wizard cancelled")

