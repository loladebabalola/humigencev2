#!/usr/bin/env python3
"""
Humigence CLI - Main entry point for all Humigence commands
"""

import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import sys

# Add the current directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from training.train_wikitext import run_training

app = typer.Typer(
    name="humigence",
    help="Your AI. Your pipeline. Zero code.",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def train_wikitext(
    model: str = typer.Option(
        ..., 
        "--model", 
        "-m", 
        help="Path or Hugging Face model name (e.g., 'gpt2' or 'microsoft/DialoGPT-small')"
    ),
    output_dir: str = typer.Option(
        ..., 
        "--output-dir", 
        "-o", 
        help="Directory where checkpoints will be saved"
    ),
    epochs: int = typer.Option(
        1, 
        "--epochs", 
        "-e", 
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        2, 
        "--batch-size", 
        "-b", 
        help="Per-device batch size"
    ),
    learning_rate: float = typer.Option(
        5e-5, 
        "--learning-rate", 
        "-lr", 
        help="Learning rate for training"
    ),
    dataset: str = typer.Option(
        "wikitext", 
        "--dataset", 
        help="Dataset name (default: wikitext)"
    ),
    dataset_config: str = typer.Option(
        "wikitext-2-raw-v1", 
        "--dataset-config", 
        help="Dataset configuration (default: wikitext-2-raw-v1)"
    ),
    max_steps: Optional[int] = typer.Option(
        None, 
        "--max-steps", 
        help="Maximum training steps (overrides epochs if set)"
    ),
    block_size: int = typer.Option(
        1024, 
        "--block-size", 
        help="Maximum sequence length"
    ),
    grad_accum: int = typer.Option(
        4, 
        "--grad-accum", 
        help="Gradient accumulation steps"
    ),
    warmup_steps: int = typer.Option(
        100, 
        "--warmup-steps", 
        help="Number of warmup steps"
    ),
    logging_steps: int = typer.Option(
        10, 
        "--logging-steps", 
        help="Logging frequency in steps"
    ),
    save_steps: int = typer.Option(
        200, 
        "--save-steps", 
        help="Model saving frequency in steps"
    ),
    eval_steps: int = typer.Option(
        200, 
        "--eval-steps", 
        help="Evaluation frequency in steps"
    ),
    lora_r: int = typer.Option(
        8, 
        "--lora-r", 
        help="LoRA rank"
    ),
    lora_alpha: int = typer.Option(
        32, 
        "--lora-alpha", 
        help="LoRA alpha parameter"
    ),
    lora_dropout: float = typer.Option(
        0.05, 
        "--lora-dropout", 
        help="LoRA dropout rate"
    ),
):
    """
    Train a model on Wikitext dataset using LoRA fine-tuning.
    
    This command fine-tunes a language model on the Wikitext dataset using LoRA (Low-Rank Adaptation)
    for efficient parameter updates. The training runs on a single GPU by default.
    
    Examples:
        # Basic training with GPT-2
        humigence train-wikitext --model gpt2 --output-dir ./out
        
        # Training with custom parameters
        humigence train-wikitext --model microsoft/DialoGPT-small --output-dir ./out --epochs 2 --batch-size 4 --learning-rate 1e-4
        
        # Training with specific steps instead of epochs
        humigence train-wikitext --model gpt2 --output-dir ./out --max-steps 1000 --batch-size 2
    """
    
    # Display training configuration
    config_panel = Panel(
        f"""[bold blue]Training Configuration[/bold blue]
        
[cyan]Model:[/cyan] {model}
[cyan]Output Directory:[/cyan] {output_dir}
[cyan]Epochs:[/cyan] {epochs}
[cyan]Batch Size:[/cyan] {batch_size}
[cyan]Learning Rate:[/cyan] {learning_rate}
[cyan]Dataset:[/cyan] {dataset}/{dataset_config}
[cyan]Max Steps:[/cyan] {max_steps if max_steps else 'Auto-calculated'}
[cyan]Block Size:[/cyan] {block_size}
[cyan]Gradient Accumulation:[/cyan] {grad_accum}
[cyan]LoRA Rank:[/cyan] {lora_r}
[cyan]LoRA Alpha:[/cyan] {lora_alpha}
[cyan]LoRA Dropout:[/cyan] {lora_dropout}""",
        title="🚀 Starting Wikitext Training",
        border_style="green"
    )
    
    console.print(config_panel)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        result = run_training(
            model=model,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dataset=dataset,
            dataset_config=dataset_config,
            max_steps=max_steps,
            block_size=block_size,
            grad_accum=grad_accum,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        if result["status"] == "success":
            console.print(Panel(
                f"""[bold green]✅ Training Completed Successfully![/bold green]
                
[cyan]Output Directory:[/cyan] {result['output_dir']}
[cyan]Model Path:[/cyan] {result['model_path']}

[bold blue]Final Metrics:[/bold blue]
[cyan]Train Loss:[/cyan] {result['metrics'].get('train_loss', 'N/A')}
[cyan]Eval Loss:[/cyan] {result['metrics'].get('eval_loss', 'N/A')}
[cyan]Total Steps:[/cyan] {result['metrics'].get('total_steps', 'N/A')}
[cyan]Epochs:[/cyan] {result['metrics'].get('epochs', 'N/A')}
[cyan]Train Runtime:[/cyan] {result['metrics'].get('train_runtime', 'N/A')}s
[cyan]Samples/Second:[/cyan] {result['metrics'].get('train_samples_per_second', 'N/A')}""",
                title="🎉 Training Results",
                border_style="green"
            ))
            raise typer.Exit(0)
        else:
            console.print(Panel(
                f"""[bold red]❌ Training Failed[/bold red]
                
[red]Error:[/red] {result.get('error', 'Unknown error')}
[cyan]Output Directory:[/cyan] {result.get('output_dir', 'N/A')}""",
                title="💥 Training Error",
                border_style="red"
            ))
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(Panel(
            f"""[bold red]❌ Unexpected Error[/bold red]
            
[red]Error:[/red] {str(e)}""",
            title="💥 Unexpected Error",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("[bold blue]Humigence v1.0.0[/bold blue]")
    console.print("[dim]Your AI. Your pipeline. Zero code.[/dim]")


@app.callback()
def main(
    version: bool = typer.Option(
        False, 
        "--version", 
        "-v", 
        help="Show version and exit"
    )
):
    """
    Humigence - Your AI. Your pipeline. Zero code.
    
    A complete MLOps suite built for makers, teams, and enterprises.
    """
    if version:
        console.print("[bold blue]Humigence v1.0.0[/bold blue]")
        console.print("[dim]Your AI. Your pipeline. Zero code.[/dim]")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()




