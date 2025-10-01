# cli/atomic_eval.py

import typer
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.production_pipeline import ProductionPipeline

app = typer.Typer()
console = Console()

@app.command()
def run(
    dataset_path: str = typer.Option(..., help="Path to the dataset file"),
    model_name: str = typer.Option("microsoft/DialoGPT-medium", help="Model name to fine-tune"),
    output_dir: str = typer.Option("runs/humigence", help="Output directory for results"),
    gpu_id: int = typer.Option(0, help="GPU ID to use for atomic evaluation"),
    use_atomic_evaluation: bool = typer.Option(True, help="Use atomic evaluation (true process isolation)"),
    per_device_train_batch_size: int = typer.Option(4, help="Training batch size per device"),
    per_device_eval_batch_size: int = typer.Option(8, help="Evaluation batch size per device"),
    num_train_epochs: int = typer.Option(3, help="Number of training epochs"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
):
    """
    Run Humigence pipeline with atomic evaluation for guaranteed device isolation
    """
    
    # Display configuration
    config_panel = Panel(
        f"""[bold cyan]Humigence Pipeline Configuration[/bold cyan]

[bold]Dataset:[/bold] {dataset_path}
[bold]Model:[/bold] {model_name}
[bold]Output Directory:[/bold] {output_dir}
[bold]GPU ID:[/bold] {gpu_id}
[bold]Atomic Evaluation:[/bold] {'✅ Enabled' if use_atomic_evaluation else '❌ Disabled'}
[bold]Training Batch Size:[/bold] {per_device_train_batch_size}
[bold]Evaluation Batch Size:[/bold] {per_device_eval_batch_size}
[bold]Epochs:[/bold] {num_train_epochs}
[bold]Learning Rate:[/bold] {learning_rate}
[bold]Max Length:[/bold] {max_length}""",
        title="🚀 Configuration",
        border_style="cyan"
    )
    
    console.print(config_panel)
    
    if use_atomic_evaluation:
        atomic_panel = Panel(
            """[bold green]Atomic Evaluation Enabled[/bold green]

✅ True process isolation
✅ No device contamination
✅ Guaranteed single-GPU evaluation
✅ Clean environment separation""",
            title="🔒 Atomic Evaluation",
            border_style="green"
        )
        console.print(atomic_panel)
    
    # Confirm before proceeding
    if not Confirm.ask("\n[bold]Proceed with pipeline execution?[/bold]"):
        console.print("[yellow]Pipeline cancelled by user[/yellow]")
        return
    
    # Create configuration
    config = {
        "dataset_path": dataset_path,
        "model_name": model_name,
        "output_dir": output_dir,
        "gpu_id": gpu_id,
        "use_atomic_evaluation": use_atomic_evaluation,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
    }
    
    try:
        # Initialize and run pipeline
        console.print("\n[bold cyan]🚀 Starting Humigence Pipeline...[/bold cyan]")
        
        pipeline = ProductionPipeline(config)
        result = pipeline.run()
        
        if result["status"] == "success":
            success_panel = Panel(
                f"""[bold green]Pipeline Completed Successfully![/bold green]

[bold]Status:[/bold] {result['status']}
[bold]Dataset Info:[/bold] {result.get('dataset_info', 'N/A')}
[bold]Report Path:[/bold] {result.get('report_path', 'N/A')}

[bold]Evaluation Results:[/bold]
{_format_evaluation_results(result.get('evaluation_results', {}))}""",
                title="🎉 Success",
                border_style="green"
            )
            console.print(success_panel)
        else:
            error_panel = Panel(
                f"""[bold red]Pipeline Failed[/bold red]

[bold]Status:[/bold] {result['status']}
[bold]Error:[/bold] {result.get('error', 'Unknown error')}""",
                title="❌ Error",
                border_style="red"
            )
            console.print(error_panel)
            
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline execution failed: {e}[/bold red]")
        raise typer.Exit(1)

def _format_evaluation_results(results):
    """Format evaluation results for display"""
    if not results:
        return "No evaluation results available"
    
    formatted = []
    for key, value in results.items():
        if isinstance(value, dict):
            if 'eval_loss' in value and 'perplexity' in value:
                formatted.append(f"  {key}: Loss={value['eval_loss']:.4f}, Perplexity={value['perplexity']:.2f}")
            else:
                formatted.append(f"  {key}: {value}")
        else:
            formatted.append(f"  {key}: {value}")
    
    return "\n".join(formatted) if formatted else "No metrics available"

if __name__ == "__main__":
    app()
