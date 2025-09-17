import typer
from rich.console import Console
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from cli
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import fine_tune

app = typer.Typer()
console = Console()

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print("[bold cyan]Humigence — Your AI. Your pipeline. Zero code.[/bold cyan]")
        console.print("[green]A complete MLOps suite built for makers, teams, and enterprises.[/green]")
        console.print()
        console.print("Options:")
        console.print("1. Supervised Fine-Tuning ✅")
        console.print("2. RAG Implementation (coming soon)")
        console.print("3. EnterpriseGPT (coming soon)")
        console.print("4. Batch Inference (coming soon)")
        console.print("5. Context Length (coming soon)")
        console.print()
        console.print("Starting Supervised Fine-Tuning...")
        fine_tune.run()

app.command()(fine_tune.run)

if __name__ == "__main__":
    app()