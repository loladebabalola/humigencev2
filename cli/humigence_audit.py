# humigence_audit.py

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def audit_run(run_dir="runs/humigence"):
    config_path = Path(run_dir) / "config.snapshot.json"
    summary_path = Path(run_dir) / "run_summary.json"
    status = "❌ Not found"

    if Path(run_dir, "ACCEPTED.txt").exists():
        status = "✅ ACCEPTED"
    elif Path(run_dir, "REJECTED.txt").exists():
        status = "❌ REJECTED"

    console.rule("[bold magenta]Humigence Run Audit")

    # Load config
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

        table = Table(title="Training Configuration", show_lines=True)
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)
    else:
        console.print("[red]❌ config.snapshot.json not found[/red]")

    # Load summary
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        console.print(f"\n[bold green]📄 Summary:[/bold green] {summary['status']}")
        console.print(json.dumps(summary, indent=2))
    else:
        console.print("[yellow]⚠️ run_summary.json not found[/yellow]")

    console.print(f"\n[bold cyan]📌 Run Status:[/bold cyan] {status}")

if __name__ == "__main__":
    audit_run()
