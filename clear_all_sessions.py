#!/usr/bin/env python3
"""
Clear All Sessions
=================

This script clears all client sessions (admin function).
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

console = Console()

def clear_all_sessions():
    """Clear all client sessions"""
    console.print("[bold red]Clear All Client Sessions[/bold red]")
    console.print("=" * 50)
    
    session_file = Path("client_session.json")
    
    if session_file.exists():
        session_file.unlink()
        console.print("[green]✅ All client sessions cleared[/green]")
        console.print("[yellow]💡 All users will need to log in again[/yellow]")
    else:
        console.print("[yellow]ℹ️ No sessions found to clear[/yellow]")
    
    console.print("\n[bold]Note:[/bold] This only clears local sessions.")
    console.print("To deactivate devices permanently, use the admin device manager.")

if __name__ == "__main__":
    clear_all_sessions()

