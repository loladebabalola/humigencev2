#!/usr/bin/env python3
"""
Clear My Session
================

This script clears your personal session so you can test the login process again.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

console = Console()

def clear_my_session():
    """Clear your personal session"""
    console.print("[bold blue]Clear My Session[/bold blue]")
    console.print("=" * 40)
    
    session_file = Path("client_session.json")
    
    if session_file.exists():
        session_file.unlink()
        console.print("[green]✅ Your session has been cleared[/green]")
        console.print("[yellow]💡 Next time you run 'humigence-client', you'll need to log in again[/yellow]")
    else:
        console.print("[yellow]ℹ️ No session found to clear[/yellow]")
    
    console.print("\n[bold]Note:[/bold] This only clears your local session.")
    console.print("Your device is still registered and active on the server.")

if __name__ == "__main__":
    clear_my_session()

