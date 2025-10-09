#!/usr/bin/env python3
"""
Test Password Input Behavior
============================

This script demonstrates how the password input works in the Humigence client.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from client_login import get_password_input
from rich.console import Console

console = Console()

def test_password_input():
    """Test the password input function"""
    console.print("[bold blue]Testing Password Input Behavior[/bold blue]")
    console.print("=" * 50)
    
    console.print("\n[yellow]This will test how password input works:[/yellow]")
    console.print("1. In interactive terminals: Characters will be hidden")
    console.print("2. In non-interactive environments: Characters will be visible")
    console.print("3. The system will automatically choose the best method")
    
    console.print("\n[bold]Test 1: Device Name Input (visible)[/bold]")
    device_name = input("Device Name: ")
    console.print(f"You entered: {device_name}")
    
    console.print("\n[bold]Test 2: API Key Input (hidden in interactive terminals)[/bold]")
    console.print("[yellow]Note: In interactive terminals, characters will be hidden as you type[/yellow]")
    api_key = get_password_input("API Key (Password)")
    console.print(f"You entered: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else '***'}")
    
    console.print("\n[green]✅ Password input test completed![/green]")
    console.print("\n[bold]Summary:[/bold]")
    console.print("- Device Name: Always visible (for user convenience)")
    console.print("- API Key: Hidden in interactive terminals (for security)")
    console.print("- Fallback: Visible in non-interactive environments (with warning)")

if __name__ == "__main__":
    test_password_input()

