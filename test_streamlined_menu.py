#!/usr/bin/env python3
"""
Test Streamlined Menu Structure
==============================

This script demonstrates the new streamlined menu structure.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

console = Console()

def show_new_menu_structure():
    """Show the new streamlined menu structure"""
    console.print("[bold blue]New Streamlined Menu Structure[/bold blue]")
    console.print("=" * 60)
    
    console.print("\n[bold cyan]Multi-Tenant Inference Control Panel[/bold cyan]")
    console.print("=" * 60)
    console.print("📊 Status: Router 🟢 Running, Llama Servers 🟢 2/2 running")
    console.print("💚 GPU Health: 0% util, 24.9GB/31.8GB VRAM, 46°C")
    
    console.print("\n[bold]Quick Actions:[/bold]")
    console.print("[bold green]1.[/bold green] Start/Stop Services")
    console.print("[bold green]2.[/bold green] Monitor (live dashboard)")
    console.print("[bold green]3.[/bold green] Tenant Management")
    console.print("[bold green]4.[/bold green] Device Management 📱")
    console.print("[bold green]5.[/bold green] Advanced Options")
    console.print("\n[bold red]0.[/bold red] Exit Wizard")
    
    console.print("\n[bold yellow]Key Improvements:[/bold yellow]")
    console.print("✅ Device Management moved to main level (option 4)")
    console.print("✅ Tenant Management streamlined (option 3)")
    console.print("✅ Removed redundant sub-menu navigation")
    console.print("✅ Preserved all functionality including Tenant Statistics")
    console.print("✅ More direct user journey")
    
    console.print("\n[bold]Tenant Management (Option 3):[/bold]")
    console.print("1. List Tenants")
    console.print("2. Add Tenant")
    console.print("3. Remove Tenant")
    console.print("4. Update Tenant")
    console.print("5. Tenant Statistics")
    console.print("0. Back to Main Menu")
    
    console.print("\n[bold]Device Management (Option 4):[/bold]")
    console.print("1. Register New Device")
    console.print("2. List Devices")
    console.print("3. Manage Device")
    console.print("4. Test Device Connection")
    console.print("5. Generate Client Scripts")
    console.print("0. Back to Main Menu")
    
    console.print("\n[bold green]🎉 Streamlined and more efficient![/bold green]")

if __name__ == "__main__":
    show_new_menu_structure()

