#!/usr/bin/env python3
"""
Test Paste Functionality
========================

This script tests how paste functionality works with different input methods.
"""

import sys
import getpass
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from client_login import get_password_input
from rich.console import Console

console = Console()

def test_paste_functionality():
    """Test paste functionality with different input methods"""
    console.print("[bold blue]Testing Paste Functionality[/bold blue]")
    console.print("=" * 50)
    
    console.print("\n[yellow]This test will check if you can paste API keys:[/yellow]")
    console.print("1. Try pasting with Ctrl+Shift+V (Linux) or Cmd+V (Mac)")
    console.print("2. Try right-click paste")
    console.print("3. Try middle-click paste")
    
    console.print("\n[bold]Test 1: Regular input (should work with paste)[/bold]")
    try:
        regular_input = input("Regular input (try pasting): ")
        console.print(f"✅ Regular input works: {regular_input[:10]}...")
    except Exception as e:
        console.print(f"❌ Regular input failed: {e}")
    
    console.print("\n[bold]Test 2: getpass (may or may not work with paste)[/bold]")
    try:
        getpass_input = getpass.getpass("getpass input (try pasting): ")
        console.print(f"✅ getpass works: {getpass_input[:10]}...")
    except Exception as e:
        console.print(f"❌ getpass failed: {e}")
    
    console.print("\n[bold]Test 3: Our custom function (should work with paste)[/bold]")
    try:
        custom_input = get_password_input("Custom input (try pasting): ")
        console.print(f"✅ Custom input works: {custom_input[:10]}...")
    except Exception as e:
        console.print(f"❌ Custom input failed: {e}")
    
    console.print("\n[bold green]Summary:[/bold green]")
    console.print("- Regular input: Usually supports paste")
    console.print("- getpass: May not support paste in all terminals")
    console.print("- Our custom function: Falls back to regular input if getpass fails")
    console.print("\n[bold yellow]Recommendation:[/bold yellow]")
    console.print("If paste doesn't work with getpass, the system will automatically")
    console.print("fall back to regular input (with a warning) which supports paste.")

if __name__ == "__main__":
    test_paste_functionality()

