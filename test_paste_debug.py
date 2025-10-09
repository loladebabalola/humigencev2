#!/usr/bin/env python3
"""
Test Paste Debug
================

This script helps debug the paste issue in the authentication.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager
from rich.console import Console

console = Console()

def test_paste_debug():
    """Test paste functionality with debug output"""
    console.print("[bold blue]Paste Debug Test[/bold blue]")
    console.print("=" * 50)
    
    console.print("This will test the exact same input processing as the client login.")
    console.print("Please enter the same credentials you used before:")
    console.print("")
    
    # Get device name
    device_name = input("Device Name: ")
    console.print(f"Received device name: '{device_name}' (length: {len(device_name)})")
    console.print(f"Device name repr: {repr(device_name)}")
    
    # Get API key
    console.print("\nNow paste your API key:")
    api_key = input("API Key: ")
    console.print(f"Received API key: '{api_key}' (length: {len(api_key)})")
    console.print(f"API key repr: {repr(api_key)}")
    
    # Test authentication
    console.print("\n[bold]Testing authentication...[/bold]")
    
    dm = DeviceManager()
    
    # Find device by API key
    device = dm.get_device_by_api_key(api_key)
    console.print(f"Device found by API key: {device is not None}")
    
    if device:
        console.print(f"Device name from API key: '{device.device_name}'")
        console.print(f"Device name match: {device.device_name.lower() == device_name.lower()}")
        console.print(f"Device active: {device.is_active}")
        
        if device.device_name.lower() == device_name.lower() and device.is_active:
            console.print("[green]✅ Authentication should succeed![/green]")
        else:
            console.print("[red]❌ Authentication failed![/red]")
            if device.device_name.lower() != device_name.lower():
                console.print(f"Name mismatch: '{device.device_name}' vs '{device_name}'")
            if not device.is_active:
                console.print("Device is inactive")
    else:
        console.print("[red]❌ No device found with that API key[/red]")
        
        # Show available devices
        console.print("\nAvailable devices:")
        for d in dm.devices.values():
            console.print(f"  {d.device_name}: {d.api_key}")

if __name__ == "__main__":
    test_paste_debug()

