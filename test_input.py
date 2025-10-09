#!/usr/bin/env python3
"""
Test Input Processing
====================

This script tests how input is processed in the authentication.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager
from rich.console import Console

console = Console()

def test_input_processing():
    """Test how input is processed"""
    console.print("[bold blue]Testing Input Processing[/bold blue]")
    console.print("=" * 50)
    
    # Simulate the exact input you provided
    device_name = "iPad"
    api_key = "SbdJ1tsf1gLcTuHrNFNQSdwSOzGzyV8_jYVTkoz6_-s"
    
    console.print(f"[blue]Simulating input:[/blue]")
    console.print(f"Device Name: '{device_name}' (length: {len(device_name)})")
    console.print(f"API Key: '{api_key}' (length: {len(api_key)})")
    
    # Test with the exact authentication logic
    dm = DeviceManager()
    
    # Find device by API key
    device = dm.get_device_by_api_key(api_key)
    
    if not device:
        console.print("[red]❌ No device found with that API key[/red]")
        return False
    
    # Verify device name matches (case-insensitive)
    if device.device_name.lower() != device_name.lower():
        console.print(f"[red]❌ Device name does not match[/red]")
        console.print(f"Expected: '{device.device_name}'")
        console.print(f"Got: '{device_name}'")
        return False
    
    # Check if device is active
    if not device.is_active:
        console.print("[red]❌ Device is inactive[/red]")
        return False
    
    console.print("[green]✅ Authentication should succeed![/green]")
    console.print(f"Device: {device.device_name}")
    console.print(f"Device ID: {device.device_id}")
    console.print(f"Endpoint: {device.endpoint}")
    
    return True

if __name__ == "__main__":
    test_input_processing()

