#!/usr/bin/env python3
"""
Test Direct Authentication
=========================

This script tests the authentication directly without input prompts.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from client_login import HumigenceClientLogin
from rich.console import Console

console = Console()

def test_direct_auth():
    """Test authentication directly"""
    console.print("[bold blue]Direct Authentication Test[/bold blue]")
    console.print("=" * 50)
    
    # Test credentials
    device_name = "iPad"
    api_key = "SbdJ1tsf1gLcTuHrNFNQSdwSOzGzyV8_jYVTkoz6_-s"
    
    console.print(f"Testing with:")
    console.print(f"Device Name: '{device_name}'")
    console.print(f"API Key: '{api_key}'")
    
    # Create client instance
    client = HumigenceClientLogin()
    
    # Manually set the credentials (bypassing input)
    client.current_device = None
    client.session_active = False
    
    # Test the authentication logic directly
    dm = client.device_manager
    
    # Find device by API key
    device = dm.get_device_by_api_key(api_key)
    console.print(f"Device found by API key: {device is not None}")
    
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
    
    # Authentication successful
    client.current_device = device
    client.session_active = True
    
    console.print("[green]✅ Authentication successful![/green]")
    console.print(f"Device: {device.device_name}")
    console.print(f"Device ID: {device.device_id}")
    console.print(f"Endpoint: {device.endpoint}")
    
    # Test AI request
    console.print("\n[bold]Testing AI request...[/bold]")
    try:
        response = client.send_ai_request("Hello, this is a test message.")
        if response:
            console.print(f"[green]✅ AI request successful: {response}[/green]")
        else:
            console.print("[yellow]⚠️ AI request failed[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ AI request error: {e}[/red]")
    
    return True

if __name__ == "__main__":
    test_direct_auth()

