#!/usr/bin/env python3
"""
Test Persistent Session
======================

This script demonstrates the persistent session functionality.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from client_login import HumigenceClientLogin
from rich.console import Console

console = Console()

def test_persistent_session():
    """Test the persistent session functionality"""
    console.print("[bold blue]Testing Persistent Session Functionality[/bold blue]")
    console.print("=" * 60)
    
    # Test 1: First login (should require authentication)
    console.print("\n[bold]Test 1: First Login[/bold]")
    client1 = HumigenceClientLogin()
    
    if client1.session_active:
        console.print("✅ Session already exists - skipping authentication")
    else:
        console.print("ℹ️ No existing session - would require authentication")
    
    # Test 2: Simulate successful authentication
    console.print("\n[bold]Test 2: Simulate Authentication[/bold]")
    
    # Get device
    from inference.device_manager import DeviceManager
    dm = DeviceManager()
    device = dm.get_device_by_api_key('SbdJ1tsf1gLcTuHrNFNQSdwSOzGzyV8_jYVTkoz6_-s')
    
    if device and device.is_active:
        client1.current_device = device
        client1.session_active = True
        client1.save_session()
        console.print(f"✅ Session saved for {device.device_name}")
    else:
        console.print("❌ Device not available")
        return
    
    # Test 3: Second login (should restore session)
    console.print("\n[bold]Test 3: Second Login (Session Restore)[/bold]")
    client2 = HumigenceClientLogin()
    
    if client2.session_active and client2.current_device:
        console.print(f"✅ Session restored for {client2.current_device.device_name}")
        console.print(f"   Device ID: {client2.current_device.device_id}")
        console.print(f"   Quota: {client2.current_device.quota_percentage}%")
    else:
        console.print("❌ Session not restored")
    
    # Test 4: Logout (should clear session)
    console.print("\n[bold]Test 4: Logout (Session Clear)[/bold]")
    client2.logout()
    
    # Test 5: Third login (should require authentication again)
    console.print("\n[bold]Test 5: Third Login (After Logout)[/bold]")
    client3 = HumigenceClientLogin()
    
    if client3.session_active:
        console.print("❌ Session should have been cleared")
    else:
        console.print("✅ Session cleared - would require authentication")
    
    console.print("\n[bold green]🎉 Persistent Session Test Complete![/bold green]")
    console.print("\n[bold]How it works:[/bold]")
    console.print("1. First login: Enter device name and API key")
    console.print("2. Session saved: Login persists until logout")
    console.print("3. Subsequent logins: Automatic session restore")
    console.print("4. Logout: Clears session, requires re-authentication")
    console.print("5. Admin removal: Session becomes invalid automatically")

if __name__ == "__main__":
    test_persistent_session()

