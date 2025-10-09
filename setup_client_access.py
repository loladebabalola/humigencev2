#!/usr/bin/env python3
"""
Humigence Client Access Setup
============================

This script helps you set up client device access to your Humigence server.
It provides an easy way to:
1. Register devices and get API credentials
2. Generate client scripts for each device
3. Test device connectivity
4. Manage device access

Usage:
    python3 setup_client_access.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich import print as rprint

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager
from inference.tenant_manager import TenantManager

console = Console()

class HumigenceClientSetup:
    """Main setup class for client access"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.tenant_manager = TenantManager()
        self.base_url = "http://localhost:8000"
    
    def show_banner(self):
        """Display the setup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Humigence Client Access Setup                      ║
║                                                                              ║
║  🚀 Register devices and get instant access to your AI inference server    ║
║  📱 Generate client scripts for easy device management                      ║
║  🔐 Secure API key-based authentication for each device                     ║
║  📊 Monitor device usage and manage quotas                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold blue"))
    
    def check_server_status(self) -> bool:
        """Check if the Humigence server is running"""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Humigence server is running[/green]")
                return True
        except Exception as e:
            console.print(f"[red]❌ Humigence server is not running: {e}[/red]")
            console.print("[yellow]💡 Please start the server first using: humigence[/yellow]")
            return False
    
    def register_new_device(self):
        """Register a new device"""
        console.print("\n[bold cyan]📱 Register New Device[/bold cyan]")
        console.print("=" * 50)
        
        # Get device name
        device_name = Prompt.ask("Device name", default="")
        if not device_name:
            device_name = None  # Auto-detect
        
        # Get quota
        quota = IntPrompt.ask("Quota percentage (0-100)", default=25)
        if not 0 <= quota <= 100:
            console.print("[red]❌ Quota must be between 0 and 100[/red]")
            return
        
        # Get priority
        priority = Prompt.ask("Priority (low/normal/high)", default="normal", choices=["low", "normal", "high"])
        
        # Register device
        try:
            device_id, api_key = self.device_manager.register_device(device_name, quota, priority)
            self.device_manager.save_devices()
            
            # Generate client script
            script_path = self.device_manager.generate_client_script(device_id)
            
            console.print(f"\n[bold green]🎉 Device Registration Complete![/bold green]")
            console.print(f"[blue]📱 Device Name: {device_name or 'Auto-detected'}[/blue]")
            console.print(f"[blue]🆔 Device ID: {device_id}[/blue]")
            console.print(f"[blue]🔑 API Key: {api_key}[/blue]")
            console.print(f"[blue]🌐 Endpoint: {self.base_url}/{device_id}[/blue]")
            console.print(f"[blue]📄 Client Script: {script_path}[/blue]")
            
            # Show usage instructions
            self.show_usage_instructions(device_id, api_key, script_path)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to register device: {e}[/red]")
    
    def show_usage_instructions(self, device_id: str, api_key: str, script_path: str):
        """Show usage instructions for the registered device"""
        instructions = f"""
[bold yellow]📋 How to Use Your Device:[/bold yellow]

[bold]1. Direct API Access:[/bold]
   curl -X POST {self.base_url}/{device_id}/v1/chat/completions \\
     -H "Authorization: Bearer {api_key}" \\
     -H "Content-Type: application/json" \\
     -d '{{"model": "gpt-oss-20b-F16", "messages": [{{"role": "user", "content": "Hello!"}}]}}'

[bold]2. Using the Client Script:[/bold]
   python3 {script_path} "Hello, how are you?"
   python3 {script_path} --status
   python3 {script_path}  # Interactive mode

[bold]3. From Another Device:[/bold]
   - Copy the client script to your device
   - The script contains all necessary credentials
   - Run it with Python 3

[bold]4. Programmatic Access:[/bold]
   - Device ID: {device_id}
   - API Key: {api_key}
   - Endpoint: {self.base_url}/{device_id}
        """
        
        console.print(Panel(instructions, title="Usage Instructions", border_style="green"))
    
    def list_devices(self):
        """List all registered devices"""
        console.print("\n[bold cyan]📱 Registered Devices[/bold cyan]")
        console.print("=" * 50)
        
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered yet[/yellow]")
            return
        
        self.device_manager.list_devices()
    
    def manage_device(self):
        """Manage existing devices"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered yet[/yellow]")
            return
        
        console.print("\n[bold cyan]🔧 Manage Device[/bold cyan]")
        console.print("=" * 50)
        
        # Show devices
        self.device_manager.list_devices()
        
        device_id = Prompt.ask("\nEnter device ID to manage")
        device = self.device_manager.get_device(device_id)
        
        if not device:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return
        
        console.print(f"\n[bold]Managing: {device.device_name} ({device_id})[/bold]")
        console.print(f"[blue]API Key: {device.api_key}[/blue]")
        console.print(f"[blue]Endpoint: {device.endpoint}[/blue]")
        console.print(f"[blue]Quota: {device.quota_percentage}%[/blue]")
        console.print(f"[blue]Priority: {device.priority}[/blue]")
        
        # Management options
        console.print("\n[bold]Management Options:[/bold]")
        console.print("1. Generate new client script")
        console.print("2. Show credentials")
        console.print("3. Test connection")
        console.print("4. Remove device")
        console.print("0. Back to main menu")
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
        
        if choice == "1":
            script_path = self.device_manager.generate_client_script(device_id)
            console.print(f"[green]✅ Generated client script: {script_path}[/green]")
        
        elif choice == "2":
            creds = self.device_manager.get_device_credentials(device_id)
            console.print(f"[green]✅ Device Credentials:[/green]")
            for key, value in creds.items():
                console.print(f"[blue]{key}: {value}[/blue]")
        
        elif choice == "3":
            self.test_device_connection(device_id)
        
        elif choice == "4":
            if Confirm.ask(f"Are you sure you want to remove device '{device_id}'?"):
                self.device_manager.remove_device(device_id)
                self.device_manager.save_devices()
                console.print(f"[green]✅ Removed device '{device_id}'[/green]")
    
    def test_device_connection(self, device_id: str):
        """Test device connection to the server"""
        device = self.device_manager.get_device(device_id)
        if not device:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return
        
        console.print(f"\n[bold]Testing connection for {device.device_name}...[/bold]")
        
        try:
            import requests
            
            # Test health endpoint
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                console.print("[green]✅ Server health check passed[/green]")
            else:
                console.print(f"[yellow]⚠️ Server health check returned status {response.status_code}[/yellow]")
            
            # Test device-specific endpoint
            device_url = f"{device.endpoint}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {device.api_key}",
                "X-Tenant-ID": device_id
            }
            
            test_payload = {
                "model": "gpt-oss-20b-F16",
                "messages": [{"role": "user", "content": "Hello, this is a test message."}],
                "max_tokens": 10,
                "temperature": 0.7
            }
            
            console.print("[blue]🔄 Sending test message...[/blue]")
            response = requests.post(device_url, headers=headers, json=test_payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    response_text = data['choices'][0].get('message', {}).get('content', '')
                    console.print(f"[green]✅ Test successful! Response: {response_text}[/green]")
                else:
                    console.print("[yellow]⚠️ Test completed but no response text received[/yellow]")
            else:
                console.print(f"[red]❌ Test failed with status {response.status_code}[/red]")
                console.print(f"[red]Response: {response.text}[/red]")
        
        except Exception as e:
            console.print(f"[red]❌ Connection test failed: {e}[/red]")
    
    def show_quick_start(self):
        """Show quick start guide"""
        guide = """
[bold yellow]🚀 Quick Start Guide[/bold yellow]

[bold]Step 1: Start the Server[/bold]
   humigence
   # Select option 3 (Multi-Tenant Inference)
   # Wait for "All services started successfully!"

[bold]Step 2: Register Your Device[/bold]
   python3 setup_client_access.py
   # Select option 1 (Register New Device)
   # Follow the prompts

[bold]Step 3: Use Your Device[/bold]
   # Copy the generated client script to your device
   python3 humigence_client_[device_id].py "Hello, AI!"
   
   # Or use direct API calls
   curl -X POST http://localhost:8000/[device_id]/v1/chat/completions \\
     -H "Authorization: Bearer [api_key]" \\
     -H "Content-Type: application/json" \\
     -d '{"model": "gpt-oss-20b-F16", "messages": [{"role": "user", "content": "Hello!"}]}'

[bold]Step 4: Add More Devices[/bold]
   # Repeat Step 2 for each additional device
   # Each device gets its own API key and endpoint

[bold]💡 Tips:[/bold]
   - Each device gets its own quota percentage
   - Higher priority devices get faster responses
   - Monitor usage with option 2 (List Devices)
   - Generate new client scripts anytime with option 3
        """
        
        console.print(Panel(guide, title="Quick Start Guide", border_style="yellow"))
    
    def main_menu(self):
        """Main menu loop"""
        while True:
            console.print("\n[bold cyan]🏠 Main Menu[/bold cyan]")
            console.print("=" * 50)
            console.print("1. Register New Device")
            console.print("2. List Devices")
            console.print("3. Manage Device")
            console.print("4. Test Server Connection")
            console.print("5. Quick Start Guide")
            console.print("0. Exit")
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5"])
            
            if choice == "0":
                console.print("[green]👋 Goodbye![/green]")
                break
            elif choice == "1":
                self.register_new_device()
            elif choice == "2":
                self.list_devices()
            elif choice == "3":
                self.manage_device()
            elif choice == "4":
                self.test_device_connection("test")
            elif choice == "5":
                self.show_quick_start()
            
            if choice != "0":
                Prompt.ask("\nPress Enter to continue...")

def main():
    """Main entry point"""
    setup = HumigenceClientSetup()
    setup.show_banner()
    
    # Check server status
    if not setup.check_server_status():
        if not Confirm.ask("Continue anyway?"):
            return
    
    # Show main menu
    setup.main_menu()

if __name__ == "__main__":
    main()

