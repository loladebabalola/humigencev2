#!/usr/bin/env python3
"""
Humigence Admin Device Manager
==============================

This script provides an admin interface for managing client device credentials.
Admins can create, update, and manage device access for SSH-based authentication.

Usage:
    python3 admin_device_manager.py
"""

import os
import sys
import json
import time
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich import print as rprint

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager

console = Console()

class AdminDeviceManager:
    """Admin interface for managing client device credentials"""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.server_url = self.device_manager.base_endpoint
    
    def show_admin_banner(self):
        """Display admin banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Humigence Admin Device Manager                       ║
║                                                                              ║
║  🔧 Manage client device credentials for SSH-based authentication          ║
║  📱 Create, update, and monitor device access                              ║
║  🔐 Generate API keys and manage device permissions                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold red"))
        console.print(f"[blue]🌐 Server: {self.server_url}[/blue]")
    
    def show_main_menu(self):
        """Display main admin menu"""
        while True:
            console.print("\n[bold red]🔧 Admin Device Manager[/bold red]")
            console.print("=" * 60)
            console.print("[bold]Device Management:[/bold]")
            console.print("[bold green]1.[/bold green] Create New Device")
            console.print("[bold green]2.[/bold green] List All Devices")
            console.print("[bold green]3.[/bold green] Update Device")
            console.print("[bold green]4.[/bold green] Remove Device")
            console.print("[bold green]5.[/bold green] Device Statistics")
            console.print("[bold green]6.[/bold green] Generate Client Credentials")
            console.print("[bold green]7.[/bold green] Test Device Connection")
            console.print("[bold green]8.[/bold green] Export Device List")
            console.print("[bold red]0.[/bold red] Exit")
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            
            if choice == "1":
                self.create_device()
            elif choice == "2":
                self.list_devices()
            elif choice == "3":
                self.update_device()
            elif choice == "4":
                self.remove_device()
            elif choice == "5":
                self.show_device_statistics()
            elif choice == "6":
                self.generate_client_credentials()
            elif choice == "7":
                self.test_device_connection()
            elif choice == "8":
                self.export_device_list()
            elif choice == "0":
                console.print("[green]👋 Goodbye![/green]")
                break
    
    def create_device(self):
        """Create a new device with credentials"""
        console.print("\n[bold green]➕ Create New Device[/bold green]")
        console.print("=" * 50)
        
        # Get device information
        device_name = Prompt.ask("Device Name (e.g., iPhone-13, MacBook-Pro, iPad-Air)")
        if not device_name.strip():
            console.print("[red]❌ Device name cannot be empty[/red]")
            return
        
        # Check if device already exists
        existing_devices = [d.device_name.lower() for d in self.device_manager.devices.values()]
        if device_name.lower() in existing_devices:
            console.print("[red]❌ Device name already exists[/red]")
            return
        
        # Get quota and priority
        quota = IntPrompt.ask("Quota percentage (0-100)", default=25)
        if not 0 <= quota <= 100:
            console.print("[red]❌ Quota must be between 0 and 100[/red]")
            return
        
        priority = Prompt.ask("Priority (low/normal/high)", choices=["low", "normal", "high"], default="normal")
        
        # Create device
        try:
            device_id, api_key = self.device_manager.register_device(device_name, quota, priority)
            self.device_manager.save_devices()
            
            console.print(f"\n[bold green]✅ Device created successfully![/bold green]")
            console.print(f"[blue]Device Name: {device_name}[/blue]")
            console.print(f"[blue]Device ID: {device_id}[/blue]")
            console.print(f"[blue]API Key: {api_key}[/blue]")
            console.print(f"[blue]Endpoint: {self.device_manager.base_endpoint}/{device_id}[/blue]")
            console.print(f"[blue]Quota: {quota}%[/blue]")
            console.print(f"[blue]Priority: {priority}[/blue]")
            
            # Show login instructions
            self.show_login_instructions(device_name, api_key)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create device: {e}[/red]")
    
    def show_login_instructions(self, device_name: str, api_key: str):
        """Show login instructions for the device"""
        instructions = f"""
[bold yellow]📋 Login Instructions for {device_name}:[/bold yellow]

[bold]1. SSH into the server:[/bold]
   ssh user@server

[bold]2. Run the client login:[/bold]
   humigence-client

[bold]3. Enter credentials:[/bold]
   Device Name: {device_name}
   API Key (Password): {api_key}

[bold]4. Start chatting with AI![/bold]

[bold]💡 Alternative - Direct API Access:[/bold]
   curl -X POST {self.server_url}/{device_name.lower().replace('-', '_')}_[device_id]/v1/chat/completions \\
     -H "Authorization: Bearer {api_key}" \\
     -H "Content-Type: application/json" \\
     -d '{{"model": "gpt-oss-20b-F16", "messages": [{{"role": "user", "content": "Hello!"}}]}}'
        """
        
        console.print(Panel(instructions, title="Login Instructions", border_style="green"))
    
    def list_devices(self):
        """List all devices with detailed information"""
        console.print("\n[bold green]📱 All Devices[/bold green]")
        console.print("=" * 50)
        
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Registered Devices")
        table.add_column("Device Name", style="cyan", width=20)
        table.add_column("Device ID", style="blue", width=25)
        table.add_column("Quota %", style="green", width=10)
        table.add_column("Priority", style="yellow", width=10)
        table.add_column("Status", style="white", width=10)
        table.add_column("Last Seen", style="white", width=15)
        
        for device in self.device_manager.devices.values():
            status = "🟢 Active" if device.is_active else "🔴 Inactive"
            last_seen = "Never"
            if device.last_accessed > 0:
                time_diff = time.time() - device.last_accessed
                if time_diff < 3600:  # Less than 1 hour
                    last_seen = f"{int(time_diff/60)}m ago"
                elif time_diff < 86400:  # Less than 1 day
                    last_seen = f"{int(time_diff/3600)}h ago"
                else:
                    last_seen = f"{int(time_diff/86400)}d ago"
            
            table.add_row(
                device.device_name,
                device.device_id,
                f"{device.quota_percentage}%",
                device.priority,
                status,
                last_seen
            )
        
        console.print(table)
    
    def update_device(self):
        """Update device settings"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        console.print("\n[bold green]✏️ Update Device[/bold green]")
        console.print("=" * 50)
        
        # Show devices
        self.list_devices()
        
        device_name = Prompt.ask("\nEnter device name to update")
        device = None
        
        # Find device by name
        for d in self.device_manager.devices.values():
            if d.device_name.lower() == device_name.lower():
                device = d
                break
        
        if not device:
            console.print(f"[red]❌ Device '{device_name}' not found[/red]")
            return
        
        console.print(f"\n[blue]Updating: {device.device_name} ({device.device_id})[/blue]")
        
        # Update quota
        new_quota = IntPrompt.ask("New quota percentage (0-100)", default=int(device.quota_percentage))
        if 0 <= new_quota <= 100:
            device.quota_percentage = new_quota
            console.print(f"[green]✅ Updated quota to {new_quota}%[/green]")
        
        # Update priority
        new_priority = Prompt.ask("New priority (low/normal/high)", 
                                choices=["low", "normal", "high"], 
                                default=device.priority)
        device.priority = new_priority
        console.print(f"[green]✅ Updated priority to {new_priority}[/green]")
        
        # Save changes
        self.device_manager.save_devices()
        console.print(f"[green]✅ Device updated successfully[/green]")
    
    def remove_device(self):
        """Remove a device"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        console.print("\n[bold red]🗑️ Remove Device[/bold red]")
        console.print("=" * 50)
        
        # Show devices
        self.list_devices()
        
        device_name = Prompt.ask("\nEnter device name to remove")
        device = None
        
        # Find device by name
        for d in self.device_manager.devices.values():
            if d.device_name.lower() == device_name.lower():
                device = d
                break
        
        if not device:
            console.print(f"[red]❌ Device '{device_name}' not found[/red]")
            return
        
        if Confirm.ask(f"Are you sure you want to remove device '{device.device_name}'?"):
            self.device_manager.remove_device(device.device_id)
            self.device_manager.save_devices()
            console.print(f"[green]✅ Device '{device.device_name}' removed successfully[/green]")
    
    def show_device_statistics(self):
        """Show device usage statistics"""
        console.print("\n[bold green]📊 Device Statistics[/bold green]")
        console.print("=" * 50)
        
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        # Calculate statistics
        total_devices = len(self.device_manager.devices)
        active_devices = len([d for d in self.device_manager.devices.values() if d.is_active])
        total_quota = sum(d.quota_percentage for d in self.device_manager.devices.values())
        
        console.print(f"[blue]Total Devices: {total_devices}[/blue]")
        console.print(f"[blue]Active Devices: {active_devices}[/blue]")
        console.print(f"[blue]Inactive Devices: {total_devices - active_devices}[/blue]")
        console.print(f"[blue]Total Quota Allocated: {total_quota}%[/blue]")
        console.print(f"[blue]Available Quota: {100 - total_quota}%[/blue]")
        
        # Show quota distribution
        console.print("\n[bold]Quota Distribution:[/bold]")
        for device in self.device_manager.devices.values():
            console.print(f"  {device.device_name}: {device.quota_percentage}% ({device.priority})")
    
    def generate_client_credentials(self):
        """Generate client credentials for a device"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        console.print("\n[bold green]🔑 Generate Client Credentials[/bold green]")
        console.print("=" * 50)
        
        # Show devices
        self.list_devices()
        
        device_name = Prompt.ask("\nEnter device name")
        device = None
        
        # Find device by name
        for d in self.device_manager.devices.values():
            if d.device_name.lower() == device_name.lower():
                device = d
                break
        
        if not device:
            console.print(f"[red]❌ Device '{device_name}' not found[/red]")
            return
        
        # Show credentials
        console.print(f"\n[bold]Credentials for {device.device_name}:[/bold]")
        console.print(f"[blue]Device Name: {device.device_name}[/blue]")
        console.print(f"[blue]API Key: {device.api_key}[/blue]")
        console.print(f"[blue]Endpoint: {device.endpoint}[/blue]")
        
        # Show login instructions
        self.show_login_instructions(device.device_name, device.api_key)
    
    def test_device_connection(self):
        """Test connection for a device"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        console.print("\n[bold green]🔄 Test Device Connection[/bold green]")
        console.print("=" * 50)
        
        # Show devices
        self.list_devices()
        
        device_name = Prompt.ask("\nEnter device name to test")
        device = None
        
        # Find device by name
        for d in self.device_manager.devices.values():
            if d.device_name.lower() == device_name.lower():
                device = d
                break
        
        if not device:
            console.print(f"[red]❌ Device '{device_name}' not found[/red]")
            return
        
        console.print(f"\n[blue]Testing connection for {device.device_name}...[/blue]")
        
        try:
            import requests
            
            # Test server health
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✅ Server is online[/green]")
            else:
                console.print(f"[yellow]⚠️ Server returned status {response.status_code}[/yellow]")
            
            # Test device endpoint
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {device.api_key}",
                "X-Tenant-ID": device.device_id
            }
            
            test_payload = {
                "model": "gpt-oss-20b-F16",
                "messages": [{"role": "user", "content": "Test connection"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{device.endpoint}/v1/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    response_text = data['choices'][0].get('message', {}).get('content', '')
                    console.print(f"[green]✅ Test successful! Response: {response_text}[/green]")
                else:
                    console.print("[yellow]⚠️ Test completed but no response text[/yellow]")
            else:
                console.print(f"[red]❌ Test failed with status {response.status_code}[/red]")
        
        except Exception as e:
            console.print(f"[red]❌ Connection test failed: {e}[/red]")
    
    def export_device_list(self):
        """Export device list to a file"""
        if not self.device_manager.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        console.print("\n[bold green]📄 Export Device List[/bold green]")
        console.print("=" * 50)
        
        filename = Prompt.ask("Export filename", default="device_list.json")
        
        try:
            export_data = {
                "exported_at": time.time(),
                "server_url": self.server_url,
                "devices": []
            }
            
            for device in self.device_manager.devices.values():
                device_data = {
                    "device_name": device.device_name,
                    "device_id": device.device_id,
                    "api_key": device.api_key,
                    "quota_percentage": device.quota_percentage,
                    "priority": device.priority,
                    "endpoint": device.endpoint,
                    "is_active": device.is_active,
                    "created_at": device.created_at
                }
                export_data["devices"].append(device_data)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"[green]✅ Device list exported to {filename}[/green]")
            console.print(f"[blue]Total devices: {len(export_data['devices'])}[/blue]")
            
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
    
    def run(self):
        """Main entry point"""
        try:
            self.show_admin_banner()
            self.show_main_menu()
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Unexpected error: {e}[/red]")

def main():
    """Main entry point"""
    admin = AdminDeviceManager()
    admin.run()

if __name__ == "__main__":
    main()

