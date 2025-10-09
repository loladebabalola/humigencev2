# inference/device_manager.py

import os
import json
import time
import secrets
import hashlib
import platform
import socket
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
import typer

console = Console()

@dataclass
class DeviceInfo:
    """Device information and capabilities"""
    device_name: str
    device_type: str  # phone, tablet, laptop, desktop, server
    os_name: str
    os_version: str
    hardware_id: str  # Unique hardware identifier
    network_info: Dict[str, str]
    capabilities: List[str]  # What the device can do
    last_seen: float = 0.0
    is_online: bool = False

@dataclass
class DeviceTenant:
    """Device-specific tenant configuration"""
    device_id: str
    device_name: str
    tenant_alias: str
    api_key: str
    device_info: DeviceInfo
    quota_percentage: float = 25.0
    priority: str = "normal"
    created_at: float = 0.0
    last_accessed: float = 0.0
    is_active: bool = True
    endpoint: Optional[str] = None

class DeviceManager:
    """Device registration and management system"""
    
    def __init__(self, devices_file: str = "registered_devices.json"):
        self.devices_file = Path(devices_file)
        self.devices: Dict[str, DeviceTenant] = {}
        self.device_stats: Dict[str, Dict[str, Any]] = {}
        self.base_endpoint = self.get_network_endpoint()
        
        # Load existing devices
        self.load_devices()
    
    def get_network_endpoint(self) -> str:
        """Get the network endpoint for the server"""
        try:
            import socket
            # Get the primary network interface IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to a remote address to determine local IP
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            
            # Use the detected IP with port 8000
            endpoint = f"http://{local_ip}:8000"
            console.print(f"[blue]🌐 Using network endpoint: {endpoint}[/blue]")
            return endpoint
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not detect network IP, using localhost: {e}[/yellow]")
            return "http://localhost:8000"
    
    def load_devices(self):
        """Load registered devices from JSON file"""
        if not self.devices_file.exists():
            console.print(f"[yellow]⚠️ Devices file not found: {self.devices_file}[/yellow]")
            return
        
        try:
            with open(self.devices_file, 'r') as f:
                data = json.load(f)
            
            devices_data = data.get('devices', {})
            
            for device_id, device_data in devices_data.items():
                # Convert device info
                device_info_data = device_data.get('device_info', {})
                device_info = DeviceInfo(**device_info_data)
                
                device_tenant = DeviceTenant(
                    device_id=device_id,
                    device_name=device_data.get('device_name', 'Unknown'),
                    tenant_alias=device_data.get('tenant_alias', ''),
                    api_key=device_data.get('api_key', ''),
                    device_info=device_info,
                    quota_percentage=device_data.get('quota_percentage', 25.0),
                    priority=device_data.get('priority', 'normal'),
                    created_at=device_data.get('created_at', time.time()),
                    last_accessed=device_data.get('last_accessed', 0.0),
                    is_active=device_data.get('is_active', True),
                    endpoint=device_data.get('endpoint')
                )
                
                self.devices[device_id] = device_tenant
                self.device_stats[device_id] = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'current_rps': 0.0,
                    'last_request_time': 0.0,
                    'error_count': 0,
                    'last_seen': 0.0
                }
            
            console.print(f"[green]✅ Loaded {len(self.devices)} registered device(s)[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load devices: {e}[/red]")
    
    def save_devices(self):
        """Save registered devices to JSON file"""
        try:
            data = {
                'devices': {},
                'metadata': {
                    'version': '1.0.0',
                    'last_updated': time.time(),
                    'total_devices': len(self.devices)
                }
            }
            
            for device_id, device in self.devices.items():
                device_data = {
                    'device_name': device.device_name,
                    'tenant_alias': device.tenant_alias,
                    'api_key': device.api_key,
                    'device_info': asdict(device.device_info),
                    'quota_percentage': device.quota_percentage,
                    'priority': device.priority,
                    'created_at': device.created_at,
                    'last_accessed': device.last_accessed,
                    'is_active': device.is_active,
                    'endpoint': device.endpoint
                }
                data['devices'][device_id] = device_data
            
            with open(self.devices_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            console.print(f"[green]✅ Saved {len(self.devices)} device(s) to {self.devices_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save devices: {e}[/red]")
    
    def get_device_info(self) -> DeviceInfo:
        """Get current device information"""
        try:
            # Get basic system info
            os_name = platform.system()
            os_version = platform.version()
            
            # Generate hardware ID based on MAC address and system info
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(5,-1,-1)])
            hardware_id = hashlib.sha256(f"{mac}_{os_name}_{os_version}".encode()).hexdigest()[:16]
            
            # Get network info
            hostname = socket.gethostname()
            try:
                local_ip = socket.gethostbyname(hostname)
            except:
                local_ip = "127.0.0.1"
            
            network_info = {
                "hostname": hostname,
                "local_ip": local_ip,
                "mac_address": mac
            }
            
            # Determine device type based on OS and other factors
            device_type = "unknown"
            if os_name == "Darwin":  # macOS
                device_type = "laptop" if "MacBook" in platform.machine() else "desktop"
            elif os_name == "Windows":
                device_type = "laptop" if "LAPTOP" in hostname.upper() else "desktop"
            elif os_name == "Linux":
                device_type = "server" if "server" in hostname.lower() else "laptop"
            
            # Basic capabilities
            capabilities = ["inference", "chat"]
            if os_name == "Darwin" or os_name == "Windows":
                capabilities.append("gui")
            
            return DeviceInfo(
                device_name=hostname,
                device_type=device_type,
                os_name=os_name,
                os_version=os_version,
                hardware_id=hardware_id,
                network_info=network_info,
                capabilities=capabilities
            )
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not detect device info: {e}[/yellow]")
            return DeviceInfo(
                device_name="unknown_device",
                device_type="unknown",
                os_name="unknown",
                os_version="unknown",
                hardware_id="unknown",
                network_info={},
                capabilities=["inference"]
            )
    
    def register_device(self, device_name: Optional[str] = None, quota_percentage: float = 25.0, priority: str = "normal") -> Tuple[str, str]:
        """Register current device and return device_id and API key"""
        device_info = self.get_device_info()
        
        # Use provided name or detected name
        if not device_name:
            device_name = device_info.device_name
        
        # Generate unique device ID
        device_id = f"{device_name}_{device_info.hardware_id[:8]}"
        
        # Check if device already exists
        if device_id in self.devices:
            existing_device = self.devices[device_id]
            console.print(f"[yellow]⚠️ Device '{device_name}' already registered![/yellow]")
            console.print(f"[blue]🔑 Existing API Key: {existing_device.api_key}[/blue]")
            console.print(f"[blue]🌐 Endpoint: {existing_device.endpoint}[/blue]")
            return device_id, existing_device.api_key
        
        # Generate API key
        api_key = self.generate_api_key()
        
        # Create device tenant
        device_tenant = DeviceTenant(
            device_id=device_id,
            device_name=device_name,
            tenant_alias=device_id,  # Use device_id as tenant alias
            api_key=api_key,
            device_info=device_info,
            quota_percentage=quota_percentage,
            priority=priority,
            created_at=time.time(),
            is_active=True,
            endpoint=f"{self.base_endpoint}/{device_id}"
        )
        
        self.devices[device_id] = device_tenant
        self.device_stats[device_id] = {
            'total_requests': 0,
            'total_tokens': 0,
            'current_rps': 0.0,
            'last_request_time': 0.0,
            'error_count': 0,
            'last_seen': time.time()
        }
        
        console.print(f"[green]✅ Registered device '{device_name}' with {quota_percentage}% quota, {priority} priority[/green]")
        console.print(f"[blue]🆔 Device ID: {device_id}[/blue]")
        console.print(f"[blue]🔑 API Key: {api_key}[/blue]")
        console.print(f"[blue]🌐 Endpoint: {device_tenant.endpoint}[/blue]")
        console.print(f"[blue]💻 Device Type: {device_info.device_type} ({device_info.os_name})[/blue]")
        
        return device_id, api_key
    
    def list_devices(self):
        """Display all registered devices"""
        if not self.devices:
            console.print("[yellow]⚠️ No devices registered[/yellow]")
            return
        
        table = Table(show_header=True, box=None, title="Registered Devices")
        table.add_column("Device ID", style="cyan", width=20)
        table.add_column("Device Name", style="white", width=20)
        table.add_column("Type", style="green", width=10)
        table.add_column("OS", style="blue", width=15)
        table.add_column("Quota %", style="yellow", width=10)
        table.add_column("Priority", style="magenta", width=10)
        table.add_column("Status", style="white", width=10)
        table.add_column("Last Seen", style="white", width=15)
        
        for device in self.devices.values():
            device_type = device.device_info.device_type
            os_info = f"{device.device_info.os_name} {device.device_info.os_version[:10]}"
            quota_pct = f"{device.quota_percentage}%"
            status = "🟢 Online" if device.is_active else "🔴 Offline"
            
            # Calculate last seen
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
                device.device_id,
                device.device_name,
                device_type,
                os_info,
                quota_pct,
                device.priority,
                status,
                last_seen
            )
        
        console.print(table)
    
    def get_device(self, device_id: str) -> Optional[DeviceTenant]:
        """Get a device by device_id"""
        return self.devices.get(device_id)
    
    def get_device_by_api_key(self, api_key: str) -> Optional[DeviceTenant]:
        """Get a device by API key"""
        for device in self.devices.values():
            if device.api_key == api_key:
                return device
        return None
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        return secrets.token_urlsafe(32)
    
    def update_device_status(self, device_id: str, is_online: bool = True):
        """Update device online status (not active status)"""
        if device_id in self.devices:
            # Only update online status, not active status
            # is_active should only be changed by admin actions
            self.devices[device_id].last_accessed = time.time()
            if device_id in self.device_stats:
                self.device_stats[device_id]['last_seen'] = time.time()
                self.device_stats[device_id]['is_online'] = is_online
    
    def deactivate_device(self, device_id: str) -> bool:
        """Deactivate a device (admin only)"""
        if device_id in self.devices:
            self.devices[device_id].is_active = False
            self.save_devices()
            console.print(f"[yellow]⚠️ Device '{device_id}' deactivated by admin[/yellow]")
            return True
        else:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return False
    
    def activate_device(self, device_id: str) -> bool:
        """Activate a device (admin only)"""
        if device_id in self.devices:
            self.devices[device_id].is_active = True
            self.save_devices()
            console.print(f"[green]✅ Device '{device_id}' activated by admin[/green]")
            return True
        else:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return False
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a registered device"""
        if device_id not in self.devices:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return False
        
        del self.devices[device_id]
        if device_id in self.device_stats:
            del self.device_stats[device_id]
        
        console.print(f"[green]✅ Removed device '{device_id}'[/green]")
        return True
    
    def get_device_credentials(self, device_id: str) -> Optional[Dict[str, str]]:
        """Get device credentials for easy access"""
        device = self.get_device(device_id)
        if not device:
            return None
        
        return {
            "device_id": device.device_id,
            "device_name": device.device_name,
            "api_key": device.api_key,
            "endpoint": device.endpoint,
            "quota_percentage": device.quota_percentage,
            "priority": device.priority
        }
    
    def generate_client_script(self, device_id: str, script_path: str = None) -> str:
        """Generate a client script for easy device access"""
        device = self.get_device(device_id)
        if not device:
            console.print(f"[red]❌ Device '{device_id}' not found![/red]")
            return ""
        
        if not script_path:
            script_path = f"humigence_client_{device_id}.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Humigence Client for {device.device_name}
Device ID: {device_id}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

import requests
import json
import sys
import argparse
from typing import Optional

class HumigenceClient:
    def __init__(self, device_id: str = "{device_id}", api_key: str = "{device.api_key}"):
        self.device_id = device_id
        self.api_key = api_key
        self.base_url = "{device.endpoint}"
        self.headers = {{
            "Content-Type": "application/json",
            "Authorization": f"Bearer {{api_key}}",
            "X-Tenant-ID": device_id
        }}
    
    def chat(self, message: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Send a chat message to the Humigence server"""
        try:
            response = requests.post(
                f"{{self.base_url}}/v1/chat/completions",
                headers=self.headers,
                json={{
                    "model": "gpt-oss-20b-F16",
                    "messages": [{{"role": "user", "content": message}}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0].get('message', {{}}).get('content', '')
            return "No response received"
            
        except Exception as e:
            return f"Error: {{e}}"
    
    def status(self) -> dict:
        """Check server status"""
        try:
            response = requests.get(f"{{self.base_url}}/health", headers=self.headers, timeout=10)
            return {{"status": "online", "response": response.json()}}
        except Exception as e:
            return {{"status": "offline", "error": str(e)}}

def main():
    parser = argparse.ArgumentParser(description="Humigence Client for {device.device_name}")
    parser.add_argument("message", nargs="?", help="Message to send to the AI")
    parser.add_argument("--max-tokens", "-t", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--status", action="store_true", help="Check server status")
    
    args = parser.parse_args()
    
    client = HumigenceClient()
    
    if args.status:
        status = client.status()
        print(json.dumps(status, indent=2))
    elif args.message:
        response = client.chat(args.message, args.max_tokens, args.temperature)
        print(response)
    else:
        # Interactive mode
        print(f"🤖 Humigence Client for {device.device_name}")
        print(f"Device ID: {device_id}")
        print(f"Endpoint: {device.endpoint}")
        print("Type 'quit' to exit\\n")
        
        while True:
            try:
                message = input("You: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                if message:
                    response = client.chat(message, args.max_tokens, args.temperature)
                    print(f"AI: {{response}}\\n")
            except KeyboardInterrupt:
                break
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
'''
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make it executable
            os.chmod(script_path, 0o755)
            
            console.print(f"[green]✅ Generated client script: {script_path}[/green]")
            console.print(f"[blue]📋 Usage: python3 {script_path} 'Hello, how are you?'[/blue]")
            console.print(f"[blue]📋 Status: python3 {script_path} --status[/blue]")
            
            return script_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to generate client script: {e}[/red]")
            return ""

# CLI Commands using Typer
app = typer.Typer(help="Device Registration and Management")

# Global device manager instance
device_manager = None

@app.command()
def register(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom device name"),
    quota: float = typer.Option(25.0, "--quota", "-q", help="Quota percentage (0-100)"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Priority level (low/normal/high)")
):
    """Register current device and get API credentials"""
    global device_manager
    if not device_manager:
        device_manager = DeviceManager()
    
    if priority not in ["low", "normal", "high"]:
        console.print("[red]❌ Priority must be 'low', 'normal', or 'high'[/red]")
        raise typer.Exit(1)
    
    if not 0 <= quota <= 100:
        console.print("[red]❌ Quota must be between 0 and 100[/red]")
        raise typer.Exit(1)
    
    device_id, api_key = device_manager.register_device(name, quota, priority)
    device_manager.save_devices()
    
    # Generate client script
    script_path = device_manager.generate_client_script(device_id)
    
    console.print(f"\\n[bold green]🎉 Device Registration Complete![/bold green]")
    console.print(f"[blue]📱 Device Name: {name or 'Auto-detected'}[/blue]")
    console.print(f"[blue]🆔 Device ID: {device_id}[/blue]")
    console.print(f"[blue]🔑 API Key: {api_key}[/blue]")
    console.print(f"[blue]🌐 Endpoint: {device_manager.base_endpoint}/{device_id}[/blue]")
    console.print(f"[blue]📄 Client Script: {script_path}[/blue]")

@app.command()
def list():
    """List all registered devices"""
    global device_manager
    if not device_manager:
        device_manager = DeviceManager()
    
    device_manager.list_devices()

@app.command()
def remove(
    device_id: str = typer.Argument(..., help="Device ID to remove")
):
    """Remove a registered device"""
    global device_manager
    if not device_manager:
        device_manager = DeviceManager()
    
    success = device_manager.remove_device(device_id)
    if success:
        device_manager.save_devices()
    else:
        raise typer.Exit(1)

@app.command()
def credentials(
    device_id: str = typer.Argument(..., help="Device ID to get credentials for")
):
    """Get device credentials"""
    global device_manager
    if not device_manager:
        device_manager = DeviceManager()
    
    creds = device_manager.get_device_credentials(device_id)
    if creds:
        console.print(f"[green]✅ Device Credentials for '{device_id}':[/green]")
        for key, value in creds.items():
            console.print(f"[blue]{key}: {value}[/blue]")
    else:
        console.print(f"[red]❌ Device '{device_id}' not found![/red]")
        raise typer.Exit(1)

@app.command()
def script(
    device_id: str = typer.Argument(..., help="Device ID to generate script for"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output script path")
):
    """Generate client script for a device"""
    global device_manager
    if not device_manager:
        device_manager = DeviceManager()
    
    script_path = device_manager.generate_client_script(device_id, output)
    if script_path:
        console.print(f"[green]✅ Generated client script: {script_path}[/green]")

def main():
    """Main entry point for CLI"""
    app()

if __name__ == "__main__":
    main()
