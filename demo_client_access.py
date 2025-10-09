#!/usr/bin/env python3
"""
Humigence Client Access Demo
============================

This script demonstrates the complete client access workflow:
1. Register a device
2. Generate a client script
3. Test the connection
4. Show usage examples

Run this to see the system in action!
"""

import os
import sys
import time
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference.device_manager import DeviceManager
from rich.console import Console
from rich.panel import Panel

console = Console()

def demo_device_registration():
    """Demonstrate device registration"""
    console.print("\n[bold cyan]📱 Device Registration Demo[/bold cyan]")
    console.print("=" * 50)
    
    # Create device manager
    dm = DeviceManager()
    
    # Register a demo device
    console.print("[blue]Registering demo device...[/blue]")
    device_id, api_key = dm.register_device(
        device_name="Demo-Device",
        quota_percentage=25.0,
        priority="normal"
    )
    
    console.print(f"[green]✅ Device registered successfully![/green]")
    console.print(f"[blue]Device ID: {device_id}[/blue]")
    console.print(f"[blue]API Key: {api_key}[/blue]")
    
    # Save the device
    dm.save_devices()
    
    return device_id, api_key

def demo_client_script_generation(device_id):
    """Demonstrate client script generation"""
    console.print("\n[bold cyan]📄 Client Script Generation Demo[/bold cyan]")
    console.print("=" * 50)
    
    dm = DeviceManager()
    
    # Generate client script
    script_path = dm.generate_client_script(device_id, f"demo_client_{device_id}.py")
    
    if script_path:
        console.print(f"[green]✅ Client script generated: {script_path}[/green]")
        
        # Show script content preview
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Show first few lines
        lines = content.split('\n')[:20]
        preview = '\n'.join(lines) + '\n...'
        
        console.print(Panel(preview, title="Client Script Preview", border_style="green"))
        
        return script_path
    
    return None

def demo_usage_examples(device_id, api_key):
    """Show usage examples"""
    console.print("\n[bold cyan]💡 Usage Examples[/bold cyan]")
    console.print("=" * 50)
    
    examples = f"""
[bold]1. Direct API Call (curl):[/bold]
curl -X POST http://localhost:8000/{device_id}/v1/chat/completions \\
  -H "Authorization: Bearer {api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "gpt-oss-20b-F16", "messages": [{{"role": "user", "content": "Hello!"}}]}}'

[bold]2. Using Python Client:[/bold]
python3 demo_client_{device_id}.py "Hello, how are you?"

[bold]3. Interactive Mode:[/bold]
python3 demo_client_{device_id}.py

[bold]4. Check Status:[/bold]
python3 demo_client_{device_id}.py --status

[bold]5. Test Connection:[/bold]
python3 demo_client_{device_id}.py --test
    """
    
    console.print(Panel(examples, title="Usage Examples", border_style="blue"))

def demo_device_management():
    """Demonstrate device management features"""
    console.print("\n[bold cyan]🔧 Device Management Demo[/bold cyan]")
    console.print("=" * 50)
    
    dm = DeviceManager()
    
    # List devices
    console.print("[blue]Listing all registered devices...[/blue]")
    dm.list_devices()
    
    # Show device credentials
    if dm.devices:
        device_id = list(dm.devices.keys())[0]
        creds = dm.get_device_credentials(device_id)
        
        console.print(f"\n[blue]Device credentials for '{device_id}':[/blue]")
        for key, value in creds.items():
            console.print(f"  {key}: {value}")

def main():
    """Main demo function"""
    console.print(Panel(
        "[bold green]Humigence Client Access System Demo[/bold green]\n"
        "This demo shows how to register devices and access the AI server.\n"
        "Make sure the Humigence server is running first!",
        title="Demo Introduction",
        border_style="green"
    ))
    
    try:
        # Step 1: Register a device
        device_id, api_key = demo_device_registration()
        
        # Step 2: Generate client script
        script_path = demo_client_script_generation(device_id)
        
        # Step 3: Show usage examples
        demo_usage_examples(device_id, api_key)
        
        # Step 4: Show device management
        demo_device_management()
        
        # Final summary
        console.print("\n[bold green]🎉 Demo Complete![/bold green]")
        console.print("=" * 50)
        console.print(f"[blue]Device ID: {device_id}[/blue]")
        console.print(f"[blue]API Key: {api_key}[/blue]")
        console.print(f"[blue]Client Script: {script_path}[/blue]")
        console.print(f"[blue]Endpoint: http://localhost:8000/{device_id}[/blue]")
        
        console.print("\n[bold yellow]Next Steps:[/bold yellow]")
        console.print("1. Start the Humigence server: humigence")
        console.print("2. Select option 3 (Multi-Tenant Inference)")
        console.print("3. Use the generated client script to test")
        console.print("4. Register more devices as needed")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        console.print("[yellow]Make sure you're in the humigencev2 directory[/yellow]")

if __name__ == "__main__":
    main()

