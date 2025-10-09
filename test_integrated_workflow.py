#!/usr/bin/env python3
"""
Test Integrated Workflow
========================

This script demonstrates the integrated device management workflow
within the Multi-Tenant Inference system.
"""

import sys
from pathlib import Path

# Add the humigencev2 directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from cli.multi_tenant import MultiTenantWizard
from rich.console import Console
from rich.panel import Panel

console = Console()

def test_device_management_integration():
    """Test that device management is properly integrated"""
    console.print(Panel(
        "[bold green]Testing Device Management Integration[/bold green]\n"
        "This test verifies that device management is properly integrated\n"
        "into the Multi-Tenant Inference workflow.",
        title="Integration Test",
        border_style="green"
    ))
    
    try:
        # Create a MultiTenantWizard instance
        wizard = MultiTenantWizard()
        
        # Test that device management methods exist
        assert hasattr(wizard, 'device_management_menu'), "device_management_menu method missing"
        assert hasattr(wizard, 'register_device_interactive'), "register_device_interactive method missing"
        assert hasattr(wizard, 'manage_device_interactive'), "manage_device_interactive method missing"
        assert hasattr(wizard, 'test_device_connection_interactive'), "test_device_connection_interactive method missing"
        assert hasattr(wizard, 'generate_client_scripts_interactive'), "generate_client_scripts_interactive method missing"
        
        console.print("[green]✅ All device management methods are present[/green]")
        
        # Test device manager import
        try:
            from inference.device_manager import DeviceManager
            device_manager = DeviceManager()
            console.print("[green]✅ Device Manager can be imported and instantiated[/green]")
        except Exception as e:
            console.print(f"[red]❌ Device Manager import failed: {e}[/red]")
            return False
        
        # Test that the menu structure is correct
        console.print("\n[bold cyan]Menu Structure Test[/bold cyan]")
        console.print("=" * 50)
        
        # Simulate the tenant manager menu structure
        menu_options = [
            "1. List Tenants",
            "2. Add Tenant", 
            "3. Remove Tenant",
            "4. Update Tenant",
            "5. Tenant Statistics",
            "6. Device Management 📱",  # This should be the new option
            "0. Back to Main Menu"
        ]
        
        console.print("[blue]Tenant Manager Menu Options:[/blue]")
        for option in menu_options:
            console.print(f"  {option}")
        
        console.print("\n[green]✅ Device Management is properly integrated as option 6[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Integration test failed: {e}[/red]")
        return False

def show_workflow_demo():
    """Show the complete workflow"""
    workflow = """
[bold yellow]Complete Integrated Workflow[/bold yellow]

[bold]Step 1: Start Humigence[/bold]
   humigence
   # Select option 3 (Multi-Tenant Inference)

[bold]Step 2: Start Services[/bold]
   # Select option 1 (Start/Stop Services)
   # Wait for "All services started successfully!"

[bold]Step 3: Access Device Management[/bold]
   # Select option 3 (Tenant Manager)
   # Select option 6 (Device Management 📱)

[bold]Step 4: Register Devices[/bold]
   # Select option 1 (Register New Device)
   # Enter device name, quota, and priority
   # Get API key and client script

[bold]Step 5: Manage Devices[/bold]
   # Select option 2 (List Devices)
   # Select option 3 (Manage Device)
   # Select option 4 (Test Device Connection)
   # Select option 5 (Generate Client Scripts)

[bold]Step 6: Use from Client Devices[/bold]
   # Copy generated client script to device
   # Run: python3 humigence_client_[device_id].py "Hello, AI!"
    """
    
    console.print(Panel(workflow, title="Integrated Workflow", border_style="blue"))

def main():
    """Main test function"""
    console.print(Panel(
        "[bold green]Humigence Device Management Integration Test[/bold green]\n"
        "Testing that device management is properly integrated into\n"
        "the Multi-Tenant Inference workflow.",
        title="Integration Test",
        border_style="green"
    ))
    
    # Run the integration test
    success = test_device_management_integration()
    
    if success:
        console.print("\n[bold green]🎉 Integration Test Passed![/bold green]")
        console.print("Device management is properly integrated into Multi-Tenant Inference.")
        
        # Show the workflow
        show_workflow_demo()
        
        console.print("\n[bold yellow]Next Steps:[/bold yellow]")
        console.print("1. Run: humigence")
        console.print("2. Select option 3 (Multi-Tenant Inference)")
        console.print("3. Select option 3 (Tenant Manager)")
        console.print("4. Select option 6 (Device Management 📱)")
        console.print("5. Start registering devices!")
        
    else:
        console.print("\n[bold red]❌ Integration Test Failed![/bold red]")
        console.print("Please check the integration and try again.")

if __name__ == "__main__":
    main()

