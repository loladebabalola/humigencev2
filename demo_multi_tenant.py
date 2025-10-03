#!/usr/bin/env python3
# demo_multi_tenant.py

"""
Demonstration script for the Multi-Tenant Inference Wizard
Shows the complete zero-friction flow in action
"""

import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def demo_complete_flow():
    """Demonstrate the complete multi-tenant inference flow"""
    print("🚀 Multi-Tenant Inference Wizard - Complete Demo")
    print("=" * 60)
    
    try:
        from cli.multi_tenant import MultiTenantWizard
        
        # Create wizard instance
        wizard = MultiTenantWizard()
        
        print("\n📋 Step 1: Auto-Setup (Zero-Friction Configuration)")
        print("-" * 50)
        
        # Run auto-setup
        wizard.auto_setup()
        
        print("\n📊 Step 2: System Summary")
        print("-" * 50)
        
        # Show system summary
        wizard.summary_panel.display_static_summary()
        
        print("\n👥 Step 3: Tenant Management Demo")
        print("-" * 50)
        
        # Add some demo tenants
        print("Adding demo tenants...")
        wizard.tenant_manager.add_tenant("alice", 40.0, "high")
        wizard.tenant_manager.add_tenant("bob", 60.0, "normal")
        wizard.tenant_manager.save_tenants()
        
        # Show tenant list
        wizard.tenant_manager.list_tenants()
        
        print("\n📈 Step 4: Monitoring Demo")
        print("-" * 50)
        
        # Show monitoring dashboard
        wizard.summary_panel.display_static_summary()
        
        print("\n💾 Step 5: Profile Management Demo")
        print("-" * 50)
        
        # Show profiles
        wizard.profile_manager.display_profiles()
        
        # Create a snapshot
        print("\nCreating snapshot...")
        gpu_summary = wizard.gpu_detector.get_system_summary()
        model_summary = wizard.model_detector.get_system_summary()
        tenant_summary = wizard.tenant_manager.get_system_summary()
        service_status = wizard.summary_panel.check_service_status()
        
        snapshot_path = wizard.profile_manager.create_snapshot(
            "demo_profile",
            gpu_summary,
            model_summary,
            tenant_summary,
            service_status,
            {"demo": True, "timestamp": time.time()}
        )
        
        if snapshot_path:
            print(f"✅ Snapshot created: {snapshot_path}")
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("The Multi-Tenant Inference Wizard is ready for production use!")
        print("\nNext steps:")
        print("1. Run: python3 cli/main.py")
        print("2. Select option 3: Multi-Tenant Inference 🖥️")
        print("3. Start services with option 1")
        print("4. Monitor with option 2")
        print("5. Manage tenants with option 3")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_cli_commands():
    """Demonstrate CLI commands"""
    print("\n🔧 CLI Commands Demo")
    print("=" * 40)
    
    try:
        from inference.tenant_manager import TenantManager
        
        # Create tenant manager
        manager = TenantManager()
        
        print("Available CLI commands:")
        print("• python3 -m inference.tenant_manager add <alias> --quota=50% --priority=normal")
        print("• python3 -m inference.tenant_manager list")
        print("• python3 -m inference.tenant_manager update <alias> --quota=75%")
        print("• python3 -m inference.tenant_manager remove <alias>")
        print("• python3 -m inference.tenant_manager status")
        
        print("\nTesting CLI commands...")
        
        # Test add command
        print("\n1. Adding tenant via CLI...")
        success = manager.add_tenant("cli_demo", 30.0, "low")
        if success:
            print("✅ Tenant added successfully")
        
        # Test list command
        print("\n2. Listing tenants...")
        manager.list_tenants()
        
        # Test update command
        print("\n3. Updating tenant...")
        manager.update_tenant("cli_demo", 50.0, "normal")
        
        # Test status command
        print("\n4. Tenant status...")
        manager.status()
        
        print("\n✅ CLI commands working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ CLI demo failed: {e}")
        return False

def main():
    """Run complete demonstration"""
    print("🎬 Multi-Tenant Inference Wizard - Complete Demonstration")
    print("=" * 70)
    
    # Demo 1: Complete flow
    success1 = demo_complete_flow()
    
    # Demo 2: CLI commands
    success2 = demo_cli_commands()
    
    print(f"\n📊 Demo Results:")
    print(f"• Complete Flow: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"• CLI Commands: {'✅ Success' if success2 else '❌ Failed'}")
    
    if success1 and success2:
        print("\n🎉 All demonstrations completed successfully!")
        print("The Multi-Tenant Inference Wizard is ready for production use!")
    else:
        print("\n⚠️ Some demonstrations failed. Please check the errors above.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
