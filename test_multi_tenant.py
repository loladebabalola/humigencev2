#!/usr/bin/env python3
# test_multi_tenant.py

"""
Test script for the new Multi-Tenant Inference Wizard
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_detection():
    """Test GPU detection"""
    print("🔍 Testing GPU Detection...")
    try:
        from inference.gpu_detector import GPUDetector
        detector = GPUDetector()
        count, gpus = detector.detect_gpus()
        print(f"✅ Detected {count} GPU(s)")
        if count > 0:
            detector.create_port_mapping()
            detector.display_gpu_summary()
        return True
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_model_detection():
    """Test model detection"""
    print("\n🤖 Testing Model Detection...")
    try:
        from inference.model_detector import ModelDetector
        detector = ModelDetector()
        count, models = detector.scan_models()
        print(f"✅ Found {count} model(s)")
        if count > 0:
            detector.display_models()
        return True
    except Exception as e:
        print(f"❌ Model detection failed: {e}")
        return False

def test_tenant_manager():
    """Test tenant manager"""
    print("\n👥 Testing Tenant Manager...")
    try:
        from inference.tenant_manager import TenantManager
        manager = TenantManager()
        
        # Test adding a tenant
        success = manager.add_tenant("test_tenant", 50.0, "normal")
        if success:
            print("✅ Tenant added successfully")
            manager.list_tenants()
            manager.save_tenants()
            return True
        else:
            print("❌ Failed to add tenant")
            return False
    except Exception as e:
        print(f"❌ Tenant manager failed: {e}")
        return False

def test_profile_manager():
    """Test profile manager"""
    print("\n📁 Testing Profile Manager...")
    try:
        from inference.profile_manager import ProfileManager
        manager = ProfileManager()
        
        # Test profile creation
        gpu_config = {"gpu_count": 1, "gpus": [{"index": 0, "name": "Test GPU"}]}
        model_config = {"model_count": 1, "default_model": {"name": "test-model"}}
        tenant_config = {"tenant_count": 1, "tenants": {"default": {"name": "Default"}}}
        
        success = manager.create_profile("test_profile", gpu_config, model_config, tenant_config)
        if success:
            print("✅ Profile created successfully")
            manager.display_profiles()
            return True
        else:
            print("❌ Failed to create profile")
            return False
    except Exception as e:
        print(f"❌ Profile manager failed: {e}")
        return False

def test_summary_panel():
    """Test summary panel"""
    print("\n📊 Testing Summary Panel...")
    try:
        from inference.summary_panel import SummaryPanel
        from inference.gpu_detector import GPUDetector
        from inference.model_detector import ModelDetector
        from inference.tenant_manager import TenantManager
        
        # Initialize components
        gpu_detector = GPUDetector()
        model_detector = ModelDetector()
        tenant_manager = TenantManager()
        
        # Create summary panel
        panel = SummaryPanel(gpu_detector, model_detector, tenant_manager)
        
        # Test static summary
        panel.display_static_summary()
        print("✅ Summary panel working")
        return True
    except Exception as e:
        print(f"❌ Summary panel failed: {e}")
        return False

def test_monitoring():
    """Test monitoring dashboard"""
    print("\n📈 Testing Monitoring Dashboard...")
    try:
        from inference.monitor import MonitoringDashboard
        from inference.gpu_detector import GPUDetector
        from inference.tenant_manager import TenantManager
        
        # Initialize components
        gpu_detector = GPUDetector()
        tenant_manager = TenantManager()
        
        # Create monitoring dashboard
        monitor = MonitoringDashboard(gpu_detector, tenant_manager)
        
        # Test static monitoring
        monitor.display_static_monitoring()
        print("✅ Monitoring dashboard working")
        return True
    except Exception as e:
        print(f"❌ Monitoring dashboard failed: {e}")
        return False

def test_multi_tenant_wizard():
    """Test the main multi-tenant wizard"""
    print("\n🚀 Testing Multi-Tenant Wizard...")
    try:
        from cli.multi_tenant import MultiTenantWizard
        wizard = MultiTenantWizard()
        
        # Test auto-setup (without running the full wizard)
        print("Testing auto-setup components...")
        wizard.auto_setup()
        print("✅ Multi-tenant wizard components working")
        return True
    except Exception as e:
        print(f"❌ Multi-tenant wizard failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Multi-Tenant Inference System")
    print("=" * 50)
    
    tests = [
        test_gpu_detection,
        test_model_detection,
        test_tenant_manager,
        test_profile_manager,
        test_summary_panel,
        test_monitoring,
        test_multi_tenant_wizard
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Multi-Tenant Inference system is ready.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
