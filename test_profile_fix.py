#!/usr/bin/env python3
"""
Test script to demonstrate the profile generation fix
"""

import sys
import subprocess
from pathlib import Path

def test_profile_generation():
    """Test that the profile generation creates a valid YAML file"""
    print("🧪 Testing Profile Generation Fix")
    print("=" * 50)
    
    # Test 1: Run auto-setup
    print("\n1️⃣ Running Auto-Setup...")
    try:
        result = subprocess.run([
            "python3", "-c", 
            """
import sys
sys.path.append('cli')
from cli.multi_tenant import MultiTenantWizard
wizard = MultiTenantWizard()
wizard.auto_setup()
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Auto-setup completed successfully")
        else:
            print(f"❌ Auto-setup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running auto-setup: {e}")
        return False
    
    # Test 2: Check if profile file exists
    print("\n2️⃣ Checking Profile File...")
    profile_path = Path("/home/joshua/humigence/inference/profiles/auto_detected.yaml")
    if profile_path.exists():
        print("✅ Profile file created")
    else:
        print("❌ Profile file not found")
        return False
    
    # Test 3: Validate profile structure
    print("\n3️⃣ Validating Profile Structure...")
    try:
        import yaml
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        required_fields = ['model', 'instances', 'context_length', 'router', 'tenants']
        missing_fields = [field for field in required_fields if field not in profile]
        
        if not missing_fields:
            print("✅ All required fields present")
        else:
            print(f"❌ Missing fields: {missing_fields}")
            return False
            
        # Check instances structure
        instances = profile.get('instances', [])
        if instances and all('gpu' in inst and 'port' in inst and 'parallel' in inst for inst in instances):
            print("✅ Instances structure valid")
        else:
            print("❌ Instances structure invalid")
            return False
            
    except Exception as e:
        print(f"❌ Error validating profile: {e}")
        return False
    
    # Test 4: Test supervisor validation
    print("\n4️⃣ Testing Supervisor Validation...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor
supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
print('Profile validation passed!')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Supervisor validation passed")
        else:
            print(f"❌ Supervisor validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing supervisor: {e}")
        return False
    
    # Test 5: Show profile contents
    print("\n5️⃣ Profile Contents:")
    print("-" * 30)
    try:
        with open(profile_path, 'r') as f:
            content = f.read()
        print(content)
    except Exception as e:
        print(f"❌ Error reading profile: {e}")
        return False
    
    print("\n🎉 All Tests Passed!")
    print("=" * 50)
    print("✅ Profile generation fix is working correctly")
    print("✅ No more 'Missing required field: model' errors")
    print("✅ Supervisor can now start services successfully")
    
    return True

if __name__ == "__main__":
    success = test_profile_generation()
    sys.exit(0 if success else 1)
