#!/usr/bin/env python3
"""
Test script to verify robust profile generation
"""

import sys
import subprocess
from pathlib import Path

def test_profile_robustness():
    """Test that profile generation always creates valid profiles"""
    print("🧪 Testing Profile Generation Robustness")
    print("=" * 50)
    
    # Test 1: Normal auto-setup
    print("\n1️⃣ Testing Normal Auto-Setup...")
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
        
        if "✅ Profile created with all required fields" in result.stdout:
            print("✅ Normal auto-setup working")
        else:
            print(f"❌ Normal auto-setup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing auto-setup: {e}")
        return False
    
    # Test 2: Profile validation
    print("\n2️⃣ Testing Profile Validation...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor

supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
print('✅ Profile loaded successfully')

# Check all required fields
required_fields = ['model', 'instances', 'context_length', 'router', 'tenants']
for field in required_fields:
    if field in supervisor.profile:
        print(f'✅ {field}: present')
    else:
        print(f'❌ {field}: missing')
        exit(1)

# Check instances structure
instances = supervisor.profile['instances']
if instances and all('gpu' in inst and 'port' in inst and 'parallel' in inst for inst in instances):
    print('✅ Instances structure valid')
else:
    print('❌ Instances structure invalid')
    exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Profile loaded successfully" in result.stdout and "✅ Instances structure valid" in result.stdout:
            print("✅ Profile validation passed")
        else:
            print(f"❌ Profile validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        return False
    
    # Test 3: Test fallback scenarios
    print("\n3️⃣ Testing Fallback Scenarios...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('cli')
from cli.multi_tenant import generate_profile_yaml

# Test with minimal data
profile = generate_profile_yaml(
    model_path='',  # Empty model path
    context_length=0,  # Invalid context length
    gpu_ports=[],  # Empty GPU ports
    tenants=None  # No tenants
)

print('✅ Fallback profile generated')

# Check that fallbacks were applied
if profile.get('model') and 'placeholder' in profile['model']:
    print('✅ Model fallback applied')
else:
    print('❌ Model fallback not applied')
    exit(1)

if profile.get('context_length') == 131072:
    print('✅ Context length fallback applied')
else:
    print('❌ Context length fallback not applied')
    exit(1)

if profile.get('instances') and len(profile['instances']) > 0:
    print('✅ GPU ports fallback applied')
else:
    print('❌ GPU ports fallback not applied')
    exit(1)

if profile.get('tenants') and 'tenant_default' in profile['tenants']:
    print('✅ Tenant fallback applied')
else:
    print('❌ Tenant fallback not applied')
    exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Fallback profile generated" in result.stdout and "✅ Tenant fallback applied" in result.stdout:
            print("✅ Fallback scenarios working")
        else:
            print(f"❌ Fallback scenarios failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing fallbacks: {e}")
        return False
    
    # Test 4: Check profile contents
    print("\n4️⃣ Checking Generated Profile...")
    try:
        import yaml
        profile_path = Path("/home/joshua/humigence/inference/profiles/auto_detected.yaml")
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        print("Profile contents:")
        print(f"  Model: {profile.get('model', 'MISSING')}")
        print(f"  Context Length: {profile.get('context_length', 'MISSING')}")
        print(f"  Instances: {len(profile.get('instances', []))} instances")
        print(f"  Tenants: {len(profile.get('tenants', {}))} tenants")
        print(f"  Router: {profile.get('router', {}).get('strategy', 'MISSING')}")
        
        # Validate all required fields
        required = ['model', 'context_length', 'instances', 'router', 'tenants']
        missing = [field for field in required if field not in profile]
        
        if not missing:
            print("✅ All required fields present")
        else:
            print(f"❌ Missing fields: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking profile: {e}")
        return False
    
    # Test 5: Test supervisor startup (dry run)
    print("\n5️⃣ Testing Supervisor Startup...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor

supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
print('✅ Supervisor initialized successfully')

# Test validation
if supervisor.validate_profile():
    print('✅ Profile validation passed')
else:
    print('❌ Profile validation failed')
    exit(1)

print('✅ Supervisor ready to start instances')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Supervisor ready to start instances" in result.stdout:
            print("✅ Supervisor startup test passed")
        else:
            print(f"❌ Supervisor startup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing supervisor: {e}")
        return False
    
    print("\n🎉 All Tests Passed!")
    print("=" * 50)
    print("✅ Profile generation is robust and reliable")
    print("✅ All required fields are always present")
    print("✅ Fallback scenarios work correctly")
    print("✅ Supervisor validation passes")
    print("✅ No more 'Missing required field' errors")
    print("✅ Ready for production use")
    
    return True

if __name__ == "__main__":
    success = test_profile_robustness()
    sys.exit(0 if success else 1)
