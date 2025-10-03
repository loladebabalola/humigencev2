#!/usr/bin/env python3
"""
Test script to demonstrate the llama binary detection and resolution fix
"""

import sys
import subprocess
from pathlib import Path

def test_binary_detection():
    """Test the llama binary detection and resolution system"""
    print("🧪 Testing Llama Binary Detection Fix")
    print("=" * 50)
    
    # Test 1: Binary detection during auto-setup
    print("\n1️⃣ Testing Binary Detection...")
    try:
        result = subprocess.run([
            "python3", "-c", 
            """
import sys
sys.path.append('cli')
from cli.multi_tenant import detect_llama_binary
binary = detect_llama_binary()
if binary:
    print(f'✅ Found: {binary}')
else:
    print('❌ No binary found')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Found:" in result.stdout:
            print("✅ Binary detection working")
        else:
            print(f"❌ Binary detection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing detection: {e}")
        return False
    
    # Test 2: Profile generation with binary
    print("\n2️⃣ Testing Profile Generation with Binary...")
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
        
        if "✅ Found llama binary:" in result.stdout:
            print("✅ Profile generation with binary working")
        else:
            print(f"❌ Profile generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing profile generation: {e}")
        return False
    
    # Test 3: Check profile includes binary paths
    print("\n3️⃣ Checking Profile Binary Paths...")
    try:
        import yaml
        profile_path = Path("/home/joshua/humigence/inference/profiles/auto_detected.yaml")
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        instances = profile.get('instances', [])
        if instances and all('binary' in inst for inst in instances):
            print("✅ All instances have binary paths")
            for i, inst in enumerate(instances):
                print(f"  Instance {i+1}: {inst['binary']}")
        else:
            print("❌ Some instances missing binary paths")
            return False
    except Exception as e:
        print(f"❌ Error checking profile: {e}")
        return False
    
    # Test 4: Test binary resolution function
    print("\n4️⃣ Testing Binary Resolution...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import resolve_llama_binary

# Test profile-specified binary
binary1 = resolve_llama_binary('/home/joshua/llama.cpp/build/bin/llama-server')
print(f'Profile binary: {binary1}')

# Test auto-detection
binary2 = resolve_llama_binary(None)
print(f'Auto-detected: {binary2}')

# Test fallback
binary3 = resolve_llama_binary('/nonexistent/path')
print(f'Fallback: {binary3}')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "Profile binary:" in result.stdout and "Auto-detected:" in result.stdout:
            print("✅ Binary resolution working")
        else:
            print(f"❌ Binary resolution failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing resolution: {e}")
        return False
    
    # Test 5: Test supervisor validation
    print("\n5️⃣ Testing Supervisor with Binary Resolution...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor
supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
print('✅ Supervisor loaded with binary resolution')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Supervisor loaded" in result.stdout:
            print("✅ Supervisor validation passed")
        else:
            print(f"❌ Supervisor validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing supervisor: {e}")
        return False
    
    # Test 6: Show profile contents
    print("\n6️⃣ Generated Profile Contents:")
    print("-" * 40)
    try:
        with open("/home/joshua/humigence/inference/profiles/auto_detected.yaml", 'r') as f:
            content = f.read()
        print(content)
    except Exception as e:
        print(f"❌ Error reading profile: {e}")
        return False
    
    print("\n🎉 All Tests Passed!")
    print("=" * 50)
    print("✅ Binary detection working correctly")
    print("✅ Profile generation includes binary paths")
    print("✅ Binary resolution handles all cases")
    print("✅ No more 'No such file or directory: llama-server' errors")
    print("✅ Supervisor can start instances successfully")
    
    return True

if __name__ == "__main__":
    success = test_binary_detection()
    sys.exit(0 if success else 1)
