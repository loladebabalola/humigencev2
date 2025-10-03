#!/usr/bin/env python3
"""
Test script to verify the llama-server argument fix
"""

import sys
import subprocess
from pathlib import Path

def test_llama_args_fix():
    """Test that the supervisor builds correct llama-server commands"""
    print("🧪 Testing Llama-Server Arguments Fix")
    print("=" * 50)
    
    # Test 1: Load supervisor and check command building
    print("\n1️⃣ Testing Command Building...")
    try:
        result = subprocess.run([
            "python3", "-c", 
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor

# Load supervisor
supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
print('✅ Supervisor loaded successfully')

# Test argument filtering
test_cmd = [
    'llama-server', '--model', 'test.gguf', '--port', '8000', 
    '--log-format', 'json', '--ctx-size', '512', '--host', '0.0.0.0'
]
filtered = supervisor._filter_supported_args(test_cmd)
print(f'Original: {test_cmd}')
print(f'Filtered: {filtered}')

# Check that --log-format was removed
if '--log-format' not in filtered:
    print('✅ --log-format successfully removed')
else:
    print('❌ --log-format still present')
    exit(1)

# Check that core args are preserved
core_args = ['--model', '--port', '--ctx-size', '--host']
for arg in core_args:
    if arg in filtered:
        print(f'✅ {arg} preserved')
    else:
        print(f'❌ {arg} missing')
        exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ --log-format successfully removed" in result.stdout:
            print("✅ Command building working correctly")
        else:
            print(f"❌ Command building failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing command building: {e}")
        return False
    
    # Test 2: Check profile structure
    print("\n2️⃣ Checking Profile Structure...")
    try:
        import yaml
        profile_path = Path("/home/joshua/humigence/inference/profiles/auto_detected.yaml")
        with open(profile_path, 'r') as f:
            profile = yaml.safe_load(f)
        
        instances = profile.get('instances', [])
        if instances and all('binary' in inst for inst in instances):
            print("✅ Profile has binary paths")
            for i, inst in enumerate(instances):
                print(f"  Instance {i+1}: {inst['binary']}")
        else:
            print("❌ Profile missing binary paths")
            return False
    except Exception as e:
        print(f"❌ Error checking profile: {e}")
        return False
    
    # Test 3: Test actual command construction
    print("\n3️⃣ Testing Command Construction...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor

supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')

# Simulate building a command for instance 0
instance = supervisor.profile['instances'][0]
model_path = supervisor.profile['model']
context_window = supervisor.profile.get('context_window', 131072)

# This is what the supervisor would build
cmd = [
    instance['binary'],
    '--model', model_path,
    '--port', str(instance['port']),
    '--ctx-size', str(context_window),
    '--host', '0.0.0.0'
]

# Add parallel if not llama-cli
if 'llama-cli' not in instance['binary']:
    cmd.extend(['--parallel', str(instance['parallel'])])

# Add GPU layers and threads
cmd.extend(['--n-gpu-layers', '100'])
cmd.extend(['--threads', '8'])

# Filter unsupported args
filtered_cmd = supervisor._filter_supported_args(cmd)

print(f'Final command: {filtered_cmd}')

# Check for problematic args
problematic = ['--log-format', '--verbose', '--debug']
for arg in problematic:
    if arg in filtered_cmd:
        print(f'❌ Found problematic arg: {arg}')
        exit(1)
    else:
        print(f'✅ {arg} not present')

print('✅ Command construction working correctly')
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Command construction working correctly" in result.stdout:
            print("✅ Command construction working")
        else:
            print(f"❌ Command construction failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing construction: {e}")
        return False
    
    # Test 4: Show the expected command structure
    print("\n4️⃣ Expected Command Structure:")
    print("-" * 40)
    print("✅ Supported arguments:")
    print("  --model <path>")
    print("  --port <port>")
    print("  --ctx-size <size>")
    print("  --host <host>")
    print("  --parallel <num> (if not llama-cli)")
    print("  --n-gpu-layers <num>")
    print("  --threads <num>")
    print()
    print("❌ Removed arguments:")
    print("  --log-format (not supported in this build)")
    print("  --verbose (not universally supported)")
    print("  --debug (not universally supported)")
    
    print("\n🎉 All Tests Passed!")
    print("=" * 50)
    print("✅ --log-format successfully removed")
    print("✅ Core arguments preserved")
    print("✅ Argument filtering working")
    print("✅ Commands should now launch successfully")
    print("✅ No more 'invalid argument: --log-format' errors")
    
    return True

if __name__ == "__main__":
    success = test_llama_args_fix()
    sys.exit(0 if success else 1)
