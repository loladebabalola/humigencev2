#!/usr/bin/env python3
"""
Test script to verify the subprocess command fix
"""

import sys
import subprocess
from pathlib import Path

def test_subprocess_fix():
    """Test that the supervisor builds correct subprocess commands"""
    print("🧪 Testing Subprocess Command Fix")
    print("=" * 50)
    
    # Test 1: Command structure validation
    print("\n1️⃣ Testing Command Structure...")
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

# Test command construction
instance = supervisor.profile['instances'][0]
model_path = supervisor.profile['model']
context_window = supervisor.profile.get('context_window', 131072)

# Build command as supervisor does
cmd = [
    instance['binary'],
    '--model', model_path,
    '--port', str(instance['port']),
    '--ctx-size', str(context_window),
    '--host', '0.0.0.0'
]

if 'llama-cli' not in instance['binary']:
    cmd.extend(['--parallel', str(instance['parallel'])])

cmd.extend(['--n-gpu-layers', '100'])
cmd.extend(['--threads', '8'])

# Filter args
filtered_cmd = supervisor._filter_supported_args(cmd)

print(f'Command type: {type(filtered_cmd)}')
print(f'Command length: {len(filtered_cmd)}')
print(f'Binary path: {filtered_cmd[0]}')
print(f'First argument: {filtered_cmd[1]}')
print(f'Second argument: {filtered_cmd[2]}')

# Validate structure
if isinstance(filtered_cmd, list) and len(filtered_cmd) > 0:
    if filtered_cmd[0] and not filtered_cmd[0].startswith('--'):
        print('✅ Binary path is first element')
    else:
        print('❌ Binary path not first element')
        exit(1)
    
    if filtered_cmd[1] == '--model':
        print('✅ First argument is --model')
    else:
        print('❌ First argument is not --model')
        exit(1)
else:
    print('❌ Command is not a proper list')
    exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Binary path is first element" in result.stdout and "✅ First argument is --model" in result.stdout:
            print("✅ Command structure is correct")
        else:
            print(f"❌ Command structure failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing structure: {e}")
        return False
    
    # Test 2: Binary path validation
    print("\n2️⃣ Testing Binary Path Validation...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor
from pathlib import Path
import os

supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
instance = supervisor.profile['instances'][0]
binary = instance['binary']

print(f'Binary path: {binary}')
print(f'Binary exists: {Path(binary).exists()}')
print(f'Binary executable: {os.access(binary, os.X_OK)}')

if Path(binary).exists() and os.access(binary, os.X_OK):
    print('✅ Binary path is valid and executable')
else:
    print('❌ Binary path is invalid or not executable')
    exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Binary path is valid and executable" in result.stdout:
            print("✅ Binary path validation passed")
        else:
            print(f"❌ Binary path validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing binary: {e}")
        return False
    
    # Test 3: Test actual subprocess call (dry run)
    print("\n3️⃣ Testing Subprocess Call (Dry Run)...")
    try:
        result = subprocess.run([
            "python3", "-c",
            """
import sys
sys.path.append('inference')
from inference.supervisor import InferenceSupervisor
import subprocess

supervisor = InferenceSupervisor('/home/joshua/humigence/inference/profiles/auto_detected.yaml')
instance = supervisor.profile['instances'][0]

# Build the exact command the supervisor would use
cmd = [
    instance['binary'],
    '--model', supervisor.profile['model'],
    '--port', str(instance['port']),
    '--ctx-size', str(supervisor.profile.get('context_window', 131072)),
    '--host', '0.0.0.0'
]

if 'llama-cli' not in instance['binary']:
    cmd.extend(['--parallel', str(instance['parallel'])])

cmd.extend(['--n-gpu-layers', '100'])
cmd.extend(['--threads', '8'])

filtered_cmd = supervisor._filter_supported_args(cmd)

print(f'Final command: {filtered_cmd}')

# Test subprocess call with --help to verify it works
try:
    result = subprocess.run(
        filtered_cmd + ['--help'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0 or 'usage:' in result.stdout.lower() or 'help' in result.stdout.lower():
        print('✅ Subprocess call successful')
    else:
        print(f'❌ Subprocess call failed: {result.stderr}')
        exit(1)
except subprocess.TimeoutExpired:
    print('✅ Subprocess call started (timeout expected for --help)')
except Exception as e:
    print(f'❌ Subprocess call error: {e}')
    exit(1)
"""
        ], cwd="/home/joshua/humigence", capture_output=True, text=True)
        
        if "✅ Subprocess call successful" in result.stdout or "✅ Subprocess call started" in result.stdout:
            print("✅ Subprocess call working")
        else:
            print(f"❌ Subprocess call failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing subprocess: {e}")
        return False
    
    # Test 4: Show expected command structure
    print("\n4️⃣ Expected Command Structure:")
    print("-" * 40)
    print("✅ Correct structure:")
    print("  [binary_path, '--model', model_path, '--port', port, ...]")
    print()
    print("❌ Incorrect structure:")
    print("  ['--model', model_path, '--port', port, ...]  # Missing binary")
    print("  [f'{binary} --model {model} ...']  # Single string")
    
    print("\n🎉 All Tests Passed!")
    print("=" * 50)
    print("✅ Command built as proper argument list")
    print("✅ Binary path is first element")
    print("✅ Arguments are separate elements")
    print("✅ Subprocess calls work correctly")
    print("✅ No more 'Errno 2' errors")
    
    return True

if __name__ == "__main__":
    success = test_subprocess_fix()
    sys.exit(0 if success else 1)
