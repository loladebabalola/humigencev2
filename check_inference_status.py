#!/usr/bin/env python3
"""
Quick status check for Multi-Tenant Inference System
"""

import requests
import json
import subprocess
import sys

def check_router():
    """Check if router is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Router: Running on port 8000")
            print(f"   Status: {data['status']}")
            print(f"   Instances: {data['instances']}")
            print(f"   Healthy: {data['healthy_instances']}")
            print(f"   Strategy: {data['strategy']}")
            return True
        else:
            print(f"❌ Router: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Router: Not running ({e})")
        return False

def check_llama_server():
    """Check if llama-server is installed"""
    try:
        result = subprocess.run(["llama-server", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ llama-server: Installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ llama-server: Not working")
            return False
    except FileNotFoundError:
        print(f"❌ llama-server: Not installed")
        return False
    except Exception as e:
        print(f"❌ llama-server: Error ({e})")
        return False

def check_gpus():
    """Check GPU availability"""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", 
                               "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"✅ GPUs: {len(lines)} detected")
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 3:
                    print(f"   GPU {parts[0]}: {parts[1]} ({parts[2]}MB)")
            return True
        else:
            print(f"❌ GPUs: nvidia-smi failed")
            return False
    except Exception as e:
        print(f"❌ GPUs: Error ({e})")
        return False

def check_model():
    """Check if model file exists"""
    import os
    model_path = "/models/gpt-oss-20b/gpt-oss-20b.Q4_K_M.gguf"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"✅ Model: Found ({size:.1f}GB)")
        return True
    else:
        print(f"❌ Model: Not found at {model_path}")
        return False

def main():
    print("🔍 Multi-Tenant Inference Status Check")
    print("=" * 40)
    
    checks = [
        ("Router", check_router),
        ("llama-server", check_llama_server),
        ("GPUs", check_gpus),
        ("Model", check_model)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n{name}:")
        if check_func():
            passed += 1
    
    print(f"\n📊 Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 System is fully operational!")
        return True
    else:
        print("⚠️ Some components need attention.")
        print("\nNext steps:")
        if not check_llama_server():
            print("- Install llama-server: ./setup_llama_server.sh")
        if not check_model():
            print("- Add your GGUF model to /models/gpt-oss-20b/")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
