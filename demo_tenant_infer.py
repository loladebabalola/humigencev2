#!/usr/bin/env python3
"""
Demo script showing the new tenant infer command functionality
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display the result"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🚀 Multi-Tenant Inference - Tenant CLI Demo")
    print("=" * 60)
    
    # Show available commands
    print("\n📋 Available Commands:")
    run_command("python3 tenant --help", "Show all available commands")
    
    # Show infer command help
    run_command("python3 tenant infer --help", "Show infer command help")
    
    # List current tenants
    run_command("python3 tenant list", "List current tenants")
    
    # Show tenant status
    run_command("python3 tenant status", "Show tenant statistics")
    
    print("\n🎯 Tenant Infer Command Examples:")
    print("=" * 60)
    print("The 'tenant infer' command allows Shellfish users to run inference directly from CLI:")
    print()
    print("1. Basic inference (default tenant):")
    print("   tenant infer tenant_default 'Hello, how are you?'")
    print()
    print("2. Inference with custom parameters:")
    print("   tenant infer alice 'Write a poem' --max-tokens 200 --temperature 0.9")
    print()
    print("3. Inference with specific tenant:")
    print("   tenant infer bob 'Explain quantum computing' --max-tokens 150")
    print()
    print("4. High-temperature creative writing:")
    print("   tenant infer charlie 'Write a story about AI' --temperature 1.2")
    
    print("\n🔑 Key Features:")
    print("=" * 60)
    print("✅ Automatic tenant authentication (API keys)")
    print("✅ Quota validation and tracking")
    print("✅ Request rate limiting")
    print("✅ Error handling and reporting")
    print("✅ Works from any SSH client (Shellfish, etc.)")
    print("✅ JSON response formatting")
    print("✅ Request statistics tracking")
    
    print("\n📱 Shellfish Integration:")
    print("=" * 60)
    print("Users can SSH into the VM and run:")
    print("  ssh user@your-vm")
    print("  cd /home/joshua/humigence")
    print("  ./tenant infer tenant_default 'Hello from my iPad!'")
    print()
    print("Or use the Python script directly:")
    print("  python3 tenant infer alice 'Hello from Alice!'")
    
    print("\n🎉 Implementation Complete!")
    print("=" * 60)
    print("The tenant infer command is now available and ready for Shellfish users!")

if __name__ == "__main__":
    main()
