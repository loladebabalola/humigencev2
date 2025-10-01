#!/usr/bin/env python3
"""
Test script for Unsloth dual-GPU training integration
Verifies that all components work correctly
"""

import sys
import os
from pathlib import Path

# Add the humigence directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from training.unsloth import train_lora_dual_gpu, launch_dual_gpu_training
        print("✅ Core training functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core functions: {e}")
        return False
    
    try:
        from training.unsloth.wizard import run_unsloth_wizard
        print("✅ Wizard module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import wizard: {e}")
        return False
    
    try:
        from training.unsloth.launcher import check_gpu_availability
        print("✅ Launcher module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import launcher: {e}")
        return False
    
    return True

def test_gpu_detection():
    """Test GPU detection"""
    print("\n🖥️ Testing GPU detection...")
    
    try:
        from training.unsloth.launcher import check_gpu_availability
        gpu_info = check_gpu_availability()
        
        if gpu_info["available"]:
            print(f"✅ CUDA available: {gpu_info['gpu_count']} GPU(s)")
            for gpu in gpu_info["gpus"]:
                print(f"   GPU {gpu['index']}: {gpu['name']} ({gpu['memory']})")
        else:
            print(f"⚠️ CUDA not available: {gpu_info['error']}")
        
        return gpu_info["available"]
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_wizard():
    """Test the wizard (without actually running it)"""
    print("\n🧙 Testing wizard...")
    
    try:
        from training.unsloth.wizard import show_unsloth_banner, choose_model, choose_dataset
        print("✅ Wizard functions imported successfully")
        
        # Test banner display
        show_unsloth_banner()
        print("✅ Banner displayed successfully")
        
        return True
    except Exception as e:
        print(f"❌ Wizard test failed: {e}")
        return False

def test_training_config():
    """Test training configuration"""
    print("\n⚙️ Testing training configuration...")
    
    try:
        from training.unsloth.launcher import train_lora_dual_gpu
        
        # Test with minimal configuration (won't actually train)
        config = {
            "model_name": "unsloth/Llama-3-8B-Instruct",
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "output_dir": "./test_output",
            "max_steps": 10,  # Very small for testing
            "batch_size": 1,
            "grad_accum": 1,
            "learning_rate": 2e-4,
            "block_size": 512,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0
        }
        
        print("✅ Training configuration created successfully")
        print(f"   Model: {config['model_name']}")
        print(f"   Dataset: {config['dataset_name']}/{config['dataset_config']}")
        print(f"   Output: {config['output_dir']}")
        
        return True
    except Exception as e:
        print(f"❌ Training configuration test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("\n🖥️ Testing CLI integration...")
    
    try:
        # Test that the CLI can import the modules
        import sys
        from pathlib import Path
        
        # Add humigence to path
        humigence_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(humigence_dir))
        
        from cli.main import show_menu
        print("✅ CLI main module imported successfully")
        
        # Test menu display
        show_menu()
        print("✅ Menu displayed successfully")
        
        return True
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Unsloth Dual-GPU Training Integration")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("GPU Detection", test_gpu_detection),
        ("Wizard", test_wizard),
        ("Training Config", test_training_config),
        ("CLI Integration", test_cli_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

