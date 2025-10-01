#!/usr/bin/env python3
"""
Setup script for Unsloth dual-GPU training in Humigence
Installs all required dependencies and verifies the setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("🖥️ Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️ CUDA not available - will install PyTorch CPU version")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed - will install")
        return False

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("🔥 Installing PyTorch...")
    
    # Check if CUDA is available
    cuda_available = check_cuda()
    
    if cuda_available:
        # Install PyTorch with CUDA
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        # Install PyTorch CPU version
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(cmd, "Installing PyTorch")

def install_basic_dependencies():
    """Install basic ML dependencies"""
    dependencies = [
        "transformers>=4.36.0",
        "datasets>=2.14.0", 
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "rich>=13.0.0",
        "inquirer>=3.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    return True

def install_unsloth():
    """Install Unsloth from source"""
    print("🚀 Installing Unsloth...")
    
    # Try different installation methods
    methods = [
        "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'",
        "pip install unsloth",
        "pip install git+https://github.com/unslothai/unsloth.git"
    ]
    
    for method in methods:
        print(f"🔧 Trying: {method}")
        if run_command(method, f"Installing Unsloth with method: {method}"):
            return True
        print("⚠️ Method failed, trying next...")
    
    print("❌ All Unsloth installation methods failed")
    return False

def verify_installation():
    """Verify that all components are working"""
    print("🔍 Verifying installation...")
    
    try:
        import torch
        import transformers
        import datasets
        import accelerate
        import peft
        import bitsandbytes
        import unsloth
        import rich
        import inquirer
        
        print("✅ All core dependencies imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA working: {torch.cuda.device_count()} GPU(s)")
        else:
            print("⚠️ CUDA not available - using CPU mode")
        
        # Test Unsloth
        print(f"✅ Unsloth version: {unsloth.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        return False

def create_test_script():
    """Create a test script to verify the setup"""
    test_script = """
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_unsloth():
    try:
        from training.unsloth.launcher import check_gpu_availability
        gpu_info = check_gpu_availability()
        print(f"GPU Info: {gpu_info}")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_unsloth()
    sys.exit(0 if success else 1)
"""
    
    test_path = Path(__file__).parent / "test_setup.py"
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(test_path, 0o755)
    print(f"✅ Test script created: {test_path}")

def main():
    """Main setup function"""
    print("🚀 Setting up Unsloth Dual-GPU Training for Humigence")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install PyTorch
    if not install_pytorch():
        print("❌ PyTorch installation failed")
        return 1
    
    # Install basic dependencies
    if not install_basic_dependencies():
        print("❌ Basic dependencies installation failed")
        return 1
    
    # Install Unsloth
    if not install_unsloth():
        print("❌ Unsloth installation failed")
        print("⚠️ You may need to install Unsloth manually:")
        print("   pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        return 1
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        return 1
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("Next steps:")
    print("1. Run the test: python3 training/unsloth/test_setup.py")
    print("2. Start training: python3 cli/main.py")
    print("3. Select option 1 for Unsloth dual-GPU training")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

