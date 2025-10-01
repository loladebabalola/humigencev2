#!/bin/bash
# Setup script for dual-GPU LoRA fine-tuning on RTX 5090s
# Creates fine-tuning-env venv and installs all dependencies

set -euo pipefail

echo "🚀 Setting up dual-GPU fine-tuning environment for RTX 5090s..."

# Check if Python 3.8+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.8+ is required but not found"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ is required, found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "fine-tuning-env" ]; then
    python3 -m venv fine-tuning-env
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source fine-tuning-env/bin/activate

# Upgrade pip and setuptools
echo "📦 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Set environment variables for RTX 5090/Blackwell
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_VISIBLE_DEVICES="0,1"

echo "🔧 Setting up CUDA environment for RTX 5090..."
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Install PyTorch with CUDA 12.8 support first
echo "🔥 Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install other requirements using the same index to avoid mismatches
echo "📚 Installing other requirements..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import torch
import sys

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available!')
    sys.exit(1)

# Test Unsloth import
try:
    import unsloth
    print(f'✅ Unsloth version: {unsloth.__version__}')
except ImportError as e:
    print(f'❌ Unsloth import failed: {e}')
    sys.exit(1)

# Test other critical imports
try:
    import transformers
    import bitsandbytes
    import accelerate
    print(f'✅ Transformers: {transformers.__version__}')
    print(f'✅ BitsAndBytes: {bitsandbytes.__version__}')
    print(f'✅ Accelerate: {accelerate.__version__}')
except ImportError as e:
    print(f'❌ Critical import failed: {e}')
    sys.exit(1)

print('🎉 All verifications passed!')
"

echo "✅ Environment setup completed!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate environment: source fine-tuning-env/bin/activate"
echo "2. Test setup: python test_setup.py"
echo "3. Launch training: ./launch_training.sh"
echo ""
echo "📁 Output will be saved to: ~/fine-tuning/out_lora_dual/"

