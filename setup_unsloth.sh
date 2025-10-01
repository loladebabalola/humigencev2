#!/bin/bash
# Setup script for Unsloth dual-GPU LoRA training
# Optimized for RTX 5090 (Blackwell architecture)

set -e

echo "🚀 Setting up Unsloth dual-GPU LoRA training environment..."

# Check if we're in the right directory
if [ ! -f "cli/main.py" ]; then
    echo "❌ Error: Please run this script from the humigence directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ Warning: nvidia-smi not found. CUDA may not be available."
else
    echo "✅ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
fi

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install transformers>=4.36.0 datasets>=2.14.0 accelerate>=0.24.0 peft>=0.7.0 bitsandbytes>=0.41.0

# Install Unsloth from source
echo "📦 Installing Unsloth from source..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install CLI dependencies
echo "📦 Installing CLI dependencies..."
pip install rich>=13.0.0 inquirer>=3.1.0 typer>=0.9.0 numpy>=1.24.0 pandas>=2.0.0 tqdm>=4.65.0

# Create output directories
echo "📁 Creating output directories..."
mkdir -p runs/humigence
mkdir -p humigence_data

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import transformers
import datasets
import accelerate
import peft
import bitsandbytes
print('✅ All core dependencies imported successfully')

# Test CUDA
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️ CUDA not available - training will be slower')

# Test Unsloth
try:
    import unsloth
    print('✅ Unsloth imported successfully')
except ImportError as e:
    print(f'❌ Unsloth import failed: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start training:"
echo "  python3 cli/main.py"
echo ""
echo "Available options:"
echo "  1. Supervised Fine-Tuning (Unsloth + Dual-GPU) 🚀"
echo "  2. Single-GPU LoRA Training ✅"
echo ""
echo "For dual-GPU training, ensure you have 2+ GPUs available."
echo "The system will automatically detect and use available GPUs."
