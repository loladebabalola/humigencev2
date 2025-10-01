#!/bin/bash
# Launch script for dual-GPU LoRA fine-tuning
# Supports both Accelerate and TorchRun launch methods

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "fine-tuning-env" ]; then
    print_error "Virtual environment not found. Please run setup_environment.sh first."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source fine-tuning-env/bin/activate

# Set environment variables for RTX 5090
export CUDA_VISIBLE_DEVICES="0,1"
export TORCH_CUDA_ARCH_LIST="12.0"

# Check if training script exists
if [ ! -f "train_lora_dual.py" ]; then
    print_error "Training script not found. Please ensure train_lora_dual.py exists."
    exit 1
fi

# Check GPU availability
print_status "Checking GPU availability..."
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
gpu_count = torch.cuda.device_count()
print(f'Available GPUs: {gpu_count}')
if gpu_count < 2:
    print('WARNING: Less than 2 GPUs detected')
    print('Dual-GPU training may not be optimal')
for i in range(gpu_count):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Create output directory if it doesn't exist
mkdir -p out_lora_dual

# Parse command line arguments
LAUNCH_METHOD="torchrun"
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            LAUNCH_METHOD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--method torchrun|accelerate]"
            echo ""
            echo "Options:"
            echo "  --method    Launch method: 'torchrun' (default) or 'accelerate'"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Use torchrun (default, recommended for dual-GPU)"
            echo "  $0 --method accelerate  # Use accelerate"
            echo ""
            echo "Dual-GPU Training Methods:"
            echo "  TorchRun: True data parallel training (2 processes, 1 GPU each)"
            echo "  Accelerate: Model sharding with device_map='balanced' (1 process, 2 GPUs)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_status "Launch method: $LAUNCH_METHOD"

# Launch training based on method
if [ "$LAUNCH_METHOD" = "torchrun" ]; then
    print_status "Launching with TorchRun (DDP - Data Parallel)..."
    print_status "Command: torchrun --nproc_per_node=2 train_lora_dual.py"
    print_status "This will launch 2 processes, each using 1 GPU"
    echo ""
    
    torchrun --nproc_per_node=2 train_lora_dual.py
    
elif [ "$LAUNCH_METHOD" = "accelerate" ]; then
    print_status "Launching with Accelerate (Model Sharding)..."
    print_status "Command: accelerate launch --gpu_ids all train_lora_dual.py"
    print_status "This will launch 1 process with model sharded across 2 GPUs"
    echo ""
    
    # Check if accelerate is installed
    if ! command -v accelerate &> /dev/null; then
        print_error "Accelerate not found. Installing..."
        pip install accelerate
    fi
    
    # Configure accelerate if not already configured
    if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
        print_status "Configuring Accelerate for multi-GPU training..."
        print_status "Run ./configure_accelerate.sh to set up Accelerate properly"
        accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml
    fi
    
    accelerate launch --gpu_ids all train_lora_dual.py
    
else
    print_error "Invalid launch method: $LAUNCH_METHOD"
    print_error "Valid options: torchrun, accelerate"
    exit 1
fi

print_success "Training completed!"
print_status "Check output directory: ~/fine-tuning/out_lora_dual/"

