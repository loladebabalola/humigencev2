# Unsloth Dual-GPU LoRA Training Module

This module provides efficient dual-GPU LoRA fine-tuning using Unsloth's FastLanguageModel, optimized for NVIDIA RTX 5090 (Blackwell architecture) GPUs.

## Features

- **Dual-GPU Training**: True data parallel training with TorchRun (DDP)
- **Unsloth Integration**: FastLanguageModel with 4-bit quantization
- **RTX 5090 Optimized**: Blackwell architecture optimizations
- **Flexible Launch Methods**: TorchRun (DDP) or Accelerate (model sharding)
- **Memory Efficient**: 4-bit quantization with LoRA adapters
- **Production Ready**: Comprehensive error handling and logging

## Quick Start

### 1. Setup Environment

```bash
cd humigence/training/unsloth
chmod +x setup_environment.sh
./setup_environment.sh
```

### 2. Launch Training

```bash
# Using TorchRun (recommended for dual-GPU)
chmod +x launch_training.sh
./launch_training.sh

# Using Accelerate
./launch_training.sh --method accelerate
```

### 3. Programmatic Usage

```python
from humigence.training.unsloth import train_lora_dual_gpu

result = train_lora_dual_gpu(
    model_name="unsloth/Llama-3-8B-Instruct",
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    output_dir="./runs/humigence/out_lora_dual",
    max_steps=1000,
    batch_size=2,
    grad_accum=4,
    learning_rate=2e-4
)

if result["status"] == "success":
    print("Training completed successfully!")
    print(f"Output saved to: {result['output_dir']}")
else:
    print(f"Training failed: {result['error']}")
```

## Training Methods

### TorchRun (Recommended)
- **Method**: Data Parallel (DDP)
- **Processes**: 2 (one per GPU)
- **Memory**: Each process uses one GPU
- **Performance**: Optimal for dual-GPU setups
- **Command**: `torchrun --nproc_per_node=2 train_lora_dual.py`

### Accelerate
- **Method**: Model Sharding
- **Processes**: 1 (model split across GPUs)
- **Memory**: Model distributed across both GPUs
- **Performance**: Good for memory-constrained setups
- **Command**: `accelerate launch --gpu_ids all train_lora_dual.py`

## Configuration

### Model Parameters
- `model_name`: Base model (default: "unsloth/Llama-3-8B-Instruct")
- `dataset_name`: Dataset name (default: "wikitext")
- `dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1")

### Training Parameters
- `max_steps`: Maximum training steps (default: 1000)
- `batch_size`: Batch size per device (default: 2)
- `grad_accum`: Gradient accumulation steps (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `block_size`: Sequence length (default: 1024)

### LoRA Parameters
- `lora_r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA alpha (default: 32)
- `lora_dropout`: LoRA dropout (default: 0.0)

## Output Structure

```
runs/humigence/out_lora_dual/
├── lora_adapter/           # LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── merged/                 # Merged model weights
│   ├── model files
│   └── tokenizer files
└── training.log           # Training logs
```

## System Requirements

- **GPUs**: 2x NVIDIA RTX 5090 (or compatible)
- **CUDA**: 12.8+ with RTX 5090 support
- **Python**: 3.8+
- **Memory**: 48GB+ VRAM total (24GB per GPU)
- **Storage**: 50GB+ free space

## Environment Variables

The module automatically sets these environment variables for optimal performance:

```bash
export CUDA_VISIBLE_DEVICES="0,1"
export TORCH_CUDA_ARCH_LIST="12.0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM="false"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` or `block_size`
   - Increase `grad_accum` to maintain effective batch size

2. **DDP Initialization Errors**
   - Ensure both GPUs are visible: `nvidia-smi`
   - Check NCCL installation: `python -c "import torch.distributed"`

3. **Unsloth Import Errors**
   - Install Unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
   - Verify CUDA compatibility

### Performance Tips

1. **Use TorchRun for dual-GPU**: Better memory utilization and performance
2. **Optimize batch size**: Start with 2 and increase if memory allows
3. **Monitor GPU utilization**: Use `nvidia-smi` to ensure both GPUs are active
4. **Check logs**: Training logs are saved to `training.log`

## Integration with Humigence CLI

This module is integrated into the Humigence CLI wizard:

1. Run `humigence` from the CLI
2. Select "1. Supervised Fine-Tuning"
3. Choose your dataset and model
4. Training will launch automatically with dual-GPU support

## License

This module is part of the Humigence project and follows the same license terms.

