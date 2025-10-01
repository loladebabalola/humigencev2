# Humigence LoRA Training System

A robust, single-GPU LoRA fine-tuning solution that works exactly like the fixed script, but generalized for all models supported by Humigence.

## 🚀 Quick Start

### Via Humigence CLI (Recommended)
```bash
# Interactive wizard with auto-detection
humigence
# Select option 2: Single-GPU LoRA Training
# The wizard will auto-detect models, datasets, and create output directories

# Direct command (for advanced users)
python3 cli/train_lora_cli.py --model meta-llama/Meta-Llama-3-8B-Instruct --output-dir ./out_lora
```

### Via Accelerate (Recommended)
```bash
accelerate launch --num_processes=1 cli/train_lora_single.py --model meta-llama/Meta-Llama-3-8B-Instruct --output-dir ./out_lora
```

## ✨ Key Features

- ✅ **Interactive Wizard** with auto-detection of models and datasets
- ✅ **Single GPU training** (safe default)
- ✅ **bf16 precision** where supported
- ✅ **Proper gradient flow** (no loss=None errors)
- ✅ **PEFT/LoRA integration** with correct target modules
- ✅ **Gradient checkpointing** enabled
- ✅ **Support for multiple models** (LLaMA, Mistral, Phi-2, etc.)
- ✅ **Comprehensive error handling** and validation
- ✅ **Auto-generated output directories** with meaningful names
- ✅ **LoRA configuration presets** for different use cases
- ✅ **Rich progress tracking** and logging

## 🧠 Supported Models

| Model Family | Example | Target Modules |
|--------------|---------|----------------|
| **LLaMA** | `meta-llama/Meta-Llama-3-8B-Instruct` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.1` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Phi** | `microsoft/Phi-2` | `q_proj`, `k_proj`, `v_proj`, `dense` |
| **TinyLlama** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Qwen** | `Qwen/Qwen1.5-0.5B` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

## 🧙‍♂️ Interactive Wizard Features

The LoRA training wizard provides:

### 🔍 Auto-Detection
- **Models**: Scans Hugging Face cache and provides popular model options
- **Datasets**: Detects local datasets and offers popular Hugging Face datasets
- **System Info**: Shows GPU memory, CUDA availability, and system specs

### ⚙️ Configuration Presets
- **Efficient (r=8, α=16)**: Fast training, lower memory usage
- **Balanced (r=16, α=32)**: Good balance of performance and speed  
- **High Quality (r=32, α=64)**: Better performance, more parameters
- **Custom**: Set your own LoRA parameters

### 📁 Smart Output Management
- **Auto-generated directories**: `out_lora_{model}_{dataset}_{timestamp}`
- **Configuration saving**: Saves all settings to `lora_config.json`
- **Reproduction scripts**: Generates `reproduce.sh` for easy re-runs

## 📋 Usage Examples

### Interactive Wizard (Recommended)
```bash
humigence
# Select option 2: Single-GPU LoRA Training
# Follow the interactive prompts
```

### Direct Command Line
```bash
python3 cli/train_lora_single.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir ./out_lora \
    --max-steps 1000 \
    --batch-size 4
```

### Custom LoRA Settings
```bash
python3 cli/train_lora_single.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --output-dir ./out_mistral \
    --max-steps 2000 \
    --batch-size 2 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1
```

### Small Model Testing
```bash
python3 cli/train_lora_single.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output-dir ./out_tinyllama \
    --max-steps 100 \
    --batch-size 8 \
    --block-size 256
```

## 🔧 Configuration Options

### Required Arguments
- `--model`: Model name or path (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--output-dir`: Output directory for trained model

### Dataset Options
- `--dataset`: Dataset name (default: `wikitext`)
- `--dataset-config`: Dataset configuration (default: `wikitext-2-raw-v1`)
- `--block-size`: Block size for text grouping (default: `512`)

### Training Options
- `--max-steps`: Maximum training steps (default: `1000`)
- `--batch-size`: Per-device batch size (default: `4`)
- `--grad-accum`: Gradient accumulation steps (default: `4`)
- `--learning-rate`: Learning rate (default: `2e-4`)

### LoRA Options
- `--lora-r`: LoRA rank (default: `16`)
- `--lora-alpha`: LoRA alpha (default: `32`)
- `--lora-dropout`: LoRA dropout (default: `0.05`)

### Other Options
- `--warmup-steps`: Number of warmup steps (default: `100`)
- `--logging-steps`: Logging frequency (default: `10`)
- `--save-steps`: Save frequency (default: `200`)
- `--eval-steps`: Evaluation frequency (default: `200`)
- `--save-total-limit`: Maximum checkpoints to keep (default: `2`)

## 🧪 Testing

Run the test suite to validate the implementation:

```bash
python3 test_lora_single.py
```

This will test:
- Model architecture support
- CLI interface
- Model and dataset validation
- Short training run

## 🔍 Validation

After training, validate your adapters:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the base model
tokenizer = AutoTokenizer.from_pretrained('./out_lora')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

# Load the LoRA adapters
model = PeftModel.from_pretrained(model, './out_lora')

print('✅ Adapters loaded successfully!')
```

## 🐛 Troubleshooting

### Common Issues

1. **"Loss does not require gradients" warning**
   - This is handled automatically by the custom `LoRATrainer` class
   - The script will force gradient computation if needed

2. **CUDA out of memory**
   - Reduce `--batch-size` (try 1 or 2)
   - Reduce `--block-size` (try 256 or 128)
   - Use gradient accumulation: increase `--grad-accum`

3. **Model not found**
   - Ensure the model name is correct
   - Check if you have internet access for downloading
   - Verify the model exists on Hugging Face Hub

4. **Dataset loading issues**
   - The script uses `wikitext-2-raw-v1` by default
   - Ensure you have the `datasets` library installed
   - Check internet connection for dataset download

### Memory Optimization

For large models, use these settings:
```bash
python3 cli/train_lora_single.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-dir ./out_lora \
    --batch-size 1 \
    --grad-accum 8 \
    --block-size 256
```

## 📊 Output Structure

After training, you'll find:

```
out_lora/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json       # Tokenizer config
├── special_tokens_map.json     # Special tokens
├── training_summary.json       # Training metrics
└── checkpoint-*/               # Training checkpoints
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── optimizer.pt
    ├── scheduler.pt
    └── trainer_state.json
```

## 🔬 Technical Details

### Key Fixes Applied

1. **Custom LoRATrainer**: Ensures proper gradient flow
2. **enable_input_require_grads()**: Critical for PEFT + gradient checkpointing
3. **Proper data collation**: Uses `DataCollatorForLanguageModeling`
4. **Model-specific target modules**: Automatically detects correct LoRA targets
5. **Non-reentrant checkpointing**: Avoids gradient issues

### Architecture Support

The system automatically detects model architectures and applies the correct LoRA target modules:

- **LLaMA/Mistral**: All attention and MLP layers
- **Phi**: Attention layers + dense layer
- **GPT**: c_attn and c_proj layers
- **Default**: Common transformer modules

## 🤝 Contributing

To add support for new model architectures:

1. Add the model name pattern to `get_model_target_modules()`
2. Specify the correct target modules
3. Test with a short training run
4. Update this documentation

## 📝 License

This code is part of the Humigence project and follows the same license terms.
