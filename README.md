# 🧠 Humigence CLI

**Your AI. Your pipeline. Zero code.**

A complete MLOps suite built for makers, teams, and enterprises. Humigence provides zero-config, GPU-aware fine-tuning with surgical precision and complete reproducibility.

## ✨ Features

- 🎯 **Zero-Config Wizard**: Interactive setup with Basic/Advanced modes
- 🖥️ **Hardware Detection**: Automatic GPU, CPU, and memory detection
- 🧪 **Training Recipes**: QLoRA, LoRA (FP16/BF16), Full Fine-tuning
- 📊 **Smart Batching**: Auto-fit micro-batch size to available VRAM
- 🔄 **Complete Reproducibility**: Config snapshots and reproduce scripts
- 📈 **Evaluation & Acceptance**: Curated prompts and quality gates
- 📦 **Artifact Export**: Structured outputs with run summaries

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/humigence.git
cd humigence

# Install dependencies
pip install -e .

# Set up CLI alias (optional)
echo "alias humigence='python3 ~/humigence/cli/main.py'" >> ~/.bashrc
source ~/.bashrc
```

### Basic Usage

```bash
# Launch the interactive wizard
humigence

# Or run directly
python3 -m cli.main

# Run training from config
python3 -m pipelines.lora_trainer runs/humigence/config.snapshot.json

# Audit a training run
python3 -m cli.humigence_audit
```

## 🎯 Training Workflow

### 1. Interactive Setup

The Humigence wizard guides you through:

- **Setup Mode**: Basic (essential config) or Advanced (full control)
- **Hardware Detection**: Automatic GPU, CPU, and memory detection
- **Model Selection**: HuggingFace cache scanning + manual entry
- **Dataset Loading**: Auto-detection from `~/humigence_data/`
- **Training Recipe**: QLoRA, LoRA, or Full Fine-tuning
- **Hyperparameters**: Learning rate, epochs, batch size, etc.

### 2. Training Execution

```bash
🚀 Humigence Trainer Starting...
✅ Configuration Loaded: [all settings]
📦 Estimated Micro-batch Size: 4
⚠️ Loading model without quantization (RTX 5090 compatibility)
✅ Model + Tokenizer Loaded: Qwen/Qwen1.5-0.5B
✅ LoRA adapters applied
📚 Loading dataset...
✅ Dataset loaded: 10 samples
🚀 Starting training...
✅ Training complete — adapters saved.
```

### 3. Evaluation & Acceptance

- **Curated Prompts**: 5 diverse evaluation questions
- **Model Inference**: Generation with temperature and sampling
- **Acceptance Criteria**: Loss threshold (< 0.8) and eval count (≥ 1)
- **Status Markers**: ACCEPTED.txt or REJECTED.txt files

### 4. Artifact Export

```
runs/humigence/
├── adapters/                # LoRA adapter weights
├── tokenizer/               # Tokenizer used
├── config.snapshot.json     # Training config
├── reproduce.sh             # Rerun script
├── ACCEPTED.txt / REJECTED.txt
├── eval_results.jsonl       # Evaluation prompt outputs
├── run_summary.json         # Structured run summary
└── artifacts.zip            # Complete export archive
```

## 🔧 Configuration

### Basic Mode (Recommended)

Essential configuration with sensible defaults:

- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Gradient Accumulation**: 4
- **Logging Steps**: 10
- **Save Steps**: 100

### Advanced Mode

Full control over all parameters:

- Gradient accumulation steps
- Learning rate
- Evaluation strategy
- Save steps
- Warmup steps
- Number of training epochs
- Logging steps
- Random seed

## 📊 Supported Models

- **Qwen/Qwen1.5-0.5B**: 77M parameters
- **microsoft/Phi-2**: 839M parameters
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: 369M parameters
- **Custom Models**: HuggingFace repos or local paths

## 🗂️ Dataset Support

- **OpenAssistant Format**: Automatic conversation pairing
- **Instruction-Response**: Standard format support
- **JSONL Files**: Line-by-line JSON processing
- **Auto-Detection**: Scans `~/humigence_data/` directory

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (RTX 5090 compatible)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and datasets
- **Python**: 3.8+ with PyTorch

## 📁 Project Structure

```
humigence/
├── cli/
│   ├── main.py              # CLI entry point
│   ├── fine_tune.py         # Interactive wizard
│   └── humigence_audit.py   # Run inspector
├── config/
│   └── default_config.json  # Fallback defaults
├── pipelines/
│   └── lora_trainer.py      # Training engine
├── templates/
│   └── accelerate_config.yaml
├── utils/
│   ├── device.py            # Hardware detection
│   ├── tokenizer.py         # Tokenizer utilities
│   └── validators.py        # Dataset validation
└── runs/
    └── <run_name>/
        ├── config.snapshot.json
        ├── reproduce.sh
        ├── adapters/
        ├── tokenizer/
        └── artifacts.zip
```

## 🔄 Reproducibility

Every training run generates:

- **Config Snapshot**: Complete configuration in JSON
- **Reproduce Script**: One-click rerun capability
- **Artifact Archive**: Complete export of all outputs
- **Run Summary**: Structured metadata for tracking

```bash
# Rerun any training
./runs/humigence/reproduce.sh

# Or use the config directly
python3 -m pipelines.lora_trainer runs/humigence/config.snapshot.json
```

## 🧪 Evaluation

### Curated Prompts

Default evaluation questions:

1. "What is the capital of France?"
2. "Explain quantum computing in simple terms."
3. "Write a short poem about artificial intelligence."
4. "How do you make a good cup of coffee?"
5. "What are the benefits of renewable energy?"

### Custom Evaluation

Create `runs/humigence/eval_prompts.jsonl`:

```json
{"instruction": "Your custom prompt here"}
{"instruction": "Another evaluation question"}
```

## 📈 Monitoring

### Run Audit

Inspect any training run:

```bash
python3 -m cli.humigence_audit
```

Shows:
- Training configuration
- Run status (ACCEPTED/REJECTED)
- Final metrics
- Evaluation results

### Run Summary

Structured JSON output:

```json
{
  "run_id": "2025-09-17T22:50:18.668019",
  "status": "accepted",
  "model": "Qwen/Qwen1.5-0.5B",
  "dataset": "/path/to/dataset.jsonl",
  "recipe": "QLoRA (4-bit NF4)",
  "epochs": "3",
  "learning_rate": "2e-5",
  "final_loss": 0.65,
  "eval_prompt_count": 5,
  "timestamp": "2025-09-17 23:31:01"
}
```

## 🛠️ Development

### Dependencies

- `typer`: CLI framework
- `rich`: Terminal formatting
- `inquirerpy`: Interactive prompts
- `transformers`: HuggingFace models
- `peft`: Parameter-efficient fine-tuning
- `bitsandbytes`: Quantization
- `accelerate`: Multi-GPU training
- `datasets`: Dataset handling
- `psutil`: System monitoring

### Installation

```bash
# Install in development mode
pip install -e .

# Or install dependencies manually
pip install typer rich inquirerpy transformers peft bitsandbytes accelerate datasets psutil
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- HuggingFace for the transformers library
- Microsoft for PEFT and LoRA implementations
- The open-source ML community

---

**Built with ❤️ for the AI community**

*Humigence — Your AI. Your pipeline. Zero code.*
