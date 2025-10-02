---
library_name: transformers
tags:
- fine-tuning
- lora
- qlora
- mlops
- cli
- wizard
- dual-gpu
- unsloth
license: mit
pipeline_tag: text-generation
---

# Humigence CLI - MLOps Toolkit for LLM Fine-tuning

**Your AI. Your pipeline. Zero code.**

Humigence is a comprehensive MLOps toolkit that provides zero-config, GPU-aware fine-tuning with surgical precision and complete reproducibility. Built for makers, teams, and enterprises who want to fine-tune large language models without the complexity of traditional MLOps setups.

## 🚀 Key Features

- **Interactive Wizard**: Step-by-step configuration with Basic/Advanced modes
- **Smart GPU Detection**: Automatic detection and selection of available GPUs
- **Dual-GPU Training**: Multi-GPU support with Unsloth + TorchRun
- **Training Recipes**: QLoRA (4-bit), LoRA (FP16/BF16), Full Fine-tuning
- **Complete Reproducibility**: Config snapshots and reproduce scripts
- **Built-in Evaluation**: Curated prompts and quality gates

## 🎯 What Makes Humigence Different?

Unlike other fine-tuning tools, Humigence focuses on:

1. **Zero-Configuration**: Interactive wizard handles all setup
2. **Hardware-Aware**: Automatically detects and optimizes for your GPU setup
3. **Production-Ready**: Complete MLOps pipeline with evaluation and monitoring
4. **Reproducible**: Every run generates complete artifacts for reproduction

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/humigence.git
cd humigence

# Install dependencies
pip install -r requirements.txt

# Set up Unsloth (required for training)
python3 training/unsloth/setup_humigence_unsloth.py

# Launch the interactive wizard
python3 cli/main.py
```

### Basic Usage

```bash
# Launch the interactive wizard
python3 cli/main.py

# The wizard will guide you through:
# 1. Model selection (Qwen, Phi-2, TinyLlama, or custom)
# 2. Dataset configuration (JSONL format)
# 3. Training parameters (learning rate, epochs, etc.)
# 4. GPU selection (single or multi-GPU)
# 5. Launch training
```

## 📊 Supported Models

Humigence supports a wide range of models:

- **Qwen/Qwen2.5-0.5B**: 77M parameters (recommended for testing)
- **microsoft/Phi-2**: 839M parameters
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0**: 369M parameters
- **Custom Models**: Any HuggingFace model or local path

## 🗂️ Dataset Support

### Supported Formats

- **JSONL Format**: Line-by-line JSON with instruction/output pairs
- **Auto-Detection**: Scans `~/humigence_data/` directory
- **Custom Paths**: Specify any local dataset file

### Dataset Format

```json
{"instruction": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence..."}
{"instruction": "Explain quantum computing", "output": "Quantum computing uses quantum mechanical phenomena..."}
```

## 🖥️ Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space

### Recommended Setup
- **GPU**: RTX 4080/4090/5090 or better
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space

### Multi-GPU Support
- **Dual-GPU**: RTX 5090 + RTX 5090 (tested)
- **Memory**: 16GB+ VRAM per GPU recommended
- **Training**: Automatic TorchRun distribution

## 🎯 Training Workflow

### 1. Interactive Setup

The Humigence wizard guides you through:

- **Setup Mode**: Basic (essential config) or Advanced (full control)
- **Hardware Detection**: Automatic GPU, CPU, and memory detection
- **Model Selection**: Choose from supported models or custom paths
- **Dataset Loading**: Auto-detection from `~/humigence_data/` or custom paths
- **Training Recipe**: QLoRA, LoRA, or Full Fine-tuning
- **GPU Selection**: Single-GPU auto-selection or multi-GPU prompting

### 2. GPU Selection

Humigence intelligently handles GPU selection:

- **Single GPU**: Automatically selects and uses the available GPU
- **Multiple GPUs**: Prompts you to choose:
  ```
  🔧 Training Mode:
  > Multi-GPU Training (all available GPUs)
    Single GPU Training (choose specific GPU)
  ```

### 3. Training Execution

```bash
🚀 Humigence Training Starting...
✅ Configuration Loaded: [all settings]
🖥️ GPU Detection: 2x RTX 5090 detected
🔧 Training Mode: Multi-GPU Training
📦 Loading model: Qwen/Qwen2.5-0.5B
✅ LoRA adapters applied
📚 Loading dataset: wikitext2 (10,000 samples)
🚀 Starting training with TorchRun...
✅ Training complete — adapters saved.
```

## 🔧 Configuration

### Basic Mode (Recommended)

Essential configuration with sensible defaults:

- **Learning Rate**: 2e-4
- **Epochs**: 1
- **Gradient Accumulation**: 4
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

### Advanced Mode

Full control over all parameters:

- LoRA configuration (rank, alpha, dropout)
- Training hyperparameters
- Data processing options
- Evaluation settings

## 📈 Evaluation & Monitoring

### Built-in Evaluation

- **Curated Prompts**: 5 diverse evaluation questions
- **Model Inference**: Generation with temperature sampling
- **Quality Gates**: Loss thresholds and evaluation metrics
- **Status Tracking**: ACCEPTED.txt or REJECTED.txt files

### Run Monitoring

```bash
# View training progress
tail -f runs/humigence/training.log

# Check evaluation results
cat runs/humigence/eval_results.jsonl

# View run summary
cat runs/humigence/run_summary.json
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
python3 training/unsloth/train_lora_dual.py --config runs/humigence/config.snapshot.json
```

## 🛠️ Development

### Dependencies

Core dependencies are pinned for stability:

```txt
transformers>=4.41.0,<5.0.0
torch>=2.1.0
unsloth @ git+https://github.com/unslothai/unsloth.git
rich>=13.0.0
inquirer>=3.1.0
```

### Local Development

```bash
# Install in development mode
pip install -e .

# Run tests
python3 -m pytest tests/

# Run specific test
python3 test_gpu_selection.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/your-username/humigence/blob/main/CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/your-username/humigence/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for fast LoRA training
- [HuggingFace](https://huggingface.co/) for the transformers library
- [Microsoft](https://github.com/microsoft) for PEFT and LoRA implementations
- The open-source ML community

## 🆚 Comparison with Other Tools

| Feature | Humigence CLI | Other Tools |
|---------|---------------|-------------|
| **Setup** | Interactive wizard | Manual config |
| **GPU Detection** | Automatic | Manual |
| **Multi-GPU** | Built-in TorchRun | Complex setup |
| **Reproducibility** | Complete snapshots | Partial |
| **Evaluation** | Built-in prompts | External tools |
| **Artifacts** | Structured export | Manual collection |

## 🐛 Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU visibility
nvidia-smi
```

**Out of memory:**
```bash
# Reduce batch size in config
# Or use QLoRA for memory efficiency
```

**Training fails:**
```bash
# Check logs
cat runs/humigence/training.log

# Verify dataset format
head -5 ~/humigence_data/your_dataset.jsonl
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-username/humigence/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/humigence/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/humigence/wiki)

## 🗺️ Roadmap

### Current Features ✅
- Interactive configuration wizard
- Single and multi-GPU training
- QLoRA and LoRA support
- Built-in evaluation
- Complete reproducibility

### Coming Soon 🚧
- RAG implementation
- EnterpriseGPT integration
- Batch inference
- Context length optimization
- Web UI interface
- Model serving

### Future Features 🔮
- Distributed training across nodes
- Advanced evaluation metrics
- Model compression
- Deployment automation

---

**Built with ❤️ for the AI community**

*Humigence — Your AI. Your pipeline. Zero code.*

## 📊 Stats

![GitHub stars](https://img.shields.io/github/stars/your-username/humigence?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/humigence?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/humigence)
![GitHub license](https://img.shields.io/github/license/your-username/humigence)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)




