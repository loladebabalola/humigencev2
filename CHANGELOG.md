# Changelog

All notable changes to Humigence will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interactive configuration wizard with Basic/Advanced modes
- Automatic GPU detection and selection
- Multi-GPU training support with TorchRun
- QLoRA and LoRA training recipes
- Built-in evaluation with curated prompts
- Complete reproducibility with config snapshots
- Artifact export and run summaries
- Support for multiple model architectures (Qwen, Phi-2, TinyLlama)
- JSONL dataset format support
- Automatic dataset detection from ~/humigence_data/

### Changed
- Moved from manual configuration to interactive wizard
- Improved GPU detection to work with Unsloth integration
- Enhanced error handling and user feedback
- Streamlined training pipeline for better reliability

### Fixed
- GPU detection issues when Unsloth is imported
- Redundant GPU selection prompts in wizard
- Configuration summary accuracy
- Training pipeline stability

## [1.0.0] - 2025-01-01

### Added
- Initial release of Humigence CLI
- Interactive wizard for configuration
- Single and multi-GPU training support
- QLoRA and LoRA fine-tuning capabilities
- Built-in evaluation system
- Complete reproducibility features
- Comprehensive documentation
- MIT license

### Features
- **Interactive Wizard**: Step-by-step configuration with Basic/Advanced modes
- **Smart GPU Detection**: Automatic detection and selection of available GPUs
- **Dual-GPU Training**: Multi-GPU support with Unsloth + TorchRun
- **Training Recipes**: QLoRA (4-bit), LoRA (FP16/BF16), Full Fine-tuning
- **Complete Reproducibility**: Config snapshots and reproduce scripts
- **Built-in Evaluation**: Curated prompts and quality gates
- **Artifact Export**: Structured outputs with run summaries

### Supported Models
- Qwen/Qwen2.5-0.5B (77M parameters)
- microsoft/Phi-2 (839M parameters)
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (369M parameters)
- Custom HuggingFace models

### Hardware Support
- Single GPU training (auto-selection)
- Multi-GPU training (TorchRun distribution)
- RTX 5090, RTX 4080, and other CUDA-compatible GPUs
- Automatic memory optimization

### Documentation
- Comprehensive README with examples
- Contributing guidelines
- Troubleshooting guide
- Hugging Face model card
- API documentation

---

## Version History

### v1.0.0 (2025-01-01)
- **Initial Release**: Complete MLOps toolkit for LLM fine-tuning
- **Core Features**: Interactive wizard, GPU detection, multi-GPU training
- **Training Support**: QLoRA, LoRA, and full fine-tuning
- **Evaluation**: Built-in prompts and quality gates
- **Reproducibility**: Complete config snapshots and reproduce scripts

---

## Contributing

To contribute to this changelog:

1. Add your changes under the `[Unreleased]` section
2. Use the following format:
   - `### Added` for new features
   - `### Changed` for changes in existing functionality
   - `### Deprecated` for soon-to-be removed features
   - `### Removed` for now removed features
   - `### Fixed` for any bug fixes
   - `### Security` for vulnerability fixes

3. When releasing a new version:
   - Move `[Unreleased]` items to the new version
   - Update the version number and date
   - Create a new `[Unreleased]` section

## Links

- [GitHub Repository](https://github.com/your-username/humigence)
- [Documentation](https://github.com/your-username/humigence/wiki)
- [Issues](https://github.com/your-username/humigence/issues)
- [Discussions](https://github.com/your-username/humigence/discussions)
