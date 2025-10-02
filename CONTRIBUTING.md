# Contributing to Humigence

Thank you for your interest in contributing to Humigence! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the troubleshooting section** in the README
3. **Provide detailed information** about your environment and the issue

When creating an issue, include:

- **Environment**: OS, Python version, GPU model, CUDA version
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error logs
- **Screenshots**: If applicable

### Suggesting Features

We welcome feature suggestions! Please:

1. **Check the roadmap** in the README first
2. **Describe the use case** and why it would be valuable
3. **Provide examples** of how it would work
4. **Consider implementation complexity**

### Code Contributions

#### Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/humigence.git
   cd humigence
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Set up development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

#### Development Guidelines

##### Code Style

- **Python**: Follow PEP 8 style guidelines
- **Line length**: 88 characters (Black formatter)
- **Imports**: Use absolute imports, group by standard library, third-party, local
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for function parameters and return values

##### Testing

- **Write tests** for new features
- **Run existing tests** before submitting:
  ```bash
  python3 -m pytest tests/
  ```
- **Test on different hardware** if possible (single GPU, multi-GPU)

##### Documentation

- **Update README.md** if adding new features
- **Add docstrings** to new functions and classes
- **Update type hints** for better IDE support
- **Include examples** in docstrings

#### Pull Request Process

1. **Ensure tests pass**:
   ```bash
   python3 -m pytest tests/
   python3 -m flake8 cli/ pipelines/ utils/
   ```

2. **Update documentation** if needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Create a Pull Request** with:
   - Clear title and description
   - Reference any related issues
   - Include screenshots if applicable
   - List any breaking changes

### Commit Message Convention

We use conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```bash
git commit -m "feat: add multi-GPU training support"
git commit -m "fix: resolve GPU detection issue"
git commit -m "docs: update installation instructions"
```

## 🏗️ Development Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Git

### Local Development

1. **Clone and install**:
   ```bash
   git clone https://github.com/your-username/humigence.git
   cd humigence
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Set up Unsloth**:
   ```bash
   python3 training/unsloth/setup_humigence_unsloth.py
   ```

3. **Run tests**:
   ```bash
   python3 -m pytest tests/
   ```

4. **Test CLI**:
   ```bash
   python3 cli/main.py
   ```

### Project Structure

```
humigence/
├── cli/                    # Command-line interface
│   ├── main.py            # Main entry point
│   ├── config_wizard.py   # Configuration wizard
│   └── lora_wizard.py     # LoRA-specific wizard
├── training/              # Training modules
│   └── unsloth/          # Unsloth integration
├── pipelines/             # Training pipelines
├── utils/                 # Utility functions
├── config/                # Configuration files
├── tests/                 # Test files
└── docs/                  # Documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_gpu_selection.py

# Run with verbose output
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=cli --cov=pipelines --cov=utils
```

### Writing Tests

- **Test files** should be named `test_*.py`
- **Test functions** should be named `test_*`
- **Use descriptive names** for test functions
- **Test both success and failure cases**
- **Mock external dependencies** when possible

Example:
```python
def test_gpu_detection_single_gpu():
    """Test GPU detection with single GPU."""
    gpu_count, gpus = detect_gpus()
    assert gpu_count == 1
    assert len(gpus) == 1
    assert gpus[0]['index'] == 0
```

## 📝 Documentation

### Code Documentation

- **Functions**: Include docstring with description, parameters, returns, and examples
- **Classes**: Include docstring with description and usage examples
- **Modules**: Include module-level docstring

Example:
```python
def detect_gpus():
    """
    Detect available GPUs on the system.
    
    Returns:
        tuple: (gpu_count, gpus) where gpus is a list of dicts
               with 'index', 'name', and 'memory' keys
    
    Example:
        >>> gpu_count, gpus = detect_gpus()
        >>> print(f"Found {gpu_count} GPUs")
    """
```

### README Updates

When adding features, update:

- **Features section** with new capabilities
- **Usage examples** with new commands
- **Configuration section** with new options
- **Troubleshooting section** with common issues

## 🐛 Bug Reports

### Before Reporting

1. **Check existing issues** for similar problems
2. **Try the latest version** from main branch
3. **Check troubleshooting section** in README
4. **Search closed issues** for solutions

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Select option '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- GPU: [e.g., RTX 5090]
- CUDA: [e.g., 11.8]

**Error Logs**
```
Paste error logs here
```

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting

1. **Check the roadmap** in README
2. **Search existing issues** for similar requests
3. **Consider implementation complexity**

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this would solve or the workflow it would improve.

**Proposed Solution**
Describe how you think this should work.

**Alternatives**
Describe any alternative solutions you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

## 🏷️ Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Release notes prepared
- [ ] Tag created

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [Contact information if available]

## 🙏 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📄 License

By contributing to Humigence, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Humigence! 🚀




