# Contributing to WSDP

Thank you for your interest in contributing to WSDP (Wi-Fi Sensing Data Processing)! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/sdp-team/wsdp.git
cd wsdp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/ -v
```

## 📝 Contribution Guidelines

### Reporting Issues

When reporting issues, please include:
- **Description**: Clear description of the issue
- **Environment**: Python version, OS, WSDP version
- **Reproduction steps**: Minimal steps to reproduce
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Logs/Tracebacks**: Any error messages or logs

### Suggesting Features

We welcome feature suggestions! Please:
- Check if the feature has already been requested
- Describe the use case and benefits
- Provide examples if possible

### Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Changes**
   - Follow the code style (PEP 8)
   - Add tests for new features
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run with coverage
   pytest tests/ --cov=wsdp --cov-report=html
   
   # Check code style
   flake8 src/wsdp
   ```

4. **Commit**
   - Use clear, descriptive commit messages
   - Reference issues if applicable: `Fix #123: description`

5. **Submit PR**
   - Provide clear description of changes
   - Link related issues
   - Ensure CI checks pass

## 🏗️ Code Structure

```
wsdp/
├── algorithms/      # Signal processing algorithms
├── readers/         # Dataset readers
├── processors/      # Data preprocessing
├── models/          # Neural network models
├── datasets/        # PyTorch datasets
├── structure/       # Data structures (CSIData, CSIFrame)
├── core.py          # Training pipeline
├── cli.py           # Command-line interface
└── inference.py     # Prediction interface
```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_algorithms.py -v

# With coverage
pytest tests/ --cov=wsdp --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Use pytest fixtures for common setup

Example:
```python
def test_wavelet_denoise():
    csi = np.random.randn(10, 100, 30, 3)
    denoised = wavelet_denoise_csi(csi)
    assert denoised.shape == csi.shape
```

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions (Google style)
- Update CHANGELOG.md for notable changes

## 🎨 Code Style

- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 100 characters
- Use descriptive variable names

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)

## 💬 Community

- GitHub Issues: Bug reports, feature requests
- GitHub Discussions: General questions, ideas

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to WSDP! 🎉
