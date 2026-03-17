# WSDP Documentation

**Wi-Fi Sensing Data Processing**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A Python library for downloading, processing, analyzing and training on Wi-Fi CSI (Channel State Information) data.

## 🚀 Features

- **Multi-dataset support**: Widar, Gait, XRF55, ElderAL, ZTE datasets
- **Intelligent preprocessing**: Wavelet denoising, phase calibration, signal resizing
- **Deep learning pipeline**: End-to-end training with CNN + Transformer architecture
- **Authentication**: JWT Token, email/password, or non-interactive mode
- **Visualization**: Heatmaps, denoising comparison, phase calibration plots
- **Inference API**: Simple `predict()` interface for deployment
- **CLI interface**: Full command-line support for batch operations

## 📦 Installation

```bash
pip install wsdp
```

Or install from source:

```bash
git clone https://github.com/sdp-team/wsdp.git
cd wsdp
pip install -e .
```

## ⚡ Quick Start

### Python API

```python
from wsdp import pipeline, download

# Download dataset
download('widar', '/data/widar', token='your-jwt-token')

# Run pipeline
pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
)
```

### CLI

```bash
# Download
wsdp download widar /data --token YOUR_JWT_TOKEN

# Run training
wsdp run /data /output widar --lr 0.001 --epochs 50

# List datasets
wsdp list --verbose
```

## 📚 Documentation Sections

- **Getting Started**: Installation and basic usage
- **User Guide**: Detailed guides for CLI and Python API
- **API Reference**: Complete API documentation
- **Datasets**: Information about supported datasets
- **Development**: Contributing guidelines and changelog

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](../development/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License.

## 📖 Citation

```bibtex
@software{wsdp2026,
  author = {Cui, Yuanhao and WSDP Team},
  title = {WSDP: Wi-Fi Sensing Data Processing},
  url = {https://github.com/sdp-team/wsdp},
  version = {0.2.0},
  year = {2026},
}
```
