# WSDP Documentation

**Wi-Fi Sensing Data Processing**

> 🌐 **Official Platform**: [SDP8.org](https://sdp8.org) | 📦 **PyPI**: [wsdp](https://pypi.org/project/wsdp/) | 💻 **GitHub**: [Source Code](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![SDP8](https://img.shields.io/badge/Platform-SDP8.org-356596)](https://sdp8.org)

A Python library for downloading, processing, analyzing and training on Wi-Fi CSI (Channel State Information) data.

**Published and maintained by [SDP8.org](https://sdp8.org)** - the official platform for reproducible wireless sensing research.

## 🚀 Features

- **Multi-dataset support**: Widar, Gait, XRF55, ElderAL, ZTE datasets
- **Intelligent preprocessing**: Wavelet denoising, phase calibration, signal resizing
- **Deep learning pipeline**: End-to-end training with CNN + Transformer architecture
- **Authentication**: JWT Token, email/password, or non-interactive mode
- **Visualization**: Heatmaps, denoising comparison, phase calibration plots
- **Inference API**: Simple `predict()` interface for deployment
- **CLI interface**: Full command-line support for batch operations

## 🧪 Signal Processing Algorithm Library

WSDP provides a comprehensive, pluggable signal preprocessing library with **16+ algorithms**:

| Category | Algorithms | Description |
|----------|-----------|-------------|
| **Denoising** | Wavelet, Butterworth, Savitzky-Golay | Remove noise while preserving features |
| **Phase Calibration** | Linear, Polynomial, STC, Robust | Correct hardware phase errors |
| **Amplitude** | Z-Score, Min-Max, IQR Outlier Removal | Normalize and clean amplitude |
| **Interpolation** | Linear, Cubic, Nearest | Resample to canonical grids |
| **Features** | Doppler Spectrum, Entropy, CSI Ratio, Tensor | Extract motion/activity features |
| **Detection** | Variance-based, Change Point | Detect activity and transitions |

**Unified API**:
```python
from wsdp.algorithms import denoise, calibrate, normalize, extract_features

denoised = denoise(csi, method='butterworth', order=5, cutoff=0.3)
calibrated = calibrate(csi, method='stc')
features = extract_features(csi, features=['doppler', 'entropy'])
```

See [Algorithm Guide](getting-started/algorithm-guide.md) for details.

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

> ⚠️ **Prerequisite**: Create a free account at [SDP8.org](https://sdp8.org) — required for dataset downloads.

### Python API

```python
from wsdp import pipeline, download

# Download dataset (SDP8.org account required)
download('widar', '/data/widar', email='you@example.com', password='yourpassword')

# Run pipeline
pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
)
```

### CLI

```bash
# Download with SDP8.org credentials
wsdp download widar /data --email you@example.com --password yourpassword

# Or with JWT token (from SDP8.org dashboard)
wsdp download widar /data --token YOUR_JWT_TOKEN

# Run training
wsdp run /data /output widar --lr 0.001 --epochs 50

# List datasets
wsdp list --verbose
```

## 📚 Documentation Sections

- [Getting Started](getting-started/installation.md): Installation and basic usage
- [User Guide](user-guide/cli.md): Detailed guides for CLI and Python API
- [API Reference](api/core.md): Complete API documentation
- [Datasets](datasets/overview.md): Information about supported datasets
- [Development](development/contributing.md): Contributing guidelines and changelog

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License.

## 📖 Citation

```bibtex
@misc{zhang2026sdpunifiedprotocolbenchmarking,
  title={SDP: A Unified Protocol and Benchmarking Framework for Reproducible Wireless Sensing}, 
  author={Di Zhang and Jiawei Huang and Yuanhao Cui and Xiaowen Cao and Tony Xiao Han and Xiaojun Jing and Christos Masouros},
  year={2026},
  eprint={2601.08463},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2601.08463}
}
```
