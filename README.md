# WSDP - Wi-Fi Sensing Data Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-blueviolet)](https://docs.pytest.org)
[![PyPI](https://img.shields.io/badge/PyPI-Coming%20Soon-orange.svg)](https://pypi.org)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://sdp-team.github.io/wsdp)
[![Colab](https://img.shields.io/badge/Colab-Tutorial-yellow.svg)](https://colab.research.google.com/github/sdp-team/wsdp/blob/main/examples/wsdp_tutorial.ipynb)

**A Python library for downloading, processing, analyzing and training on Wi-Fi CSI (Channel State Information) data.**

[English](#english) | [中文](#中文)

</div>

---

<a id="english"></a>

## 🚀 Features

- **Multi-dataset support**: Widar, Gait, XRF55, ElderAL, ZTE datasets
- **Intelligent preprocessing**: Wavelet denoising, phase calibration, signal resizing
- **Deep learning pipeline**: End-to-end training with CNN + Transformer architecture
- **Authentication**: JWT Token, email/password, or non-interactive mode
- **Visualization**: Heatmaps, denoising comparison, phase calibration plots
- **Inference API**: Simple `predict()` interface for deployment
- **CLI interface**: Full command-line support for batch operations

## 🤔 Why WSDP?

| Feature | WSDP | Raw CSI | Other Tools |
|---------|------|---------|-------------|
| **Standardized Format** | ✅ Unified CSIFrame | ❌ Hardware-specific | ⚠️ Partial |
| **Multi-Dataset** | ✅ 5 datasets built-in | ❌ Manual parsing | ⚠️ 2-3 datasets |
| **Preprocessing** | ✅ Wavelet + Phase Calib | ❌ DIY | ⚠️ Basic only |
| **Deep Learning** | ✅ CNN+Transformer | ❌ From scratch | ⚠️ Limited |
| **Reproducibility** | ✅ Deterministic seeds | ❌ Random | ⚠️ Varies |
| **CLI Interface** | ✅ Full CLI support | ❌ None | ⚠️ Partial |

**WSDP** bridges the gap between raw CSI data and machine learning, providing a complete pipeline from data download to model deployment. No more writing boilerplate code for each new dataset!

## 📦 Installation

```bash
# From source
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## ⚡ Quick Start

### Python API

```python
from wsdp import pipeline, download, predict

# Download a dataset (with JWT token)
download('widar', '/data/widar', token='your-jwt-token')

# Run the full pipeline
pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
    learning_rate=1e-3,  # optional override
)

# Inference
import numpy as np
csi = np.random.randn(5, 200, 30, 3) + 1j * np.random.randn(5, 200, 30, 3)
predictions = predict(csi, 'best_checkpoint.pth', num_classes=6)
```

### Algorithms

```python
from wsdp.algorithms import wavelet_denoise_csi, phase_calibration
from wsdp.algorithms.visualization import plot_csi_heatmap

# Phase calibration
calibrated = phase_calibration(csi_data)

# Wavelet denoising
denoised = wavelet_denoise_csi(csi_data)

# Visualization
plot_csi_heatmap(csi_data, antenna_idx=0, save_path='heatmap.png')
```

### CLI

```bash
# Download with JWT token (non-interactive)
wsdp download widar /data --token YOUR_JWT_TOKEN

# Download with email/password
wsdp download elderAL /data --email user@example.com --password secret

# Run pipeline with default hyperparameters
wsdp run /data /output widar

# Run pipeline with custom hyperparameters
wsdp run /data /output widar --lr 0.001 --epochs 50 --batch-size 64

# Run with config file
wsdp run /data /output widar --config config.yaml

# List available datasets
wsdp list
wsdp list --verbose  # with metadata

# Version
wsdp --version
```

## 📊 Supported Datasets

| Dataset | Format | Subcarriers | Complex | Description |
|---------|--------|-------------|---------|-------------|
| Widar | .dat (bfee) | 30 | ✅ | Gesture recognition with Intel IWL5300 |
| Gait | .dat (bfee) | 30 | ✅ | Gait recognition |
| XRF55 | .npy | 30 | ✅ | Human activity recognition |
| ElderAL | .csv | varies | ❌ | Elderly activity & location |
| ZTE | .csv | 512 | ✅ | CSI with I/Q components |

## 🔧 Architecture

```
wsdp/
├── algorithms/          # Signal processing algorithms
│   ├── denoising.py     # Wavelet-based denoising
│   ├── phase_calibration.py  # Phase error correction
│   └── visualization.py # Plotting utilities
├── readers/             # Dataset-specific file readers
├── processors/          # Data preprocessing pipeline
├── models/              # Neural network models
├── datasets/            # PyTorch Dataset wrappers
├── structure/           # CSIData, CSIFrame data structures
├── inference.py         # Prediction interface
├── core.py              # Training pipeline
├── download.py          # Dataset downloading
└── cli.py               # Command-line interface
```

## 🧪 Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=wsdp --cov-report=html
```

## 📄 License

MIT License

---

<a id="中文"></a>

# WSDP - Wi-Fi 感知数据处理库

**用于下载、处理、分析和训练 Wi-Fi CSI（信道状态信息）数据的 Python 库。**

## ✨ 功能特性

- **多数据集支持**：Widar、Gait、XRF55、ElderAL、ZTE
- **智能预处理**：小波去噪、相位校准、信号重采样
- **深度学习流程**：端到端 CNN + Transformer 训练
- **认证方式**：JWT Token、邮箱密码、非交互模式
- **可视化工具**：热力图、去噪对比、相位校准图
- **推理接口**：简洁的 `predict()` API
- **命令行工具**：支持批量操作

## ⚡ 快速上手

```python
from wsdp import pipeline, download

# 下载数据集（JWT 认证）
download('widar', '/data/widar', token='your-jwt-token')

# 运行完整流程
pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
)
```

```bash
# CLI 使用
wsdp download widar /data --token YOUR_JWT_TOKEN
wsdp run /data /output widar
wsdp list --verbose
```

## 📝 超参数配置

支持通过函数参数或 YAML 文件覆盖默认超参数：

```python
pipeline(
    input_path='/data',
    output_folder='/output',
    dataset='widar',
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
    config_file='config.yaml',  # 可选
)
```

```yaml
# config.yaml
widar:
  batch: 64
  lr: 0.001
  num_epochs: 50
```

## 📚 Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Changelog](CHANGELOG.md) - Version history

## 📖 Citation

If you use WSDP in your research, please cite:

```bibtex
@software{wsdp2026,
  author = {Cui, Yuanhao and WSDP Team},
  title = {WSDP: Wi-Fi Sensing Data Processing},
  url = {https://github.com/sdp-team/wsdp},
  version = {0.2.0},
  year = {2026},
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape WSDP
- Inspired by the wireless sensing research community
- Built with PyTorch, NumPy, and other open-source tools

