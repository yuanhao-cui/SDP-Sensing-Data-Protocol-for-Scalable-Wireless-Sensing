# SDP: Sensing Data Protocol for Scalable Wireless Sensing

<div align="center">

[![SDP Website](https://img.shields.io/badge/SDP_Website-Click_here-356596)](https://sdp8.org/)
[![PyPI](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/yuanhao-cui/Sensing-Data-Protocol/refs/heads/main/pyproject.toml&query=%24.project.name&logo=pypi&label=pip)](https://pypi.org/project/wsdp/)
[![License](https://img.shields.io/github/license/yuanhao-cui/Sensing-Data-Protocol?color=green)](https://github.com/yuanhao-cui/Sensing-Data-Protocol/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-pytest-blueviolet)](https://docs.pytest.org)
[![Docs](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://sdp-team.github.io/wsdp)
[![Colab](https://img.shields.io/badge/Colab-Tutorial-yellow.svg)](https://colab.research.google.com/github/sdp-team/wsdp/blob/main/examples/wsdp_tutorial.ipynb)

**[English](#english) | [中文](#中文)**

</div>

---

## 📖 Citation

If you use SDP in your research, please cite:

```bibtex
@software{wsdp2026,
  author = {Cui, Yuanhao and WSDP Team},
  title = {SDP: Sensing Data Protocol for Scalable Wireless Sensing},
  url = {https://github.com/yuanhao-cui/Sensing-Data-Protocol-for-Scalable-Wireless-Sensing},
  version = {0.2.0},
  year = {2026},
}
```

---

<a id="english"></a>
# 🇬🇧 English

## 🎯 What is SDP?

SDP is a **protocol-level abstraction** and unified benchmark for **reproducible wireless sensing**.

> ⚠️ **SDP is not a new neural network**, but a standardized protocol that unifies CSI representations for fair comparison.

### Key Principles

Instead of improving accuracy through hidden preprocessing tricks, SDP ensures that:

- ✅ Every dataset follows the same sanitization rules
- ✅ Every model receives the same canonical tensor  
- ✅ Every experiment is reproducible

SDP acts as a **protocol-level middleware** between raw CSI and learning models.

### Performance Highlights

<div align="center">

| Metric | Result |
|:------:|:------:|
| **Accuracy** | State-of-the-art on 5 datasets |
| **Reproducibility** | 5-seed evaluation standard |
| **Stability** | Low variance across runs |

![Accuracy](./img/accuracy.png)
*Figure 1: Accuracy comparison across datasets*

![Reproducibility](./img/reproducibility_and_stability.png)
*Figure 2: Reproducibility and stability analysis*

![Ablation](./img/ablation_rank.png)
*Figure 3: Ablation study results*

</div>

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install

```bash
pip install wsdp
```

### Step 2: Download Dataset

Download from [SDP Website](https://sdp8.org/) or use CLI:

```bash
# Recommended: elderAL (smallest, fastest for testing)
wsdp download elderAL ./data

# Or other datasets: widar, gait, xrf55, zte
wsdp download widar ./data
```

**Dataset Organization:**
```
data/
├── elderAL/
│   ├── action0_static_new/
│   │   ├── user0_position1_activity0/
│   │   └── ...
│   └── action1_walk_new/
├── widar/
├── gait/
├── xrf55/
└── zte/
```

### Step 3: Train & Evaluate

**Python API:**
```python
from wsdp import pipeline

pipeline("./data/elderAL", "./output", "elderAL")
```

**CLI:**
```bash
wsdp run ./data/elderAL ./output elderAL
```

**With Custom Hyperparameters:**
```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50 --batch-size 64
```

**Output Files:**
- `best_model.pth` - Trained model checkpoint
- `confusion_matrix.png` - Evaluation visualization
- `output.log` - Training logs

---

## 🔬 Research & Modification

### Plug in Your Own Model

Create `custom_model.py`:
```python
import torch
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # x shape: (Batch, Timestamp, Frequency, Antenna)
        # Your forward pass
        return output

# Required: expose model class
model = YourCustomModel
```

Run with custom model:
```bash
wsdp run ./data/elderAL ./output elderAL -m custom_model.py
```

### Use Your Own Dataset

Organize your data:
```
data/
└── my_dataset/
    ├── user0_pos0_action0/
    │   ├── sample1.csv
    │   └── ...
    └── user0_pos0_action1/
        └── ...
```

Then run:
```bash
wsdp run ./data/my_dataset ./output my_dataset
```

---

## 📚 Documentation

- [Full Documentation](https://sdp-team.github.io/wsdp)
- [API Reference](docs/API_REFERENCE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<a id="中文"></a>
# 🇨🇳 中文

## 🎯 SDP 是什么？

SDP 是一个**协议级抽象**框架，用于**可复现的无线感知研究**。

> ⚠️ **SDP 不是一个新的神经网络**，而是一个标准化协议，统一 CSI 表示以实现公平比较。

### 核心原则

SDP 不通过隐藏预处理技巧来提高准确率，而是确保：

- ✅ 每个数据集遵循相同的清洗规则
- ✅ 每个模型接收相同的规范张量
- ✅ 每个实验都是可复现的

SDP 作为原始 CSI 和学习模型之间的**协议级中间件**。

### 性能亮点

<div align="center">

| 指标 | 结果 |
|:----:|:----:|
| **准确率** | 5 个数据集上达到 SOTA |
| **可复现性** | 5 种子评估标准 |
| **稳定性** | 多次运行方差低 |

![准确率](./img/accuracy.png)
*图 1：跨数据集准确率对比*

![可复现性](./img/reproducibility_and_stability.png)
*图 2：可复现性与稳定性分析*

![消融实验](./img/ablation_rank.png)
*图 3：消融实验结果*

</div>

---

## 🚀 快速开始（3 步）

### 第 1 步：安装

```bash
pip install wsdp
```

### 第 2 步：下载数据集

从 [SDP 官网](https://sdp8.org/) 下载或使用 CLI：

```bash
# 推荐：elderAL（最小，测试最快）
wsdp download elderAL ./data

# 或其他数据集：widar、gait、xrf55、zte
wsdp download widar ./data
```

**数据集组织：**
```
data/
├── elderAL/
│   ├── action0_static_new/
│   │   ├── user0_position1_activity0/
│   │   └── ...
│   └── action1_walk_new/
├── widar/
├── gait/
├── xrf55/
└── zte/
```

### 第 3 步：训练与评估

**Python API：**
```python
from wsdp import pipeline

pipeline("./data/elderAL", "./output", "elderAL")
```

**命令行：**
```bash
wsdp run ./data/elderAL ./output elderAL
```

**自定义超参数：**
```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50 --batch-size 64
```

**输出文件：**
- `best_model.pth` - 训练好的模型检查点
- `confusion_matrix.png` - 评估可视化
- `output.log` - 训练日志

---

## 🔬 研究与修改

### 接入你自己的模型

创建 `custom_model.py`：
```python
import torch
import torch.nn as nn

class YourCustomModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # 你的架构代码
        
    def forward(self, x):
        # x 形状: (Batch, Timestamp, Frequency, Antenna)
        # 你的前向传播
        return output

# 必需：暴露模型类
model = YourCustomModel
```

使用自定义模型运行：
```bash
wsdp run ./data/elderAL ./output elderAL -m custom_model.py
```

### 使用你自己的数据集

组织你的数据：
```
data/
└── my_dataset/
    ├── user0_pos0_action0/
    │   ├── sample1.csv
    │   └── ...
    └── user0_pos0_action1/
        └── ...
```

然后运行：
```bash
wsdp run ./data/my_dataset ./output my_dataset
```

---

## 📚 文档

- [完整文档](https://sdp-team.github.io/wsdp)
- [API 参考](docs/API_REFERENCE.md)
- [贡献指南](CONTRIBUTING.md)
- [更新日志](CHANGELOG.md)

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

---

## 📄 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🗺️ Roadmap

- [x] 5 个数据集支持 (Widar, Gait, XRF55, ElderAL, ZTE)
- [x] 标准化预处理流程
- [x] CLI 工具
- [ ] 更多数据集（持续添加中）
- [ ] 在线演示平台
- [ ] PyPI 发布

---

<div align="center">

**Made with ❤️ by the WSDP Team**

</div>
