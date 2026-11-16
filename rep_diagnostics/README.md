# rep_diagnostics — 表三表征层诊断（CKA / MMD²）计算脚本

对应论文 *SDP: A Unified Protocol and Benchmarking Framework for Reproducible
Wi-Fi Sensing*（IEEE TMC, 2026）Table III 的两项指标：

| 指标 | 含义 | 统计口径 |
|---|---|---|
| CKA | 潜在表征跨随机种子的可复现性 | 5 个种子两两配对（C(5,2)=10 对）线性 CKA 的均值 ± 标准差 |
| MMD² | 类别平衡的 train–test 表征差异 | 每类等数采样（RBF 核 + median heuristic 带宽），每个种子算一个值后取均值 ± 标准差 |

纯 numpy 实现，指标计算不需要 GPU / torch。

## 文件结构

```
rep_diagnostics/
├── metrics.py             # CKA、MMD² 核心计算（纯 numpy）
├── feature_extraction.py  # 从 checkpoint 提取潜在特征（惰性 import torch，可 CPU 运行）
├── run_analysis.py        # 端到端流水线：逐种子特征 .npz -> 表三的一行
├── tests/                 # unittest 测试（TDD，20 个用例）
│   ├── test_metrics.py
│   └── test_run_analysis.py
└── results/               # 结果输出目录
    ├── diagnostics.json
    └── diagnostics.csv    # 注意：当前内容为合成数据演示输出，非真实实验结果
```

## 在服务器上的使用流程（有 checkpoint 的设备）

**1. 提取特征**（在存有 checkpoint 和数据的服务器上；CPU 即可）：

```python
from feature_extraction import save_seed_features
import torch

model = MyModel(...)  # 与训练时一致的结构
for seed in [543, 785, 530, 558, 805]:
    model.load_state_dict(torch.load(
        f"./checkpoints/best_checkpoint_{seed}_1.pth",
        map_location="cpu"))
    save_seed_features(
        checkpoint_path=None, model=model,
        data_by_split={"test": (test_data, test_labels),
                       "train": (train_data, train_labels)},
        out_dir="./features", seed=seed, device="cpu")
```

注意：5 个种子的 `X_test` 必须是**同一批评测样本**（取各种子测试集交集，
或在全数据集上计算），否则 CKA 无意义。

**2. 计算指标**（任意机器，拷贝 features_seed*.npz 即可）：

```bash
python rep_diagnostics/run_analysis.py \
    --features-dir ./features \
    --out-dir rep_diagnostics/results \
    --seeds 543 785 530 558 805 --n-per-class 10
```

输出 `diagnostics.json` / `diagnostics.csv`：`cka_mean, cka_std, mmd2_mean,
mmd2_std` —— 即表三某一（数据集 × 骨干）行。

## 运行测试

```bash
python -m unittest discover -s rep_diagnostics/tests -v
```

当前 20 个用例全部通过。测试只依赖 numpy，可在无 torch 的环境运行。
