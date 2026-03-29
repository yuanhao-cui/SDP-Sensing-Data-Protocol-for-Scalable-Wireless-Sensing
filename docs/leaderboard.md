# Benchmark Leaderboard

WSDP provides a standardized benchmarking framework across all supported datasets. Results are collected through community submissions and verified via automated CI checks.

All reported accuracies use **3+ random seeds** to ensure reproducibility. Models are ranked by mean accuracy (descending).

!!! info "Benchmarks In Progress"
    v0.4.0 fixed critical scientific bugs (subcarrier indices, data leakage, SSM equation, etc.) that invalidate prior results. Official baselines will be re-run on GPU and published here. Use `scripts/benchmark_all_models.py` to run locally. Community submissions welcome!

---

## Widar (Gesture Recognition)

<!-- LEADERBOARD_START:widar -->
| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |
|------|-------|------------------------|------------|-------|--------|
| 1 | VisionTransformerCSI | TBD | ~5.0 | 3 | Official |
| 2 | MambaCSI | TBD | ~3.0 | 3 | Official |
| 3 | ResNet1D | TBD | ~2.0 | 3 | Official |
| 4 | BiLSTMAttention | TBD | ~1.5 | 3 | Official |
| 5 | MLPModel | TBD | ~1.0 | 3 | Official |
<!-- LEADERBOARD_END:widar -->

## Gait (Gait Recognition)

<!-- LEADERBOARD_START:gait -->
| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |
|------|-------|------------------------|------------|-------|--------|
| 1 | VisionTransformerCSI | TBD | ~5.0 | 3 | Official |
| 2 | MambaCSI | TBD | ~3.0 | 3 | Official |
| 3 | ResNet1D | TBD | ~2.0 | 3 | Official |
| 4 | BiLSTMAttention | TBD | ~1.5 | 3 | Official |
| 5 | MLPModel | TBD | ~1.0 | 3 | Official |
<!-- LEADERBOARD_END:gait -->

## XRF55 (Activity Recognition)

<!-- LEADERBOARD_START:xrf55 -->
| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |
|------|-------|------------------------|------------|-------|--------|
| 1 | VisionTransformerCSI | TBD | ~5.0 | 3 | Official |
| 2 | MambaCSI | TBD | ~3.0 | 3 | Official |
| 3 | ResNet1D | TBD | ~2.0 | 3 | Official |
| 4 | BiLSTMAttention | TBD | ~1.5 | 3 | Official |
| 5 | MLPModel | TBD | ~1.0 | 3 | Official |
<!-- LEADERBOARD_END:xrf55 -->

## ElderAL (Elder Activity Recognition)

<!-- LEADERBOARD_START:elderAL -->
| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |
|------|-------|------------------------|------------|-------|--------|
| 1 | VisionTransformerCSI | TBD | ~5.0 | 3 | Official |
| 2 | MambaCSI | TBD | ~3.0 | 3 | Official |
| 3 | ResNet1D | TBD | ~2.0 | 3 | Official |
| 4 | BiLSTMAttention | TBD | ~1.5 | 3 | Official |
| 5 | MLPModel | TBD | ~1.0 | 3 | Official |
<!-- LEADERBOARD_END:elderAL -->

## ZTE (Sensing)

<!-- LEADERBOARD_START:zte -->
| Rank | Model | Accuracy (mean +/- std) | Params (M) | Seeds | Source |
|------|-------|------------------------|------------|-------|--------|
| 1 | VisionTransformerCSI | TBD | ~5.0 | 3 | Official |
| 2 | MambaCSI | TBD | ~3.0 | 3 | Official |
| 3 | ResNet1D | TBD | ~2.0 | 3 | Official |
| 4 | BiLSTMAttention | TBD | ~1.5 | 3 | Official |
| 5 | MLPModel | TBD | ~1.0 | 3 | Official |
<!-- LEADERBOARD_END:zte -->

---

## How to Submit Results

We welcome community benchmark submissions. To submit your results:

### Step 1: Run Your Benchmark

Run your model on one or more WSDP datasets using **at least 3 different random seeds**. Record the mean and standard deviation of accuracy.

### Step 2: Create a Submission File

Create a JSON file following the [submission schema](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/benchmarks/schema.json). Example:

```json
{
  "model": "ResNet1D",
  "dataset": "widar",
  "accuracy_mean": 0.872,
  "accuracy_std": 0.015,
  "seeds": [42, 123, 456],
  "params_M": 2.1,
  "training_config": {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "paper_url": "https://arxiv.org/abs/xxxx.xxxxx",
  "code_url": "https://github.com/your-username/your-repo",
  "submitter": "Your Name",
  "date": "2026-03-29"
}
```

### Step 3: Submit a Pull Request

1. **Fork** the [WSDP repository](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing)
2. Add your JSON file to `benchmarks/submissions/` with the naming convention:
   ```
   benchmarks/submissions/{dataset}_{model}_{submitter}.json
   ```
3. Open a **Pull Request** targeting the `main` branch
4. The CI will automatically verify your submission against the schema
5. A maintainer will review and merge your PR

### Submission Requirements

- **Minimum 3 seeds**: All results must be averaged over at least 3 random seeds
- **Accuracy range**: Mean accuracy must be between 0 and 1
- **Valid dataset**: Must be one of: `widar`, `gait`, `xrf55`, `elderAL`, `zte`
- **Date format**: Must use ISO 8601 format (`YYYY-MM-DD`)
- **Reproducibility**: Including `training_config` and `code_url` is strongly encouraged

!!! tip "Naming Convention"
    Name your submission file as `{dataset}_{model}_{submitter}.json`, for example: `widar_ResNet1D_john.json`
