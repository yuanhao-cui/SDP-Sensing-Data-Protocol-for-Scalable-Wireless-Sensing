# XRF55

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=705e08e7-637e-49a1-aff1-b2f9644467ae)

## Overview

**XRF55** is a radio frequency dataset for human indoor action analysis with 55 activity categories.

| Property | Value |
|----------|-------|
| **Formats** | .npy (SDP8) / .dat (Kaggle raw) |
| **Subcarriers** | 30 |
| **Complex** | .dat: ✅ (I/Q) · .npy: ❌ (real-valued) |
| **Reader** | `XrfReader` |
| **Classes** | 55 activities |
| **Samples** | 9,900 |
| **Size** | ~3GB |

## Raw Data

- `.npy`: real-valued `(270, 1000)` array (270 = 3 Rx × 3 Ant × 30 subcarriers flattened,
  1000 time steps), reshaped internally to (rx, subcarrier, antenna, time).
- `.dat` (Kaggle): int16 binary — 40-value header + 199 packets × 270 complex I/Q values;
  paths follow `Scene_X/{lb|nb}/YY_AA_BB.dat` (parsed into `CSIData._xrf55_labels`).

## After the Reader

One `CSIData` per file (all 3 receivers kept in a single sample); frames are `BaseFrame`
with `(30, 9)` `csi_array` (subcarrier × 3 Rx × 3 Ant). `to_numpy()` → `(1000, 30, 9)`
float64 for `.npy`, `(199, 30, 9)` complex64 for `.dat`.

Filenames follow `user_action_trial` (e.g. `03_01_01`): **label** = `action`,
**group** = `trial`. Default split is the official repetition protocol — train 1–12,
val 13–16, test 17–20 (repetition-disjoint `GroupShuffleSplit` fallback on trial subsets).

## Usage

```bash
wsdp download xrf55 ./data --email you@example.com --password yourpassword
wsdp run ./data/xrf55 ./output xrf55
```

```python
from wsdp import pipeline
pipeline('./data/xrf55', './output', 'xrf55')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
