# Widar3.0

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=028828f9-1997-48df-895c-9724551a22ae)

## Overview

**Widar3.0** is a Wi-Fi-based hand gesture recognition dataset collected with Intel IWL5300 NICs.

| Property | Value |
|----------|-------|
| **Format** | .dat (bfee binary) |
| **Subcarriers** | 30 |
| **Complex** | ✅ |
| **Reader** | `BfeeReader` |
| **Classes** | 6 gestures |
| **Samples** | 12,000 |
| **Size** | ~2GB |

## Raw Data

Flat `.dat` files named `user{user_id}-{gesture}-{torso_position}-{orientation}-{serial}-r{receiver}.dat`.
Each file is a stream of Intel 5300 bfee records; every `0xBB` record carries one
`(30, n_rx, n_tx)` complex CSI matrix (8-bit I/Q) plus RSSI/noise/AGC metadata.

## After the Reader

One `CSIData` per file; frames are `BfeeFrame` (`timestamp`, `(30, n_rx*n_tx)` complex64
`csi_array`, plus `n_rx/n_tx/rssi_a/b/c/noise/agc/...`). `to_numpy()` → `(T, 30, 3)`
complex64 typically (3 Rx × 1 Tx; T ≈ 1–3k, e.g. `(1934, 30, 3)`).

**Label** = `gesture`; **group** = `torso_position*1000 + orientation*100 + receiver`
(condition-based split).

## Usage

```bash
wsdp download widar ./data --email you@example.com --password yourpassword
wsdp run ./data/widar ./output widar
```

```python
from wsdp import pipeline
pipeline('./data/widar', './output', 'widar')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
