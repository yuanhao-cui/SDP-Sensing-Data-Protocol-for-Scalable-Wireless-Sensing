# GaitID

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=87a65da2-18cb-4b8f-a1ec-c9696890172b)

## Overview

**GaitID** is a Wi-Fi-based human gait recognition dataset for identity verification through walking patterns.

| Property | Value |
|----------|-------|
| **Format** | .dat (bfee binary) |
| **Subcarriers** | 30 |
| **Complex** | ✅ |
| **Reader** | `BfeeReader` |
| **Classes** | 11 users |
| **Samples** | 22,500 |
| **Size** | ~1GB |

## Raw Data

Flat `.dat` files named `user{user_id}-{track_id}-{repetition}-r{receiver}.dat`.
Same bfee binary format as [Widar](widar.md).

## After the Reader

One `CSIData` per file; frames are `BfeeFrame` with `(30, n_rx*n_tx)` complex64
`csi_array`. `to_numpy()` → `(T, 30, 3)` complex64 typically (3 Rx × 1 Tx, e.g.
`(2917, 30, 3)`).

**Label** = `user_id`; **group** = `track_id*100 + receiver` (held-out condition split).

## Usage

```bash
wsdp download gait ./data --email you@example.com --password yourpassword
wsdp run ./data/gait ./output gait
```

```python
from wsdp import pipeline
pipeline('./data/gait', './output', 'gait')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
