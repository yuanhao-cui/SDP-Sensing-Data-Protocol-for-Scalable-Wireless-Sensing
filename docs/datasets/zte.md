# ZTE

> 📥 Download: [sdp8.org](https://sdp8.org)

## Overview

**ZTE** is a CSI dataset with I/Q components collected by ZTE Corporation.

| Property | Value |
|----------|-------|
| **Format** | .csv |
| **Subcarriers** | 512 |
| **Complex** | ✅ (I/Q pairs) |
| **Reader** | `ZTEReader` |
| **Size** | ~4GB |

## Raw Data

CSVs nested as `user{u}_pos{p}_action{a}/*.csv`. Columns: `timestamp`, `rx_chain_num`
(e.g. `rx0-...-tx0`), and `csi_i_0..511` / `csi_q_0..511` I/Q values; each row is one
(timestamp, rx chain) measurement.

## After the Reader

One `CSIData` per CSV file; only `tx0` rows are kept, grouped by `timestamp` into frames
of `(512, 3)` complex64 (subcarrier × rx chain, `I + j·Q`). `to_numpy()` → `(T, 512, 3)`
complex-valued.

**Label** = `action`; **group** = `pos` (location-held-out split).

## Usage

```bash
wsdp download zte ./data --email you@example.com --password yourpassword
wsdp run ./data/zte ./output zte
```

```python
from wsdp import pipeline
pipeline('./data/zte', './output', 'zte')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
