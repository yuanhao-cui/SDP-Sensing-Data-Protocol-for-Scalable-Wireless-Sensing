# GaitID

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=87a65da2-18cb-4b8f-a1ec-c9696890172b)

## Overview

**GaitID** is a Wi-Fi-based human gait recognition dataset for identity verification through walking patterns.

| Property | Value |
|----------|-------|
| **Format** | .dat (bfee) |
| **Subcarriers** | 30 |
| **CSI Shape** | (Time, 30, 1, 3) |
| **Complex** | ✅ |
| **Classes** | 11 gait patterns |
| **Samples** | 22,500 |
| **Size** | ~1GB |

## Usage

```bash
wsdp download gait ./data
wsdp run ./data/gait ./output gait
```

```python
from wsdp import pipeline
pipeline('./data/gait', './output', 'gait')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
