# Datasets Overview

WSDP supports 5 built-in datasets for wireless sensing research, all hosted and maintained on **[SDP8.org](https://sdp8.org)** - the official SDP platform.

> 🌐 **Browse all datasets**: [sdp8.org](https://sdp8.org) | 📥 **Download via CLI**: `wsdp download <dataset> ./data`

| Dataset | Format | Subcarriers | Complex | Scenarios | Size |
|---------|--------|-------------|---------|-----------|------|
| Widar | .dat (bfee) | 30 | ✅ | Gesture recognition | ~2GB |
| Gait | .dat (bfee) | 30 | ✅ | Gait recognition | ~1GB |
| XRF55 | .npy | 30 | ✅ | Human activity | ~3GB |
| ElderAL | .csv | varies | ❌ | Elderly activity | ~500MB |
| ZTE | .csv | 512 | ✅ | CSI with I/Q | ~4GB |

## Download

```bash
# From CLI
wsdp download elderAL ./data

# From Python
from wsdp import download
download('widar', './data', token='your-jwt-token')
```

## Dataset Structure

Organize data as:
```
data/
├── elderAL/
│   ├── action0_static_new/
│   │   ├── user0_position1_activity0/
│   │   └── ...
│   └── action1_walk_new/
└── ...
```
