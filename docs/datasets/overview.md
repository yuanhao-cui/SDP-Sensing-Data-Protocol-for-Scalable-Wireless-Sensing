# Datasets Overview

WSDP supports 5 built-in datasets for wireless sensing research, all hosted and maintained on **[SDP8.org](https://sdp8.org)** - the official SDP platform.

> Browse all datasets: [sdp8.org](https://sdp8.org) | Download via CLI: `wsdp download <dataset> ./data`

| Dataset | Format | Subcarriers | Complex | Scenarios | Size | Hardware |
|---------|--------|-------------|---------|-----------|------|----------|
| Widar | .dat (bfee) | 30 | Yes | Gesture recognition | ~2GB | Intel 5300 NIC |
| Gait | .dat (bfee) | 30 | Yes | Gait recognition | ~1GB | Intel 5300 NIC |
| XRF55 | .npy | 30 | Yes | Human activity | ~3GB | Intel 5300 NIC |
| ElderAL | .csv | 512 | No | Elderly activity | ~867MB | Commercial AP |
| ZTE | .csv | 512 | Yes | CSI with I/Q | ~4GB | ZTE 5G platform |

**Note on ElderAL**: v0.4.0 includes a subcarrier mapping fix for the ElderAL dataset. The dataset contains 512 subcarriers (not 30), and the loader now correctly maps them to the canonical grid. If you used ElderAL with a previous version, re-download or re-run preprocessing to apply the fix.

## Download

> **Authentication**: All datasets require a free **[SDP8.org](https://sdp8.org)** account.

```bash
# With email/password
wsdp download elderAL ./data --email you@example.com --password yourpassword

# With JWT token
wsdp download elderAL ./data --token YOUR_JWT_TOKEN

# From Python
from wsdp import download
download('widar', './data', email='you@example.com', password='yourpassword')
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
