# Datasets Overview

WSDP supports 5 built-in datasets for wireless sensing research, all hosted and maintained on **[SDP8.org](https://sdp8.org)** - the official SDP platform.

> Browse all datasets: [sdp8.org](https://sdp8.org) | Download via CLI: `wsdp download <dataset> ./data`

| Dataset | Format | Subcarriers | Complex | Scenarios | Size | Hardware |
|---------|--------|-------------|---------|-----------|------|----------|
| Widar | .dat (bfee) | 30 | Yes | Gesture recognition | ~2GB | Intel 5300 NIC |
| Gait | .dat (bfee) | 30 | Yes | Gait recognition | ~1GB | Intel 5300 NIC |
| XRF55 | .npy / .dat (raw) | 30 | .dat: Yes · .npy: amplitude | Human activity | ~3GB | Intel 5300 NIC |
| ElderAL | .csv | 512 | No (amplitude only) | Elderly activity | ~500MB | Commercial AP |
| ZTE | .csv | 512 | Yes | CSI with I/Q | ~4GB | ZTE 5G platform |

ElderAL/ZTE keep their 512-subcarrier resolution through the default pipeline; add an
`interpolate` step (`target_K=30`) to align with the Intel 5300 grid if needed.

## Download

> **Authentication**: Widar, Gait and ElderAL require a free **[SDP8.org](https://sdp8.org)** account (email/password or JWT token). XRF55 raw data is additionally mirrored on Kaggle and is tried first without SDP8 credentials.

```bash
# With email/password
wsdp download elderAL ./data --email you@example.com --password yourpassword

# With JWT token
wsdp download elderAL ./data --token YOUR_JWT_TOKEN

# From Python
from wsdp import download
download('widar', './data', email='you@example.com', password='yourpassword')
```

## From Raw Files to Tensors

Every dataset follows the same loading contract (`wsdp.readers.load_data(path, dataset)`):
a dataset-specific **Reader** parses each raw file into one `CSIData` holding a list of
frames (each with a `timestamp` and a `(F, A)` `csi_array` — subcarriers × antennas);
`CSIData.to_numpy()` stacks frames sorted by timestamp into a `(T, F, A)` tensor. The
processor then runs the configured algorithm steps on it and derives **label** and
**group** from the file/folder name.

| Dataset | Reader | `to_numpy()` output | Label | Group |
|---------|--------|---------------------|-------|-------|
| Widar | `BfeeReader` | (T, 30, 3) complex64 | gesture | torso·1000 + orientation·100 + receiver |
| Gait | `BfeeReader` | (T, 30, 3) complex64 | user_id | track·100 + receiver |
| XRF55 | `XrfReader` | (1000, 30, 9) · .dat: (199, 30, 9) complex64 | action_id | repetition_id |
| ElderAL | `ElderReader` | (T, 512, 3) float64 | activity | position_id |
| ZTE | `ZTEReader` | (T, 512, 3) complex64 | action_id | position_id |
