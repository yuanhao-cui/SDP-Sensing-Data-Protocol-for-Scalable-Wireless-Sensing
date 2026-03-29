"""
OFDM subcarrier index mappings for supported WiFi hardware platforms.

The subcarrier indices represent the actual OFDM tone positions in the
frequency domain. These are NOT sequential 0..N-1 integers — they follow
the IEEE 802.11 standard's subcarrier numbering relative to the DC tone.

Reference:
    IEEE 802.11n-2009, Table 7-25f (feedback subcarrier grouping).
    Halperin D, et al. "Tool release: Gathering 802.11n traces with
    channel state information." ACM SIGCOMM CCR, 2011.
"""
import numpy as np
from typing import Optional

# ============================================================================
# Intel IWL5300 — 802.11n, 20 MHz, Ng=2 (30 subcarriers out of 56)
# Source: IEEE 802.11n-2009 Table 7-25f
# The 56 OFDM subcarriers span indices -28 to +28 (excluding DC at 0).
# With grouping Ng=2, every 2nd subcarrier is reported, except near DC
# where the spacing is irregular: ..., -2, -1, +1, +3, ...
# ============================================================================
IWL5300_SUBCARRIER_INDICES_20MHZ = np.array([
    -28, -26, -24, -22, -20, -18, -16, -14, -12, -10,
     -8,  -6,  -4,  -2,  -1,   1,   3,   5,   7,   9,
     11,  13,  15,  17,  19,  21,  23,  25,  27,  28
], dtype=np.float64)

# ============================================================================
# Intel IWL5300 — 802.11n, 40 MHz, Ng=4 (30 subcarriers out of 114)
# Source: IEEE 802.11n-2009 Table 7-25f
# ============================================================================
IWL5300_SUBCARRIER_INDICES_40MHZ = np.array([
    -58, -54, -50, -46, -42, -38, -34, -30, -26, -22,
    -18, -14, -10,  -6,  -2,   2,   6,  10,  14,  18,
     22,  26,  30,  34,  38,  42,  46,  50,  54,  58
], dtype=np.float64)

# ============================================================================
# Dataset → subcarrier index mapping registry
# ============================================================================
_SUBCARRIER_REGISTRY = {
    'widar': IWL5300_SUBCARRIER_INDICES_20MHZ,
    'gait': IWL5300_SUBCARRIER_INDICES_20MHZ,
    'xrf55': IWL5300_SUBCARRIER_INDICES_20MHZ,
    # ElderAL and ZTE use 512 subcarriers with approximately uniform spacing.
    # Exact OFDM indices depend on the router firmware; we use centered indices.
    'elderAL': None,  # → will generate uniform indices
    'zte': None,       # → will generate uniform indices
}


def get_subcarrier_indices(dataset: Optional[str] = None,
                           num_subcarriers: Optional[int] = None) -> np.ndarray:
    """
    Get the OFDM subcarrier indices for a dataset or hardware platform.

    For Intel IWL5300-based datasets (widar, gait, xrf55), returns the
    non-uniform 30-element index array from IEEE 802.11n Table 7-25f.
    For other datasets, generates centered uniform indices.

    Args:
        dataset: Dataset name. If provided, looks up the registry.
        num_subcarriers: Number of subcarriers. Used as fallback when
            dataset is None or not in registry.

    Returns:
        np.ndarray: 1D array of subcarrier indices (float64).

    Raises:
        ValueError: If neither dataset nor num_subcarriers is provided.

    Reference:
        IEEE 802.11n-2009, Table 7-25f.
        Halperin D, et al. ACM SIGCOMM CCR, 2011.
    """
    if dataset is not None and dataset in _SUBCARRIER_REGISTRY:
        indices = _SUBCARRIER_REGISTRY[dataset]
        if indices is not None:
            return indices.copy()
        # Registry entry is None → generate uniform for this dataset
        if num_subcarriers is None:
            raise ValueError(
                f"Dataset '{dataset}' requires num_subcarriers to generate "
                f"uniform indices (subcarrier count is hardware-dependent)."
            )

    if num_subcarriers is not None:
        # Generate centered uniform indices: [-N/2, ..., N/2-1]
        return np.arange(num_subcarriers, dtype=np.float64) - num_subcarriers / 2.0

    raise ValueError(
        "Either 'dataset' or 'num_subcarriers' must be provided."
    )
