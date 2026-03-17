"""
Feature extraction algorithms for CSI data.

Provides spectral, statistical, and tensor-based features for activity
recognition, localization, and other CSI-based applications.
"""
import numpy as np
from scipy.signal import stft


def doppler_spectrum(csi, n_fft=64, hop_length=32):
    """
    Compute Doppler spectrum from CSI time series using STFT.

    The Doppler spectrum captures frequency components of CSI amplitude
    variations, which correspond to object motion velocities via the
    Doppler effect. High energy at low frequencies = slow/stationary,
    high energy at higher frequencies = fast motion.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex
        n_fft: FFT size for STFT (default: 64)
        hop_length: Hop size between STFT windows (default: 32)

    Returns:
        np.ndarray: Doppler spectrogram of shape (n_freq, n_time, F[, A])
            where n_freq = n_fft//2 + 1

    Reference:
        Ali K, et al. "Keystroke Recognition Using WiFi Signals."
        ACM MobiCom, 2015.
        Wang W, et al. "Understanding and Modeling of WiFi Signal Based 
        Human Activity Recognition." ACM MobiCom, 2015.
    """
    if csi.size == 0:
        return np.array([])
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    amplitude = np.abs(csi)

    if amplitude.ndim == 2:
        # (T, F)
        T, F = amplitude.shape
        n_freq = n_fft // 2 + 1
        # Compute STFT along time axis for each subcarrier
        _, _, Zxx = stft(amplitude[:, 0], nperseg=n_fft, noverlap=n_fft - hop_length)
        n_time = Zxx.shape[1]
        spectrum = np.zeros((n_freq, n_time, F))
        for f in range(F):
            _, _, Zxx = stft(amplitude[:, f], nperseg=n_fft, noverlap=n_fft - hop_length)
            spectrum[:, :, f] = np.abs(Zxx)
    elif amplitude.ndim == 3:
        # (T, F, A)
        T, F, A = amplitude.shape
        n_freq = n_fft // 2 + 1
        _, _, Zxx = stft(amplitude[:, 0, 0], nperseg=n_fft, noverlap=n_fft - hop_length)
        n_time = Zxx.shape[1]
        spectrum = np.zeros((n_freq, n_time, F, A))
        for f in range(F):
            for a in range(A):
                _, _, Zxx = stft(amplitude[:, f, a], nperseg=n_fft, noverlap=n_fft - hop_length)
                spectrum[:, :, f, a] = np.abs(Zxx)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return spectrum


def entropy_features(csi, bins=50):
    """
    Compute information entropy features from CSI amplitude distribution.

    The Shannon entropy of CSI amplitude distribution quantifies the
    randomness/uncertainty in the signal. Higher entropy = more diverse
    amplitude values (complex environment), lower entropy = concentrated
    values (stable environment).

    Computes per-subcarrier entropy along the time axis.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        bins: Number of histogram bins for entropy estimation (default: 50)

    Returns:
        np.ndarray: Entropy values of shape (F,) or (F, A)

    Reference:
        Shannon CE. "A Mathematical Theory of Communication."
        Bell System Technical Journal, 1948.
        Wang H, et al. "Human Activity Recognition Using CSI Entropy 
        Features." IEEE Access, 2019.
    """
    if csi.size == 0:
        return np.array([])
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")

    amplitude = np.abs(csi)

    def _compute_entropy(signal_1d):
        hist, _ = np.histogram(signal_1d, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-12))

    if amplitude.ndim == 2:
        # (T, F) -> (F,)
        T, F = amplitude.shape
        entropy = np.zeros(F)
        for f in range(F):
            entropy[f] = _compute_entropy(amplitude[:, f])
    elif amplitude.ndim == 3:
        # (T, F, A) -> (F, A)
        T, F, A = amplitude.shape
        entropy = np.zeros((F, A))
        for f in range(F):
            for a in range(A):
                entropy[f, a] = _compute_entropy(amplitude[:, f, a])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    return entropy


def csi_ratio(csi, antenna_pairs=None):
    """
    Compute CSI ratio between antenna pairs.

    The ratio between CSI from two antennas eliminates phase errors
    that are common to both antennas (e.g., CFO), while preserving
    the relative phase/amplitude differences caused by multipath.
    This is useful for spatial diversity-based sensing.

    Args:
        csi: CSI array of shape (T, F, A) — must have A >= 2
        antenna_pairs: List of (ant1, ant2) tuples. If None, uses
            consecutive pairs: [(0,1), (1,2), ...]

    Returns:
        np.ndarray: CSI ratios with shape (T, F, n_pairs) where n_pairs
            is the number of antenna pairs

    Reference:
        Halperin D, et al. "Tool release: Gathering 802.11n traces with 
        channel state information." ACM SIGCOMM CCR, 2011.
        Xiong J, Jamieson K. "ArrayTrack: A Fine-grained Indoor Location 
        System." USENIX NSDI, 2013.
    """
    if csi.size == 0:
        return np.array([])
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")

    T, F, A = csi.shape

    if A < 2:
        raise ValueError(f"Need at least 2 antennas for ratio, got {A}")

    if antenna_pairs is None:
        antenna_pairs = [(i, i + 1) for i in range(A - 1)]

    n_pairs = len(antenna_pairs)
    result = np.zeros((T, F, n_pairs), dtype=np.complex128)

    for idx, (ant1, ant2) in enumerate(antenna_pairs):
        if ant1 >= A or ant2 >= A or ant1 < 0 or ant2 < 0:
            raise ValueError(f"Antenna index out of range [0, {A}): ({ant1}, {ant2})")
        denom = csi[:, :, ant2]
        # Avoid division by zero
        denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        result[:, :, idx] = csi[:, :, ant1] / denom_safe

    return result


def tensor_decomposition(csi, rank=10, method='cp'):
    """
    Decompose CSI tensor using CP or Tucker decomposition.

    Tensor decomposition extracts latent factors from multi-dimensional
    CSI data, capturing temporal, frequency, and spatial patterns
    simultaneously. Useful for feature dimensionality reduction and
    noise removal.

    Args:
        csi: CSI array of shape (T, F, A) — complex or real
        rank: Rank of the decomposition (number of components) (default: 10)
        method: Decomposition method
            - 'cp': Canonical Polyadic (CP) decomposition
            - 'tucker': Tucker decomposition

    Returns:
        np.ndarray: Reconstructed low-rank CSI tensor of same shape as input

    Reference:
        Kolda TG, Bader BW. "Tensor Decompositions and Applications."
        SIAM Review, vol. 51, no. 3, pp. 455-500, 2009.
        For CSI: Wang X, et al. "Tensor-Based Low-Rank 
        Representation for WiFi Sensing." IEEE IoT Journal, 2022.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")

    valid_methods = ('cp', 'tucker')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")

    T, F, A = csi.shape
    rank = min(rank, T, F, A)

    # Use real part for decomposition (handle complex by concatenating)
    if np.iscomplexobj(csi):
        # Stack real and imaginary as additional "channels" in a 4th dimension
        # then do 3D decomposition on each
        real_decomp = _simple_cp_decomposition(np.real(csi), rank)
        imag_decomp = _simple_cp_decomposition(np.imag(csi), rank)

        if method == 'cp':
            reconstructed = real_decomp['reconstructed'] + 1j * imag_decomp['reconstructed']
        else:  # tucker
            reconstructed = real_decomp['reconstructed'] + 1j * imag_decomp['reconstructed']
        return reconstructed
    else:
        return _simple_cp_decomposition(csi, rank) if method == 'cp' \
            else _simple_tucker_decomposition(csi, rank)


def _simple_cp_decomposition(tensor, rank):
    """Simple CP decomposition using SVD-based ALS initialization."""
    T, F, A = tensor.shape
    rank = min(rank, T, F, A)

    # Use HOSVD-like initialization then reconstruct
    # Factor matrices via SVD unfolding
    U1 = _svd_factor(tensor.reshape(T, -1), rank)  # temporal
    U2 = _svd_factor(np.transpose(tensor, (1, 0, 2)).reshape(F, -1), rank)  # freq
    U3 = _svd_factor(np.transpose(tensor, (2, 0, 1)).reshape(A, -1), rank)  # spatial

    # Khatri-Rao product for CP reconstruction
    # Z = sum_r lambda_r * u1_r ⊗ u2_r ⊗ u3_r
    weights = np.ones(rank)
    reconstructed = np.zeros_like(tensor)
    for r in range(rank):
        outer = np.outer(U1[:, r], U2[:, r])
        reconstructed += weights[r] * np.einsum('ij,k->ijk', outer, U3[:, r])

    return {
        'weights': weights,
        'factors': [U1, U2, U3],
        'reconstructed': reconstructed
    }


def _simple_tucker_decomposition(tensor, rank):
    """Simple Tucker decomposition using HOSVD."""
    T, F, A = tensor.shape
    rank_t = min(rank, T)
    rank_f = min(rank, F)
    rank_a = min(rank, A)

    U1 = _svd_factor(tensor.reshape(T, -1), rank_t)
    U2 = _svd_factor(np.transpose(tensor, (1, 0, 2)).reshape(F, -1), rank_f)
    U3 = _svd_factor(np.transpose(tensor, (2, 0, 1)).reshape(A, -1), rank_a)

    # Core tensor: G = tensor ×1 U1^T ×2 U2^T ×3 U3^T
    temp = np.einsum('ijk,ia->ajk', tensor, U1)
    temp = np.einsum('ajk,jb->abk', temp, U2)
    core = np.einsum('abk,kc->abc', temp, U3)

    # Reconstruct: ≈ G ×1 U1 ×2 U2 ×3 U3
    temp = np.einsum('abc,ia->ibc', core, U1)
    temp = np.einsum('ibc,jb->ijc', temp, U2)
    reconstructed = np.einsum('ijc,kc->ijk', temp, U3)

    return {
        'core': core,
        'factors': [U1, U2, U3],
        'reconstructed': reconstructed
    }


def _svd_factor(matrix, rank):
    """Extract top-r left singular vectors."""
    rank = min(rank, min(matrix.shape))
    U, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank]
