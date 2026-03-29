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

    # Doppler frequency shift f_d = (1/2π)·dφ/dt resides in the phase
    # of the complex CSI signal. STFT must operate on complex (or phase)
    # data — using amplitude discards the Doppler information entirely.
    # For real-valued input, fall back to amplitude-based STFT.
    if np.iscomplexobj(csi):
        signal = csi
    else:
        signal = np.abs(csi)

    if signal.ndim == 2:
        # (T, F)
        T, F = signal.shape
        _, _, Zxx_first = stft(signal[:, 0], nperseg=min(n_fft, T), noverlap=min(n_fft, T) - hop_length)
        n_freq, n_time = Zxx_first.shape
        spectrum = np.zeros((n_freq, n_time, F))
        spectrum[:, :, 0] = np.abs(Zxx_first)
        for f in range(1, F):
            _, _, Zxx = stft(signal[:, f], nperseg=min(n_fft, T), noverlap=min(n_fft, T) - hop_length)
            if Zxx.shape == (n_freq, n_time):
                spectrum[:, :, f] = np.abs(Zxx)
            else:
                z = np.abs(Zxx)
                if z.shape[0] < n_freq:
                    z = np.pad(z, ((0, n_freq - z.shape[0]), (0, 0)))
                if z.shape[1] < n_time:
                    z = np.pad(z, ((0, 0), (0, n_time - z.shape[1])))
                spectrum[:, :, f] = z[:n_freq, :n_time]
    elif signal.ndim == 3:
        # (T, F, A)
        T, F, A = signal.shape
        _, _, Zxx_first = stft(signal[:, 0, 0], nperseg=min(n_fft, T), noverlap=min(n_fft, T) - hop_length)
        n_freq, n_time = Zxx_first.shape
        spectrum = np.zeros((n_freq, n_time, F, A))
        for f in range(F):
            for a in range(A):
                _, _, Zxx = stft(signal[:, f, a], nperseg=min(n_fft, T), noverlap=min(n_fft, T) - hop_length)
                z = np.abs(Zxx)
                if z.shape[0] < n_freq:
                    z = np.pad(z, ((0, n_freq - z.shape[0]), (0, 0)))
                if z.shape[1] < n_time:
                    z = np.pad(z, ((0, 0), (0, n_time - z.shape[1])))
                spectrum[:, :, f, a] = z[:n_freq, :n_time]
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
        # Shannon entropy: H = -Σ p·log2(p), where p is probability mass.
        # density=True returns probability density f(x) where ∫f(x)dx=1,
        # NOT probability mass Σp=1. Using density as probability is wrong.
        counts, _ = np.histogram(signal_1d, bins=bins)
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

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


def tensor_decomposition(csi, rank=10, method='cp', n_iter=0):
    """
    Decompose CSI tensor using HOSVD-based approximation or Tucker decomposition.

    Note: The 'cp' method uses an HOSVD-based rank-1 factor initialization,
    NOT iterative CP-ALS.  When ``n_iter`` > 0, Alternating Least Squares
    (ALS) refinement iterations are applied after the HOSVD initialization
    to improve the CP approximation quality.  With ``n_iter=0`` (default)
    the result is a single-shot HOSVD approximation which is fast but may
    be less accurate than true iterative CP decomposition.

    Args:
        csi: CSI array of shape (T, F, A) — complex or real
        rank: Rank of the decomposition (number of components) (default: 10)
        method: Decomposition method
            - 'cp': HOSVD-based CP approximation (+ optional ALS refinement)
            - 'tucker': Tucker decomposition (HOSVD)
        n_iter: Number of ALS refinement iterations for 'cp' method
            (default: 0 = pure HOSVD, no ALS). Ignored for 'tucker'.

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
        real_decomp = _hosvd_cp_approximation(np.real(csi), rank, n_iter)
        imag_decomp = _hosvd_cp_approximation(np.imag(csi), rank, n_iter)

        if method == 'cp':
            reconstructed = real_decomp['reconstructed'] + 1j * imag_decomp['reconstructed']
        else:  # tucker
            reconstructed = real_decomp['reconstructed'] + 1j * imag_decomp['reconstructed']
        return reconstructed
    else:
        return _hosvd_cp_approximation(csi, rank, n_iter) if method == 'cp' \
            else _simple_tucker_decomposition(csi, rank)


def _hosvd_cp_approximation(tensor, rank, n_iter=0):
    """HOSVD-based CP approximation with optional ALS refinement.

    Initializes factor matrices via truncated SVD of each mode unfolding
    (HOSVD). When n_iter > 0, refines with Alternating Least Squares (ALS).

    Args:
        tensor: 3D array of shape (T, F, A).
        rank: Number of rank-1 components.
        n_iter: Number of ALS refinement iterations (default 0).
    """
    T, F, A = tensor.shape
    rank = min(rank, T, F, A)

    # HOSVD initialization: factor matrices via SVD unfolding
    U1 = _svd_factor(tensor.reshape(T, -1), rank)  # temporal
    U2 = _svd_factor(np.transpose(tensor, (1, 0, 2)).reshape(F, -1), rank)  # freq
    U3 = _svd_factor(np.transpose(tensor, (2, 0, 1)).reshape(A, -1), rank)  # spatial

    # Optional ALS refinement
    for _ in range(n_iter):
        # Mode-0: fix U2, U3, solve for U1
        # Unfold tensor along mode 0: X_(0) = (T, F*A)
        X0 = tensor.reshape(T, F * A)
        # Khatri-Rao product of U3 and U2: (F*A, rank)
        kr_32 = np.einsum('ir,jr->ijr', U2, U3).reshape(F * A, rank)
        U1, _ = np.linalg.qr(X0 @ kr_32)
        U1 = U1[:, :rank]

        # Mode-1: fix U1, U3, solve for U2
        X1 = np.transpose(tensor, (1, 0, 2)).reshape(F, T * A)
        kr_31 = np.einsum('ir,jr->ijr', U1, U3).reshape(T * A, rank)
        U2, _ = np.linalg.qr(X1 @ kr_31)
        U2 = U2[:, :rank]

        # Mode-2: fix U1, U2, solve for U3
        X2 = np.transpose(tensor, (2, 0, 1)).reshape(A, T * F)
        kr_12 = np.einsum('ir,jr->ijr', U1, U2).reshape(T * F, rank)
        U3, _ = np.linalg.qr(X2 @ kr_12)
        U3 = U3[:, :rank]

    # Compute weights: λ_r = tensor contracted with u1_r, u2_r, u3_r
    weights = np.zeros(rank)
    for r in range(rank):
        weights[r] = np.einsum('ijk,i,j,k->', tensor, U1[:, r], U2[:, r], U3[:, r])

    # Reconstruct: Z = sum_r λ_r * u1_r ⊗ u2_r ⊗ u3_r
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


def conjugate_multiply(csi, ref_antenna=0):
    """
    Compute conjugate multiplication between antenna pairs.

    Eliminates common CFO/SFO phase errors shared across antennas by
    computing the channel ratio relative to a reference antenna:

        H_ratio_i = H_i * conj(H_ref) / |H_ref|^2

    This preserves the differential phase caused by multipath propagation
    while canceling transmitter-side phase noise.

    Args:
        csi: CSI array of shape (T, F, A) — must be complex, A >= 2
        ref_antenna: Reference antenna index (default: 0)

    Returns:
        np.ndarray: (T, F, A-1) complex conjugate multiplication results,
            excluding the reference antenna

    Reference:
        Li X, et al. "IndoTrack: Device-Free Indoor Human Tracking with
        Commodity Wi-Fi." Proc. ACM MobiCom, 2017.
    """
    if csi.size == 0:
        return np.array([], dtype=np.complex128)
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")

    T, F, A = csi.shape

    if A < 2:
        raise ValueError(f"Need at least 2 antennas for conjugate multiply, got {A}")
    if not (0 <= ref_antenna < A):
        raise ValueError(
            f"ref_antenna must be in [0, {A}), got {ref_antenna}"
        )
    if not np.iscomplexobj(csi):
        raise ValueError("CSI must be complex-valued for conjugate multiplication")

    # H_ref: (T, F)
    h_ref = csi[:, :, ref_antenna]
    h_ref_conj = np.conj(h_ref)
    # |H_ref|^2 with numerical safety
    h_ref_power = np.abs(h_ref) ** 2
    h_ref_power = np.where(h_ref_power < 1e-20, 1e-20, h_ref_power)

    # Select non-reference antenna indices
    other_antennas = [a for a in range(A) if a != ref_antenna]
    result = np.empty((T, F, len(other_antennas)), dtype=np.complex128)

    for idx, a in enumerate(other_antennas):
        # H_ratio = H_i * conj(H_ref) / |H_ref|^2
        result[:, :, idx] = csi[:, :, a] * h_ref_conj / h_ref_power

    return result


def pca_subcarrier_fusion(csi, n_components=5):
    """
    PCA along subcarrier dimension to extract motion components.

    For each antenna stream, treats (T, F) as a matrix where columns are
    subcarrier time series. After centering, SVD extracts the top-K
    principal components that capture dominant motion patterns.

    The first component typically represents the strongest motion signal,
    while higher components capture progressively weaker motion or noise.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real.
            If complex, amplitude is used.
        n_components: Number of principal components to keep (default: 5).
            Must be <= min(T, F).

    Returns:
        np.ndarray: (T, n_components, A) or (T, n_components) projected data

    Reference:
        Wang W, et al. "Understanding and Modeling of WiFi Signal Based
        Human Activity Recognition." Proc. ACM MobiCom (CARM), 2015.
    """
    if csi.size == 0:
        return np.array([])
    if csi.ndim < 2 or csi.ndim > 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    # Work on amplitude for complex input
    data = np.abs(csi) if np.iscomplexobj(csi) else csi.copy()

    squeezed = False
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
        squeezed = True

    T, F, A = data.shape
    K = min(n_components, T, F)

    result = np.empty((T, K, A), dtype=np.float64)

    for a in range(A):
        mat = data[:, :, a].astype(np.float64)  # (T, F)
        # Center columns (subcarriers)
        col_mean = np.mean(mat, axis=0, keepdims=True)
        mat_centered = mat - col_mean
        # SVD: mat_centered = U * diag(S) * Vt
        U, S, _ = np.linalg.svd(mat_centered, full_matrices=False)
        # Project: top-K components = U[:, :K] * S[:K]
        result[:, :, a] = U[:, :K] * S[:K]

    if squeezed:
        result = result[:, :, 0]

    return result
