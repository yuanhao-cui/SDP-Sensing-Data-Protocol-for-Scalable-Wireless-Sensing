"""
Motion detection and change point detection for CSI data.

Algorithms for detecting human activity and identifying transitions
in CSI-based sensing applications.
"""
import numpy as np


def detect_activity(csi, window=32, threshold=0.1):
    """
    Detect activity using sliding window variance analysis.

    Computes the variance of CSI amplitude within a sliding window.
    High variance indicates motion (activity), low variance indicates
    a static environment. The method normalizes variance by signal
    energy to provide a robust detection metric.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        window: Sliding window size in samples (default: 32)
        threshold: Detection threshold for normalized variance (default: 0.1)

    Returns:
        np.ndarray: Boolean array of shape (T,) indicating activity at each time step

    Reference:
        Zhou Z, et al. "Device-Free Passive Localization for Human 
        Activity Recognition." IEEE Communications Magazine, 2013.
        Youssef M, et al. "Challenges: Device-free Passive Localization 
        for Wireless Environments." ACM MobiCom, 2007.
    """
    if csi.size == 0:
        return np.array([], dtype=bool)
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    amplitude = np.abs(csi)
    T = amplitude.shape[0]

    # Compute sliding variance over all subcarriers and antennas
    if amplitude.ndim == 3:
        # Mean across subcarriers and antennas first
        signal = np.mean(amplitude, axis=(1, 2))
    else:
        signal = np.mean(amplitude, axis=1)

    # Sliding window variance
    half_w = window // 2
    variance_trace = np.zeros(T)
    for t in range(T):
        start = max(0, t - half_w)
        end = min(T, t + half_w + 1)
        window_data = signal[start:end]
        if len(window_data) > 1:
            # Normalized variance: var / mean^2
            mean_val = np.mean(window_data)
            if mean_val > 1e-10:
                variance_trace[t] = np.var(window_data) / (mean_val ** 2)
            else:
                variance_trace[t] = 0.0
        else:
            variance_trace[t] = 0.0

    return variance_trace > threshold


def change_point_detection(csi, method='mean_shift_ratio'):
    """
    Detect change points in CSI time series.

    Identifies time indices where the statistical properties of the
    CSI signal change significantly, indicating transitions between
    different activities or environmental states.

    Args:
        csi: CSI array of shape (T, F, A) or (T, F) — complex or real
        method: Detection method
            - 'mean_shift_ratio': Mean-shift ratio test — compares
              distribution means before/after each candidate point.
              (Note: NOT Bayesian Online Changepoint Detection.)
            - 'cusum': Cumulative Sum (CUSUM) control chart
            - 'variance': Variance-based detection

    Returns:
        np.ndarray: Array of time indices where change points detected

    Reference:
        Page ES. "Continuous Inspection Schemes." Biometrika, 1954. (CUSUM)
    """
    if csi.size == 0:
        return np.array([], dtype=int)
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")

    valid_methods = ('mean_shift_ratio', 'cusum', 'variance')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Supported: {valid_methods}")

    amplitude = np.abs(csi)

    # Flatten to single time series (mean across subcarriers and antennas)
    if amplitude.ndim == 3:
        signal = np.mean(amplitude, axis=(1, 2))
    else:
        signal = np.mean(amplitude, axis=1)

    T = len(signal)

    if T < 4:
        return np.array([], dtype=int)

    if method == 'mean_shift_ratio':
        return _mean_shift_changepoint(signal)
    elif method == 'cusum':
        return _cusum_changepoint(signal)
    else:
        return _variance_changepoint(signal)


def _mean_shift_changepoint(signal):
    """
    Mean-shift ratio change point detection.
    Compares mean difference between historical and recent segments,
    normalized by combined standard deviation.
    """
    T = len(signal)
    scores = np.zeros(T)
    min_historical = 10

    for t in range(min_historical, T - min_historical):
        # Historical: signal before t
        historical = signal[:t]
        # Recent: signal after t
        recent = signal[t:]

        mu_hist, std_hist = np.mean(historical), np.std(historical) + 1e-10
        mu_rec, std_rec = np.mean(recent), np.std(recent) + 1e-10

        # Log-likelihood ratio: measures how different the distributions are
        llr = abs(mu_rec - mu_hist) / (std_hist + std_rec)
        scores[t] = llr

    # Adaptive threshold: mean + 2*std of scores
    valid_scores = scores[min_historical:T - min_historical]
    if len(valid_scores) > 0:
        thresh = np.mean(valid_scores) + 2 * np.std(valid_scores)
    else:
        thresh = 0

    change_points = np.where(scores > thresh)[0]

    return change_points


def _cusum_changepoint(signal):
    """
    CUSUM (Cumulative Sum) change point detection.
    Detects shifts in mean level.
    """
    T = len(signal)
    mu = np.mean(signal)
    sigma = np.std(signal) + 1e-10

    # CUSUM statistic
    cusum_pos = np.zeros(T)
    cusum_neg = np.zeros(T)
    slack = 0.5 * sigma  # Allowable slack

    for t in range(1, T):
        deviation = (signal[t] - mu) / sigma
        cusum_pos[t] = max(0, cusum_pos[t - 1] + deviation - slack)
        cusum_neg[t] = max(0, cusum_neg[t - 1] - deviation - slack)

    # Threshold: 5*sigma
    threshold = 5.0
    change_points = np.where((cusum_pos > threshold) | (cusum_neg > threshold))[0]

    # Deduplicate nearby change points (keep first of each cluster)
    if len(change_points) > 1:
        deduped = [change_points[0]]
        for cp in change_points[1:]:
            if cp - deduped[-1] > 10:  # minimum gap of 10 samples
                deduped.append(cp)
        change_points = np.array(deduped)

    scores = cusum_pos + cusum_neg
    return change_points


def _variance_changepoint(signal):
    """
    Variance-based change point detection.
    Uses ratio of local variances before/after each candidate point.
    """
    T = len(signal)
    scores = np.zeros(T)
    min_window = 5

    for t in range(min_window, T - min_window):
        var_before = np.var(signal[max(0, t - min_window):t])
        var_after = np.var(signal[t:min(T, t + min_window)])
        total_var = var_before + var_after

        if total_var > 1e-10:
            # Ratio-based score
            scores[t] = abs(var_before - var_after) / total_var

    # Adaptive threshold
    valid_scores = scores[min_window:T - min_window]
    if len(valid_scores) > 0:
        thresh = np.mean(valid_scores) + 2 * np.std(valid_scores)
    else:
        thresh = 0

    change_points = np.where(scores > thresh)[0]

    # Deduplicate
    if len(change_points) > 1:
        deduped = [change_points[0]]
        for cp in change_points[1:]:
            if cp - deduped[-1] > 10:
                deduped.append(cp)
        change_points = np.array(deduped)

    return change_points
