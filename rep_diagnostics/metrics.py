"""Representation-level diagnostics for the SDP benchmark.

Implements the two metrics reported in Table III of the paper
("SDP: A Unified Protocol and Benchmarking Framework for Reproducible
Wi-Fi Sensing"):

* CKA  - latent representation reproducibility across independently
         trained seeds (linear CKA over all seed pairs);
* MMD2 - class-balanced train--test representation discrepancy
         (squared maximum mean discrepancy with an RBF kernel).

Pure numpy; no torch / GPU required at this stage.
"""
from itertools import combinations

import numpy as np

__all__ = [
    "biased_hsic",
    "linear_cka",
    "cka_across_seeds",
    "median_heuristic_sigma",
    "rbf_kernel",
    "mmd2",
    "class_balanced_subsample",
    "summarize",
]


# ---------------------------------------------------------------- CKA ----

def biased_hsic(K, L):
    """Biased (V-statistic) HSIC between two kernel/Gram matrices."""
    K = np.asarray(K, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if K.shape != L.shape:
        raise ValueError("K and L must have the same shape")
    n = K.shape[0]
    ones = np.ones((n, 1))
    return float((K * L).sum() / n ** 2
                 - 2.0 * (K @ L @ ones).sum() / n ** 3
                 + (K @ ones @ ones.T @ L.T).sum() / n ** 4)


def linear_cka(X, Y):
    """Linear CKA between two representation matrices (N x d each).

    Both matrices must describe the *same* N samples (aligned rows).
    Returns a value in [-1, 1]; identical representations give 1.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same sample count, got {X.shape[0]} and {Y.shape[0]}"
        )
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = biased_hsic(K, L)
    hsic_kk = biased_hsic(K, K)
    hsic_ll = biased_hsic(L, L)
    if hsic_kk <= 0 or hsic_ll <= 0:
        raise ValueError("degenerate representation (zero HSIC self-similarity)")
    return hsic_kl / np.sqrt(hsic_kk * hsic_ll)


def cka_across_seeds(features_by_seed):
    """Mean +- std of linear CKA over all seed pairs.

    Parameters
    ----------
    features_by_seed : list of ndarray, shape (N, d)
        Latent features of the SAME N evaluation samples extracted from
        models trained with different seeds. Needs at least 2 entries,
        all with identical N.

    Returns
    -------
    mean, std, pairs : float, float, list[float]
        `pairs` holds every individual C(S,2) CKA value; `std` uses ddof=1.
    """
    feats = [np.asarray(f, dtype=np.float64) for f in features_by_seed]
    if len(feats) < 2:
        raise ValueError("need latent features from at least two seeds")
    n_samples = feats[0].shape[0]
    if any(f.shape[0] != n_samples for f in feats):
        raise ValueError("all seeds must provide features for the same samples")
    pairs = [linear_cka(feats[i], feats[j])
             for i, j in combinations(range(len(feats)), 2)]
    return float(np.mean(pairs)), float(np.std(pairs, ddof=1)), pairs


# ---------------------------------------------------------------- MMD ----

def median_heuristic_sigma(X, Y):
    """RBF bandwidth from the median pairwise distance between X and Y."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sq_dists = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
    median_sq = np.median(sq_dists)
    if median_sq <= 0:
        raise ValueError("degenerate inputs: all pairwise distances are zero")
    return float(np.sqrt(median_sq))


def rbf_kernel(X, Y, sigma):
    """Gaussian kernel matrix k(x, y) = exp(-||x-y||^2 / (2 sigma^2))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sq_dists = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def mmd2(X, Y, sigma=None):
    """Squared MMD (biased V-statistic) between samples X and Y.

    With the RBF kernel and the median-heuristic bandwidth when sigma
    is None. Identical samples give 0; larger values mean more shift.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if sigma is None:
        sigma = median_heuristic_sigma(X, Y)
    k_xx = rbf_kernel(X, X, sigma)
    k_yy = rbf_kernel(Y, Y, sigma)
    k_xy = rbf_kernel(X, Y, sigma)
    m, n = X.shape[0], Y.shape[0]
    return float(k_xx.sum() / m ** 2 + k_yy.sum() / n ** 2
                 - 2.0 * k_xy.sum() / (m * n))


def class_balanced_subsample(labels, n_per_class, rng=None):
    """Indices with exactly `n_per_class` samples drawn per class."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(rng)
    idx_all = []
    for c in np.unique(labels):
        idx_c = np.flatnonzero(labels == c)
        if len(idx_c) < n_per_class:
            raise ValueError(
                f"class {c} has only {len(idx_c)} samples, "
                f"cannot draw {n_per_class}"
            )
        idx_all.append(rng.choice(idx_c, size=n_per_class, replace=False))
    return np.concatenate(idx_all)


def mmd2_class_balanced(X_train, y_train, X_test, y_test, n_per_class, rng=None):
    """Class-balanced squared MMD between train and test features.

    Draws `n_per_class` samples from every class on BOTH sides (RBF
    bandwidth from the balanced draw), as in the paper's Table III.
    """
    idx_tr = class_balanced_subsample(y_train, n_per_class, rng)
    idx_te = class_balanced_subsample(y_test, n_per_class, rng)
    X_tr = np.asarray(X_train, dtype=np.float64)[idx_tr]
    X_te = np.asarray(X_test, dtype=np.float64)[idx_te]
    sigma = median_heuristic_sigma(X_tr, X_te)
    return mmd2(X_tr, X_te, sigma=sigma)


# ----------------------------------------------------------- summary -----

def summarize(values):
    """mean +- sample std (ddof=1) of a non-empty sequence of floats."""
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    return float(vals.mean()), float(vals.std(ddof=1))
