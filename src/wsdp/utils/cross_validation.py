"""Cross-validation utilities for WSDP."""

import logging

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def group_kfold_split(data, labels, groups, n_splits=5):
    """Generate GroupKFold splits for person-independent evaluation.

    Uses :class:`sklearn.model_selection.GroupKFold` which ensures that no
    group (e.g. participant / user_id) appears in both the train and test
    sets within any single fold.

    Parameters
    ----------
    data : array-like of shape (N, ...)
        Feature array.
    labels : array-like of shape (N,)
        Target labels.
    groups : array-like of shape (N,)
        Group identifiers (e.g. ``user_id``).  Samples with the same group
        value are always placed together in either train or test.
    n_splits : int, default=5
        Number of folds.

    Yields
    ------
    fold_idx : int
        Zero-based fold index.
    train_idx : ndarray
        Indices for the training set.
    test_idx : ndarray
        Indices for the test set.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    if n_splits > n_groups:
        logger.warning(
            "n_splits=%d exceeds the number of unique groups (%d). "
            "Reducing n_splits to %d.",
            n_splits,
            n_groups,
            n_groups,
        )
        n_splits = n_groups

    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, labels, groups)):
        logger.info(
            "Fold %d/%d  —  train: %d samples, test: %d samples",
            fold_idx + 1,
            n_splits,
            len(train_idx),
            len(test_idx),
        )
        yield fold_idx, train_idx, test_idx
