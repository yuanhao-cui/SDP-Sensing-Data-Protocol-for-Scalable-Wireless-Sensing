"""File-based preprocessing cache for WSDP to avoid re-processing on repeated runs."""

import hashlib
import json
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_cache_key(input_path, dataset, pad_len):
    """Generate a cache key from input parameters and file modification times.

    The key is a hex digest derived from the dataset name, padding length,
    and sorted (filename, mtime) pairs of every file under *input_path*.
    """
    file_info = []
    for root, _dirs, files in os.walk(input_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = 0
            rel = os.path.relpath(fpath, input_path)
            file_info.append((rel, mtime))

    file_info.sort()
    hasher = hashlib.sha256()
    hasher.update(f"dataset={dataset}|pad_len={pad_len}".encode())
    for rel_name, mtime in file_info:
        hasher.update(f"|{rel_name}:{mtime}".encode())

    return hasher.hexdigest()


def load_cache(cache_dir, cache_key):
    """Load cached preprocessed data if available.

    Returns a dict with keys (processed_data, labels, groups, unique_labels)
    or None on cache miss / stale cache.
    """
    if cache_dir is None or cache_key is None:
        return None

    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    json_path = os.path.join(cache_dir, f"{cache_key}.json")

    if not os.path.isfile(npz_path) or not os.path.isfile(json_path):
        return None

    try:
        data = np.load(npz_path)
        with open(json_path, 'r') as f:
            meta = json.load(f)
        logger.debug(f"Cache loaded from {npz_path}")
        return {
            'processed_data': data['processed_data'],
            'labels': data['labels'],
            'groups': data['groups'],
            'unique_labels': meta['unique_labels'],
        }
    except Exception as e:
        logger.warning(f"Failed to load cache ({e}), will re-process")
        return None


def save_cache(cache_dir, cache_key, processed_data, labels, groups, unique_labels):
    """Save preprocessed data to cache.

    Stores a .npz file for arrays and a .json sidecar for unique_labels.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
        json_path = os.path.join(cache_dir, f"{cache_key}.json")

        np.savez(npz_path, processed_data=processed_data, labels=labels, groups=groups)
        with open(json_path, 'w') as f:
            json.dump({'unique_labels': list(unique_labels)}, f)

        logger.debug(f"Cache saved to {npz_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache ({e}), continuing without caching")
