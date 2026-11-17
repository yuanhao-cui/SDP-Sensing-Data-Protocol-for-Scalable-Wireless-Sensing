"""File-based preprocessing cache for WSDP to avoid re-processing on repeated runs."""

import hashlib
import json
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_cache_key(input_path, dataset, pad_len, preprocess_config=None):
    """Generate a cache key from input parameters and file modification times.

    The key is a hex digest derived from the dataset name, padding length,
    optional preprocessing configuration, and sorted (filename, mtime)
    pairs of every file under *input_path*.
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
    if preprocess_config is not None:
        config_json = json.dumps(preprocess_config, sort_keys=True, default=str)
        hasher.update(f"|preprocess_config={config_json}".encode())
    for rel_name, mtime in file_info:
        hasher.update(f"|{rel_name}:{mtime}".encode())

    return hasher.hexdigest()


def load_cache(cache_dir, cache_key):
    """Load cached data, mapping the large processed array from disk."""
    if cache_dir is None or cache_key is None:
        return None

    npy_path = os.path.join(cache_dir, f"{cache_key}.npy")
    npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
    json_path = os.path.join(cache_dir, f"{cache_key}.json")
    if not os.path.isfile(npz_path) or not os.path.isfile(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)
        with np.load(npz_path) as data:
            labels = data['labels']
            groups = data['groups']
            if os.path.isfile(npy_path):
                processed_data = np.load(npy_path, mmap_mode='r')
            elif 'processed_data' in data:
                # Backward compatibility with caches created before memmaps.
                processed_data = data['processed_data']
            else:
                return None
        logger.debug(f"Cache loaded from {cache_dir}")
        return {
            'processed_data': processed_data,
            'labels': labels,
            'groups': groups,
            'unique_labels': meta['unique_labels'],
        }
    except Exception as e:
        logger.warning(f"Failed to load cache ({e}), will re-process")
        return None


def save_cache(cache_dir, cache_key, processed_data, labels, groups, unique_labels):
    """Save a disk-mappable ``.npy`` array and small metadata files."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        npy_path = os.path.join(cache_dir, f"{cache_key}.npy")
        npz_path = os.path.join(cache_dir, f"{cache_key}.npz")
        json_path = os.path.join(cache_dir, f"{cache_key}.json")

        source_path = getattr(processed_data, 'filename', None)
        same_file = (
            source_path is not None
            and os.path.abspath(source_path) == os.path.abspath(npy_path)
        )
        if not same_file:
            np.save(npy_path, processed_data)
        np.savez(npz_path, labels=labels, groups=groups)
        with open(json_path, 'w') as f:
            json.dump({'unique_labels': list(unique_labels)}, f)

        logger.debug(f"Cache saved to {npy_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache ({e}), continuing without caching")
