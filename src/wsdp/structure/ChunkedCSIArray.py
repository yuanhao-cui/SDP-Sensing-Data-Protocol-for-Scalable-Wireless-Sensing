"""Disk-backed CSI array stored as independent NumPy shards."""

import bisect
import json
import operator
from pathlib import Path

import numpy as np


class ChunkedCSIArray:
    """Present in-memory chunks or ``.npy`` shards as one sample array."""

    def __init__(self, paths, counts, shape, dtype, arrays=None):
        self.paths = [Path(path).resolve() for path in paths]
        self.arrays = arrays
        self.counts = [int(count) for count in counts]
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self._offsets = [0]
        for count in self.counts:
            self._offsets.append(self._offsets[-1] + count)
        sources = self.arrays if self.arrays is not None else self.paths
        if sources and (
            len(sources) != len(self.counts) or self._offsets[-1] != self.shape[0]
        ):
            raise ValueError("Chunk counts do not match the chunked array shape")
        self._maps = {}

    @property
    def disk_backed(self):
        return self.arrays is None

    def __len__(self):
        return self.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        return state

    def _get_map(self, shard_index):
        if self.arrays is not None:
            return self.arrays[shard_index]
        if shard_index not in self._maps:
            self._maps[shard_index] = np.load(
                self.paths[shard_index], mmap_mode='r'
            )
        return self._maps[shard_index]

    def _get_scalar(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("ChunkedCSIArray index out of range")
        shard_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[shard_index]
        return self._get_map(shard_index)[local_index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self[np.arange(len(self), dtype=np.int64)[index]]
        try:
            return self._get_scalar(index)
        except TypeError:
            pass

        indices = np.asarray(index)
        if indices.ndim == 0:
            return self._get_scalar(indices.item())
        if indices.dtype == np.bool_:
            indices = np.flatnonzero(indices)
        indices = indices.astype(np.int64, copy=False).reshape(-1)
        result = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        for output_index, source_index in enumerate(indices):
            result[output_index] = self._get_scalar(source_index)
        return result

    def save_manifest(self, manifest_path):
        if not self.disk_backed:
            raise ValueError("In-memory chunks do not have a disk manifest")
        manifest_path = Path(manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format': 'wsdp-chunked-csi-v1',
            'shape': self.shape,
            'dtype': self.dtype.str,
            'counts': self.counts,
            'files': [
                path.relative_to(manifest_path.parent).as_posix()
                for path in self.paths
            ],
        }
        with open(manifest_path, 'w', encoding='utf-8') as file:
            json.dump(payload, file)

    @classmethod
    def from_manifest(cls, manifest_path):
        manifest_path = Path(manifest_path).resolve()
        with open(manifest_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
        if payload.get('format') != 'wsdp-chunked-csi-v1':
            raise ValueError(f"Unsupported chunked CSI manifest: {manifest_path}")
        paths = [manifest_path.parent / name for name in payload['files']]
        return cls(paths, payload['counts'], payload['shape'], payload['dtype'])
