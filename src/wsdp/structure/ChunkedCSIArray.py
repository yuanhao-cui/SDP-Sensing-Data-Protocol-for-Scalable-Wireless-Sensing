"""Chunked CSI storage with shared-memory and disk-backed implementations."""

import bisect
import json
import operator
import os
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


def _untrack_shared_memory(handle):
    """Prevent non-owning POSIX workers from unlinking a shared block."""
    if os.name == 'posix':
        from multiprocessing import resource_tracker

        resource_tracker.unregister(handle._name, 'shared_memory')


class ChunkedCSIArray:
    """Present independent CSI chunks as one logical, fixed-length array.

    Disk shards retain the historical dense representation. Shared-memory
    blocks instead pack only real frames and store each sample's true length;
    zero padding is created only for the requested sample or batch. Packed
    blocks contain the final real-valued model input, not an unused complex
    intermediate representation.
    """

    def __init__(
        self,
        paths=None,
        counts=None,
        shape=None,
        dtype=None,
        arrays=None,
        *,
        shared_blocks=None,
        shared_handles=None,
        model_input=False,
    ):
        self.paths = [Path(path).resolve() for path in (paths or [])]
        self.arrays = arrays
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.model_input = bool(model_input)
        self._maps = {}
        self._handles = {}
        self._closed = False
        self._shared_owner = shared_blocks is not None
        self.shared_blocks = shared_blocks
        self.sample_lengths = None

        if shared_blocks is not None:
            self._init_shared_blocks(shared_blocks)
            by_name = {handle.name: handle for handle in (shared_handles or [])}
            for index, name in enumerate(self._shared_names):
                if name in by_name:
                    self._handles[index] = by_name[name]
            return

        self._shared_names = []
        self.counts = [int(count) for count in counts]
        self._offsets = [0]
        for count in self.counts:
            self._offsets.append(self._offsets[-1] + count)
        sources = self.arrays if self.arrays is not None else self.paths
        if sources and (
            len(sources) != len(self.counts) or self._offsets[-1] != self.shape[0]
        ):
            raise ValueError("Chunk counts do not match the chunked array shape")

    def _init_shared_blocks(self, blocks):
        if not blocks:
            raise ValueError("At least one shared-memory block is required")
        self.arrays = None
        self._shared_names = [block['name'] for block in blocks]
        self._block_shapes = [tuple(block['shape']) for block in blocks]
        self.counts = [len(block['sample_lengths']) for block in blocks]
        self._offsets = [0]
        for count in self.counts:
            self._offsets.append(self._offsets[-1] + count)
        if self._offsets[-1] != self.shape[0]:
            raise ValueError("Shared-memory sample counts do not match its shape")

        lengths = []
        self._sample_blocks = []
        self._sample_frame_offsets = []
        for block_index, block in enumerate(blocks):
            block_lengths = np.asarray(block['sample_lengths'], dtype=np.int64)
            if np.any(block_lengths < 0) or np.any(block_lengths > self.shape[1]):
                raise ValueError("Invalid sample length in shared-memory block")
            if int(block_lengths.sum()) != self._block_shapes[block_index][0]:
                raise ValueError("Shared-memory frame count does not match samples")
            frame_offsets = np.concatenate((
                np.zeros(1, dtype=np.int64), np.cumsum(block_lengths)
            ))
            lengths.append(block_lengths)
            self._sample_blocks.extend([block_index] * len(block_lengths))
            self._sample_frame_offsets.extend(frame_offsets[:-1].tolist())
        self.sample_lengths = np.concatenate(lengths)
        self._sample_blocks = np.asarray(self._sample_blocks, dtype=np.int64)
        self._sample_frame_offsets = np.asarray(
            self._sample_frame_offsets, dtype=np.int64
        )
        self.nbytes = sum(
            int(np.prod(block_shape)) * self.dtype.itemsize
            for block_shape in self._block_shapes
        )

    def __len__(self):
        return self.shape[0]

    @property
    def disk_backed(self):
        return self.shared_blocks is None and self.arrays is None

    @property
    def shared_backed(self):
        return self.shared_blocks is not None

    @property
    def sample_nbytes(self):
        return int(np.prod(self.shape[1:])) * self.dtype.itemsize

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        state['_handles'] = {}
        state['_shared_owner'] = False
        return state

    def _get_map(self, shard_index):
        if self._closed:
            raise RuntimeError("The CSI array's shared memory has been released")
        if self.shared_backed:
            if shard_index not in self._maps:
                handle = self._handles.get(shard_index)
                if handle is None:
                    handle = shared_memory.SharedMemory(
                        name=self._shared_names[shard_index], create=False
                    )
                    if not self._shared_owner:
                        _untrack_shared_memory(handle)
                    self._handles[shard_index] = handle
                view = np.ndarray(
                    self._block_shapes[shard_index],
                    dtype=self.dtype,
                    buffer=handle.buf,
                )
                self._maps[shard_index] = view
            return self._maps[shard_index]
        if self.arrays is not None:
            return self.arrays[shard_index]
        if shard_index not in self._maps:
            self._maps[shard_index] = np.load(
                self.paths[shard_index], mmap_mode='r'
            )
        return self._maps[shard_index]

    @staticmethod
    def _normalize_index(index, length):
        index = operator.index(index)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("ChunkedCSIArray index out of range")
        return index

    def _get_dense_scalar(self, index):
        index = self._normalize_index(index, len(self))
        shard_index = bisect.bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[shard_index]
        return self._get_map(shard_index)[local_index]

    def _copy_shared_scalar(self, index, output):
        index = self._normalize_index(index, len(self))
        block_index = int(self._sample_blocks[index])
        length = int(self.sample_lengths[index])
        if length:
            frame_offset = int(self._sample_frame_offsets[index])
            output[:length] = self._get_map(block_index)[
                frame_offset:frame_offset + length
            ]
        return output

    def _get_scalar(self, index):
        if self.shared_backed:
            output = np.zeros(self.shape[1:], dtype=self.dtype)
            return self._copy_shared_scalar(index, output)
        return self._get_dense_scalar(index)

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
        if self.shared_backed:
            result = np.zeros((len(indices), *self.shape[1:]), dtype=self.dtype)
            for output_index, source_index in enumerate(indices):
                self._copy_shared_scalar(source_index, result[output_index])
            return result
        result = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        for output_index, source_index in enumerate(indices):
            result[output_index] = self._get_scalar(source_index)
        return result

    def close(self):
        """Release shared blocks owned by the parent process."""
        if self._closed:
            return
        self._closed = True
        self._maps.clear()
        for block_index, handle in list(self._handles.items()):
            if self._shared_owner:
                try:
                    handle.unlink()
                except FileNotFoundError:
                    pass
                if os.name == 'posix':
                    from multiprocessing import resource_tracker
                    try:
                        resource_tracker.unregister(handle._name, 'shared_memory')
                    except KeyError:
                        pass
            handle.close()
        self._handles.clear()
        if self._shared_owner:
            for name in self._shared_names:
                # Blocks never opened for reading still need explicit cleanup.
                try:
                    handle = shared_memory.SharedMemory(name=name, create=False)
                except FileNotFoundError:
                    continue
                try:
                    handle.unlink()
                except FileNotFoundError:
                    pass
                if os.name == 'posix':
                    from multiprocessing import resource_tracker
                    try:
                        resource_tracker.unregister(handle._name, 'shared_memory')
                    except KeyError:
                        pass
                handle.close()
        self._shared_owner = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
