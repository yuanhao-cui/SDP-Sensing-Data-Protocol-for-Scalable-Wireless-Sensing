"""Bounded-memory, multiprocess CSI reading and preprocessing."""

from pathlib import Path
import shutil
import tempfile

import numpy as np

from wsdp import readers
from wsdp.utils.resize import resize_csi_to_fixed_array


def preprocess_to_memmap(
    input_path,
    dataset,
    pad_len,
    processor,
    reader_name,
    storage_path,
    *,
    batch_size=8,
    read_workers=4,
):
    """Read and process CSI in batches, returning a disk-backed array.

    Reader and processor pools both remain parallel, but only a bounded number
    of files/samples can be in flight. Batch shards are combined into one
    ``.npy`` file and loaded as a read-only memmap before returning.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(tempfile.mkdtemp(
        prefix=f".{storage_path.stem}-shards-", dir=str(storage_path.parent)
    ))

    shard_paths = []
    shard_dtypes = []
    labels = []
    groups = []
    sample_shape = None
    total_samples = 0
    batch = []

    def flush_batch():
        nonlocal sample_shape, total_samples
        if not batch:
            return

        processed, batch_labels, batch_groups = processor.process(
            batch,
            dataset=dataset,
            release_inputs=True,
        )
        if processed:
            array = resize_csi_to_fixed_array(processed, target_length=pad_len)
            if sample_shape is None:
                sample_shape = array.shape[1:]
            elif array.shape[1:] != sample_shape:
                raise ValueError(
                    f"Processed sample shape changed from {sample_shape} "
                    f"to {array.shape[1:]}"
                )

            shard_path = shard_dir / f"{len(shard_paths):06d}.npy"
            np.save(shard_path, array)
            shard_paths.append(shard_path)
            shard_dtypes.append(array.dtype)
            total_samples += len(array)
            labels.extend(batch_labels)
            groups.extend(batch_groups)
        batch.clear()

    try:
        for csi_data in readers.iter_data(
            input_path,
            reader_name,
            n_workers=read_workers,
            max_pending=read_workers * 2,
        ):
            batch.append(csi_data)
            if len(batch) >= batch_size:
                flush_batch()
        flush_batch()

        if not shard_paths:
            raise IOError(f"no valid CSI samples found in: {input_path}")

        dtype = np.result_type(*shard_dtypes)
        output = np.lib.format.open_memmap(
            storage_path,
            mode="w+",
            dtype=dtype,
            shape=(total_samples, *sample_shape),
        )
        offset = 0
        for shard_path in shard_paths:
            shard = np.load(shard_path, mmap_mode="r")
            count = len(shard)
            output[offset:offset + count] = shard
            offset += count
            del shard
            shard_path.unlink()
        output.flush()
        del output

        return (
            np.load(storage_path, mmap_mode="r"),
            labels,
            groups,
        )
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
