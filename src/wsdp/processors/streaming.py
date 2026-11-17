"""Bounded-memory, multiprocess CSI reading and preprocessing."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import numpy as np

from wsdp import readers
from wsdp.structure import ChunkedCSIArray
from wsdp.utils.resize import resize_csi_to_fixed_array


def _process_one_file(reader, file_path, processor, dataset, pad_len):
    """Read and fully preprocess one file inside the same worker."""
    file_name, data, err = readers._process_file(reader, file_path)
    if err is not None:
        return file_name, None, [], [], err

    # ProcessPool workers are daemonic and cannot create nested process pools.
    if hasattr(processor, 'n_workers'):
        processor.n_workers = 1
    if hasattr(processor, '_modular'):
        processor._modular.n_workers = 1

    processed, labels, groups = processor.process(
        data,
        dataset=dataset,
        release_inputs=True,
    )
    if not processed:
        return file_name, None, [], [], None
    array = resize_csi_to_fixed_array(processed, target_length=pad_len)
    return file_name, array, labels, groups, None


def _partition_files(files, worker_count):
    """Assign whole files to disjoint, size-balanced worker partitions."""
    sized_files = []
    for file_path in files:
        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0
        sized_files.append((file_path, size))

    partitions = [[] for _ in range(worker_count)]
    weights = [0] * worker_count
    for file_path, size in sorted(
        sized_files, key=lambda item: (-item[1], str(item[0]))
    ):
        worker_id = min(range(worker_count), key=lambda i: (weights[i], i))
        partitions[worker_id].append(file_path)
        weights[worker_id] += size
    return [partition for partition in partitions if partition]


def _process_partition(
    worker_id,
    files,
    reader,
    processor,
    dataset,
    pad_len,
    shard_dir,
    max_shard_bytes,
):
    """Process a static file partition and write bounded local shards."""
    shard_dir = Path(shard_dir)
    if hasattr(processor, 'n_workers'):
        processor.n_workers = 1
    if hasattr(processor, '_modular'):
        processor._modular.n_workers = 1

    shard_paths = []
    shard_dtypes = []
    labels = []
    groups = []
    errors = []
    sample_shape = None
    total_samples = 0

    batch_arrays = []
    batch_labels = []
    batch_groups = []
    batch_bytes = 0

    def flush_batch():
        nonlocal batch_bytes, total_samples
        if not batch_arrays:
            return
        array = (
            batch_arrays[0]
            if len(batch_arrays) == 1
            else np.concatenate(batch_arrays, axis=0)
        )
        shard_path = shard_dir / (
            f"worker-{worker_id:02d}-{len(shard_paths):06d}.npy"
        )
        np.save(shard_path, array)
        shard_paths.append(shard_path)
        shard_dtypes.append(array.dtype)
        labels.extend(batch_labels)
        groups.extend(batch_groups)
        total_samples += len(array)
        batch_arrays.clear()
        batch_labels.clear()
        batch_groups.clear()
        batch_bytes = 0

    for file_path in files:
        file_name, array, file_labels, file_groups, err = _process_one_file(
            reader, file_path, processor, dataset, pad_len
        )
        if err is not None:
            errors.append((file_name, err))
            continue
        if array is None:
            continue

        if sample_shape is None:
            sample_shape = array.shape[1:]
        elif array.shape[1:] != sample_shape:
            raise ValueError(
                f"Processed sample shape changed from {sample_shape} "
                f"to {array.shape[1:]}"
            )

        if batch_arrays and batch_bytes + array.nbytes > max_shard_bytes:
            flush_batch()
        batch_arrays.append(array)
        batch_labels.extend(file_labels)
        batch_groups.extend(file_groups)
        batch_bytes += array.nbytes
        print(f"√ processed: {file_name}\n")

    flush_batch()
    return {
        'shard_paths': shard_paths,
        'shard_dtypes': shard_dtypes,
        'labels': labels,
        'groups': groups,
        'errors': errors,
        'sample_shape': sample_shape,
        'total_samples': total_samples,
    }


def preprocess_to_memmap(
    input_path,
    dataset,
    pad_len,
    processor,
    reader_name,
    storage_path,
    *,
    read_workers=8,
    max_shard_bytes=64 * 1024 * 1024,
):
    """Process static file partitions into a directly readable shard set.

    Every worker owns a disjoint file list and writes bounded shards. The
    parent creates only a small manifest, never a second full merged array.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = storage_path.with_suffix('.chunks.json')
    shard_dir = storage_path.parent / f"{storage_path.stem}_chunks"

    # Remove only artifacts owned by this output path, freeing stale data before
    # requesting any new storage.
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir()
    storage_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)

    files = readers._list_input_files(input_path)
    reader = readers.create_reader(reader_name)
    worker_count = max(1, min(read_workers, len(files)))
    partitions = _partition_files(files, worker_count)

    shard_paths = []
    shard_dtypes = []
    labels = []
    groups = []
    sample_shape = None
    total_samples = 0
    skipped = 0
    succeeded = False

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _process_partition,
                    worker_id,
                    partition,
                    reader,
                    processor,
                    dataset,
                    pad_len,
                    str(shard_dir),
                    max_shard_bytes,
                ): worker_id
                for worker_id, partition in enumerate(partitions)
            }
            for future in as_completed(futures):
                result = future.result()
                if result['sample_shape'] is not None:
                    if sample_shape is None:
                        sample_shape = result['sample_shape']
                    elif result['sample_shape'] != sample_shape:
                        raise ValueError(
                            f"Processed sample shape changed from {sample_shape} "
                            f"to {result['sample_shape']}"
                        )
                shard_paths.extend(result['shard_paths'])
                shard_dtypes.extend(result['shard_dtypes'])
                labels.extend(result['labels'])
                groups.extend(result['groups'])
                total_samples += result['total_samples']
                for file_name, err in result['errors']:
                    if err == 'format_mismatch':
                        skipped += 1
                    else:
                        print(f"× unable to process {file_name}: {err}\n")

        if not shard_paths:
            raise IOError(f"no valid CSI samples found in: {input_path}")

        dtype = np.result_type(*shard_dtypes)
        counts = [
            len(np.load(shard_path, mmap_mode='r'))
            for shard_path in shard_paths
        ]
        chunked = ChunkedCSIArray(
            shard_paths,
            counts,
            (total_samples, *sample_shape),
            dtype,
        )
        chunked.save_manifest(manifest_path)

        if skipped > 0:
            print(
                f"[Info] skipped {skipped} file(s) "
                f"(format mismatch for {reader_name} reader)"
            )
        succeeded = True
        return chunked, labels, groups
    finally:
        if not succeeded:
            shutil.rmtree(shard_dir, ignore_errors=True)
            manifest_path.unlink(missing_ok=True)
