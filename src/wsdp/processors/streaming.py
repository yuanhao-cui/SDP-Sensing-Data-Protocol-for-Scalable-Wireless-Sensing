"""Bounded-memory, multiprocess CSI reading and preprocessing."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import shutil
import tempfile

import numpy as np

from wsdp import readers
from wsdp.utils.resize import resize_csi_to_fixed_array


def _process_one_file(reader, file_path, processor, dataset, pad_len):
    """Read and fully preprocess one file inside the same worker."""
    file_name, data, err = readers._process_file(reader, file_path)
    if err is not None:
        return file_name, None, [], [], err

    # ProcessPool workers are daemonic and cannot create nested process pools.
    # Each worker therefore runs the per-sample algorithm chain serially, while
    # different files remain parallel across the outer pool.
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


def preprocess_to_memmap(
    input_path,
    dataset,
    pad_len,
    processor,
    reader_name,
    storage_path,
    *,
    read_workers=8,
):
    """Read and process files concurrently, returning a disk-backed array.

    Each worker owns one complete file task, so reading and preprocessing never
    alternate between separate pools. At most ``2 * read_workers`` files are in
    flight, bounding memory while keeping all workers supplied with work.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(tempfile.mkdtemp(
        prefix=f".{storage_path.stem}-shards-", dir=str(storage_path.parent)
    ))

    files = readers._list_input_files(input_path)
    reader = readers.create_reader(reader_name)
    worker_count = max(1, read_workers)
    pending_limit = worker_count * 2

    shard_paths = []
    shard_dtypes = []
    labels = []
    groups = []
    sample_shape = None
    total_samples = 0
    skipped = 0

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending = {}
            next_index = 0
            while next_index < len(files) and len(pending) < pending_limit:
                future = executor.submit(
                    _process_one_file,
                    reader,
                    files[next_index],
                    processor,
                    dataset,
                    pad_len,
                )
                pending[future] = files[next_index]
                next_index += 1

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    file_name, array, file_labels, file_groups, err = future.result()

                    if err == 'format_mismatch':
                        skipped += 1
                    elif err is not None:
                        print(f"× unable to process {file_name}: {err}\n")
                    elif array is not None:
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
                        labels.extend(file_labels)
                        groups.extend(file_groups)
                        print(f"√ processed: {file_name}\n")

                    if next_index < len(files):
                        next_future = executor.submit(
                            _process_one_file,
                            reader,
                            files[next_index],
                            processor,
                            dataset,
                            pad_len,
                        )
                        pending[next_future] = files[next_index]
                        next_index += 1

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

        if skipped > 0:
            print(
                f"[Info] skipped {skipped} file(s) "
                f"(format mismatch for {reader_name} reader)"
            )
        return np.load(storage_path, mmap_mode="r"), labels, groups
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
