"""Static multiprocess CSI preprocessing with shared-memory results."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
import os
import shutil
import threading

import numpy as np

from wsdp import readers
from wsdp.dataset_policy import convert_to_model_input
from wsdp.structure import ChunkedCSIArray
from wsdp.structure.ChunkedCSIArray import _untrack_shared_memory
from wsdp.utils.resize import resize_csi_to_fixed_array


def _read_and_process(reader, file_path, processor, dataset):
    """Read and fully process one file without fixed-length zero padding."""
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
    return file_name, processed, labels, groups, None


def _process_one_file(reader, file_path, processor, dataset, pad_len):
    """Compatibility helper for explicitly requested dense disk shards."""
    file_name, processed, labels, groups, err = _read_and_process(
        reader, file_path, processor, dataset
    )
    if err is not None or not processed:
        return file_name, None, labels, groups, err
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


def _shared_memory_allocator(request_queue, response_queues, handles):
    """Allocate blocks in the parent while workers continue processing."""
    while True:
        request = request_queue.get()
        if request is None:
            return
        worker_id, request_id, size = request
        try:
            handle = shared_memory.SharedMemory(create=True, size=max(1, size))
        except Exception as exc:
            response_queues[worker_id].put((request_id, None, str(exc)))
            continue
        handles.append(handle)
        response_queues[worker_id].put((request_id, handle.name, None))


def _close_shared_handles(handles, unlink=False):
    for handle in handles:
        if unlink:
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


_ALLOCATION_REQUESTS = None
_ALLOCATION_RESPONSES = None


def _init_shared_worker(allocation_requests, allocation_responses):
    """Inherit allocator queues at process creation, including on spawn."""
    global _ALLOCATION_REQUESTS, _ALLOCATION_RESPONSES
    _ALLOCATION_REQUESTS = allocation_requests
    _ALLOCATION_RESPONSES = allocation_responses


def _process_shared_partition(
    worker_id,
    files,
    reader,
    processor,
    dataset,
    pad_len,
    max_shard_bytes,
    storage_dtype,
):
    """Process a static partition directly into parent-owned shared blocks."""
    allocation_requests = _ALLOCATION_REQUESTS
    allocation_responses = _ALLOCATION_RESPONSES[worker_id]
    if hasattr(processor, 'n_workers'):
        processor.n_workers = 1
    if hasattr(processor, '_modular'):
        processor._modular.n_workers = 1

    labels = []
    groups = []
    errors = []
    blocks = []
    batch_arrays = []
    batch_bytes = 0
    sample_shape = None
    request_id = 0

    def flush_batch():
        nonlocal batch_bytes, request_id
        if not batch_arrays:
            return

        lengths = [len(array) for array in batch_arrays]
        frame_count = sum(lengths)
        request_id += 1
        allocation_requests.put((worker_id, request_id, batch_bytes))
        response_id, name, error = allocation_responses.get()
        if response_id != request_id:
            raise RuntimeError("Shared-memory allocation response is out of order")
        if error is not None:
            raise MemoryError(f"Unable to allocate shared preprocessing block: {error}")

        # The parent created and retains this block. POSIX workers must not
        # unregister it from the parent's tracker or unlink it on worker exit.
        handle = shared_memory.SharedMemory(name=name, create=False)
        _untrack_shared_memory(handle)
        packed = None
        try:
            packed = np.ndarray(
                (frame_count, *sample_shape), dtype=storage_dtype, buffer=handle.buf
            )
            frame_offset = 0
            for array in batch_arrays:
                next_offset = frame_offset + len(array)
                packed[frame_offset:next_offset] = array
                frame_offset = next_offset
            blocks.append({
                'name': name,
                'shape': packed.shape,
                'sample_lengths': lengths,
            })
        finally:
            if packed is not None:
                del packed
            handle.close()
        batch_arrays.clear()
        batch_bytes = 0

    for file_path in files:
        file_name, samples, file_labels, file_groups, err = _read_and_process(
            reader, file_path, processor, dataset
        )
        if err is not None:
            errors.append((file_name, err))
            continue
        if not samples:
            continue

        for sample, label, group in zip(samples, file_labels, file_groups):
            array = convert_to_model_input(
                sample[:pad_len],
                dataset,
                getattr(processor, 'pipeline_steps', None),
                output_dtype=storage_dtype,
            )
            if array.ndim != 3:
                raise ValueError(
                    f"Expected a 3D CSI sample, got shape {array.shape}"
                )
            current_shape = array.shape[1:]
            if sample_shape is None:
                sample_shape = current_shape
            elif current_shape != sample_shape:
                raise ValueError(
                    f"Processed sample shape changed from {sample_shape} "
                    f"to {current_shape}"
                )

            if batch_arrays and batch_bytes + array.nbytes > max_shard_bytes:
                flush_batch()
            batch_arrays.append(array)
            batch_bytes += array.nbytes
            labels.append(label)
            groups.append(group)
            if batch_bytes >= max_shard_bytes:
                flush_batch()
        print(f"√ processed: {file_name}\n")

    flush_batch()
    return {
        'blocks': blocks,
        'labels': labels,
        'groups': groups,
        'errors': errors,
    }


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
    """Compatibility path for explicitly requested disk-backed processing."""
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


def _clean_stale_outputs(storage_path):
    if storage_path is None:
        return None, None, None
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = storage_path.parent / f"{storage_path.stem}_chunks"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    for stale_path in (
        storage_path,
        storage_path.with_suffix('.chunks.json'),
        storage_path.with_suffix('.npz'),
        storage_path.with_suffix('.json'),
    ):
        stale_path.unlink(missing_ok=True)
    return storage_path, shard_dir, storage_path.with_suffix('.chunks.json')


def _report_errors(errors, reader_name):
    skipped = 0
    for file_name, err in errors:
        if err == 'format_mismatch':
            skipped += 1
        else:
            print(f"× unable to process {file_name}: {err}\n")
    if skipped:
        print(
            f"[Info] skipped {skipped} file(s) "
            f"(format mismatch for {reader_name} reader)"
        )


def _preprocess_shared(
    files,
    dataset,
    pad_len,
    processor,
    reader,
    reader_name,
    read_workers,
    max_shard_bytes,
    storage_dtype,
):
    worker_count = max(1, min(read_workers, len(files)))
    partitions = _partition_files(files, worker_count)

    # Use one spawn context for both queues and workers so semaphore locks
    # are never created under the platform-default fork context.
    worker_context = mp.get_context('spawn')
    allocation_requests = worker_context.Queue()
    allocation_responses = [worker_context.Queue() for _ in partitions]
    shared_handles = []
    allocator = threading.Thread(
        target=_shared_memory_allocator,
        args=(allocation_requests, allocation_responses, shared_handles),
        daemon=True,
    )
    allocator.start()
    results = []
    failures = []
    try:
        # Spawn after the allocator thread starts would be safe on every
        # platform; forking a multithreaded parent is not. Spawn also prevents
        # each preprocessing worker from inheriting the parent's full heap.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=worker_context,
            initializer=_init_shared_worker,
            initargs=(allocation_requests, allocation_responses),
        ) as executor:
            futures = [
                executor.submit(
                    _process_shared_partition,
                    worker_id,
                    partition,
                    reader,
                    processor,
                    dataset,
                    pad_len,
                    max_shard_bytes,
                    storage_dtype,
                )
                for worker_id, partition in enumerate(partitions)
            ]
            # Collect every result so that a failed worker never leaves blocks
            # allocated by successful sibling workers untracked or leaked.
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append(exc)
    finally:
        allocation_requests.put(None)
        allocator.join()
        allocation_requests.close()
        for queue in allocation_responses:
            queue.close()

    if failures:
        _close_shared_handles(shared_handles, unlink=True)
        raise failures[0]

    blocks = []
    labels = []
    groups = []
    errors = []
    for result in results:
        blocks.extend(result['blocks'])
        labels.extend(result['labels'])
        groups.extend(result['groups'])
        errors.extend(result['errors'])
    _report_errors(errors, reader_name)
    if not blocks:
        _close_shared_handles(shared_handles, unlink=True)
        raise IOError("no valid CSI samples found")

    sample_shapes = {tuple(block['shape'][1:]) for block in blocks}
    if len(sample_shapes) != 1:
        _close_shared_handles(shared_handles, unlink=True)
        raise ValueError(f"Inconsistent processed sample shapes: {sample_shapes}")
    total_samples = sum(len(block['sample_lengths']) for block in blocks)
    try:
        processed = ChunkedCSIArray(
            shape=(total_samples, pad_len, *sample_shapes.pop()),
            dtype=storage_dtype,
            shared_blocks=blocks,
            shared_handles=shared_handles,
            model_input=True,
        )
    except Exception:
        _close_shared_handles(shared_handles, unlink=True)
        raise

    raw_bytes = sum(file_path.stat().st_size for file_path in files)
    ratio = processed.nbytes / raw_bytes if raw_bytes else 0.0
    padding_bytes = processed.size * processed.dtype.itemsize - processed.nbytes
    print(
        f"[Info] shared preprocessing: workers={worker_count}, "
        f"dtype={processed.dtype}, raw={raw_bytes / 1024**3:.2f} GiB, "
        f"stored={processed.nbytes / 1024**3:.2f} GiB "
        f"({ratio:.2f}x raw), padding deferred="
        f"{padding_bytes / 1024**3:.2f} GiB"
    )
    return processed, labels, groups


def preprocess_to_memmap(
    input_path,
    dataset,
    pad_len,
    processor,
    reader_name,
    storage_path,
    *,
    read_workers=16,
    max_shard_bytes=16 * 1024 * 1024,
    storage_mode='memory',
    storage_dtype='float32',
):
    """Preprocess disjoint static file partitions without duplicate arrays.

    The default ``memory`` mode stores real, unpadded float32 model inputs in
    shared memory and creates no preprocessing data files. ``auto`` is a
    backward-compatible alias for ``memory``; it no longer estimates sizes or
    falls back to disk. ``disk`` remains available only when explicitly chosen.
    """
    if storage_mode == 'auto':
        storage_mode = 'memory'
    if storage_mode not in {'memory', 'disk'}:
        raise ValueError("storage_mode must be 'memory' or 'disk'")
    if read_workers < 1:
        raise ValueError("read_workers must be at least 1")
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be at least 1")
    storage_dtype = np.dtype(storage_dtype)
    if storage_dtype not in {np.dtype('float16'), np.dtype('float32')}:
        raise ValueError("storage_dtype must be 'float16' or 'float32'")

    if storage_mode == 'disk' and storage_path is None:
        raise ValueError("storage_path is required for explicit disk mode")
    storage_path, shard_dir, manifest_path = _clean_stale_outputs(storage_path)
    files = readers._list_input_files(input_path)
    reader = readers.create_reader(reader_name)

    if storage_mode == 'memory':
        return _preprocess_shared(
            files,
            dataset,
            pad_len,
            processor,
            reader,
            reader_name,
            read_workers,
            max_shard_bytes,
            storage_dtype,
        )

    shard_dir.mkdir()
    shard_paths = []
    shard_dtypes = []
    labels = []
    groups = []
    errors = []
    sample_shape = None
    total_samples = 0
    succeeded = False
    try:
        worker_count = max(1, min(read_workers, len(files)))
        partitions = _partition_files(files, worker_count)
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
                errors.extend(result['errors'])

        _report_errors(errors, reader_name)
        if not shard_paths:
            raise IOError(f"no valid CSI samples found in: {input_path}")
        counts = [len(np.load(path, mmap_mode='r')) for path in shard_paths]
        chunked = ChunkedCSIArray(
            shard_paths,
            counts,
            (total_samples, *sample_shape),
            np.result_type(*shard_dtypes),
        )
        chunked.save_manifest(manifest_path)
        succeeded = True
        return chunked, labels, groups
    finally:
        if not succeeded:
            shutil.rmtree(shard_dir, ignore_errors=True)
            manifest_path.unlink(missing_ok=True)
