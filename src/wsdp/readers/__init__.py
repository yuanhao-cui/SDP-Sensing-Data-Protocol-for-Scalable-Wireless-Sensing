from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterator, List, Type

from wsdp.structure import CSIData
from .base import BaseReader
from .bfee_reader import BfeeReader
from .xrf_reader import XrfReader
from .elder_reader import ElderReader
from .zte_reader import ZTEReader

# ^^^ import future reader above ^^^


_READER_REGISTRY: dict[str, Type[BaseReader]] = {
    'widar': BfeeReader,
    'gait': BfeeReader,
    'xrf55': XrfReader,
    'elderAL': ElderReader,
    "zte": ZTEReader,
}


def register_reader(
    dataset: str,
    reader_class: Type[BaseReader],
    *,
    replace: bool = False,
) -> None:
    """Register a dataset reader so raw-data loading is pluggable.

    Args:
        dataset: Canonical dataset name.
        reader_class: ``BaseReader`` subclass that handles the format.
        replace: If True, allow replacing an existing registration.

    Raises:
        TypeError: If ``reader_class`` is not a ``BaseReader`` subclass.
        ValueError: If the dataset is already registered and ``replace`` is False.
    """
    if not isinstance(reader_class, type) or not issubclass(reader_class, BaseReader):
        raise TypeError("reader_class must inherit from BaseReader")
    if dataset in _READER_REGISTRY and not replace:
        raise ValueError(f"reader already registered for dataset: {dataset}")
    _READER_REGISTRY[dataset] = reader_class


def unregister_reader(dataset: str) -> bool:
    """Remove a reader registration, returning whether it existed."""
    return _READER_REGISTRY.pop(dataset, None) is not None


def create_reader(dataset: str) -> BaseReader:
    """Create a reader instance for a dataset name."""
    reader_cls = _READER_REGISTRY.get(dataset)
    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")
    return reader_cls()


def get_reader_class(dataset: str) -> Type[BaseReader]:
    """Return the proper reader class according to dataset."""
    reader_cls = _READER_REGISTRY.get(dataset)
    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")
    return reader_cls


def list_datasets() -> List[str]:
    """Return sorted names of all registered datasets."""
    return sorted(_READER_REGISTRY.keys())


def get_all_reader_metadata(dataset: str) -> dict:
    """Get metadata from the reader for a specific dataset."""
    reader = create_reader(dataset)
    return reader.get_metadata()


def _process_file(reader: BaseReader, file_path: Path) -> tuple[str, list[CSIData], str | None]:
    """Read one file in a worker process."""
    try:
        if not reader.sniff(str(file_path)):
            return file_path.name, [], "format_mismatch"
        data = reader.read_file(str(file_path))
        return file_path.name, data if isinstance(data, list) else [data], None
    except Exception as e:
        return file_path.name, [], str(e)


def _list_input_files(file_path: str) -> list[Path]:
    input_path = Path(file_path)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"invalid file path: {input_path}")
    files = [f for f in input_path.rglob("*") if f.is_file() and "truth" not in f.name]
    if not files:
        raise IOError(f"no file in folder: {input_path}")
    return files


def iter_data(
    file_path: str,
    dataset: str,
    *,
    n_workers: int = 16,
    max_pending: int | None = None,
) -> Iterator[CSIData]:
    """Yield CSI samples using a bounded number of in-flight file reads."""
    files = _list_input_files(file_path)
    reader = create_reader(dataset)
    worker_count = max(1, n_workers)
    pending_limit = max_pending or worker_count * 2
    pending_limit = max(1, pending_limit)
    skipped = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        pending = {}
        next_index = 0
        while next_index < len(files) and len(pending) < pending_limit:
            future = executor.submit(_process_file, reader, files[next_index])
            pending[future] = files[next_index]
            next_index += 1

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                file_path_obj = pending.pop(future)
                file_name, data, err = future.result()
                if err is None:
                    for sample in data:
                        yield sample
                    if data:
                        print(f"√ processed: {file_name}\n")
                elif err == "format_mismatch":
                    skipped += 1
                else:
                    print(f"× unable to process {file_name}: {err}\n")

                if next_index < len(files):
                    next_future = executor.submit(
                        _process_file, reader, files[next_index]
                    )
                    pending[next_future] = files[next_index]
                    next_index += 1

    if skipped > 0:
        print(f"[Info] skipped {skipped} file(s) (format mismatch for {dataset} reader)")


def load_data(file_path: str, dataset: str) -> List[CSIData]:
    """Load all samples for backward compatibility.

    New large-data pipelines should consume :func:`iter_data` incrementally.
    """
    return list(iter_data(file_path, dataset))


__all__ = [
    "BaseReader",
    "BfeeReader",
    "XrfReader",
    "ElderReader",
    "ZTEReader",
    "create_reader",
    "get_reader_class",
    "list_datasets",
    "get_all_reader_metadata",
    "iter_data",
    "load_data",
    "register_reader",
    "unregister_reader",
]
