"""Dataset reader registry for MVP CSI input formats."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sdp_mvp.structure import CSIData

from .base import BaseReader
from .bfee_reader import BfeeReader
from .elder_reader import ElderReader
from .xrf_reader import XrfReader
from .zte_reader import ZTEReader

_READER_REGISTRY: dict[str, type[BaseReader]] = {
    "widar": BfeeReader,
    "gait": BfeeReader,
    "xrf55": XrfReader,
    "elderAL": ElderReader,
    "zte": ZTEReader,
}

_READER_ALIASES = {
    "elderal": "elderAL",
    "elder_al": "elderAL",
    "xrf": "xrf55",
    "bfee": "widar",
}


def _canonical_dataset(dataset: str) -> str:
    if dataset in _READER_REGISTRY:
        return dataset
    normalized = dataset.strip()
    if normalized in _READER_REGISTRY:
        return normalized
    return _READER_ALIASES.get(normalized.lower(), normalized)


def register_reader(dataset: str, reader_class: type[BaseReader], *, aliases: list[str] | None = None, replace: bool = False) -> None:
    """Register a dataset reader so raw-data loading is pluggable."""

    if not issubclass(reader_class, BaseReader):
        raise TypeError("reader_class must inherit BaseReader")
    if dataset in _READER_REGISTRY and not replace:
        raise ValueError(f"reader already registered for dataset: {dataset}")
    _READER_REGISTRY[dataset] = reader_class
    for alias in aliases or []:
        _READER_ALIASES[alias.lower()] = dataset


def unregister_reader(dataset: str) -> bool:
    """Remove a reader registration, returning whether it existed."""

    canonical = _canonical_dataset(dataset)
    removed = _READER_REGISTRY.pop(canonical, None) is not None
    for alias, target in list(_READER_ALIASES.items()):
        if target == canonical:
            del _READER_ALIASES[alias]
    return removed


def get_reader_class(dataset: str | type[BaseReader] | BaseReader) -> type[BaseReader]:
    """Return the reader class for a dataset name, class, or instance."""

    if isinstance(dataset, type) and issubclass(dataset, BaseReader):
        return dataset
    if isinstance(dataset, BaseReader):
        return dataset.__class__
    canonical = _canonical_dataset(str(dataset))
    reader_cls = _READER_REGISTRY.get(canonical)
    if reader_cls is None:
        raise ValueError(f"not supported dataset: {dataset}")
    return reader_cls


def list_datasets() -> list[str]:
    """List available dataset names."""

    return sorted(_READER_REGISTRY.keys())


def get_all_reader_metadata(dataset: str) -> dict[str, Any]:
    """Return metadata for the reader associated with ``dataset``."""

    return get_reader_class(dataset)().get_metadata()


def _process_file(reader_class: type[BaseReader], file_path: Path) -> tuple[str, list[CSIData], str | None]:
    try:
        reader = reader_class()
        if not reader.sniff(str(file_path)):
            return file_path.name, [], "format_mismatch"
        data = reader.read_file(str(file_path))
        if isinstance(data, list):
            return file_path.name, data, None
        return file_path.name, [data], None
    except Exception as exc:  # pragma: no cover - surfaced to caller output
        return file_path.name, [], str(exc)


def load_data(file_path: str, dataset: str | type[BaseReader] | BaseReader, max_workers: int | None = 16) -> list[CSIData]:
    """Load CSIData objects from one file or all matching files in a folder."""

    input_path = Path(file_path)
    if not input_path.exists():
        raise ValueError(f"invalid file path: {input_path}")

    reader_class = get_reader_class(dataset)
    if input_path.is_file():
        reader = reader_class()
        if not reader.sniff(str(input_path)):
            raise ValueError(f"file format does not match {dataset}: {input_path}")
        data = reader.read_file(str(input_path))
        return data if isinstance(data, list) else [data]

    files = [path for path in input_path.rglob("*") if path.is_file() and "truth" not in path.name]
    if not files:
        raise IOError(f"no file in folder: {input_path}")

    csi_data_list: list[CSIData] = []
    skipped = 0
    workers = max_workers if max_workers and max_workers > 0 else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, reader_class, path): path for path in files}
        for future in as_completed(futures):
            file_name, data, err = future.result()
            if err is None:
                csi_data_list.extend(data)
                print(f"processed: {file_name}")
            elif err == "format_mismatch":
                skipped += 1
            else:
                print(f"unable to process {file_name}: {err}")

    if skipped > 0:
        print(f"[Info] skipped {skipped} file(s) (format mismatch for {dataset} reader)")
    return csi_data_list


__all__ = [
    "BaseReader",
    "BfeeReader",
    "ElderReader",
    "XrfReader",
    "ZTEReader",
    "register_reader",
    "unregister_reader",
    "get_reader_class",
    "list_datasets",
    "get_all_reader_metadata",
    "load_data",
]
