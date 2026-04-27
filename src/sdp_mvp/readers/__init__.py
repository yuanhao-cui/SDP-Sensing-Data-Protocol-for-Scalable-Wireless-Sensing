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


def get_reader_class(dataset: str) -> type[BaseReader]:
    """Return the reader class for a dataset name."""

    canonical = _canonical_dataset(dataset)
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


def load_data(file_path: str, dataset: str, max_workers: int | None = 16) -> list[CSIData]:
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
    "get_reader_class",
    "list_datasets",
    "get_all_reader_metadata",
    "load_data",
]
