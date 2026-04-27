"""Reader base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sdp_mvp.structure import CSIData


class BaseReader(ABC):
    """Base class for dataset-specific CSI readers."""

    @abstractmethod
    def read_file(self, file_path: str) -> CSIData | list[CSIData]:
        """Read one source file."""

    def sniff(self, file_path: str) -> bool:
        """Return whether this reader can plausibly read ``file_path``."""

        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return reader and format metadata."""

        return {
            "reader": self.__class__.__name__,
            "format": self.__class__.__name__.replace("Reader", "").lower(),
        }
