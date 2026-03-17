from abc import ABC, abstractmethod
from typing import Dict, Any
from wsdp.structure import CSIData


class BaseReader(ABC):
    """
    Base class for Readers
    One reader handles specified type of file
    """

    @abstractmethod
    def read_file(self, file_path: str) -> CSIData:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about this reader and its supported format.

        Returns:
            dict: Metadata including reader name, supported format, etc.
        """
        return {
            "reader": self.__class__.__name__,
            "format": self.__class__.__name__.replace("Reader", "").lower(),
        }
