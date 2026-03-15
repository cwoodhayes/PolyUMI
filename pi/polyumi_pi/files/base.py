"""Abstraction layer for the various files recorded by the PolyUMI rpi."""

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SessionDataABC(ABC):
    """Base class for the data files recorded during data collection."""

    path: pathlib.Path
    """Path to this file/folder"""

    @classmethod
    @abstractmethod
    def from_file(cls, path: pathlib.Path) -> 'SessionDataABC':
        """Load the data from a given file."""
        pass
