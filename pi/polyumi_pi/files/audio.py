"""WAV audio file abstraction for PolyUMI data collection."""

from __future__ import annotations

import pathlib
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from polyumi_pi.files.base import SessionDataABC


@dataclass
class AudioFile(SessionDataABC):
    """Abstraction for a WAV audio file recorded during data collection."""

    sample_rate: int
    """Sample rate in Hz."""

    channels: int
    """Number of audio channels."""

    sample_width: int
    """Sample width in bytes (e.g. 2 for 16-bit PCM)."""

    @classmethod
    def from_file(cls, path: pathlib.Path) -> AudioFile:
        """Load audio parameters from an existing WAV file."""
        with wave.open(str(path), 'rb') as wf:
            return cls(
                path=path,
                sample_rate=wf.getframerate(),
                channels=wf.getnchannels(),
                sample_width=wf.getsampwidth(),
            )

    @contextmanager
    def recording(self) -> Generator[wave.Wave_write, None, None]:
        """
        Context manager that opens the WAV file for writing.

        Yields the open :class:`wave.Wave_write` object so that audio samples
        can be written incrementally via ``writeframes()``.

        Example::

            audio = AudioFile(path=path, sample_rate=44100, channels=1, sample_width=2)
            with audio.recording() as wf:
                for chunk in source:
                    wf.writeframes(chunk)
        """
        with wave.open(str(self.path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            yield wf
