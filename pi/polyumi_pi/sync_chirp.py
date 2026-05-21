"""Sync chirp generation and playback for audio time-alignment."""

import time

import numpy as np
import sounddevice as sd

DURATION_S = 0.5
F0 = 440.0
F1 = 7000.0


def generate(sample_rate: int) -> np.ndarray:
    """
    Generate a linear frequency chirp.

    Returns a float32 mono array of length int(sample_rate * DURATION_S).
    """
    n = int(sample_rate * DURATION_S)
    t = np.linspace(0, DURATION_S, n, endpoint=False)
    k = (F1 - F0) / DURATION_S
    return np.sin(2 * np.pi * (F0 * t + 0.5 * k * t**2)).astype(np.float32)


def play(sample_rate: int, device: int | str | None = None) -> int:
    """
    Play the sync chirp on the given device (non-blocking).

    Returns time.time_ns() captured just before playback starts.
    The WM8960 requires stereo output, so the mono chirp is duplicated.
    """
    mono = generate(sample_rate)
    stereo = np.column_stack([mono, mono])
    ts = time.time_ns()
    sd.play(stereo, samplerate=sample_rate, device=device, blocking=False)
    return ts
