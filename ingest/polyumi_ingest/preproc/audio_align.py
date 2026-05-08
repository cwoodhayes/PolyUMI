"""Audio alignment algorithms for time synchronization."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AudioAligner(ABC):
    """
    Base class for audio lag estimation.

    Implementations receive two resampled, overlapping mono signals on the same
    sample grid and return the integer lag (in samples) of `sig` relative to
    `refsig`, plus a scalar confidence value whose scale is algorithm-specific.
    """

    @abstractmethod
    def estimate_lag(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        max_lag_samples: int | tuple[int, int] | None = None,
    ) -> tuple[int, float]:
        """
        Estimate the sample lag of sig relative to refsig.

        Parameters
        ----------
        sig:
            Signal to align (float32/64, 1-D, already resampled to common grid).
        refsig:
            Reference signal (same grid as sig).
        max_lag_samples:
            If given, restrict the search window. An ``int`` n restricts to
            ±n (symmetric). A ``(lo, hi)`` tuple allows an asymmetric range,
            e.g. ``(-100, 200)``.

        Returns
        -------
        (lag_samples, confidence)
            lag_samples: positive means sig leads refsig.
            confidence: algorithm-specific scalar; higher is better.

        """


class GCCPHATAligner(AudioAligner):
    """
    GCC-PHAT cross-correlation aligner with tunable spectral weighting.

    The normalisation exponent `alpha` controls how aggressively the cross-spectrum
    is whitened before back-transforming:

    * ``alpha=1.0`` (default) — full PHAT: divides by |X·Y*|, equalising all
      frequencies.  Good time resolution but ignores signal power.
    * ``alpha=0.0`` — standard cross-correlation: no whitening, so loud transients
      dominate the peak.
    * ``0 < alpha < 1`` — intermediate; try 0.5–0.75 to up-weight transients while
      retaining some frequency spreading.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize.

        Parameters
        ----------
        alpha:
            Spectral whitening exponent in [0, 1].

        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')
        self.alpha = alpha

    def estimate_lag(
        self,
        sig: np.ndarray,
        refsig: np.ndarray,
        max_lag_samples: int | tuple[int, int] | None = None,
    ) -> tuple[int, float]:
        """Estimate lag using Generalised Cross-Correlation with configurable weighting."""
        sig = np.asarray(sig, dtype=np.float64)
        refsig = np.asarray(refsig, dtype=np.float64)
        n_sig = len(sig)

        sig = sig - sig.mean()
        refsig = refsig - refsig.mean()
        sig = sig / (float(np.std(sig)) or 1.0)
        refsig = refsig / (float(np.std(refsig)) or 1.0)

        n = len(sig) + len(refsig)
        nfft = 1 << (n - 1).bit_length()
        sig_fft = np.fft.rfft(sig, n=nfft)
        ref_fft = np.fft.rfft(refsig, n=nfft)
        cross_power = sig_fft * np.conj(ref_fft)
        cross_power /= np.maximum(np.abs(cross_power), 1e-12) ** self.alpha
        cc = np.fft.irfft(cross_power, n=nfft)

        max_shift = nfft // 2
        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        shifts = np.arange(-max_shift, max_shift + 1)

        if max_lag_samples is not None:
            lo, hi = (-max_lag_samples, max_lag_samples) if isinstance(max_lag_samples, int) else max_lag_samples
            mask = (shifts >= lo) & (shifts <= hi)
            cc = cc[mask]
            shifts = shifts[mask]

        best_index = int(np.argmax(cc))
        return int(shifts[best_index]), float(cc[best_index]) / n_sig
