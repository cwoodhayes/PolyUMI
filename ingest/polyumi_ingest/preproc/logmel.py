"""
Log-mel spectrograms for the contact-mic diagnostic, in numpy/scipy alone.

**Nothing trains on the output of this module.** It exists so the catalog and the quality pass
can answer "did the contact mic actually record anything", and so a human can look at an episode
without a GPU box. The spectrogram the *policy* sees is computed inside the training container
from the raw waveform ``export-polyumi`` ships, after waveform-domain augmentation — see
``docs/maniwav-audio-policy.md``. The two are not required to agree bit-for-bit, and this one
must never become an input to training, because a precomputed mel cannot be augmented.

Parameters follow ManiWAV's so the picture resembles what the model will see: 64 HTK mel bins
over a 25 ms frame at a 10 ms hop, natural log of the power spectrum. The filterbank is
triangular and unnormalised, which is the Kaldi/AST convention their ``ASTFeatureExtractor``
uses rather than librosa's area-normalised default.
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Convert Hz to the HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Convert HTK mel back to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Triangular HTK mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``.

    Unnormalised: each triangle peaks at 1.0 rather than being scaled to unit area. That is the
    Kaldi convention, and it matters because the log of an area-normalised bank sits at a
    different offset — a diagnostic drawn one way and a model fed the other would not look alike.

    Raises:
        ValueError: if the band is empty or ``fmax`` exceeds Nyquist.

    """
    nyquist = sample_rate / 2.0
    if not 0.0 <= fmin < fmax:
        raise ValueError(f'Need 0 <= fmin < fmax, got fmin={fmin}, fmax={fmax}')
    if fmax > nyquist:
        raise ValueError(f'fmax {fmax} Hz exceeds Nyquist {nyquist} Hz for sample_rate={sample_rate}')

    # n_mels + 2 edges give n_mels overlapping triangles, each spanning three consecutive edges.
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bin_hz = np.linspace(0.0, nyquist, n_fft // 2 + 1)

    left, centre, right = edges_hz[:-2, None], edges_hz[1:-1, None], edges_hz[2:, None]
    up = (bin_hz[None, :] - left) / (centre - left)
    down = (right - bin_hz[None, :]) / (right - centre)
    return np.maximum(0.0, np.minimum(up, down))


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    log_offset: float,
) -> np.ndarray:
    """
    Log-mel spectrogram of a mono waveform, shape ``(n_hops, n_mels)`` float32.

    Time-major to match the layout ``ASTFeatureExtractor`` produces, so the diagnostic and the
    model's features are read the same way round.

    Args:
        audio: (N,) mono waveform.
        sample_rate: sample rate of ``audio`` in Hz.
        n_fft: FFT size; also the analysis frame length before any zero-padding.
        hop_length: samples between consecutive frames.
        win_length: Hann window length; padded symmetrically to ``n_fft`` when shorter.
        n_mels: number of mel bins.
        fmin: low edge of the filterbank in Hz.
        fmax: high edge of the filterbank in Hz.
        log_offset: added inside the log to floor silence.

    Returns:
        (n_hops, n_mels) float32. Empty ``(0, n_mels)`` when the waveform is shorter than one
        frame — a real case for a truncated episode, and not worth raising over.

    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if win_length > n_fft:
        raise ValueError(f'win_length {win_length} exceeds n_fft {n_fft}')
    if len(audio) < n_fft:
        return np.zeros((0, n_mels), dtype=np.float32)

    window = hann(win_length, sym=False)
    if win_length < n_fft:
        pad = n_fft - win_length
        window = np.pad(window, (pad // 2, pad - pad // 2))

    # p0/p1 with fft_mode='onesided' and no padding gives exactly the frames that fit inside the
    # signal, i.e. Kaldi's snip_edges=True — no synthesised half-frames at either end.
    sft = ShortTimeFFT(win=window, hop=hop_length, fs=sample_rate, fft_mode='onesided')
    n_hops = 1 + (len(audio) - n_fft) // hop_length
    spectrum = sft.stft(audio, p0=0, p1=n_hops, k_offset=0)  # (n_freq, n_hops) complex
    power = np.abs(spectrum) ** 2

    fbank = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)  # (n_mels, n_freq)
    mel_power = fbank @ power  # (n_mels, n_hops)
    return np.log(mel_power + log_offset).T.astype(np.float32)
