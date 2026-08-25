"""Tests for the diagnostic log-mel spectrogram."""

import numpy as np
import pytest
from polyumi_ingest.preproc.logmel import hz_to_mel, log_mel_spectrogram, mel_filterbank, mel_to_hz

SR = 16_000
N_FFT = 400
HOP = 160
N_MELS = 64

_PARAMS = dict(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=8000.0, log_offset=1e-6)


def _tone(freq_hz: float, duration_s: float) -> np.ndarray:
    t = np.arange(int(SR * duration_s), dtype=np.float64) / SR
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)


def test_mel_scale_round_trips() -> None:
    """hz_to_mel and mel_to_hz are inverses across the audible band."""
    hz = np.array([0.0, 20.0, 700.0, 1000.0, 8000.0])
    assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-9, abs=1e-9)


def test_filterbank_shape_and_triangles() -> None:
    """Triangles are non-negative, bounded by 1, ordered, and none is left empty."""
    fbank = mel_filterbank(SR, N_FFT, N_MELS, 20.0, 8000.0)

    assert fbank.shape == (N_MELS, N_FFT // 2 + 1)
    assert (fbank >= 0.0).all()
    assert (fbank <= 1.0).all()
    # Non-decreasing, not strictly increasing: at n_fft=400 the FFT resolves 40 Hz while the
    # lowest mel bins are narrower than that, so neighbouring bins can peak on the same FFT bin.
    assert (np.diff(fbank.argmax(axis=1)) >= 0).all()
    # Same cause, and this is the part that would actually hurt: a mel bin falling entirely
    # between two FFT bins would be a dead row in every spectrogram we ever draw.
    assert (fbank.max(axis=1) > 0.0).all()


def test_filterbank_is_unnormalised() -> None:
    """
    Triangles peak at 1.0, the Kaldi/AST convention, not scaled to unit area.

    Checked at an FFT size fine enough to sample the peaks; area normalisation would shift the
    log of every bin by a different constant and stop the diagnostic resembling what AST sees.
    """
    fbank = mel_filterbank(SR, 2048, N_MELS, 20.0, 8000.0)
    # Never exactly 1.0 — no FFT bin lands precisely on a mel centre — but an area-normalised
    # bank would put these an order of magnitude away, not a thousandth.
    assert fbank.max() == pytest.approx(1.0, abs=1e-3)
    assert fbank.max(axis=1) == pytest.approx(np.ones(N_MELS), abs=0.1)
    assert (np.diff(fbank.argmax(axis=1)) > 0).all()


def test_filterbank_rejects_bad_band() -> None:
    """A band beyond Nyquist or with fmin >= fmax is a config error, not something to clamp."""
    with pytest.raises(ValueError, match='Nyquist'):
        mel_filterbank(SR, N_FFT, N_MELS, 20.0, 9000.0)
    with pytest.raises(ValueError, match='fmin < fmax'):
        mel_filterbank(SR, N_FFT, N_MELS, 4000.0, 1000.0)


def test_hop_count_matches_kaldi_snip_edges() -> None:
    """Only whole frames inside the signal — no synthesised half-frames at either end."""
    audio = _tone(1000.0, 2.0)
    spec = log_mel_spectrogram(audio, SR, **_PARAMS)

    assert spec.shape == (1 + (len(audio) - N_FFT) // HOP, N_MELS)
    assert spec.dtype == np.float32
    # 2 s at a 10 ms hop is the 198 frames ManiWAV's AST path zero-pads to 200; this pins that
    # our diagnostic sits on the same grid theirs does.
    assert spec.shape[0] == 198


def test_tone_peaks_in_its_mel_bin() -> None:
    """A pure tone concentrates in the bin the HTK mel scale puts it in."""
    spec = log_mel_spectrogram(_tone(1000.0, 1.0), SR, **_PARAMS)

    centres_hz = mel_to_hz(np.linspace(hz_to_mel(20.0), hz_to_mel(8000.0), N_MELS + 2))[1:-1]
    expected = int(np.argmin(np.abs(centres_hz - 1000.0)))
    assert abs(int(np.argmax(spec.mean(axis=0))) - expected) <= 1


def test_silence_floors_at_the_log_offset() -> None:
    """Digital silence maps to log(log_offset) rather than -inf."""
    spec = log_mel_spectrogram(np.zeros(SR, dtype=np.float32), SR, **_PARAMS)

    assert np.isfinite(spec).all()
    assert spec == pytest.approx(np.full_like(spec, np.log(1e-6)), abs=1e-4)


def test_short_audio_returns_empty_not_error() -> None:
    """A truncated episode is a real case; it has no frames, which is not a failure."""
    spec = log_mel_spectrogram(np.zeros(N_FFT - 1, dtype=np.float32), SR, **_PARAMS)
    assert spec.shape == (0, N_MELS)


def test_first_frame_covers_the_whole_window_not_half_zero_padding() -> None:
    """
    The first hop is centred so its window is exactly audio[:n_fft] — snip_edges=True.

    A window centred on sample 0 instead would zero-pad the first half, so the first hop would
    be blind to whatever the second half of audio[:n_fft] contains. Two signals differing only
    there must then produce different first hops.
    """
    audio_silent_tail = np.zeros(N_FFT * 3, dtype=np.float32)
    audio_loud_tail = audio_silent_tail.copy()
    audio_loud_tail[N_FFT // 2 : N_FFT] = _tone(1000.0, (N_FFT - N_FFT // 2) / SR)[: N_FFT - N_FFT // 2]

    spec_silent = log_mel_spectrogram(audio_silent_tail, SR, **_PARAMS)
    spec_loud = log_mel_spectrogram(audio_loud_tail, SR, **_PARAMS)

    assert not np.allclose(spec_silent[0], spec_loud[0])
