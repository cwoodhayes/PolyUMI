"""Unit tests for AudioFile."""

import struct
import wave

import pytest
from polyumi_pi.files.audio import AudioFile


@pytest.fixture
def audio_params():
    return dict(sample_rate=16000, channels=1, sample_width=2)


def test_recording_writes_valid_wav(tmp_path, audio_params):
    """recording() context manager writes a readable WAV with correct headers."""
    path = tmp_path / 'audio.wav'
    audio = AudioFile(path=path, **audio_params)

    # two frames of silence (16-bit, 1 channel)
    frames = struct.pack('<hh', 0, 0)

    with audio.recording() as wf:
        wf.writeframes(frames)

    assert path.is_file()

    with wave.open(str(path), 'rb') as wf:
        assert wf.getframerate() == audio_params['sample_rate']
        assert wf.getnchannels() == audio_params['channels']
        assert wf.getsampwidth() == audio_params['sample_width']
        assert wf.getnframes() == 2


def test_from_file_roundtrip(tmp_path, audio_params):
    """from_file() recovers the same parameters written by recording()."""
    path = tmp_path / 'audio.wav'
    audio = AudioFile(path=path, **audio_params)

    with audio.recording() as wf:
        wf.writeframes(b'\x00\x00')

    loaded = AudioFile.from_file(path)

    assert loaded.path == path
    assert loaded.sample_rate == audio_params['sample_rate']
    assert loaded.channels == audio_params['channels']
    assert loaded.sample_width == audio_params['sample_width']


def test_recording_incremental_writes(tmp_path, audio_params):
    """Frames written across multiple writeframes() calls accumulate correctly."""
    path = tmp_path / 'audio.wav'
    audio = AudioFile(path=path, **audio_params)

    frame = struct.pack('<h', 1000)  # one 16-bit sample

    with audio.recording() as wf:
        for _ in range(5):
            wf.writeframes(frame)

    with wave.open(str(path), 'rb') as wf:
        assert wf.getnframes() == 5
