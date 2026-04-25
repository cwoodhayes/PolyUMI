"""Unit tests for pi/files/metadata.py."""

import json
import pathlib
from datetime import datetime, timezone

import pytest
from polyumi_pi.files.metadata import SessionMetadata


def test_to_file_and_from_file_roundtrip(tmp_path):
    """SessionMetadata written to disk can be read back identically."""
    path = tmp_path / 'metadata.json'
    sync_time = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
    original = SessionMetadata(
        path=path,
        pi_hostname='test-pi',
        camera_fps=30,
        camera_resolution=(1920, 1080),
        audio_start_time_ns=1_000_000_000,
        audio_sample_rate=44100,
        audio_channels=2,
        audio_chunk_ms=100,
        duration_s=5.0,
        n_video_frames=150,
        n_audio_chunks=50,
        video_dropped_frames=0,
        audio_dropped_chunks=0,
        led_brightness=0.8,
        gopro_sync_time=sync_time,
        first_frame_metadata={'exposure_time': 5000, 'analogue_gain': 1.0},
        notes='unit test',
        task='pick',
        robot='franka',
    )
    original.to_file()

    loaded = SessionMetadata.from_file(path)

    assert loaded.session_id == original.session_id
    assert loaded.created_at == original.created_at
    assert loaded.camera_resolution == (1920, 1080)
    assert loaded.audio_start_time_ns == 1_000_000_000
    assert loaded.duration_s == 5.0
    assert loaded.led_brightness == 0.8
    assert loaded.gopro_sync_time == sync_time
    assert loaded.first_frame_metadata == {'exposure_time': 5000, 'analogue_gain': 1.0}
    assert loaded.notes == 'unit test'
    assert loaded.file_version == 1


def test_gopro_sync_time_none_roundtrip(tmp_path):
    """gopro_sync_time=None survives the JSON roundtrip as None."""
    path = tmp_path / 'metadata.json'
    original = SessionMetadata(path=path, gopro_sync_time=None)
    original.to_file()
    loaded = SessionMetadata.from_file(path)
    assert loaded.gopro_sync_time is None


def test_gopro_sync_time_preserves_timezone(tmp_path):
    """gopro_sync_time retains its timezone offset after serialization."""
    path = tmp_path / 'metadata.json'
    from datetime import timedelta
    jst = timezone(timedelta(hours=9))
    sync_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=jst)
    original = SessionMetadata(path=path, gopro_sync_time=sync_time)
    original.to_file()
    loaded = SessionMetadata.from_file(path)
    assert loaded.gopro_sync_time == sync_time
    assert loaded.gopro_sync_time.utcoffset() == timedelta(hours=9)


def test_first_frame_metadata_none_roundtrip(tmp_path):
    """first_frame_metadata=None survives the JSON roundtrip as None."""
    path = tmp_path / 'metadata.json'
    original = SessionMetadata(path=path, first_frame_metadata=None)
    original.to_file()
    loaded = SessionMetadata.from_file(path)
    assert loaded.first_frame_metadata is None


def test_invalid_filename_raises():
    """SessionMetadata rejects paths that are not named metadata.json."""
    with pytest.raises(ValueError, match='metadata.json'):
        SessionMetadata(path=pathlib.Path('/tmp/session/data.json'))
