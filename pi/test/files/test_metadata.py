"""Unit tests for pi/files/metadata.py."""

import json
import pathlib
from datetime import datetime, timezone

import pytest
from polyumi_pi.files.metadata import SessionMetadata


def test_to_file_and_from_file_roundtrip(tmp_path):
    """SessionMetadata written to disk can be read back identically."""
    path = tmp_path / 'metadata.json'
    original = SessionMetadata(
        path=path,
        pi_hostname='test-pi',
        camera_fps=30,
        camera_resolution=(1920, 1080),
        audio_sample_rate=44100,
        audio_channels=2,
        audio_chunk_ms=100,
        duration_s=5.0,
        n_video_frames=150,
        n_audio_chunks=50,
        video_dropped_frames=0,
        audio_dropped_chunks=0,
        notes='unit test',
        task='pick',
        robot='franka',
    )
    original.to_file()

    loaded = SessionMetadata.from_file(path)

    assert loaded.session_id == original.session_id
    assert loaded.created_at == original.created_at
    assert loaded.camera_resolution == (1920, 1080)
    assert loaded.duration_s == 5.0
    assert loaded.notes == 'unit test'
    assert loaded.file_version == 1


def test_invalid_filename_raises():
    """SessionMetadata rejects paths that are not named metadata.json."""
    with pytest.raises(ValueError, match='metadata.json'):
        SessionMetadata(path=pathlib.Path('/tmp/session/data.json'))
