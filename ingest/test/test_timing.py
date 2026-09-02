"""Unit tests for scene/dataset time accounting."""

from datetime import datetime, timedelta, timezone

import pytest
from polyumi_pi.files.metadata import SessionMetadata, SessionType

from polyumi_ingest import timing

SCENE_START = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)


def _session(scene_dir, name, *, offset_s, duration_s, scene_started_at=SCENE_START):
    """Write one session directory whose metadata.json carries the given times."""
    session_dir = scene_dir / name
    session_dir.mkdir(parents=True)
    SessionMetadata(
        path=session_dir / 'metadata.json',
        created_at=SCENE_START + timedelta(seconds=offset_s),
        duration_s=duration_s,
        scene_started_at=scene_started_at,
        session_type=SessionType.EPISODE,
    ).to_file()
    return session_dir


@pytest.fixture
def scene_dir(tmp_path):
    """Build a scene of two 60 s sessions five minutes apart, starting 30 s before the first."""
    scene = tmp_path / 'scene_2026-08-29_12-00-30_abcd'
    _session(scene, 'session_a', offset_s=30, duration_s=60)
    _session(scene, 'session_b', offset_s=330, duration_s=60)
    return scene


def test_scene_span_covers_dead_time(scene_dir):
    """The span runs from the scene start, not the first session, to the last session's end."""
    # Deliberately more than the 120 s recorded: the rest is the setup and the pause between.
    assert timing.scene_span_seconds(scene_dir) == 390.0  # 0 -> 330 + 60


def test_scene_span_falls_back_to_first_session(tmp_path):
    """A scene recorded before scene_started_at existed still gets a span."""
    scene = tmp_path / 'scene_old'
    _session(scene, 'session_a', offset_s=30, duration_s=60, scene_started_at=None)
    assert timing.scene_span_seconds(scene) == 60.0


def test_scene_span_unknown_without_finalized_sessions(tmp_path):
    """A scene whose sessions never finalized has no end, so no span."""
    scene = tmp_path / 'scene_unfinished'
    _session(scene, 'session_a', offset_s=30, duration_s=None)
    assert timing.scene_span_seconds(scene) is None


def test_dataset_totals_count_each_session_once(scene_dir):
    """A split session is counted once as episode time; a session left out counts not at all."""
    provenance = [
        {'scene': scene_dir.name, 'session': 'session_a', 'duration_s': 20.0},
        {'scene': scene_dir.name, 'session': 'session_a', 'duration_s': 25.0},
        {'scene': scene_dir.name, 'session': 'session_b', 'duration_s': 50.0},
    ]
    assert timing.dataset_time_totals([scene_dir], provenance) == {
        'scene_seconds': 390.0,
        'episode_seconds': 120.0,  # not 180: session_a counted once despite two segments
        'exported_seconds': 95.0,
    }

    # Drop session_b from the export and its 60 s of recording goes with it.
    assert timing.dataset_time_totals([scene_dir], provenance[:1])['episode_seconds'] == 60.0


def test_dataset_totals_report_unknown_scene_time_as_none(scene_dir, tmp_path):
    """One member scene with no measurable span makes the whole rig total unknown, not zero."""
    unfinished = tmp_path / 'scene_unfinished'
    _session(unfinished, 'session_a', offset_s=30, duration_s=None)
    totals = timing.dataset_time_totals([scene_dir, unfinished], [])
    assert totals['scene_seconds'] is None
    assert totals['episode_seconds'] == 0.0
