"""Tests for GoPro SD-card clip matching, and the memo that keeps it from being quadratic."""

from __future__ import annotations

import datetime
import pathlib
import unittest.mock as mock

import pytest
from polyumi_ingest import gopro_fetch
from polyumi_ingest.gopro_fetch import GOPRO_VIDEO_SUBDIR, find_gopro_video

_EPOCH = datetime.datetime(2026, 8, 17, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the module-level memo so one test's answers cannot leak into the next."""
    gopro_fetch._START_TIME_CACHE.clear()
    yield
    gopro_fetch._START_TIME_CACHE.clear()


def _card(tmp_path: pathlib.Path, n_clips: int) -> pathlib.Path:
    """Build a mount point holding ``n_clips`` MP4s, one per minute from _EPOCH."""
    video_dir = tmp_path / 'card' / GOPRO_VIDEO_SUBDIR
    video_dir.mkdir(parents=True)
    for i in range(n_clips):
        (video_dir / f'GX01{i:04d}.MP4').write_bytes(b'not really an mp4')
    return tmp_path / 'card'


def _fake_probe(path: pathlib.Path) -> datetime.datetime:
    """Clip N started N minutes after _EPOCH — stands in for the ffprobe call."""
    return _EPOCH + datetime.timedelta(minutes=int(path.stem[-4:]))


def test_finds_the_clip_matching_the_sync_time(tmp_path: pathlib.Path) -> None:
    """Baseline: the closest clip by recording start wins."""
    card = _card(tmp_path, 5)

    with mock.patch.object(gopro_fetch, '_probe_start_time', side_effect=_fake_probe):
        match = find_gopro_video(_EPOCH + datetime.timedelta(minutes=3), mount_point=card)

    assert match.name == 'GX010003.MP4'


def test_each_clip_is_probed_once_across_many_sessions(tmp_path: pathlib.Path) -> None:
    """
    The whole point: N sessions against M clips must cost M probes, not N*M.

    A real fetch ran 20 sessions against a 451-clip card — 9020 ffprobe launches at ~100 ms
    each (process startup, not I/O) for 451 distinct answers, turning 45 s into 15 minutes.
    """
    card = _card(tmp_path, 10)

    with mock.patch.object(gopro_fetch, '_probe_start_time', side_effect=_fake_probe) as probe:
        for minute in range(6):  # six "sessions", each scanning the whole card
            find_gopro_video(_EPOCH + datetime.timedelta(minutes=minute), mount_point=card)

    assert probe.call_count == 10  # not 60


def test_a_rewritten_clip_is_reprobed(tmp_path: pathlib.Path) -> None:
    """
    Keyed on stat, not path alone, so a swapped card cannot serve the previous card's answer.

    GoPro filenames restart per card, so path alone would collide across cards.
    """
    card = _card(tmp_path, 1)
    clip = card / GOPRO_VIDEO_SUBDIR / 'GX010000.MP4'

    with mock.patch.object(gopro_fetch, '_probe_start_time', side_effect=_fake_probe) as probe:
        find_gopro_video(_EPOCH, mount_point=card)
        clip.write_bytes(b'a different clip, same filename')  # changes size and mtime
        find_gopro_video(_EPOCH, mount_point=card)

    assert probe.call_count == 2


def test_a_failing_probe_is_not_cached(tmp_path: pathlib.Path) -> None:
    """An unreadable clip is skipped with a warning, and retried rather than memoized."""
    card = _card(tmp_path, 1)

    with mock.patch.object(gopro_fetch, '_probe_start_time', side_effect=ValueError('bad file')) as probe:
        for _ in range(3):
            with pytest.raises(RuntimeError, match='Could not determine recording start time'):
                find_gopro_video(_EPOCH, mount_point=card)

    assert probe.call_count == 3


def test_an_explicit_mount_point_skips_card_detection(tmp_path: pathlib.Path) -> None:
    """With the mount resolved by the caller, no udisksctl/lsblk probing happens per session."""
    card = _card(tmp_path, 2)

    with (
        mock.patch.object(gopro_fetch, '_probe_start_time', side_effect=_fake_probe),
        mock.patch.object(gopro_fetch, 'find_gopro_mount') as detect,
    ):
        find_gopro_video(_EPOCH, mount_point=card)

    detect.assert_not_called()
