"""Tests for reading per-episode SLAM tracking-quality stats out of pzarr (Phase 4)."""

from __future__ import annotations

import pathlib

import zarr
from polyumi_catalog import episode_quality


def _make_pzarr_with_slam(scene_dir: pathlib.Path, episodes: list[tuple[str, dict | None]]) -> None:
    """
    Build a scene.zarr with one episode group per (session_dirname, slam_attrs) pair.

    ``slam_attrs`` of ``None`` leaves that episode without an ``annotations/slam``
    group at all, simulating an episode SLAM hasn't run for yet.
    """
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = len(episodes)
    for i, (dirname, slam_attrs) in enumerate(episodes):
        ep = root.require_group(f'episode_{i}')
        ep.attrs['session_dir'] = dirname
        if slam_attrs is not None:
            slam_grp = ep.require_group('annotations').require_group('slam')
            for k, v in slam_attrs.items():
                slam_grp.attrs[k] = v


def _slam_attrs(n_total: int, n_lost: int, n_reloc: int = 0) -> dict:
    return {
        'n_frames_total': n_total,
        'n_frames_lost': n_lost,
        'tracking_ratio': (n_total - n_lost) / n_total,
        'n_relocalization_events': n_reloc,
    }


def test_scene_quality_by_session_dir_without_pzarr_returns_empty(tmp_path: pathlib.Path):
    """No scene.zarr at all (pzarr not yet built) resolves to an empty map, not an error."""
    scene_dir = tmp_path / 'scene_a'
    scene_dir.mkdir()
    assert episode_quality.scene_quality_by_session_dir(scene_dir) == {}


def test_scene_quality_by_session_dir_reads_good_and_low_quality_episodes(tmp_path: pathlib.Path):
    """Tracking ratio below the threshold is flagged low_quality; above it, not."""
    scene_dir = tmp_path / 'scene_b'
    scene_dir.mkdir()
    _make_pzarr_with_slam(
        scene_dir,
        [
            ('session_good', _slam_attrs(100, 2)),  # 98% tracked
            ('session_bad', _slam_attrs(100, 50)),  # 50% tracked
        ],
    )
    quality = episode_quality.scene_quality_by_session_dir(scene_dir)
    assert quality['session_good']['tracking_ratio'] == 0.98
    assert quality['session_good']['low_quality'] is False
    assert quality['session_bad']['tracking_ratio'] == 0.5
    assert quality['session_bad']['low_quality'] is True


def test_scene_quality_by_session_dir_skips_episodes_without_slam(tmp_path: pathlib.Path):
    """An episode with no annotations/slam group (SLAM hasn't run) is simply omitted."""
    scene_dir = tmp_path / 'scene_c'
    scene_dir.mkdir()
    _make_pzarr_with_slam(scene_dir, [('session_pending', None), ('session_done', _slam_attrs(10, 0))])
    quality = episode_quality.scene_quality_by_session_dir(scene_dir)
    assert 'session_pending' not in quality
    assert quality['session_done']['tracking_ratio'] == 1.0


def test_session_quality_resolves_single_session(tmp_path: pathlib.Path):
    """session_quality returns one session's stats by dirname."""
    scene_dir = tmp_path / 'scene_d'
    scene_dir.mkdir()
    _make_pzarr_with_slam(scene_dir, [('session_1', _slam_attrs(50, 5, n_reloc=2))])
    quality = episode_quality.session_quality(scene_dir, 'session_1')
    assert quality is not None
    assert quality['n_frames_lost'] == 5
    assert quality['n_relocalization_events'] == 2


def test_session_quality_unknown_session_returns_none(tmp_path: pathlib.Path):
    """A session with no matching episode (or no pzarr) resolves to None."""
    scene_dir = tmp_path / 'scene_e'
    scene_dir.mkdir()
    assert episode_quality.session_quality(scene_dir, 'session_1') is None


def test_scene_quality_summary_aggregates_across_episodes(tmp_path: pathlib.Path):
    """The scene-level summary averages tracking ratio and counts low-quality episodes."""
    scene_dir = tmp_path / 'scene_f'
    scene_dir.mkdir()
    _make_pzarr_with_slam(
        scene_dir,
        [
            ('session_1', _slam_attrs(100, 0)),  # 100%
            ('session_2', _slam_attrs(100, 80)),  # 20% -> low quality
        ],
    )
    summary = episode_quality.scene_quality_summary(scene_dir, ['session_1', 'session_2'])
    assert summary['n_episodes_with_slam'] == 2
    assert summary['avg_tracking_ratio'] == 0.6
    assert summary['n_low_quality'] == 1


def test_scene_quality_summary_with_no_slam_data_returns_zeros(tmp_path: pathlib.Path):
    """No episodes with SLAM results yet yields a summary with None/0, not a crash."""
    scene_dir = tmp_path / 'scene_g'
    scene_dir.mkdir()
    summary = episode_quality.scene_quality_summary(scene_dir, ['session_1'])
    assert summary == {'n_episodes_with_slam': 0, 'avg_tracking_ratio': None, 'n_low_quality': 0}
