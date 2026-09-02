"""
Tests for `pingest copy-map`: reusing one scene's ORB-SLAM3 atlas in another.

The point of the command is that the target ends up in the *source's* SLAM frame, so the two
things worth pinning down are that the map actually moves across and that nothing rebuilds it
afterwards — a rebuild would silently put the target back in its own frame while every log
line still says "step 2 complete".
"""

from __future__ import annotations

import pathlib
import unittest.mock as mock

import numpy as np
import zarr
from polyumi_ingest.main import app
from polyumi_ingest.preproc.slam_step import ATLAS_SOURCE_ATTR, OrbSlam3Step
from polyumi_ingest.preproc.step_base import preprocessing_steps_done
from typer.testing import CliRunner

from test_slam_step import _calibrated_settings, _make_episode

runner = CliRunner()


def _make_scene(scene_dir: pathlib.Path, *, with_atlas: bool, steps_done: list[int]) -> pathlib.Path:
    scene_dir.mkdir(parents=True)
    scene_zarr = scene_dir / 'scene.zarr'
    root = zarr.open_group(str(scene_zarr), mode='w', zarr_format=2)
    _make_episode(root, 'episode_0', session_type='MAPPING')
    _make_episode(root, 'episode_1', session_type='EPISODE')
    root.attrs['preprocessing_steps'] = steps_done
    for key in ('episode_0', 'episode_1'):
        root[key].attrs['preprocessing_steps'] = steps_done
    if with_atlas:
        (scene_dir / f'{scene_dir.name}.atlas.osa').write_bytes(b'ATLAS')
    return scene_zarr


def test_copy_map_moves_the_atlas_and_reopens_step_2(tmp_path: pathlib.Path) -> None:
    """The atlas lands under the target's own name, stamped, with step 2 no longer complete."""
    _make_scene(tmp_path / 'scene_src', with_atlas=True, steps_done=[1, 2, 3])
    dst_zarr = _make_scene(tmp_path / 'scene_dst', with_atlas=False, steps_done=[1, 2, 3])

    result = runner.invoke(app, ['copy-map', str(tmp_path / 'scene_src'), str(tmp_path / 'scene_dst')])
    assert result.exit_code == 0, result.output

    atlas = tmp_path / 'scene_dst' / 'scene_dst.atlas.osa'
    assert atlas.read_bytes() == b'ATLAS'

    root = zarr.open_group(str(dst_zarr), mode='r')
    assert root.attrs[ATLAS_SOURCE_ATTR] == 'scene_src'
    assert preprocessing_steps_done(root) == [1, 3]
    # Per-episode marks too, or the re-run skips every episode and stamps the step done anyway.
    assert root['episode_1'].attrs['preprocessing_steps'] == [1, 3]


def test_copy_map_refuses_to_clobber_without_force(tmp_path: pathlib.Path) -> None:
    """An existing atlas is a 40-minute map build; overwriting it takes --force."""
    _make_scene(tmp_path / 'scene_src', with_atlas=True, steps_done=[2])
    _make_scene(tmp_path / 'scene_dst', with_atlas=False, steps_done=[2])
    dst_atlas = tmp_path / 'scene_dst' / 'scene_dst.atlas.osa'
    dst_atlas.write_bytes(b'OWN MAP')

    args = ['copy-map', str(tmp_path / 'scene_src'), str(tmp_path / 'scene_dst')]
    assert runner.invoke(app, args).exit_code == 1
    assert dst_atlas.read_bytes() == b'OWN MAP'

    assert runner.invoke(app, [*args, '--force']).exit_code == 0
    assert dst_atlas.read_bytes() == b'ATLAS'


def test_borrowed_atlas_survives_a_forced_rerun(tmp_path: pathlib.Path) -> None:
    """
    ``pingest pp 2 --force`` must relocalize against the borrowed map, not rebuild it.

    Without the guard, force deletes the atlas and maps from the target's own MAPPING session,
    quietly undoing the copy and returning the scene to its own frame.
    """
    _make_scene(tmp_path / 'scene_src', with_atlas=True, steps_done=[2])
    dst_zarr = _make_scene(tmp_path / 'scene_dst', with_atlas=False, steps_done=[2])
    assert runner.invoke(app, ['copy-map', str(tmp_path / 'scene_src'), str(tmp_path / 'scene_dst')]).exit_code == 0

    settings = _calibrated_settings(tmp_path)
    step = OrbSlam3Step(settings_yaml=settings)
    built, localized = [], []

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, scene_zarr):
        localized.append(ep_grp.name)
        from polyumi_ingest.preproc.slam_step import _write_slam_results

        n = ep_grp['timestamps/gopro'].shape[0]
        poses = np.zeros((n, 7), dtype=np.float64)
        poses[:, 6] = 1.0
        _write_slam_results(ep_grp, poses, settings, atlas_path)

    with (
        mock.patch.object(step, '_build_map', side_effect=lambda *a: built.append(a)),
        mock.patch.object(step, '_localize_episode', side_effect=_fake_localize),
    ):
        step.run_step(dst_zarr, force=True)

    assert built == []
    assert localized == ['/episode_1']
    assert (tmp_path / 'scene_dst' / 'scene_dst.atlas.osa').read_bytes() == b'ATLAS'


def test_deleting_a_borrowed_atlas_returns_the_scene_to_its_own_map(tmp_path: pathlib.Path) -> None:
    """
    The documented escape hatch — delete the file by hand — must also drop the borrow marker.

    Otherwise the marker outlives the atlas it described: the rebuild happens, but every later
    `--force` sees a scene still flagged as borrowing and refuses to rebuild the *local* map,
    while the log keeps calling it copied.
    """
    _make_scene(tmp_path / 'scene_src', with_atlas=True, steps_done=[2])
    dst_zarr = _make_scene(tmp_path / 'scene_dst', with_atlas=False, steps_done=[2])
    assert runner.invoke(app, ['copy-map', str(tmp_path / 'scene_src'), str(tmp_path / 'scene_dst')]).exit_code == 0

    (tmp_path / 'scene_dst' / 'scene_dst.atlas.osa').unlink()  # the escape hatch

    settings = _calibrated_settings(tmp_path)
    step = OrbSlam3Step(settings_yaml=settings)
    built = []

    def _fake_localize(ep_grp, episode_index, atlas_path, log_dir, scene_zarr):
        from polyumi_ingest.preproc.slam_step import _write_slam_results

        n = ep_grp['timestamps/gopro'].shape[0]
        poses = np.zeros((n, 7), dtype=np.float64)
        poses[:, 6] = 1.0
        _write_slam_results(ep_grp, poses, settings, atlas_path)

    with (
        mock.patch.object(step, '_build_map', side_effect=lambda *a: built.append(a)),
        mock.patch.object(step, '_localize_episode', side_effect=_fake_localize),
    ):
        step.run_step(dst_zarr, force=True)

    assert len(built) == 1  # it did map from this scene
    assert ATLAS_SOURCE_ATTR not in zarr.open_group(str(dst_zarr), mode='r').attrs
