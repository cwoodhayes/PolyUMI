"""
Tests for the UMI-format diffusion-policy exporter.

The exporter's contract is UMI's ``ReplayBuffer`` layout, and the coupling is by key *name*:
``UmiDataset``/``sampler`` count robots via ``key.endswith('eef_pos')`` and raise on any
low-dim key they can't name-match. These tests pin the schema (names, shapes, dtypes) and the
two easy-to-break transforms — quaternion→rotvec and the broadcast ``demo_start/end_pose`` —
so a regression fails here rather than deep inside a training run.
"""

import pathlib
import zipfile

import cv2
import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from polyumi_ingest.export.dp import export_scene_to_dp, export_scenes_to_dp
from polyumi_ingest.manifests import SceneManifest
from polyumi_ingest.preproc import available_preprocessing_steps

RES = 224
RATE = 59.94

#: The full registered pipeline; a scene must have all of these marked complete to pass the
#: exporter's enforce_preprocessing check.
ALL_STEPS = sorted(cls.step_number for cls in available_preprocessing_steps())


def _write_gopro_mp4(path: pathlib.Path, n: int, h: int = 240, w: int = 320) -> None:
    """Write an n-frame gopro.mp4 sidecar (content arbitrary — frames are read by index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    if not writer.isOpened():
        pytest.skip('cv2.VideoWriter (mp4v) unavailable in this environment')
    for i in range(n):
        writer.write(np.full((h, w, 3), i % 256, dtype=np.uint8))
    writer.release()


EXPECTED_KEYS = {
    'camera0_rgb': ((RES, RES, 3), np.uint8),
    'robot0_eef_pos': ((3,), np.float32),
    'robot0_eef_rot_axis_angle': ((3,), np.float32),
    'robot0_gripper_width': ((1,), np.float32),
    'robot0_demo_start_pose': ((6,), np.float32),
    'robot0_demo_end_pose': ((6,), np.float32),
}


def _build_scene(
    tmp_path: pathlib.Path,
    *,
    n: int = 120,
    session_type: str = 'EPISODE',
    nan_rows: slice | None = None,
    preprocessing_steps: list[int] | None = ALL_STEPS,
    gopro_chirp_end_s: float | None = None,
) -> pathlib.Path:
    """
    Build a one-episode scene.zarr with GoPro-grid frames, eef/pose, and gripper width.

    ``preprocessing_steps`` marks the scene's completed steps (defaults to every registered
    step, so export's enforce_preprocessing check passes out of the box); pass None to omit
    the attr entirely (simulating a scene that's never been preprocessed). ``gopro_chirp_end_s``
    writes the step-1 chirp-end marker the exporter uses to trim the start.
    """
    scene = tmp_path / 'scene.zarr'
    root = zarr.open_group(str(scene), mode='w', zarr_format=2)
    root.attrs['n_episodes'] = 1
    if preprocessing_steps is not None:
        root.attrs['preprocessing_steps'] = preprocessing_steps
    ep = root.create_group('episode_0')
    ep.attrs['session_type'] = session_type
    # v3: GoPro frames live in the gopro.mp4 sidecar, resolved via the session_dir attr.
    ep.attrs['session_dir'] = 'session_0'
    _write_gopro_mp4(tmp_path / 'session_0' / 'gopro.mp4', n)

    gopro_ts = np.arange(n, dtype=np.float64) / RATE
    ep.create_group('timestamps').create_array('gopro', data=gopro_ts)

    # A pose trajectory that translates along x and yaws, so pos and rotvec are both non-trivial.
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(0, 0.5, n)
    yaw = np.linspace(0, np.pi / 2, n)
    quat = Rotation.from_rotvec(yaw[:, None] * np.array([0.0, 0.0, 1.0])).as_quat()
    pose = np.concatenate([pos, quat], axis=1)
    if nan_rows is not None:
        pose[nan_rows] = np.nan
    ep.create_group('eef').create_array('pose', data=pose)
    ep['eef'].attrs['source'] = 'optitrack'

    widths = np.linspace(0.02, 0.08, n).astype(np.float32)
    ep.create_group('annotations').create_group('gripper_width').create_array('width_m', data=widths)

    if gopro_chirp_end_s is not None:
        ep['annotations'].create_group('time_sync').attrs['gopro_chirp_end_s'] = gopro_chirp_end_s

    return scene


def _open_zip(path: pathlib.Path) -> zarr.Group:
    """Open a .zarr.zip the way UmiDataset does — through a ZipStore."""
    return zarr.open_group(zarr.storage.ZipStore(str(path), mode='r'), mode='r')


def test_export_produces_umi_schema(tmp_path: pathlib.Path) -> None:
    """Every expected key is present with UMI's shape and dtype, and lengths agree."""
    scene = _build_scene(tmp_path, n=120)
    out = tmp_path / 'buf.zarr.zip'

    n_eps = export_scene_to_dp(scene, out)

    assert n_eps == 1
    g = _open_zip(out)
    ends = g['meta/episode_ends'][:]
    assert ends.dtype == np.int64 and len(ends) == 1
    t = int(ends[-1])
    for key, (per_step_shape, dtype) in EXPECTED_KEYS.items():
        a = g[f'data/{key}']
        assert a.shape == (t,) + per_step_shape, key
        assert a.dtype == dtype, key


def test_export_omits_action_and_wrt_start(tmp_path: pathlib.Path) -> None:
    """Action and *_wrt_start must NOT be stored — the dataset synthesises/derives them."""
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)

    keys = set(_open_zip(out)['data'].keys())
    assert 'action' not in keys
    assert not any(k.endswith('_wrt_start') for k in keys)


def test_synthesised_action_is_7d(tmp_path: pathlib.Path) -> None:
    """
    The sampler builds action = [eef_pos, eef_rot_axis_angle, gripper_width]; that must be 7-wide.

    We replicate that concatenation here so the stored keys are proven to compose into the
    7-vector the sampler expects (pos3 + rotvec3 + gripper1), independent of UMI being importable.
    """
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    action = np.concatenate(
        [g[f'data/robot0_{c}'][:] for c in ('eef_pos', 'eef_rot_axis_angle', 'gripper_width')],
        axis=-1,
    )
    assert action.shape[1] == 7


def test_rotvec_matches_source_quaternion(tmp_path: pathlib.Path) -> None:
    """robot0_eef_rot_axis_angle is the rotvec of the source quaternion, row for row."""
    scene = _build_scene(tmp_path, n=90)
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    rotvec = g['data/robot0_eef_rot_axis_angle'][:]
    pos = g['data/robot0_eef_pos'][:]
    # Recover the rotation and confirm it is a pure z-yaw ramp, matching the source trajectory.
    eul = Rotation.from_rotvec(rotvec).as_euler('xyz')
    assert np.allclose(eul[:, :2], 0, atol=1e-5)  # no roll/pitch
    assert eul[0, 2] == pytest.approx(0.0, abs=1e-5)
    assert eul[-1, 2] == pytest.approx(np.pi / 2, abs=1e-5)  # last frame kept — no resampling
    assert pos[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert pos[-1, 0] == pytest.approx(0.5, abs=1e-5)


def test_demo_poses_are_first_and_last_repeated(tmp_path: pathlib.Path) -> None:
    """demo_start/end_pose are the episode's first/last tcp pose, broadcast across all rows."""
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    pos = g['data/robot0_eef_pos'][:]
    rot = g['data/robot0_eef_rot_axis_angle'][:]
    tcp = np.concatenate([pos, rot], axis=1)
    start = g['data/robot0_demo_start_pose'][:]
    end = g['data/robot0_demo_end_pose'][:]

    assert np.all(start == start[0])  # constant across the episode
    assert np.all(end == end[0])
    np.testing.assert_allclose(start[0], tcp[0], atol=1e-6)
    np.testing.assert_allclose(end[0], tcp[-1], atol=1e-6)


def test_skips_mapping_session(tmp_path: pathlib.Path) -> None:
    """A MAPPING-only scene has nothing to export and raises rather than writing an empty store."""
    scene = _build_scene(tmp_path, session_type='MAPPING')
    out = tmp_path / 'buf.zarr.zip'
    with pytest.raises(RuntimeError, match='no EPISODE sessions'):
        export_scene_to_dp(scene, out)


def test_skips_episode_marked_unusable_in_scene_json(tmp_path: pathlib.Path) -> None:
    """An episode whose session dir is listed in scene.json's unusable_episodes is skipped."""
    scene = _build_scene(tmp_path)
    SceneManifest(scene_id='x', unusable_episodes=['session_0']).write_to_scene_dir(tmp_path)
    out = tmp_path / 'buf.zarr.zip'
    with pytest.raises(RuntimeError, match='no EPISODE sessions'):
        export_scene_to_dp(scene, out)


def test_missing_eef_pose_raises(tmp_path: pathlib.Path) -> None:
    """Exporting before step 5 has run points the user at the missing step, not a KeyError."""
    scene = _build_scene(tmp_path)
    root = zarr.open_group(str(scene), mode='a')
    del root['episode_0']['eef']
    out = tmp_path / 'buf.zarr.zip'
    with pytest.raises(RuntimeError, match='step 5'):
        export_scene_to_dp(scene, out)


def test_nan_span_is_dropped_keeping_longest_run(tmp_path: pathlib.Path) -> None:
    """
    Mid-episode tracking loss (NaN pose rows) shortens the export to the longest gap-free run.

    This is the SLAM-only failure the old exporter could not express — it clipped windows to
    OptiTrack timestamps and raised. Here the valid span is 80 rows out of 120.
    """
    scene = _build_scene(tmp_path, n=120, nan_rows=slice(80, 100))
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)

    t = int(_open_zip(out)['meta/episode_ends'][-1])
    assert t == 80  # longest gap-free run is rows [0, 80); frames taken as-is, none resampled


def test_output_is_a_valid_zip(tmp_path: pathlib.Path) -> None:
    """The artifact is a real zip (UmiDataset opens it with zarr.ZipStore)."""
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'
    export_scene_to_dp(scene, out)
    assert zipfile.is_zipfile(out)


def test_export_scene_to_dp_is_the_single_scene_case(tmp_path: pathlib.Path) -> None:
    """export_scene_to_dp is just export_scenes_to_dp with a one-element list."""
    scene = _build_scene(tmp_path, n=60)
    out_single = tmp_path / 'single.zarr.zip'
    out_multi = tmp_path / 'multi.zarr.zip'

    n1 = export_scene_to_dp(scene, out_single)
    n2 = export_scenes_to_dp([scene], out_multi)

    assert n1 == n2 == 1
    ends_single = _open_zip(out_single)['meta/episode_ends'][:].tolist()
    ends_multi = _open_zip(out_multi)['meta/episode_ends'][:].tolist()
    assert ends_single == ends_multi


def test_export_scenes_to_dp_concatenates_episode_ends_across_scenes(tmp_path: pathlib.Path) -> None:
    """Two scenes' episodes land in one buffer, with episode_ends accumulating across both."""
    scene_a = _build_scene(tmp_path / 'a', n=50)
    scene_b = _build_scene(tmp_path / 'b', n=70)
    out = tmp_path / 'combined.zarr.zip'

    n_eps = export_scenes_to_dp([scene_a, scene_b], out)

    assert n_eps == 2
    g = _open_zip(out)
    ends = g['meta/episode_ends'][:].tolist()
    assert ends == [50, 120]
    for key, (per_step_shape, dtype) in EXPECTED_KEYS.items():
        a = g[f'data/{key}']
        assert a.shape == (120,) + per_step_shape
        assert a.dtype == dtype


def test_export_scenes_to_dp_skips_mapping_per_scene(tmp_path: pathlib.Path) -> None:
    """A MAPPING scene among the given scenes contributes nothing, but others still export."""
    scene_a = _build_scene(tmp_path / 'a', n=40)
    scene_mapping = _build_scene(tmp_path / 'b', n=30, session_type='MAPPING')
    out = tmp_path / 'combined.zarr.zip'

    n_eps = export_scenes_to_dp([scene_a, scene_mapping], out)

    assert n_eps == 1
    assert _open_zip(out)['meta/episode_ends'][:].tolist() == [40]


def test_export_scenes_to_dp_raises_if_no_scenes_given() -> None:
    """An empty scene list is rejected rather than silently producing an empty buffer."""
    with pytest.raises(ValueError, match='No scenes'):
        export_scenes_to_dp([], pathlib.Path('unused.zarr.zip'))


def test_export_scenes_to_dp_raises_if_every_scene_is_mapping_only(tmp_path: pathlib.Path) -> None:
    """All-MAPPING input across every scene still raises rather than writing an empty buffer."""
    scene_a = _build_scene(tmp_path / 'a', session_type='MAPPING')
    scene_b = _build_scene(tmp_path / 'b', session_type='MAPPING')
    out = tmp_path / 'combined.zarr.zip'

    with pytest.raises(RuntimeError, match='no EPISODE sessions'):
        export_scenes_to_dp([scene_a, scene_b], out)


def test_export_scenes_to_dp_raises_for_missing_scene(tmp_path: pathlib.Path) -> None:
    """A scene path with no scene.zarr raises FileNotFoundError, not a partial export."""
    scene_a = _build_scene(tmp_path / 'a')
    missing = tmp_path / 'does-not-exist'
    out = tmp_path / 'combined.zarr.zip'

    with pytest.raises(FileNotFoundError):
        export_scenes_to_dp([scene_a, missing], out)
    assert not out.exists()


def test_chirp_end_trims_leading_frames(tmp_path: pathlib.Path) -> None:
    """Frames before gopro_chirp_end_s (the idle 'waiting for chirp' prefix) are dropped."""
    n = 120
    gopro_ts = np.arange(n, dtype=np.float64) / RATE
    scene = _build_scene(tmp_path, n=n, gopro_chirp_end_s=float(gopro_ts[10]))
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    t = int(g['meta/episode_ends'][-1])
    assert t == n - 10
    # First exported pos is the trajectory's row 10, not row 0 — the chirp prefix is gone.
    pos = g['data/robot0_eef_pos'][:]
    expected_first_x = np.linspace(0, 0.5, n)[10]
    assert pos[0, 0] == pytest.approx(expected_first_x, abs=1e-5)
    start = g['data/robot0_demo_start_pose'][:]
    assert start[0, 0] == pytest.approx(expected_first_x, abs=1e-5)


def test_chirp_end_beyond_valid_span_is_not_trimmed(tmp_path: pathlib.Path) -> None:
    """A chirp end past the whole episode (bad detection) is ignored rather than dropping everything."""
    n = 50
    scene = _build_scene(tmp_path, n=n, gopro_chirp_end_s=1000.0)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    assert int(_open_zip(out)['meta/episode_ends'][-1]) == n


def test_missing_chirp_end_annotation_exports_without_trim(tmp_path: pathlib.Path) -> None:
    """No annotations/time_sync group at all (step 1 never ran) — export proceeds untrimmed."""
    n = 40
    scene = _build_scene(tmp_path, n=n)  # gopro_chirp_end_s left unset
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out, enforce_preprocessing=False)

    assert int(_open_zip(out)['meta/episode_ends'][-1]) == n


def test_enforce_preprocessing_raises_when_step_incomplete(tmp_path: pathlib.Path) -> None:
    """A scene missing a registered preprocessing step fails fast, naming the missing step."""
    missing_step = ALL_STEPS[-1]
    scene = _build_scene(tmp_path, preprocessing_steps=[s for s in ALL_STEPS if s != missing_step])
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match=r'preprocessing steps \[' + str(missing_step)):
        export_scene_to_dp(scene, out)
    assert not out.exists()


def test_enforce_preprocessing_raises_when_never_preprocessed(tmp_path: pathlib.Path) -> None:
    """A scene with no preprocessing_steps attr at all is treated as fully unprocessed."""
    scene = _build_scene(tmp_path, preprocessing_steps=None)
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match='preprocessing steps'):
        export_scene_to_dp(scene, out)


def test_enforce_preprocessing_false_bypasses_the_check(tmp_path: pathlib.Path) -> None:
    """--no-enforce-preprocessing exports a partially preprocessed scene instead of raising."""
    scene = _build_scene(tmp_path, n=30, preprocessing_steps=None)
    out = tmp_path / 'buf.zarr.zip'

    n_eps = export_scene_to_dp(scene, out, enforce_preprocessing=False)

    assert n_eps == 1
    assert int(_open_zip(out)['meta/episode_ends'][-1]) == 30


def test_export_scenes_to_dp_error_identifies_the_failing_scene(tmp_path: pathlib.Path) -> None:
    """
    A failure in the second of several scenes names *that scene's own directory*, not 'scene.zarr'.

    Every scene's zarr store is named identically ('scene.zarr') inside its own directory, so
    labeling by that name alone would make every scene's episode_0 indistinguishable in a
    multi-scene export's errors/logs. It must use the scene directory name instead.
    """
    scene_a = _build_scene(tmp_path / 'a')
    scene_b = _build_scene(tmp_path / 'b')
    root_b = zarr.open_group(str(scene_b), mode='a')
    del root_b['episode_0']['eef']
    out = tmp_path / 'combined.zarr.zip'

    with pytest.raises(RuntimeError, match='step 5') as exc_info:
        export_scenes_to_dp([scene_a, scene_b], out)

    assert 'b/episode_0' in str(exc_info.value)
    assert 'scene.zarr/episode_0' not in str(exc_info.value)
