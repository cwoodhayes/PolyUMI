"""
Tests for the UMI-format diffusion-policy exporter.

The exporter's contract is UMI's ``ReplayBuffer`` layout, and the coupling is by key *name*:
``UmiDataset``/``sampler`` count robots via ``key.endswith('eef_pos')`` and raise on any
low-dim key they can't name-match. These tests pin the schema (names, shapes, dtypes) and the
two easy-to-break transforms — quaternion→rotvec and the broadcast ``demo_start/end_pose`` —
so a regression fails here rather than deep inside a training run.
"""

import logging
import pathlib
import zipfile

import cv2
import numpy as np
import pytest
import zarr
from scipy.spatial.transform import Rotation

from polyumi_catalog import episode_quality
from polyumi_ingest.export.dp import buffer
from polyumi_ingest.export.dp import export_scene_to_dp as _export_scene_to_dp
from polyumi_ingest.manifests import SceneManifest
from polyumi_ingest.preproc import available_preprocessing_steps

from export_floor import export_scene_to_dp, export_scenes_to_dp

RES = 224
RATE = 59.94

#: The full registered pipeline.
ALL_STEPS = sorted(cls.step_number for cls in available_preprocessing_steps())
#: The subset the visuomotor export actually reads, and so the only steps whose absence makes
#: the plain visuomotor export refuse. Steps feeding an optional modality (contact-audio) are excluded, which
#: is what keeps scenes preprocessed before those steps existed exportable.
DP_REQUIRED_STEPS = sorted(cls.step_number for cls in available_preprocessing_steps() if cls.required_for_export)


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
    with_slam: bool = False,
) -> pathlib.Path:
    """
    Build a one-episode scene.zarr with GoPro-grid frames, eef/pose_<source>, and gripper width.

    ``preprocessing_steps`` marks the scene's completed steps (defaults to every registered
    step, so export's enforce_preprocessing check passes out of the box); pass None to omit
    the attr entirely (simulating a scene that's never been preprocessed). ``gopro_chirp_end_s``
    writes the step-1 chirp-end marker the exporter uses to trim the start. ``eef/pose_optitrack``
    is always written (and is the default source); ``with_slam`` also writes a distinguishable
    ``eef/pose_slam`` (offset +10 in x) so tests can tell which source an export actually used.
    ``nan_rows`` NaNs out the optitrack array only.
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
    opti_pose = np.concatenate([pos, quat], axis=1)
    if nan_rows is not None:
        opti_pose[nan_rows] = np.nan

    eef_grp = ep.create_group('eef')
    opti_arr = eef_grp.create_array('pose_optitrack', data=opti_pose)
    opti_arr.attrs['world_frame'] = 'optitrack'
    opti_arr.attrs['n_interp_filled'] = 0
    available = ['optitrack']

    if with_slam:
        # Offset +10m in x so it's trivially distinguishable from the optitrack trajectory.
        slam_pose = opti_pose.copy()
        slam_pose[:, 0] += 10.0
        slam_arr = eef_grp.create_array('pose_slam', data=slam_pose)
        slam_arr.attrs['world_frame'] = 'slam'
        slam_arr.attrs['n_interp_filled'] = 3
        available.append('slam')

    eef_grp.attrs['available_sources'] = available
    eef_grp.attrs['default_source'] = available[0]  # optitrack preferred, matches EefPoseStep

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

    n_eps, provenance = export_scene_to_dp(scene, out)

    assert n_eps == 1
    assert len(provenance) == 1
    assert provenance[0]['source'] == 'optitrack'
    assert provenance[0]['episode'] == 'episode_0'
    assert provenance[0]['session'] == 'session_0'
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


def _make_slam_only(scene: pathlib.Path, **slam_attrs: object) -> None:
    """
    Strip OptiTrack from the built scene and give it SLAM quality attrs.

    ``_build_scene`` always writes ``pose_optitrack`` and lists it in
    ``available_sources``, which exempts the episode from the SLAM-derived usability
    checks — so a test of those checks has to remove it first.
    """
    root = zarr.open_group(str(scene), mode='a')
    ep = root['episode_0']
    eef = ep['eef']
    if 'pose_optitrack' in eef:
        del eef['pose_optitrack']
    eef.attrs['available_sources'] = ['slam']
    eef.attrs['default_source'] = 'slam'
    ep['annotations'].require_group('slam').attrs.update(slam_attrs)


def _slam_counts(n_fed: int, n_lost: int) -> dict:
    """Post-chirp fed-grid counts as step 2 records them (see _write_slam_results)."""
    return {
        'frame_stride': 1,
        'n_frames_fed': n_fed,
        'n_frames_fed_post_chirp': n_fed,
        'n_frames_fed_lost_post_chirp': n_lost,
        'chirp_gated': True,
        'tracking_ratio': (n_fed - n_lost) / n_fed,
    }


def _add_pose_jump(scene: pathlib.Path, at: int, metres: float, source: str = 'slam') -> None:
    """
    Displace every pose from row ``at`` onward, making one over-threshold step at ``at``.

    Shifting the whole tail rather than a single row models what a relocalization actually does
    — the trajectory continues, consistently, in a different place — and leaves exactly one
    oversized inter-frame step, at the seam.
    """
    root = zarr.open_group(str(scene), mode='a')
    pose = root['episode_0']['eef'][f'pose_{source}']
    data = pose[:]
    data[at:, 0] += metres
    pose[:] = data


def test_a_session_with_lost_frames_still_exports(tmp_path: pathlib.Path) -> None:
    """
    Whole-episode SLAM metrics never discard a session; only segmentation decides what survives.

    25 lost of 120 fed frames is a patchy demo, not a worthless one — segmentation cuts around
    the holes and keeps the runs either side, so nothing here may reject it up front.
    """
    scene = _build_scene(tmp_path, with_slam=True)
    _make_slam_only(scene, **_slam_counts(n_fed=120, n_lost=25))

    n_eps, _ = export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')

    assert n_eps == 1


def test_a_pose_jump_splits_a_session_and_costs_no_frames(tmp_path: pathlib.Path) -> None:
    """
    A teleport is a bad *transition*, so it cuts the demo in two and keeps both poses bracketing it.

    The frame-range half fails if the break is ever implemented by marking a step invalid —
    that would silently drop a good frame and put the boundary one step off. Both halves
    together must still account for every valid step.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n, with_slam=True)
    _make_slam_only(scene, max_pose_jump_m=3.0, **_slam_counts(n_fed=n, n_lost=0))
    _add_pose_jump(scene, at=60, metres=1.0)
    out = tmp_path / 'buf.zarr.zip'

    n_eps, provenance = export_scene_to_dp(scene, out, min_segment_steps=24)

    assert n_eps == 2
    assert provenance[0]['cut_end'] == 'pose_jump'
    assert provenance[1]['cut_start'] == 'pose_jump'
    ends = list(_open_zip(out)['meta/episode_ends'][:])
    assert ends == [60, n]  # no step lost at the seam
    assert provenance[0]['frame_range'] == [0, 59]
    assert provenance[1]['frame_range'] == [60, n - 1]


def test_a_pose_step_under_the_threshold_does_not_split(tmp_path: pathlib.Path) -> None:
    """The converse: ordinary fast motion is not a teleport."""
    n = 120
    scene = _build_scene(tmp_path, n=n, with_slam=True)
    _make_slam_only(scene, **_slam_counts(n_fed=n, n_lost=0))
    _add_pose_jump(scene, at=60, metres=0.01)  # 1 cm, well under the 8 cm threshold

    n_eps, _ = export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip', min_segment_steps=24)

    assert n_eps == 1


def test_a_jump_beside_a_tracking_gap_cuts_once_not_twice(tmp_path: pathlib.Path) -> None:
    """
    A pose pair spanning a NaN run produces a NaN distance, so the gap alone does the cutting.

    Without that, the frames either side of a dropout would look like a teleport to the jump
    check and split an already-split run again, emitting a spurious empty-ish episode.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n, nan_rows=slice(60, 66), with_slam=True)
    _make_slam_only(scene, **_slam_counts(n_fed=n, n_lost=6))
    # The trajectory resumes 1 m away — across the gap, which must NOT register as a jump.
    _add_pose_jump(scene, at=66, metres=1.0)

    n_eps, provenance = export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip', min_segment_steps=24)

    assert n_eps == 2
    assert [p['cut_end'] for p in provenance] == ['pose_gap', 'episode_end']


def test_catalog_unusable_implies_the_export_yields_nothing(tmp_path: pathlib.Path) -> None:
    """
    The surviving half of "one rule, two consumers", now that export has no veto of its own.

    ``quality.auto_unusable_reasons`` can no longer be *the* export rule — the real rule needs
    the pose array, and the catalog is attrs-only by design. What must still hold is the
    direction that matters to a user reading the UI: if the catalog predicts an episode
    contributes nothing, the export must actually produce nothing from it.
    """
    scene = _build_scene(tmp_path, n=40, with_slam=True)
    _make_slam_only(scene, **_slam_counts(n_fed=40, n_lost=0))

    catalog_quality = episode_quality.scene_quality_by_session_dir(tmp_path)['session_0']
    assert catalog_quality['auto_unusable'] is True  # too few tracked frames to yield a segment

    # The production floor, not this module's small one — the prediction is only meaningful
    # against the floor an actual export uses.
    with pytest.raises(RuntimeError, match='no EPISODE sessions'):
        _export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')


def test_a_bad_chirp_detection_does_not_make_an_exportable_episode_unusable(tmp_path: pathlib.Path) -> None:
    """
    The catalog must follow the exporter into its bad-chirp fallback, or the implication breaks.

    A chirp end that leaves fewer steps than the floor is treated by the exporter as a wrong
    detection, so it ships the episode *untrimmed*. Judging such an episode on its post-chirp
    counts would badge it unusable while the export produces a full episode from it.
    """
    n = 200
    post_chirp = 20
    scene = _build_scene(tmp_path, n=n, with_slam=True, gopro_chirp_end_s=(n - post_chirp) / RATE)
    counts = _slam_counts(n_fed=n, n_lost=0)
    counts['n_frames_fed_post_chirp'] = post_chirp
    _make_slam_only(scene, **counts)

    n_eps, provenance = _export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')

    assert n_eps == 1
    assert provenance[0]['n_steps'] == n  # untrimmed: the marker was distrusted
    assert episode_quality.scene_quality_by_session_dir(tmp_path)['session_0']['auto_unusable'] is False


def test_the_shortest_exportable_episode_is_not_called_unusable(tmp_path: pathlib.Path) -> None:
    """
    The boundary the prediction above rests on, exercised rather than asserted on constants.

    An episode of exactly ``MIN_SEGMENT_STEPS`` tracked frames is the shortest thing that
    exports at all. The catalog must not condemn it — which holds only while
    ``min_tracked_frames <= MIN_SEGMENT_STEPS``. Raise the catalog's number above the export's
    and this fails, because the catalog would be excluding an episode that does export.
    """
    n = buffer.MIN_SEGMENT_STEPS
    scene = _build_scene(tmp_path, n=n, with_slam=True)
    _make_slam_only(scene, **_slam_counts(n_fed=n, n_lost=0))

    assert episode_quality.scene_quality_by_session_dir(tmp_path)['session_0']['auto_unusable'] is False
    n_eps, _ = _export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')
    assert n_eps == 1


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
    # Floor pinned above the 20-step tail so only the long run survives — the point of the test.
    export_scene_to_dp(scene, out, min_segment_steps=24)

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

    n1, prov1 = export_scene_to_dp(scene, out_single)
    n2, prov2 = export_scenes_to_dp([scene], out_multi)

    assert n1 == n2 == 1
    assert prov1 == prov2
    ends_single = _open_zip(out_single)['meta/episode_ends'][:].tolist()
    ends_multi = _open_zip(out_multi)['meta/episode_ends'][:].tolist()
    assert ends_single == ends_multi


def test_export_scenes_to_dp_concatenates_episode_ends_across_scenes(tmp_path: pathlib.Path) -> None:
    """Two scenes' episodes land in one buffer, with episode_ends accumulating across both."""
    scene_a = _build_scene(tmp_path / 'a', n=50)
    scene_b = _build_scene(tmp_path / 'b', n=70)
    out = tmp_path / 'combined.zarr.zip'

    n_eps, provenance = export_scenes_to_dp([scene_a, scene_b], out)

    assert n_eps == 2
    assert [p['scene'] for p in provenance] == ['a', 'b']
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

    n_eps, provenance = export_scenes_to_dp([scene_a, scene_mapping], out)

    assert n_eps == 1
    assert [p['scene'] for p in provenance] == ['a']
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
    """A scene missing a step the export depends on fails fast, naming the missing step."""
    missing_step = DP_REQUIRED_STEPS[-1]
    scene = _build_scene(tmp_path, preprocessing_steps=[s for s in ALL_STEPS if s != missing_step])
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match=r'preprocessing steps \[' + str(missing_step)):
        export_scene_to_dp(scene, out)
    assert not out.exists()


def test_export_dp_ignores_steps_it_does_not_read(tmp_path: pathlib.Path) -> None:
    """A scene missing a step the visuomotor export never reads (e.g. contact-audio) still exports fine."""
    audio_only = sorted(set(ALL_STEPS) - set(DP_REQUIRED_STEPS))
    assert audio_only, 'expected at least one step marked required_for_export = False'
    scene = _build_scene(tmp_path, n=30, preprocessing_steps=DP_REQUIRED_STEPS)
    out = tmp_path / 'buf.zarr.zip'

    n_eps, _ = export_scene_to_dp(scene, out)

    assert n_eps == 1
    assert set(_open_zip(out)['data'].keys()) == set(EXPECTED_KEYS)


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

    n_eps, _ = export_scene_to_dp(scene, out, enforce_preprocessing=False)

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


def test_default_pose_source_is_optitrack_when_both_available(tmp_path: pathlib.Path) -> None:
    """With no override, an episode with both sources exports from its default_source (optitrack)."""
    scene = _build_scene(tmp_path, n=60, with_slam=True)
    out = tmp_path / 'buf.zarr.zip'

    n_eps, provenance = export_scene_to_dp(scene, out)

    assert n_eps == 1
    assert provenance[0]['source'] == 'optitrack'
    pos = _open_zip(out)['data/robot0_eef_pos'][:]
    assert pos[0, 0] == pytest.approx(0.0, abs=1e-5)  # optitrack trajectory starts at x=0, not x=10


def test_pose_source_override_routes_to_slam(tmp_path: pathlib.Path) -> None:
    """A scene.json pose_source_overrides entry (keyed by session dir) routes that episode to slam."""
    scene = _build_scene(tmp_path, n=60, with_slam=True)
    SceneManifest(scene_id='x', pose_source_overrides={'session_0': 'slam'}).write_to_scene_dir(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    n_eps, provenance = export_scene_to_dp(scene, out)

    assert n_eps == 1
    assert provenance[0]['source'] == 'slam'
    assert provenance[0]['world_frame'] == 'slam'
    pos = _open_zip(out)['data/robot0_eef_pos'][:]
    assert pos[0, 0] == pytest.approx(10.0, abs=1e-5)  # slam trajectory is offset +10 in x


def test_pose_source_override_to_unavailable_source_raises(tmp_path: pathlib.Path) -> None:
    """Overriding to a source the episode never computed (no slam here) fails clearly, not a KeyError."""
    scene = _build_scene(tmp_path, n=30, with_slam=False)
    SceneManifest(scene_id='x', pose_source_overrides={'session_0': 'slam'}).write_to_scene_dir(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match='pose_source_overrides'):
        export_scene_to_dp(scene, out)


def test_pose_provenance_embedded_in_meta_attrs(tmp_path: pathlib.Path) -> None:
    """The .zarr.zip carries the same provenance the function returns, as meta attrs."""
    scene_a = _build_scene(tmp_path / 'a', n=30, with_slam=True)
    scene_b = _build_scene(tmp_path / 'b', n=30)
    SceneManifest(scene_id='a', pose_source_overrides={'session_0': 'slam'}).write_to_scene_dir(tmp_path / 'a')
    out = tmp_path / 'combined.zarr.zip'

    n_eps, provenance = export_scenes_to_dp([scene_a, scene_b], out)

    assert n_eps == 2
    meta = _open_zip(out)['meta']
    assert meta.attrs['pose_provenance'] == provenance
    assert meta.attrs['episode_pose_source'] == ['slam', 'optitrack']


# ---------------------------------------------------------------------------
# Fed-grid steps and segment-per-episode (the UMI discard/trim/split migration)
# ---------------------------------------------------------------------------


def _set_stride(scene: pathlib.Path, stride: int) -> None:
    """Record the localizer's frame stride, which is the grid poses actually exist on."""
    root = zarr.open_group(str(scene), mode='a')
    root['episode_0']['annotations'].require_group('slam').attrs['frame_stride'] = stride


def test_steps_are_the_fed_grid_not_every_frame(tmp_path: pathlib.Path) -> None:
    """
    At stride 2 the export holds half the GoPro frames, one per frame SLAM was fed.

    Poses only exist on that grid — nothing is interpolated since pzarr v4 — so exporting the
    full grid would make every other row NaN and leave no contiguous run to sample from.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n)
    _set_stride(scene, 2)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    assert int(_open_zip(out)['meta/episode_ends'][-1]) == n // 2


def test_exported_step_period_matches_the_stride(tmp_path: pathlib.Path) -> None:
    """
    Consecutive steps are stride/rate apart, uniformly.

    This is the number the training config's obs_down_sample_steps divides: if the stored
    period changes and that factor doesn't, the policy trains at a different rate than it runs.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n)
    _set_stride(scene, 2)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    pos = _open_zip(out)['data/robot0_eef_pos'][:]
    # The fixture translates linearly along x, so equal steps in x mean equal steps in time.
    dx = np.diff(pos[:, 0])
    assert np.allclose(dx, dx[0], atol=1e-6)
    # Half as many steps over the same span => each covers twice the ground of a stride-1 step.
    assert dx[0] == pytest.approx(0.5 / (n - 1) * 2, rel=1e-3)


def test_pose_gap_splits_one_session_into_two_episodes(tmp_path: pathlib.Path) -> None:
    """
    A hole in the pose source ends one episode and starts another, rather than truncating.

    Bridging the gap would put a step of the wrong duration inside an episode, which UMI's
    fixed-rate sampler cannot see; keeping only the longer side would throw away real data.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n, nan_rows=slice(50, 56))
    out = tmp_path / 'buf.zarr.zip'

    n_eps, provenance = export_scene_to_dp(scene, out)

    assert n_eps == 2
    ends = list(_open_zip(out)['meta/episode_ends'][:])
    assert ends == [50, 50 + (n - 56)]
    assert [p['segment'] for p in provenance] == [0, 1]
    assert provenance[0]['frame_range'] == [0, 49]
    assert provenance[1]['frame_range'] == [56, n - 1]


def test_segment_shorter_than_the_floor_is_dropped(tmp_path: pathlib.Path) -> None:
    """A run too short to sample a horizon from is discarded, not emitted as an episode."""
    n = 120
    # 10 valid steps, then a hole, then the rest: the first run is under the floor set below.
    scene = _build_scene(tmp_path, n=n, nan_rows=slice(10, 20))
    out = tmp_path / 'buf.zarr.zip'

    n_eps, provenance = export_scene_to_dp(scene, out, min_segment_steps=24)

    assert n_eps == 1
    assert provenance[0]['frame_range'] == [20, n - 1]


def test_min_segment_steps_is_configurable(tmp_path: pathlib.Path) -> None:
    """Lowering the floor keeps the short run that the default would have dropped."""
    scene = _build_scene(tmp_path, n=120, nan_rows=slice(10, 20))
    n_eps, _ = export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip', min_segment_steps=5)
    assert n_eps == 2


def test_episode_with_no_long_enough_run_is_skipped(tmp_path: pathlib.Path) -> None:
    """An episode chopped into fragments by tracking losses exports nothing at all."""
    scene = _build_scene(tmp_path, n=120)
    root = zarr.open_group(str(scene), mode='a')
    pose = root['episode_0']['eef']['pose_optitrack'][:]
    pose[::3] = np.nan  # no run longer than 2 steps survives
    root['episode_0']['eef']['pose_optitrack'][:] = pose

    with pytest.raises(RuntimeError, match='no EPISODE sessions'):
        export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')


def test_mixing_frame_strides_in_one_buffer_raises(tmp_path: pathlib.Path) -> None:
    """
    Two scenes localized at different strides can't share a buffer.

    UmiDataset reads one episode_ends array and assumes a single uniform Δt across all of it,
    so a mixed-rate buffer would train on two different notions of "one step" with nothing
    recording which episode used which.
    """
    scene_a = _build_scene(tmp_path / 'a', n=120)
    scene_b = _build_scene(tmp_path / 'b', n=120)
    _set_stride(scene_a, 1)
    _set_stride(scene_b, 2)

    with pytest.raises(RuntimeError, match='mixed-rate'):
        export_scenes_to_dp([scene_a, scene_b], tmp_path / 'combined.zarr.zip')


def test_gripper_width_is_exported_as_opening_from_closed(tmp_path: pathlib.Path, monkeypatch) -> None:
    """
    Widths must leave the exporter shifted by closed width, matching UMI's convention.

    The fixture writes raw ArUco tag separation (what step 4 stores); the buffer must contain
    opening-from-closed. Skipping this subtraction is invisible — the numbers stay plausible and
    the policy just learns a frame offset from every commanded width.
    """
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.0052)
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    exported = np.asarray(g['data/robot0_gripper_width'][:]).ravel()
    # The fixture's raw series is linspace(0.02, 0.08, n), subsampled by the frame stride.
    assert exported.min() == pytest.approx(0.02 - 0.0052, abs=1e-6)
    assert float(g['meta'].attrs['gripper_closed_width_m']) == pytest.approx(0.0052)


def test_exported_width_reaches_zero_when_the_gripper_was_fully_closed(tmp_path: pathlib.Path, monkeypatch) -> None:
    """The point of the convention: a fully-closed gripper reads 0, not 'whatever the tags say'."""
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.02)
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    exported = np.asarray(_open_zip(out)['data/robot0_gripper_width'][:]).ravel()
    assert exported.min() == pytest.approx(0.0, abs=1e-6)


def test_detections_below_s_closed_clamp_to_zero_rather_than_going_negative(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """
    The closed width is a percentile, so ~1% of real detections sit under it and would go negative.

    UMI clamps here too, though not visibly: get_gripper_calibration_interpolator (in the upstream
    repo's umi/common/interpolation_util.py) builds an interp1d over [min_width, max_width] with
    fill_value=(x[0], x[-1]), so a measured width under the calibrated minimum saturates to 0
    instead of running past it.
    """
    # 0.03 lands mid-series, so the fixture's linspace(0.02, 0.08) straddles it.
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.03)
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    exported = np.asarray(_open_zip(out)['data/robot0_gripper_width'][:]).ravel()
    assert exported.min() == 0.0, 'widths below closed width must clamp, not go negative'
    # Well inside the calibrated stroke, so the top clamp doesn't bite here.
    assert exported.max() == pytest.approx(0.08 - 0.03, abs=1e-6)


def test_widths_above_the_calibrated_stroke_clamp_to_it(tmp_path: pathlib.Path, monkeypatch) -> None:
    """
    The gripper has a hard stop, so a separation past fully-open is a misread tag, not a wider demo.

    Left unclamped it corrupts the whole channel rather than one frame: UMI normalizes
    gripper_width by min/max, so a single bad detection sets the top of the range and squeezes
    the real signal into a fraction of it. red_trapezoid_mug_v4 had exactly one such sample, at
    2.4x the p99.9, and it cost ~59% of the channel's resolution.
    """
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.02)
    monkeypatch.setattr(buffer, 'load_open_width_m', lambda: 0.05)  # stroke = 30 mm
    scene = _build_scene(tmp_path)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    g = _open_zip(out)
    exported = np.asarray(g['data/robot0_gripper_width'][:]).ravel()
    # Raw series runs to 0.08 -> 0.06 of opening, which is past the 0.03 stroke.
    assert exported.max() == pytest.approx(0.03, abs=1e-6)
    assert exported.min() == pytest.approx(0.0, abs=1e-6)
    assert float(g['meta'].attrs['gripper_open_width_m']) == pytest.approx(0.05)


def test_clamping_the_top_end_warns(tmp_path: pathlib.Path, monkeypatch, caplog) -> None:
    """
    A miscalibrated open_mm must announce itself rather than silently reshaping every width.

    This is what replaced simply not clamping. The original objection to a top clamp was that it
    would hide a demo opening wider than the calibration recording — the warning keeps that
    information visible while still bounding the data.
    """
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.02)
    monkeypatch.setattr(buffer, 'load_open_width_m', lambda: 0.05)
    scene = _build_scene(tmp_path)

    with caplog.at_level(logging.WARNING, logger='export.dp'):
        export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')

    assert any('exceed the calibrated stroke' in r.message for r in caplog.records)


def test_no_warning_when_every_width_is_within_the_stroke(tmp_path: pathlib.Path, monkeypatch, caplog) -> None:
    """The warning has to stay quiet on healthy data, or it trains people to ignore it."""
    monkeypatch.setattr(buffer, 'load_closed_width_m', lambda: 0.02)
    monkeypatch.setattr(buffer, 'load_open_width_m', lambda: 0.20)  # stroke far exceeds the fixture
    scene = _build_scene(tmp_path)

    with caplog.at_level(logging.WARNING, logger='export.dp'):
        export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip')

    assert not any('exceed the calibrated stroke' in r.message for r in caplog.records)
