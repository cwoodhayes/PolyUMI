"""
Tests for the finger-camera modality: `finger/frames` cropped onto the exported step grid.

The two things worth pinning here are the ones no shape assertion catches. **Alignment** — each
step must get the finger frame nearest it *after* the GoPro→finger clock hop, and the fixture
encodes each frame's index in its pixels so a wrong pairing is nameable rather than merely
different. **Coverage** — `nearest_idx` clamps, so a finger stream that stops early would hand
every remaining step the same frozen frame; the staleness guard is what turns that into a
failure instead of a plausible-looking dataset.
"""

import pathlib

import numpy as np
import pytest
import zarr
from polyumi_ingest.config import load_finger_camera_config
from export_floor import export_scene_to_dp, export_scenes_to_polyumi
from test_dp_export import EXPECTED_KEYS, _build_scene, _open_zip
from test_polyumi_export import FINGER_FPS, FINGER_OFFSET_S, _add_contact_audio, _add_finger_camera


CFG = load_finger_camera_config()
CROP = CFG['crop']
X_MIN = int(CROP['x_min'])


def _export(tmp_path: pathlib.Path, scene: pathlib.Path, name: str = 'buf.zarr.zip') -> zarr.Group:
    """Run export_scenes_to_polyumi and open the result the way UmiDataset would."""
    out = tmp_path / name
    export_scenes_to_polyumi([scene], out)
    return _open_zip(out)


def _expected_source_frames(scene: pathlib.Path, stride: int) -> np.ndarray:
    """
    Derive, independently of the code, the finger frame index each exported step should carry.

    Rounding the elapsed time onto the finger grid rather than reusing ``nearest_idx`` — a test
    that calls the function under test to compute its own expectation asserts nothing.
    """
    ep = zarr.open_group(str(scene), mode='r')['episode_0']
    gopro_ts = np.asarray(ep['timestamps/gopro'][:], dtype=np.float64)[::stride]
    finger_ts = np.asarray(ep['timestamps/finger'][:], dtype=np.float64)
    elapsed = (gopro_ts - FINGER_OFFSET_S) - finger_ts[0]
    return np.clip(np.round(elapsed * FINGER_FPS), 0, len(finger_ts) - 1).astype(np.int64)


def _row_values(finger: np.ndarray) -> np.ndarray:
    """Each exported row is a constant image; recover that constant per row."""
    flat = finger.reshape(finger.shape[0], -1)
    assert (flat == flat[:, :1]).all(), 'a row is not a single source frame'
    return flat[:, 0].astype(np.int64)


def test_finger_rgb_rides_alongside_the_other_keys(tmp_path: pathlib.Path) -> None:
    """export_scenes_to_polyumi emits everything the plain visuomotor export does, plus both PolyUMI modalities."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, width=400, height=120)

    data = _export(tmp_path, scene)['data']

    assert set(data.keys()) == set(EXPECTED_KEYS) | {'mic_0', 'finger_rgb'}
    finger = data['finger_rgb']
    assert finger.shape == (60, 120, 400 - X_MIN, 3)
    assert finger.dtype == np.uint8


def test_shape_at_the_real_finger_resolution(tmp_path: pathlib.Path) -> None:
    """
    1152x648 with the shipped crop is (648, 982) — the shape a policy config has to name.

    The real recorded resolution, which is NOT ``cam_streamer``'s ``VIEW_WIDTH``/``VIEW_HEIGHT``
    (620x480) — those size the preview stream, not the stored JPEGs.
    """
    scene = _build_scene(tmp_path, n=30)
    _add_contact_audio(scene)
    _add_finger_camera(scene, width=1152, height=648)

    finger = _export(tmp_path, scene)['data/finger_rgb']

    assert finger.shape == (30, 648, 1152 - X_MIN, 3)


def test_exported_pixels_are_the_cropped_source(tmp_path: pathlib.Path) -> None:
    """The kept columns are exactly the source minus the occluded strip, unresampled."""
    scene = _build_scene(tmp_path, n=30)
    _add_contact_audio(scene)
    _add_finger_camera(scene, width=400, height=120)
    source = np.asarray(zarr.open_group(str(scene), mode='r')['episode_0/finger/frames'][:])

    finger = np.asarray(_export(tmp_path, scene)['data/finger_rgb'][:])
    rows = _expected_source_frames(scene, stride=1)

    assert np.array_equal(finger, source[rows][:, :, X_MIN:])


def test_each_step_gets_the_finger_frame_nearest_it_in_the_finger_clock(tmp_path: pathlib.Path) -> None:
    """
    The alignment contract, and the reason the clock hop is not decorative.

    The fixture puts the finger stream 5 s away from the GoPro clock, so a hop that dropped or
    flipped ``gopro_to_finger_offset_s`` would land outside the stream entirely — clamped to one
    frame, which is exactly what this compares against a per-step expectation.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    finger = np.asarray(_export(tmp_path, scene)['data/finger_rgb'][:])

    assert np.array_equal(_row_values(finger), _expected_source_frames(scene, stride=1))


def test_a_slow_camera_repeats_frames_rather_than_interpolating(tmp_path: pathlib.Path) -> None:
    """
    10 fps against a ~60 Hz grid means ~6 consecutive steps share a source frame.

    Nothing is blended: a policy must see images the camera actually produced. The duplication is
    the honest consequence, and is why the buffer records the source rate.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    finger = np.asarray(_export(tmp_path, scene)['data/finger_rgb'][:])
    values = _row_values(finger)

    assert len(np.unique(values)) < len(values) // 4
    assert np.all(np.diff(values) >= 0), 'source frames must advance monotonically'


def test_stride_2_still_pairs_every_step_with_its_own_frame(tmp_path: pathlib.Path) -> None:
    """The step grid is the frames SLAM was fed; the finger lookup follows it, not the full grid."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    _add_finger_camera(scene)

    finger = np.asarray(_export(tmp_path, scene)['data/finger_rgb'][:])

    assert finger.shape[0] == 30
    assert np.array_equal(_row_values(finger), _expected_source_frames(scene, stride=2))


def test_a_finger_stream_that_stops_early_is_trimmed_not_rejected(tmp_path: pathlib.Path) -> None:
    """
    The real recording pattern: the finger camera stops ~0.65 s before the GoPro, every episode.

    Measured over 111 episodes in three scenes. Rejecting on it would reject the entire corpus,
    so the uncovered tail is excluded from the export the way a pose dropout is. What must NOT
    happen is exporting it: `nearest_idx` clamps, so those steps would all carry the same frozen
    frame against moving proprioception — invisible to every shape and dtype assertion.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, truncate_s=0.6)

    finger = np.asarray(_export(tmp_path, scene)['data/finger_rgb'][:])

    assert 24 <= finger.shape[0] < 60, 'the uncovered tail should be trimmed, not kept or fatal'
    values = _row_values(finger)
    # The give-away for a frozen tail: the last source frame repeated far more than the ~3 rows
    # a 10 fps camera legitimately occupies on a ~60 Hz grid.
    assert (values == values[-1]).sum() <= 12


def test_trimming_keeps_every_surviving_step_within_tolerance(tmp_path: pathlib.Path) -> None:
    """Whatever survives the trim is a real observation, which is the property policies rely on."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, truncate_s=0.6)

    _, provenance = export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')

    assert provenance[0]['modalities']['finger_rgb']['max_staleness_s'] <= float(CFG['max_staleness_s'])


def test_an_episode_with_no_usable_coverage_is_dropped(tmp_path: pathlib.Path) -> None:
    """
    Trimming past the length floor leaves nothing to export, and that has to be a clean skip.

    The existing segmentation already handles this for pose dropouts; the finger camera reuses
    it rather than inventing a second way for an episode to be unusable.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, truncate_s=0.8)

    with pytest.raises(RuntimeError, match='no EPISODE sessions to export'):
        # Floor pinned above what the truncated finger stream can cover, so the trim empties it.
        export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip', min_segment_steps=24)


def test_a_gap_inside_tolerance_trims_nothing(tmp_path: pathlib.Path) -> None:
    """The guard must not bite on the ordinary case — half a frame period is normal at 10 fps."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, truncate_s=0.05)

    finger = _export(tmp_path, scene)['data/finger_rgb']

    assert finger.shape[0] == 60


def test_a_session_without_a_finger_camera_names_itself(tmp_path: pathlib.Path) -> None:
    """A buffer cannot be half a modality, so this fails rather than exporting a short episode."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)

    with pytest.raises(RuntimeError, match='no finger/frames'):
        export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')


def test_missing_time_sync_refuses_rather_than_pairing_arbitrary_frames(tmp_path: pathlib.Path) -> None:
    """
    Without step 1's chirp offset the two clocks are unrelated epochs.

    Defaulting it to zero would pair each step with whatever finger frame happens to sit at the
    same number of seconds since two different devices' epochs — wrong by seconds, and
    undetectable downstream.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene)
    ep = zarr.open_group(str(scene), mode='a')['episode_0']
    del ep['annotations']['time_sync']

    with pytest.raises(RuntimeError, match='time_sync'):
        export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')


def test_meta_attrs_describe_the_crop_contract(tmp_path: pathlib.Path) -> None:
    """A checkpoint has to be able to say which crop it trained under."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene, width=400, height=120)

    attrs = dict(_export(tmp_path, scene)['meta'].attrs)

    assert attrs['finger_rgb_crop'] == {k: CROP[k] for k in ('x_min', 'x_max', 'y_min', 'y_max')}
    assert attrs['finger_rgb_output_size'] == (list(CFG['output_size']) if CFG['output_size'] else None)
    assert attrs['finger_rgb_shape'] == [120, 400 - X_MIN, 3]
    assert attrs['finger_rgb_source'] == 'finger/frames'
    assert attrs['finger_rgb_source_rate_hz'] == pytest.approx(FINGER_FPS)
    assert attrs['finger_rgb_max_staleness_s'] == pytest.approx(float(CFG['max_staleness_s']))


def test_provenance_records_the_coverage_it_actually_got(tmp_path: pathlib.Path) -> None:
    """Per-segment provenance is where a marginal-but-passing episode becomes visible."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    _, provenance = export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')

    finger = provenance[0]['modalities']['finger_rgb']
    assert finger['max_staleness_s'] <= float(CFG['max_staleness_s'])
    assert finger['median_staleness_s'] <= finger['max_staleness_s']
    assert 0 < finger['n_source_frames'] < 60


def test_export_dp_on_the_same_scene_carries_no_finger_rgb(tmp_path: pathlib.Path) -> None:
    """
    The plain visuomotor export's contract is frozen: the same scene exports the six visuomotor keys and nothing else.

    This is the invariant the modality seam exists to protect — an export with no modalities has
    to be byte-identical to one from before any of them existed.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    export_scene_to_dp(scene, tmp_path / 'vis.zarr.zip')
    _, provenance = export_scene_to_dp(scene, tmp_path / 'vis2.zarr.zip')

    assert set(_open_zip(tmp_path / 'vis.zarr.zip')['data'].keys()) == set(EXPECTED_KEYS)
    assert 'modalities' not in provenance[0]
