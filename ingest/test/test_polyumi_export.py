"""Tests for `export-polyumi`: the visuomotor buffer plus PolyUMI's extra modalities."""

import pathlib

import numpy as np
import pytest
import zarr
from polyumi_ingest.config import load_contact_audio_config
from polyumi_ingest.export.dp import export_scene_to_dp, export_scenes_to_polyumi
from polyumi_ingest.preproc import available_preprocessing_steps

from test_dp_export import ALL_STEPS, EXPECTED_KEYS, _build_scene, _open_zip

BLOCKS = load_contact_audio_config()['blocks']
BLOCK_WIDTH = int(BLOCKS['samples_per_gopro_frame'])
ALIGNMENT = str(BLOCKS['block_alignment'])
AUDIO_STEP = next(cls.step_number for cls in available_preprocessing_steps() if not cls.required_for_export)


def _add_contact_audio(scene: pathlib.Path, *, frame_stride: int = 1, n_frames: int | None = None) -> None:
    """
    Attach step-6 output whose block *k* holds the integers ``[k*W, (k+1)*W)``.

    Blocks carry their own source-sample indices, so a test can read a `mic_0` row and say
    exactly which frames it was assembled from — and whether the concatenation left a hole.
    Real anchors overlap slightly; a clean tiling is the stricter case for the gap assertions
    and makes the arithmetic legible.
    """
    root = zarr.open_group(str(scene), mode='a')
    ep = root['episode_0']
    n = n_frames if n_frames is not None else ep['timestamps/gopro'].shape[0]

    starts = np.arange(n, dtype=np.int64) * BLOCK_WIDTH
    blocks = (starts[:, None] + np.arange(BLOCK_WIDTH, dtype=np.int64)[None, :]).astype(np.float32)

    grp = ep['annotations'].require_group('contact_audio')
    grp.create_array('frame_blocks', data=blocks)
    grp.create_array('frame_block_start_idx', data=starts)
    grp.attrs['sample_rate_hz'] = int(BLOCKS['sample_rate_hz'])
    grp.attrs['samples_per_gopro_frame'] = BLOCK_WIDTH

    if frame_stride != 1:
        ep['annotations'].require_group('slam').attrs['frame_stride'] = frame_stride


def test_mic_0_rides_alongside_the_visuomotor_keys(tmp_path: pathlib.Path) -> None:
    """export-polyumi emits everything export-dp does, plus mic_0."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    out = tmp_path / 'buf.zarr.zip'

    n_eps, _ = export_scenes_to_polyumi([scene], out)

    data = _open_zip(out)['data']
    assert n_eps == 1
    assert set(data.keys()) == set(EXPECTED_KEYS) | {'mic_0'}
    mic = data['mic_0']
    assert mic.shape == (60, BLOCK_WIDTH)  # stride 1 -> one frame-block per step
    assert mic.dtype == np.float32


def test_mic_0_row_width_follows_the_stride(tmp_path: pathlib.Path) -> None:
    """A step spans `stride` frames, so its audio row is that many blocks wide."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    out = tmp_path / 'buf.zarr.zip'

    export_scenes_to_polyumi([scene], out)

    root = _open_zip(out)
    mic = root['data/mic_0']
    assert mic.shape == (30, 2 * BLOCK_WIDTH)
    assert int(root['meta'].attrs['mic_0_samples_per_step']) == 2 * BLOCK_WIDTH


def test_consecutive_rows_form_a_gapless_waveform(tmp_path: pathlib.Path) -> None:
    """
    The property the whole design rests on: flattening mic_0 row-major loses no audio.

    ManiWAV's audio path concatenates the rows back into one signal before the mel, so a hole
    between steps would be inaudible in any shape assertion and wrong in every spectrogram.
    """
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    out = tmp_path / 'buf.zarr.zip'

    export_scenes_to_polyumi([scene], out)

    mic = np.asarray(_open_zip(out)['data/mic_0'][:]).astype(np.int64)
    # Every step after the first is fully covered by real blocks, and they abut exactly.
    stream = mic[1:].reshape(-1)
    assert np.array_equal(stream, np.arange(stream[0], stream[0] + len(stream)))
    # ...and the first row runs straight into the second, so the join is seamless too.
    assert mic[0, -1] + 1 == mic[1, 0]


def test_first_causal_step_is_padded_with_silence(tmp_path: pathlib.Path) -> None:
    """
    The one step with no history gets zeros, not a duplicate of the audio that follows it.

    Repeating the nearest block would splice a copy of real audio into the waveform, where it
    would read as a genuine contact event. ManiWAV zero-pads audio at episode start likewise.
    """
    if ALIGNMENT != 'causal':
        pytest.skip(f'block_alignment is {ALIGNMENT!r}; only causal alignment pads at the start')
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    out = tmp_path / 'buf.zarr.zip'

    export_scenes_to_polyumi([scene], out)

    mic = np.asarray(_open_zip(out)['data/mic_0'][:])
    assert (mic[0, :BLOCK_WIDTH] == 0.0).all()
    assert (mic[0, BLOCK_WIDTH:] != 0.0).any()
    assert (mic[1:] != 0.0).any()


def test_causal_alignment_never_reads_ahead_of_the_step(tmp_path: pathlib.Path) -> None:
    """Causal rows end at their step: no audio the policy could not have heard yet."""
    if ALIGNMENT != 'causal':
        pytest.skip(f'block_alignment is {ALIGNMENT!r}; this pins the causal contract')
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    out = tmp_path / 'buf.zarr.zip'

    export_scenes_to_polyumi([scene], out)

    mic = np.asarray(_open_zip(out)['data/mic_0'][:]).astype(np.int64)
    # Step k is GoPro frame 2k, whose block ends at (2k+1)*W - 1. A forward-aligned row would
    # end a full stride later, i.e. at (2k+2)*W - 1.
    steps = np.arange(mic.shape[0])
    assert np.array_equal(mic[:, -1], (2 * steps + 1) * BLOCK_WIDTH - 1)


def test_meta_attrs_describe_the_audio_contract(tmp_path: pathlib.Path) -> None:
    """A checkpoint has to be able to say what geometry it was trained under."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)
    out = tmp_path / 'buf.zarr.zip'

    export_scenes_to_polyumi([scene], out)

    attrs = _open_zip(out)['meta'].attrs
    assert int(attrs['mic_0_sample_rate_hz']) == int(BLOCKS['sample_rate_hz'])
    assert int(attrs['mic_0_samples_per_gopro_frame']) == BLOCK_WIDTH
    assert int(attrs['mic_0_samples_per_step']) == 2 * BLOCK_WIDTH
    assert attrs['mic_0_block_alignment'] == ALIGNMENT
    assert attrs['mic_0_source'] == 'finger/finger_piezo'


def test_provenance_records_the_modality(tmp_path: pathlib.Path) -> None:
    """Per-episode provenance gains a modalities entry; export-dp's must stay without one."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, frame_stride=2)

    _, audio_prov = export_scenes_to_polyumi([scene], tmp_path / 'audio.zarr.zip')
    _, visuo_prov = export_scene_to_dp(scene, tmp_path / 'visuo.zarr.zip')

    assert audio_prov[0]['modalities']['mic_0'] == {
        'samples_per_step': 2 * BLOCK_WIDTH,
        'block_alignment': ALIGNMENT,
    }
    assert 'modalities' not in visuo_prov[0]


def test_export_dp_on_the_same_scene_has_no_mic(tmp_path: pathlib.Path) -> None:
    """The visuomotor export is unchanged by a scene happening to carry audio."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    out = tmp_path / 'buf.zarr.zip'

    export_scene_to_dp(scene, out)

    root = _open_zip(out)
    assert set(root['data'].keys()) == set(EXPECTED_KEYS)
    assert not [k for k in root['meta'].attrs if k.startswith('mic_0')]


def test_missing_step_6_names_the_command_to_run(tmp_path: pathlib.Path) -> None:
    """The modality demands its own step, and says how to satisfy it."""
    scene = _build_scene(tmp_path, n=60, preprocessing_steps=[s for s in ALL_STEPS if s != AUDIO_STEP])
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match=rf'preprocessing steps \[{AUDIO_STEP}\]'):
        export_scenes_to_polyumi([scene], out)


def test_missing_blocks_with_the_step_marked_done_still_raises(tmp_path: pathlib.Path) -> None:
    """A step-6 mark without its output is a broken store, not a reason to export silently."""
    scene = _build_scene(tmp_path, n=60)  # marked complete, but no contact_audio group written
    out = tmp_path / 'buf.zarr.zip'

    with pytest.raises(RuntimeError, match='pingest pp 6'):
        export_scenes_to_polyumi([scene], out)


def test_block_width_mismatch_refuses_to_mix(tmp_path: pathlib.Path) -> None:
    """One buffer, one block width — a mismatch is a re-run, not something to reshape around."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene)
    root = zarr.open_group(str(scene), mode='a')
    grp = root['episode_0/annotations/contact_audio']
    del grp['frame_blocks']
    grp.create_array('frame_blocks', data=np.zeros((60, BLOCK_WIDTH - 1), dtype=np.float32))

    with pytest.raises(RuntimeError, match='samples wide'):
        export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')


def test_block_count_must_match_the_gopro_grid(tmp_path: pathlib.Path) -> None:
    """Blocks live on the GoPro frame grid; a short array means step 6 ran on different input."""
    scene = _build_scene(tmp_path, n=60)
    _add_contact_audio(scene, n_frames=50)

    with pytest.raises(RuntimeError, match='GoPro grid'):
        export_scenes_to_polyumi([scene], tmp_path / 'buf.zarr.zip')
