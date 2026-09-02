"""
Tests for `pingest export --dry-run`: the export preview.

The preview is only worth having if it is cheap and if it selects the same episodes a real
export would. Both are asserted here rather than assumed. What it *reports* per segment needs
no test of its own — it prints ``EpisodePlan.segment_record``, the same call the real export
builds its provenance from, so the two cannot disagree by construction.
"""

import json
import pathlib

from polyumi_ingest.export.dp import buffer
from polyumi_ingest.main import app
from typer.testing import CliRunner

from test_dp_export import _add_pose_jump, _build_scene, _make_slam_only, _slam_counts
from test_polyumi_export import _add_contact_audio, _add_finger_camera

runner = CliRunner()

#: Small enough for the fixtures below, which are 120 steps. See export_floor.py.
FLOOR = ('--min-segment-steps', '24')


def _dry_run_json(scene: pathlib.Path, *extra: str) -> list[dict]:
    result = runner.invoke(app, ['export', str(scene), '--dry-run', '--json', *FLOOR, *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_preview_finds_the_same_segments_the_export_writes(tmp_path: pathlib.Path) -> None:
    """
    Same episode selection, same cuts, same spans — on a session cut by both causes.

    The shared ``segment_record`` guarantees the *fields* agree; what this pins is that the
    preview walks the same sessions and plans them the same way, which is the part that could
    still drift as ``iter_exportable_episodes`` grows.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n, nan_rows=slice(50, 56), with_slam=True)
    _make_slam_only(scene, max_pose_jump_m=3.0, **_slam_counts(n_fed=n, n_lost=6))
    _add_pose_jump(scene, at=90, metres=1.0)

    preview = _dry_run_json(scene)
    _, provenance = buffer.export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip', min_segment_steps=24)

    compared = (
        'scene',
        'episode',
        'segment',
        'source',
        'n_steps',
        'frame_range',
        'frame_stride',
        'duration_s',
        'cut_start',
        'cut_end',
    )
    assert [{k: r[k] for k in compared} for r in preview] == [{k: r[k] for k in compared} for r in provenance]
    assert [r['cut_end'] for r in preview] == ['pose_gap', 'pose_jump', 'episode_end']


def test_preview_decodes_no_frames(tmp_path: pathlib.Path, monkeypatch) -> None:
    """
    The whole point of the preview is that it costs seconds, not the minutes an export does.

    Frame decode is the expensive part, so the guard is simply that the decode path is never
    reached — a preview that opened the mp4 per session would be a preview nobody runs.
    """
    scene = _build_scene(tmp_path, n=120)

    def _boom(*args, **kwargs):
        raise AssertionError('a dry run must not decode frames')

    monkeypatch.setattr(buffer, '_decode_resized_frames', _boom)
    monkeypatch.setattr(buffer, 'open_gopro_frames', _boom)

    assert len(_dry_run_json(scene)) == 1


def test_preview_writes_nothing(tmp_path: pathlib.Path) -> None:
    """--dry-run needs no -o, and must not leave a buffer behind."""
    scene = _build_scene(tmp_path, n=120)

    result = runner.invoke(app, ['export', str(scene), '--dry-run', *FLOOR])

    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob('*.zarr.zip')) == []


def test_export_without_output_is_an_error(tmp_path: pathlib.Path) -> None:
    """The converse: -o stays required for a real export, rather than silently doing nothing."""
    scene = _build_scene(tmp_path, n=120)

    result = runner.invoke(app, ['export', str(scene), *FLOOR])

    assert result.exit_code == 1


def test_preview_reports_runs_dropped_by_the_floor(tmp_path: pathlib.Path) -> None:
    """A run below the floor is reported as dropped rather than silently vanishing."""
    scene = _build_scene(tmp_path, n=120, nan_rows=slice(10, 20))

    result = runner.invoke(app, ['export', str(scene), '--dry-run', *FLOOR])

    assert result.exit_code == 0, result.output
    assert '1 run(s)' in result.stdout  # the 10-step head, under the 24-step floor


def test_preview_type_polyumi_attaches_the_modalities(tmp_path: pathlib.Path) -> None:
    """
    --type polyumi must build modality *instances*, as the real exporter does.

    A modality stashes per-episode state on ``self``, so passing the classes through instead
    fails at the first ``prepare_episode`` — and only on the non-default flag, where nothing
    else would catch it.
    """
    scene = _build_scene(tmp_path, n=120)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    result = runner.invoke(app, ['export', str(scene), '--dry-run', '--type', 'polyumi', *FLOOR])

    assert result.exit_code == 0, result.output
    assert '1 segment(s)' in result.stdout


def test_preview_exits_nonzero_on_a_missing_scene(tmp_path: pathlib.Path) -> None:
    """A bad path fails the command rather than reporting an empty, reassuring plan."""
    result = runner.invoke(app, ['export', str(tmp_path / 'nope'), '--dry-run'])
    assert result.exit_code == 1
