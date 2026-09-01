"""
Tests for `pingest segments`: the export preview.

The preview is only worth having if it agrees with the export it predicts, and only cheap if
it never touches the video. Both are asserted here rather than assumed, because both are easy
to break without noticing — the planner and the export loop live in the same module and either
could grow a step the other doesn't have.
"""

import json
import pathlib

from polyumi_ingest.export.dp import buffer
from polyumi_ingest.main import app
from typer.testing import CliRunner

from test_dp_export import _add_pose_jump, _build_scene, _make_slam_only, _slam_counts
from test_polyumi_export import _add_contact_audio, _add_finger_camera

runner = CliRunner()

#: Small enough for the fixtures below, which are 120 steps. See test_dp_export.py.
FLOOR = ('--min-segment-steps', '24')


def _segments_json(scene: pathlib.Path, *extra: str) -> list[dict]:
    result = runner.invoke(app, ['segments', str(scene), '--json', *FLOOR, *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_preview_matches_what_the_export_actually_writes(tmp_path: pathlib.Path) -> None:
    """
    Every segment the preview reports is a segment the export produces, field for field.

    This equality is the only thing keeping the preview honest as the exporter changes; without
    it the two can drift and the preview quietly becomes a lie that costs someone a training run.
    """
    n = 120
    scene = _build_scene(tmp_path, n=n, nan_rows=slice(50, 56), with_slam=True)
    _make_slam_only(scene, max_pose_jump_m=3.0, **_slam_counts(n_fed=n, n_lost=6))
    _add_pose_jump(scene, at=90, metres=1.0)

    preview = _segments_json(scene)
    _, provenance = buffer.export_scene_to_dp(scene, tmp_path / 'buf.zarr.zip', min_segment_steps=24)

    compared = ('scene', 'episode', 'segment', 'source', 'n_steps', 'frame_range', 'cut_start', 'cut_end')
    assert [{k: r[k] for k in compared} for r in preview] == [{k: r[k] for k in compared} for r in provenance]
    # And it really did find the two cut causes, not just agree about an unsplit episode.
    assert len(preview) == 3
    assert [r['cut_end'] for r in preview] == ['pose_gap', 'pose_jump', 'episode_end']


def test_preview_decodes_no_frames(tmp_path: pathlib.Path, monkeypatch) -> None:
    """
    The whole point of the preview is that it costs seconds, not the minutes an export does.

    Frame decode is the expensive part, so the guard is simply that the decode path is never
    reached — a preview that opened the mp4 per session would be a preview nobody runs.
    """
    scene = _build_scene(tmp_path, n=120)

    def _boom(*args, **kwargs):
        raise AssertionError('pingest segments must not decode frames')

    monkeypatch.setattr(buffer, '_decode_resized_frames', _boom)
    monkeypatch.setattr(buffer, 'open_gopro_frames', _boom)

    assert len(_segments_json(scene)) == 1


def test_preview_reports_runs_dropped_by_the_floor(tmp_path: pathlib.Path) -> None:
    """A run below the floor is reported as dropped rather than silently vanishing."""
    scene = _build_scene(tmp_path, n=120, nan_rows=slice(10, 20))

    result = runner.invoke(app, ['segments', str(scene), *FLOOR])

    assert result.exit_code == 0, result.output
    assert '1 run(s)' in result.stdout  # the 10-step head, under the 24-step floor


def test_preview_type_polyumi_attaches_the_modalities(tmp_path: pathlib.Path) -> None:
    """
    --type polyumi must build modality *instances*, as the real exporter does.

    A modality stashes per-episode state on ``self``, so passing the classes through instead
    fails at the first ``prepare_episode`` — which is exactly what this preview did before, and
    only on the non-default flag, where nothing else would have caught it.
    """
    scene = _build_scene(tmp_path, n=120)
    _add_contact_audio(scene)
    _add_finger_camera(scene)

    result = runner.invoke(app, ['segments', str(scene), '--type', 'polyumi', *FLOOR])

    assert result.exit_code == 0, result.output
    assert '1 segment(s)' in result.stdout


def test_preview_exits_nonzero_on_a_missing_scene(tmp_path: pathlib.Path) -> None:
    """A bad path fails the command rather than reporting an empty, reassuring plan."""
    result = runner.invoke(app, ['segments', str(tmp_path / 'nope')])
    assert result.exit_code == 1
