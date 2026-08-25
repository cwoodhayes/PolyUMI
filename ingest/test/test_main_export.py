"""Tests for the `pingest export` CLI command: --type dispatch and validation."""

import pathlib

from polyumi_ingest.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _invoke(tmp_path: pathlib.Path, *extra_args: str) -> object:
    scene = tmp_path / 'scene'
    scene.mkdir()
    out = tmp_path / 'out.zarr.zip'
    return runner.invoke(app, ['export', str(scene), '-o', str(out), *extra_args])


def test_default_type_dispatches_to_export_scenes_to_dp(tmp_path: pathlib.Path, monkeypatch) -> None:
    """`pingest export` with no --type runs the visuomotor-only exporter."""
    calls = []
    monkeypatch.setattr(
        'polyumi_ingest.export.dp.export_scenes_to_dp', lambda *a, **kw: calls.append(('dp', a, kw)) or (0, [])
    )
    monkeypatch.setattr(
        'polyumi_ingest.export.dp.export_scenes_to_polyumi',
        lambda *a, **kw: calls.append(('polyumi', a, kw)) or (0, []),
    )

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ['dp']


def test_type_polyumi_dispatches_to_export_scenes_to_polyumi(tmp_path: pathlib.Path, monkeypatch) -> None:
    """`--type polyumi` runs the modality-carrying exporter instead."""
    calls = []
    monkeypatch.setattr(
        'polyumi_ingest.export.dp.export_scenes_to_dp', lambda *a, **kw: calls.append(('dp', a, kw)) or (0, [])
    )
    monkeypatch.setattr(
        'polyumi_ingest.export.dp.export_scenes_to_polyumi',
        lambda *a, **kw: calls.append(('polyumi', a, kw)) or (0, []),
    )

    result = _invoke(tmp_path, '--type', 'polyumi')

    assert result.exit_code == 0, result.output
    assert [c[0] for c in calls] == ['polyumi']


def test_unknown_type_is_rejected_not_silently_defaulted(tmp_path: pathlib.Path) -> None:
    """A typo in --type is a usage error, not a quiet fall-through to the visuomotor exporter."""
    result = _invoke(tmp_path, '--type', 'bogus')

    assert result.exit_code != 0
    assert 'bogus' in result.output
