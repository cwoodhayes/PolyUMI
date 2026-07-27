"""
Tests for the Phase 2.5 MCAP/Foxglove glue.

``export_session_to_mcap`` and ``open_in_foxglove`` are thin wrappers around
``polyumi_ingest``'s real exporter and the local ``foxglove-studio`` binary; those
are exercised elsewhere (ingest's own test suite, and manual smoke testing), so here
we monkeypatch them and test only the glue this module actually adds: resolving a
session to its pzarr episode index, and the pzarr/mcap presence checks that drive
the UI's button states.
"""

from __future__ import annotations

import pathlib

import pytest
import zarr
from polyumi_catalog import mcap_tools


def _make_pzarr(scene_dir: pathlib.Path, session_dirnames: list[str]) -> None:
    """Build a minimal scene.zarr with one episode group per session dirname."""
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['n_episodes'] = len(session_dirnames)
    for i, dirname in enumerate(session_dirnames):
        ep = root.require_group(f'episode_{i}')
        ep.attrs['session_dir'] = dirname


def test_pzarr_exists(tmp_path: pathlib.Path):
    """pzarr_exists reflects whether scene.zarr is present, nothing more."""
    scene_dir = tmp_path / 'scene_a'
    scene_dir.mkdir()
    assert mcap_tools.pzarr_exists(scene_dir) is False
    _make_pzarr(scene_dir, ['session_1'])
    assert mcap_tools.pzarr_exists(scene_dir) is True


def test_resolve_episode_index_matches_session_dir(tmp_path: pathlib.Path):
    """The episode index is recovered by matching session_dir, not by dict order alone."""
    scene_dir = tmp_path / 'scene_b'
    scene_dir.mkdir()
    _make_pzarr(scene_dir, ['session_mapping', 'session_ep0', 'session_ep1'])

    assert mcap_tools.resolve_episode_index(scene_dir, 'session_ep1') == 2
    assert mcap_tools.resolve_episode_index(scene_dir, 'session_mapping') == 0


def test_resolve_episode_index_without_pzarr_returns_none(tmp_path: pathlib.Path):
    """No scene.zarr at all (pzarr not yet built) resolves to None, not an error."""
    scene_dir = tmp_path / 'scene_c'
    scene_dir.mkdir()
    assert mcap_tools.resolve_episode_index(scene_dir, 'session_1') is None


def test_resolve_episode_index_unknown_session_returns_none(tmp_path: pathlib.Path):
    """A session_dir with no matching episode group resolves to None."""
    scene_dir = tmp_path / 'scene_d'
    scene_dir.mkdir()
    _make_pzarr(scene_dir, ['session_1'])
    assert mcap_tools.resolve_episode_index(scene_dir, 'session_nonexistent') is None


def test_mcap_path_for_session_absent_vs_present(tmp_path: pathlib.Path):
    """mcap_path_for_session returns None until the .mcap file actually exists on disk."""
    scene_dir = tmp_path / 'scene_e'
    scene_dir.mkdir()
    _make_pzarr(scene_dir, ['session_1'])

    assert mcap_tools.mcap_path_for_session(scene_dir, 'session_1') is None
    (scene_dir / 'episode_0.mcap').write_bytes(b'\x89MCAP0\r\n')
    path = mcap_tools.mcap_path_for_session(scene_dir, 'session_1')
    assert path == scene_dir / 'episode_0.mcap'


def test_export_session_to_mcap_raises_without_pzarr_episode(tmp_path: pathlib.Path):
    """Exporting a session with no matching pzarr episode raises McapError, not a crash."""
    scene_dir = tmp_path / 'scene_f'
    scene_dir.mkdir()
    with pytest.raises(mcap_tools.McapError):
        mcap_tools.export_session_to_mcap(scene_dir, 'session_1')


def test_export_session_to_mcap_calls_ingest_exporter_with_resolved_index(tmp_path: pathlib.Path, monkeypatch):
    """The resolved episode index is passed through to polyumi_ingest's real exporter."""
    scene_dir = tmp_path / 'scene_g'
    scene_dir.mkdir()
    _make_pzarr(scene_dir, ['session_mapping', 'session_ep0'])

    calls = []

    def fake_export_scene_to_mcap(scene_path, episode=None, **kwargs):
        calls.append((scene_path, episode))
        out = scene_path / f'episode_{episode}.mcap'
        out.write_bytes(b'fake')
        return [out]

    monkeypatch.setattr('polyumi_ingest.export.mcap.export_scene_to_mcap', fake_export_scene_to_mcap)

    result = mcap_tools.export_session_to_mcap(scene_dir, 'session_ep0')
    assert calls == [(scene_dir, 1)]
    assert result == scene_dir / 'episode_1.mcap'
    assert result.is_file()


def test_export_session_to_mcap_raises_if_exporter_writes_nothing(tmp_path: pathlib.Path, monkeypatch):
    """If the real exporter returns an empty list, that's surfaced as McapError."""
    scene_dir = tmp_path / 'scene_h'
    scene_dir.mkdir()
    _make_pzarr(scene_dir, ['session_1'])
    monkeypatch.setattr('polyumi_ingest.export.mcap.export_scene_to_mcap', lambda *a, **k: [])

    with pytest.raises(mcap_tools.McapError):
        mcap_tools.export_session_to_mcap(scene_dir, 'session_1')


def test_open_in_foxglove_raises_for_missing_file(tmp_path: pathlib.Path):
    """Launching Foxglove on a nonexistent path is rejected before shelling out."""
    with pytest.raises(mcap_tools.McapError):
        mcap_tools.open_in_foxglove(tmp_path / 'does_not_exist.mcap')


def test_open_in_foxglove_raises_if_binary_missing(tmp_path: pathlib.Path, monkeypatch):
    """If foxglove-studio isn't on PATH, that's a clear McapError, not a subprocess crash."""
    mcap_path = tmp_path / 'episode_0.mcap'
    mcap_path.write_bytes(b'fake')
    monkeypatch.setattr('shutil.which', lambda name: None)

    with pytest.raises(mcap_tools.McapError):
        mcap_tools.open_in_foxglove(mcap_path)


def test_open_in_foxglove_launches_subprocess(tmp_path: pathlib.Path, monkeypatch):
    """When the binary is found, it's launched (detached) with the mcap path as its argument."""
    mcap_path = tmp_path / 'episode_0.mcap'
    mcap_path.write_bytes(b'fake')
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/foxglove-studio')

    calls = []
    monkeypatch.setattr('subprocess.Popen', lambda args, **kwargs: calls.append((args, kwargs)))

    mcap_tools.open_in_foxglove(mcap_path)
    assert calls == [(['/usr/bin/foxglove-studio', str(mcap_path)], {'start_new_session': True})]
