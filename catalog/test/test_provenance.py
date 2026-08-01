"""Tests for the scene detail pane's recording/preprocessing commit provenance."""

from __future__ import annotations

import json
import pathlib

import zarr
from polyumi_catalog import provenance


def _make_session(scene_dir: pathlib.Path, name: str, polyumi_version: str | None) -> pathlib.Path:
    """
    Write a minimal metadata.json directly rather than via SessionMetadata.

    provenance reads the file with plain json, and these tests exercise cases
    (missing key, unparseable file) that the dataclass writer can't produce.
    """
    sd = scene_dir / name
    sd.mkdir(parents=True)
    meta: dict = {'session_id': name, 'scene_id': 'scene-1'}
    if polyumi_version is not None:
        meta['polyumi_version'] = polyumi_version
    (sd / 'metadata.json').write_text(json.dumps(meta))
    return sd


def test_pi_versions_collapses_identical_deploys(tmp_path: pathlib.Path):
    """Sessions recorded off one deploy report a single recording commit, not one per session."""
    scene_dir = tmp_path / 'scene_a'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', 'a' * 40)
    _make_session(scene_dir, 'session_2', 'a' * 40)

    assert provenance.pi_versions(scene_dir) == ['a' * 40]


def test_pi_versions_reports_every_deploy_when_they_differ(tmp_path: pathlib.Path):
    """A mid-scene redeploy shows as multiple commits rather than being collapsed away."""
    scene_dir = tmp_path / 'scene_b'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', 'a' * 40)
    _make_session(scene_dir, 'session_2', 'b' * 40)

    assert set(provenance.pi_versions(scene_dir)) == {'a' * 40, 'b' * 40}


def test_pi_versions_tolerates_missing_and_unreadable_metadata(tmp_path: pathlib.Path):
    """A session with no version, no metadata.json, or corrupt JSON is skipped, not fatal."""
    scene_dir = tmp_path / 'scene_c'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', 'a' * 40)
    _make_session(scene_dir, 'session_2', None)
    (scene_dir / 'session_3').mkdir()
    bad = scene_dir / 'session_4'
    bad.mkdir()
    (bad / 'metadata.json').write_text('{not json')

    assert provenance.pi_versions(scene_dir) == ['a' * 40]


def test_scene_provenance_without_pzarr_reports_recording_commit_only(tmp_path: pathlib.Path):
    """A synced-but-unbuilt scene still knows what recorded it; the pzarr fields are None."""
    scene_dir = tmp_path / 'scene_d'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', 'a' * 40)

    prov = provenance.scene_provenance(scene_dir)

    assert prov['pi_versions'] == ['a' * 40]
    assert prov['pzarr_git_sha'] is None
    assert prov['pipeline_version'] is None
    assert prov['pzarr_created_at'] is None


def test_scene_provenance_reads_pzarr_build_attrs(tmp_path: pathlib.Path):
    """The pzarr's build-time git_sha/pipeline_version/created_at come through as stored."""
    scene_dir = tmp_path / 'scene_e'
    scene_dir.mkdir()
    _make_session(scene_dir, 'session_1', 'a' * 40)
    root = zarr.open_group(str(scene_dir / 'scene.zarr'), mode='w')
    root.attrs['git_sha'] = 'c' * 40
    root.attrs['pipeline_version'] = '0.1.0'
    root.attrs['created_at'] = '2026-08-01T00:00:00+00:00'

    prov = provenance.scene_provenance(scene_dir)

    assert prov['pzarr_git_sha'] == 'c' * 40
    assert prov['pipeline_version'] == '0.1.0'
    assert prov['pzarr_created_at'] == '2026-08-01T00:00:00+00:00'


def test_short_sha_abbreviates_and_passes_through_absent_values():
    """None and the 'unknown' sentinel become None so templates render a dash, not the word."""
    assert provenance.short_sha('a' * 40) == 'a' * provenance.SHORT_SHA_LEN
    assert provenance.short_sha(None) is None
    assert provenance.short_sha('unknown') is None
    assert provenance.short_sha('') is None
