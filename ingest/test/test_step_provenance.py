"""Tests for the per-step commit provenance recorded alongside preprocessing_steps."""

import pathlib
import unittest.mock as mock

import zarr

from polyumi_ingest import gitinfo
from polyumi_ingest.preproc import preprocessing_step_versions
from polyumi_ingest.preproc.step_base import _mark_preprocessing_step


def _root(tmp_path: pathlib.Path) -> zarr.Group:
    return zarr.open_group(str(tmp_path / 'scene.zarr'), mode='w', zarr_format=2)


def test_marking_a_step_records_the_commit_and_time(tmp_path: pathlib.Path) -> None:
    """Completing a step stamps which commit produced it, keyed by step number as a string."""
    root = _root(tmp_path)

    _mark_preprocessing_step(root, 2)

    versions = preprocessing_step_versions(root)
    assert set(versions) == {'2'}
    assert versions['2']['git_sha'] == gitinfo.git_sha()
    assert versions['2']['completed_at'].startswith('20')


def test_re_marking_a_step_overwrites_its_recorded_commit(tmp_path: pathlib.Path) -> None:
    """
    A re-run replaces the step's stamp rather than appending.

    The point of the stamp is "which code produced what's in the store right now", so a
    stale entry from the previous run would be actively misleading.
    """
    root = _root(tmp_path)
    root.attrs['preprocessing_steps'] = [2]
    root.attrs['preprocessing_step_versions'] = {'2': {'git_sha': 'old', 'completed_at': '2020-01-01T00:00:00+00:00'}}

    _mark_preprocessing_step(root, 2)

    versions = preprocessing_step_versions(root)
    assert versions['2']['git_sha'] != 'old'
    assert versions['2']['completed_at'] != '2020-01-01T00:00:00+00:00'


def test_marking_a_step_preserves_other_steps_stamps(tmp_path: pathlib.Path) -> None:
    """Re-running one step must not wipe the provenance of the steps around it."""
    root = _root(tmp_path)
    root.attrs['preprocessing_steps'] = [1]
    root.attrs['preprocessing_step_versions'] = {'1': {'git_sha': 'sha-one', 'completed_at': 'then'}}

    _mark_preprocessing_step(root, 2)

    versions = preprocessing_step_versions(root)
    assert versions['1'] == {'git_sha': 'sha-one', 'completed_at': 'then'}
    assert '2' in versions


def test_preprocessing_step_versions_empty_for_stores_predating_it(tmp_path: pathlib.Path) -> None:
    """Stores processed before this existed report {} rather than raising."""
    root = _root(tmp_path)
    root.attrs['preprocessing_steps'] = [1, 2]

    assert preprocessing_step_versions(root) == {}


def test_preprocessing_step_versions_ignores_a_malformed_attr(tmp_path: pathlib.Path) -> None:
    """A non-dict attr, or entries that aren't dicts, are dropped instead of crashing the read."""
    root = _root(tmp_path)
    root.attrs['preprocessing_step_versions'] = ['not', 'a', 'dict']
    assert preprocessing_step_versions(root) == {}

    root.attrs['preprocessing_step_versions'] = {'1': 'bare-sha', '2': {'git_sha': 'ok'}}
    assert preprocessing_step_versions(root) == {'2': {'git_sha': 'ok'}}


def test_git_sha_resolves_against_the_repo_not_the_cwd(tmp_path: pathlib.Path, monkeypatch) -> None:
    """
    The sha must not depend on where the process was launched from.

    The catalog server and ingest CLI are routinely run from elsewhere; a
    cwd-relative lookup would stamp whatever unrelated repo the shell happened to be in.
    """
    expected = gitinfo._resolve_git_sha()
    monkeypatch.chdir(tmp_path)

    assert gitinfo._resolve_git_sha() == expected


def test_git_sha_is_fixed_at_import_not_re_read_per_call() -> None:
    """
    The stamp must describe the code that is running, not HEAD at the moment it is asked.

    A process that outlives a commit would otherwise stamp a commit it never executed.
    That is not hypothetical: a catalog server up since before a batch of commits ran the
    old eef-pose step while reporting the new HEAD, and scene 30ed was stamped with a
    commit whose code had never touched it.
    """
    before = gitinfo.git_sha()

    with mock.patch.object(gitinfo, '_resolve_git_sha', return_value='f' * 40) as resolve:
        assert gitinfo.git_sha() == before

    resolve.assert_not_called()
