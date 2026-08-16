"""Tests for the Pi fetch client's host resolution and incremental session fetching."""

import importlib
import pathlib
import subprocess
import types

from polyumi_ingest import pi_fetch
from polyumi_ingest.pi_fetch import PiFetch


def test_default_host_comes_from_env(monkeypatch):
    """POLYUMI_PI_HOST overrides the built-in default for every consumer of DEFAULT_HOST."""
    monkeypatch.setenv('POLYUMI_PI_HOST', 'pi@other-pi.local')
    try:
        assert importlib.reload(pi_fetch).DEFAULT_HOST == 'pi@other-pi.local'
    finally:
        monkeypatch.delenv('POLYUMI_PI_HOST')
        importlib.reload(pi_fetch)


def test_default_host_falls_back_without_env(monkeypatch):
    """Unset (or empty) POLYUMI_PI_HOST leaves the built-in default in place."""
    monkeypatch.setenv('POLYUMI_PI_HOST', '')
    try:
        # matches fr3_session.sh's own POLYUMI_PI_HOST default, so nothing has to be exported
        assert importlib.reload(pi_fetch).DEFAULT_HOST == 'polyumi-pi'
    finally:
        monkeypatch.delenv('POLYUMI_PI_HOST')
        importlib.reload(pi_fetch)


# ---------------------------------------------------------------------------
# Incremental, session-level fetching
# ---------------------------------------------------------------------------

# What `grep -L '"duration_s": null' .../session_*/metadata.json` prints: the sessions whose
# metadata does NOT still say null, i.e. the ones finalize() has closed out. The session that
# is still recording is absent from this list, which is the whole point.
_GREP_OUT = """\
/home/conor/recordings/scene_A/session_1/metadata.json
/home/conor/recordings/scene_A/session_2/metadata.json
"""


def _fake_run(stdout: str, returncode: int = 0):
    def run(cmd, **kwargs):
        run.cmd = cmd
        return types.SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)

    return run


def test_list_remote_sessions_returns_only_finalized(monkeypatch):
    """A session still being recorded has duration_s null and must not be fetched yet."""
    monkeypatch.setattr(subprocess, 'run', _fake_run(_GREP_OUT))

    assert PiFetch('pi').list_remote_sessions('scene_A') == ['session_1', 'session_2']


def test_list_remote_sessions_tolerates_grep_exit_1(monkeypatch):
    """Every session still recording makes grep exit 1 — that's "nothing ready", not an error."""
    monkeypatch.setattr(subprocess, 'run', _fake_run('', returncode=1))

    assert PiFetch('pi').list_remote_sessions('scene_A') == []


def test_missing_sessions_subtracts_what_is_already_local(tmp_path: pathlib.Path, monkeypatch):
    """A scene directory existing locally says nothing; only the sessions in it count."""
    monkeypatch.setattr(subprocess, 'run', _fake_run(_GREP_OUT))
    local = tmp_path / 'scene_A'
    (local / 'session_1').mkdir(parents=True)
    (local / 'session_1' / 'metadata.json').write_text('{}')
    # Host-only artifacts must not be mistaken for sessions.
    (local / 'scene.zarr').mkdir()
    (local / 'scene.json').write_text('{}')

    assert PiFetch('pi').missing_sessions('scene_A', local) == ['session_2']


def test_missing_sessions_refetches_a_half_transferred_session(tmp_path: pathlib.Path, monkeypatch):
    """
    A session directory without metadata.json was interrupted mid-transfer, so it's not done.

    tar extracts in place and a killed transfer runs no cleanup, so existence alone would
    strand the session as permanently "already fetched" and silently truncated.
    """
    monkeypatch.setattr(subprocess, 'run', _fake_run(_GREP_OUT))
    local = tmp_path / 'scene_A'
    (local / 'session_1').mkdir(parents=True)
    (local / 'session_1' / 'metadata.json').write_text('{}')
    # what a dropped connection leaves: some payload, no metadata
    (local / 'session_2' / 'video').mkdir(parents=True)

    assert PiFetch('pi').missing_sessions('scene_A', local) == ['session_2']


def test_missing_sessions_on_an_unfetched_scene(tmp_path: pathlib.Path, monkeypatch):
    """No local directory at all means every finalized session is missing."""
    monkeypatch.setattr(subprocess, 'run', _fake_run(_GREP_OUT))

    assert PiFetch('pi').missing_sessions('scene_A', tmp_path / 'nope') == ['session_1', 'session_2']


def test_copy_sessions_tars_only_the_named_members(tmp_path: pathlib.Path, monkeypatch):
    """
    The tar stream carries just the new sessions.

    That is what leaves the local scene.zarr, scene.json, slam_logs/ and the SLAM atlas alone
    when a scene is re-fetched after growing.
    """
    sent: dict[str, list[str]] = {}

    class _Proc:
        stdout = None

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        sent['remote'] = cmd
        proc = _Proc()
        proc.stdout = types.SimpleNamespace(close=lambda: None)
        return proc

    monkeypatch.setattr(subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(subprocess, 'run', lambda cmd, **kw: types.SimpleNamespace(returncode=0))

    PiFetch('pi').copy_sessions('scene_A', ['session_2', 'session_3'], tmp_path)

    assert sent['remote'][-2:] == ['scene_A/session_2', 'scene_A/session_3']


def test_copy_sessions_with_nothing_to_do_makes_no_connection(monkeypatch):
    """An empty session list is a no-op, not an ssh round trip that tars the whole scene."""

    def explode(*a, **kw):
        raise AssertionError('should not have opened an ssh stream')

    monkeypatch.setattr(subprocess, 'Popen', explode)

    PiFetch('pi').copy_sessions('scene_A', [], pathlib.Path('/tmp'))
