"""
Tests for ``config/policy_select.sh``.

The selector stands between a typo in a ``config/policy.<name>.env`` and a docker build that fails
twenty minutes into a conda solve, so its contract is worth asserting from outside bash. The
contract tests run against a synthetic repo root -- they describe what an env file may say, not
what today's two forks happen to say -- and one test lints the real files.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECT_SH = REPO_ROOT / 'config' / 'policy_select.sh'
REAL_ENV_FILES = sorted((REPO_ROOT / 'config').glob('policy.*.env'))

REQUIRED_KEYS = ('POLICY_DIR', 'IMAGE', 'TRAIN_CMD')


def run_select(repo_root: Path, policy: str, **overrides: str) -> subprocess.CompletedProcess:
    """
    Source the selector, resolve ``policy``, and dump what it set as ``KEY=value`` lines.

    ``overrides`` become environment variables, i.e. the calling shell's values.
    """
    script = f"""
        set -euo pipefail
        source {SELECT_SH}
        policy_select {repo_root} {policy}
        echo "POLICY_DIR=${{POLICY_DIR}}"
        echo "IMAGE=${{IMAGE}}"
        echo "TRAIN_CMD=${{TRAIN_CMD}}"
        echo "SERVE_CMD=${{SERVE_CMD}}"
        echo "CONFIG_NAME=${{CONFIG_NAME:-}}"
    """
    env = {k: v for k, v in os.environ.items() if k not in ('IMAGE', 'CONFIG_NAME', 'SERVE_CMD')}
    env.update(overrides)
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env)


def resolved(proc: subprocess.CompletedProcess) -> dict[str, str]:
    """Parse run_select's stdout into a dict, asserting the call succeeded."""
    assert proc.returncode == 0, proc.stderr
    return dict(line.split('=', 1) for line in proc.stdout.strip().splitlines())


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build a repo root with two policies: one complete, one with only the required keys."""
    (tmp_path / 'config').mkdir()
    for name in ('full', 'minimal'):
        (tmp_path / 'external' / name).mkdir(parents=True)
        (tmp_path / 'external' / name / 'Dockerfile').write_text('FROM scratch\n')

    (tmp_path / 'config' / 'policy.full.env').write_text(
        'POLICY_DIR=external/full\n'
        'IMAGE=img-full\n'
        'TRAIN_CMD="bash docker/train.sh"\n'
        'SERVE_CMD="bash docker/serve.sh"\n'
        'CONFIG_NAME=full_workspace\n'
    )
    (tmp_path / 'config' / 'policy.minimal.env').write_text(
        'POLICY_DIR=external/minimal\nIMAGE=img-minimal\nTRAIN_CMD="bash docker/train.sh"\n'
    )
    return tmp_path


def test_unknown_policy_names_the_ones_that_exist(fake_repo: Path) -> None:
    """An unrecognised POLICY fails, and says what it could have been."""
    proc = run_select(fake_repo, 'nope')
    assert proc.returncode != 0
    assert "unknown POLICY 'nope'" in proc.stderr
    assert 'full' in proc.stderr and 'minimal' in proc.stderr


def test_env_file_values_reach_the_caller(fake_repo: Path) -> None:
    """Every key an env file sets lands in the sourcing shell."""
    assert resolved(run_select(fake_repo, 'full')) == {
        'POLICY_DIR': 'external/full',
        'IMAGE': 'img-full',
        'TRAIN_CMD': 'bash docker/train.sh',
        'SERVE_CMD': 'bash docker/serve.sh',
        'CONFIG_NAME': 'full_workspace',
    }


def test_calling_shell_overrides_the_file(fake_repo: Path) -> None:
    """IMAGE and CONFIG_NAME from the environment win, for a one-off tag or workspace."""
    got = resolved(run_select(fake_repo, 'full', IMAGE='one-off', CONFIG_NAME='other_workspace'))
    assert got['IMAGE'] == 'one-off'
    assert got['CONFIG_NAME'] == 'other_workspace'


def test_optional_keys_stay_empty_rather_than_leaking(fake_repo: Path) -> None:
    """Keys the env file omits resolve empty, not to whatever the environment held."""
    # A stale SERVE_CMD in the environment must not make a fork that cannot serve look like it can;
    # an absent CONFIG_NAME must not be an unbound-variable error under `set -u`.
    got = resolved(run_select(fake_repo, 'minimal', SERVE_CMD='bash docker/serve.sh'))
    assert got['SERVE_CMD'] == ''
    assert got['CONFIG_NAME'] == ''


def test_uninitialised_submodule_is_caught_before_the_build(fake_repo: Path) -> None:
    """A fork with no Dockerfile fails with the command that would fix it."""
    (fake_repo / 'external' / 'full' / 'Dockerfile').unlink()
    proc = run_select(fake_repo, 'full')
    assert proc.returncode != 0
    assert 'git submodule update --init external/full' in proc.stderr


@pytest.mark.parametrize('env_file', REAL_ENV_FILES, ids=lambda p: p.stem)
def test_real_env_files_declare_a_fork_this_repo_has(env_file: Path) -> None:
    """Each shipped env file resolves, and names a fork and entrypoints that exist."""
    policy = env_file.name[len('policy.') : -len('.env')]
    proc = run_select(REPO_ROOT, policy)
    if 'git submodule update --init' in proc.stderr:
        pytest.skip(f'{policy} submodule not checked out')
    got = resolved(proc)
    for key in REQUIRED_KEYS:
        assert got[key], f'{env_file.name} sets no {key}'
    assert (REPO_ROOT / got['POLICY_DIR']).is_dir()
    for cmd in (got['TRAIN_CMD'], got['SERVE_CMD']):
        if cmd:
            # "bash docker/train.sh" -- the script must exist in the fork, or the container dies
            # on a command not found after the image has already been built.
            assert (REPO_ROOT / got['POLICY_DIR'] / cmd.split()[-1]).is_file(), cmd
