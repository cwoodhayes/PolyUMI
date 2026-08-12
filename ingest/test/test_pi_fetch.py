"""Tests for the Pi fetch client's host resolution."""

import importlib

from polyumi_ingest import pi_fetch


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
