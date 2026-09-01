"""Tests for the headphone muting around sync chirp playback."""

import subprocess

from polyumi_pi import sync_chirp

CGET_OUT = """numid=11,iface=MIXER,name='Headphone Playback Volume'
  ; type=INTEGER,access=rw---R--,values=2,min=0,max=127,step=0
  : values=95,95
  | dBscale-min=-121.00dB,step=1.00dB,mute=1
"""


def test_hp_volume_reads_and_writes(monkeypatch):
    """A read parses amixer's value line; a write passes the value through as the last arg."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=CGET_OUT, stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)

    assert sync_chirp._hp_volume() == '95,95'
    assert 'cget' in calls[0]

    sync_chirp._hp_volume('0')
    assert 'cset' in calls[1]
    assert calls[1][-1] == '0'


def test_hp_volume_survives_missing_amixer(monkeypatch):
    """A missing amixer must degrade to a warning, never break chirp playback."""
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('amixer')))

    assert sync_chirp._hp_volume() == ''
    assert sync_chirp._hp_volume('') == ''  # restoring after a failed read is a no-op
