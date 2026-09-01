"""Sync chirp generation and playback for audio time-alignment."""

import logging
import subprocess
import time

import numpy as np

log = logging.getLogger('pi_sync_chirp')

# The WM8960's headphone and speaker drivers are both fed from the same output mixers, so the
# chirp cannot be routed to one and not the other. Muting the headphone volume for the duration
# is the only lever. Below 0101111 the driver mutes with the pin held at VREF, so 0 is clickless.
_CARD = 'wm8960soundcard'
_HP_VOL = 'Headphone Playback Volume'


def _hp_volume(value: str | None = None) -> str:
    """Read the headphone volume, or set it when `value` is given. Returns '' if amixer fails."""
    if value == '':
        return ''
    action = 'cget' if value is None else 'cset'
    args = ['amixer', '-c', _CARD, action, f'name={_HP_VOL}'] + ([value] if value else [])
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        return out.split(': values=')[1].split('\n', 1)[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        log.warning(f'Could not {action} {_HP_VOL} ({exc}); chirp will play through the headphones.')
        return ''


DURATION_S = 0.5
F0 = 440.0
F1 = 7000.0


def generate(sample_rate: int) -> np.ndarray:
    """
    Generate a linear frequency chirp.

    Returns a float32 mono array of length int(sample_rate * DURATION_S).
    """
    n = int(sample_rate * DURATION_S)
    t = np.linspace(0, DURATION_S, n, endpoint=False)
    k = (F1 - F0) / DURATION_S
    return np.sin(2 * np.pi * (F0 * t + 0.5 * k * t**2)).astype(np.float32)


BEEP_FREQ_HZ = 880.0
BEEP_DURATION_S = 0.1
BEEP_GAP_S = 0.05


def beep(count: int, sample_rate: int, device: int | str | None = None) -> None:
    """Play `count` short beeps on the given device (blocking)."""
    import sounddevice as sd

    n = int(sample_rate * BEEP_DURATION_S)
    t = np.linspace(0, BEEP_DURATION_S, n, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * BEEP_FREQ_HZ * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    for i in range(count):
        sd.play(stereo, samplerate=sample_rate, device=device, blocking=True)
        if i < count - 1:
            time.sleep(BEEP_GAP_S)


def play(sample_rate: int, device: int | str | None = None) -> int:
    """
    Play the sync chirp on the given device (blocking for DURATION_S).

    The headphone output is muted for the duration so the chirp goes to the speaker alone; it
    blocks so the volume can be restored inline. Live mic monitoring is muted along with it,
    since the headphone volume sits downstream of the output mixer the bypass feeds.

    Returns the epoch nanoseconds at which playback was *requested* — one output buffer ahead of
    the chirp actually sounding, and so not directly comparable to the capture instants stamped
    on the streams. That is fine because nothing measures against it: ``ChirpTimeSyncStep`` uses
    it only to centre a +/-3 s search window, and recovers the real onset by matched-filtering
    the recorded audio.

    The WM8960 requires stereo output, so the mono chirp is duplicated.
    """
    import sounddevice as sd

    mono = generate(sample_rate)
    stereo = np.column_stack([mono, mono])
    prev_hp = _hp_volume()
    _hp_volume('0')
    try:
        ts = time.time_ns()
        sd.play(stereo, samplerate=sample_rate, device=device, blocking=True)
    finally:
        _hp_volume(prev_hp)
    return ts
