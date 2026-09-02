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

#: Where to restore the headphones when the pre-chirp read failed and there is no measured
#: value to put back. Left ear at +1 dB (122 on the control's 0-127 scale), right muted —
#: monitoring convention, and the same pair `pi/alsa_preset` stores. Needed because a failed
#: read must not leave the mute in place: the chirp would be the last thing ever heard.
_HP_DEFAULT = '122,0'


def _hp_volume_get() -> str:
    """Read the headphone volume as amixer's ``'L,R'`` string; ``''`` if amixer fails."""
    args = ['amixer', '-c', _CARD, 'cget', f'name={_HP_VOL}']
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
        return out.split(': values=')[1].split('\n', 1)[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        log.warning(f'Could not read {_HP_VOL} ({exc}); will restore it to {_HP_DEFAULT} after the chirp.')
        return ''


def _hp_volume_set(value: str) -> None:
    """Set the headphone volume from an amixer ``'L,R'`` string; a failure is logged, not raised."""
    args = ['amixer', '-c', _CARD, 'cset', f'name={_HP_VOL}', value]
    try:
        subprocess.run(args, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning(f'Could not set {_HP_VOL} to {value} ({exc}).')


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
    # Resolved before the mute, so a failed read still restores to something audible rather
    # than leaving the headphones dead for the rest of the session.
    restore_to = _hp_volume_get() or _HP_DEFAULT
    _hp_volume_set('0')
    try:
        ts = time.time_ns()
        sd.play(stereo, samplerate=sample_rate, device=device, blocking=True)
    finally:
        _hp_volume_set(restore_to)
    return ts
