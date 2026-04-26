"""
ingest/gopro_fetch.py - Find GoPro MP4 files on SD card matching a session timestamp.

GoPro cameras embed a UTC creation_time tag in the MP4 container that records the exact
moment the shutter was pressed. We prefer this over filesystem mtime, which is stored in
local time on FAT32 and subject to timezone misinterpretation when read on Linux.

Fallback: if the tag is absent, infer start time as mtime - duration.
"""

import datetime
import json
import logging
import pathlib
import subprocess

log = logging.getLogger(__name__)

GOPRO_VIDEO_SUBDIR = pathlib.Path('DCIM') / '100GOPRO'
DEFAULT_THRESHOLD_MS = 1000.0

_MOUNT_ROOTS = [
    pathlib.Path('/media'),
    pathlib.Path('/run/media'),
    pathlib.Path('/mnt'),
]


def _find_gopro_mount() -> pathlib.Path | None:
    """Scan common Linux auto-mount roots for a volume containing DCIM/100GOPRO."""
    for root in _MOUNT_ROOTS:
        if not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except (PermissionError, OSError):
            continue
        for child in children:
            try:
                if not child.is_dir():
                    continue
                # Direct mount (e.g. /mnt/gopro) or one level deeper (/media/<user>/<label>)
                if (child / GOPRO_VIDEO_SUBDIR).is_dir():
                    return child
                for grandchild in child.iterdir():
                    if grandchild.is_dir() and (grandchild / GOPRO_VIDEO_SUBDIR).is_dir():
                        return grandchild
            except (PermissionError, OSError):
                continue
    return None


def _recording_start_time(video_path: pathlib.Path) -> datetime.datetime:
    """
    Return the UTC recording start time for a GoPro MP4.

    Prefers the creation_time tag embedded in the MP4 container (written by the GoPro
    at the moment of shutter press, in UTC). Falls back to filesystem mtime minus
    duration if the tag is absent.
    """
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    fmt = json.loads(result.stdout)['format']

    ct_str = fmt.get('tags', {}).get('creation_time')
    if ct_str:
        return datetime.datetime.fromisoformat(ct_str.replace('Z', '+00:00'))

    # FAT32 mtime is in local time but Linux reads it without timezone correction,
    # so this path may be off by the GoPro's UTC offset. Use only as a last resort.
    log.warning(f'{video_path.name}: no creation_time tag; falling back to mtime - duration')
    duration_s = float(fmt['duration'])
    mtime = datetime.datetime.fromtimestamp(video_path.stat().st_mtime, tz=datetime.timezone.utc)
    return mtime - datetime.timedelta(seconds=duration_s)


def find_gopro_video(
    start_time: datetime.datetime,
    mount_point: pathlib.Path | None = None,
    threshold_ms: float = DEFAULT_THRESHOLD_MS,
) -> pathlib.Path:
    """
    Find the GoPro MP4 whose recording start best matches *start_time*.

    Args:
        start_time: Nominal recording start time (gopro_sync_time from session metadata).
            Timezone-naive values are treated as local time.
        mount_point: SD card mount point. When None, scanned automatically from
            common Linux auto-mount roots (/media, /run/media, /mnt).
        threshold_ms: Maximum allowed difference in milliseconds between the
            file's recording start and *start_time*. Raises RuntimeError if the
            best match exceeds this.

    Returns:
        Path to the best-matching MP4 file on the SD card.

    Raises:
        FileNotFoundError: SD card not found, or DCIM/100GOPRO is missing/empty.
        RuntimeError: No file within *threshold_ms* of *start_time*.

    """
    if mount_point is None:
        mount_point = _find_gopro_mount()
        if mount_point is None:
            raise FileNotFoundError(
                'No GoPro SD card found under /media, /run/media, or /mnt.\n'
                'Are you sure you\'ve both inserted AND mounted the SD card? '
                'You can also pass --mount-point explicitly if it is mounted elsewhere.'
            )
        log.info(f'Auto-detected GoPro SD card at {mount_point}')

    video_dir = mount_point / GOPRO_VIDEO_SUBDIR
    if not video_dir.is_dir():
        raise FileNotFoundError(f'GoPro video directory not found: {video_dir}')

    mp4_files = sorted(video_dir.glob('*.MP4')) + sorted(video_dir.glob('*.mp4'))
    if not mp4_files:
        raise FileNotFoundError(f'No MP4 files found in {video_dir}')

    if start_time.tzinfo is None:
        start_time = start_time.astimezone(datetime.timezone.utc)
    else:
        start_time = start_time.astimezone(datetime.timezone.utc)

    best_path: pathlib.Path | None = None
    best_delta_ms = float('inf')

    for mp4 in mp4_files:
        try:
            recording_start = _recording_start_time(mp4)
        except (subprocess.CalledProcessError, KeyError, ValueError) as exc:
            log.warning(f'Skipping {mp4.name}: could not read start time ({exc})')
            continue

        delta_ms = abs((recording_start - start_time).total_seconds()) * 1000
        log.debug(f'{mp4.name}: start={recording_start.isoformat()}, delta={delta_ms:.0f}ms')

        if delta_ms < best_delta_ms:
            best_delta_ms = delta_ms
            best_path = mp4

    if best_path is None:
        raise RuntimeError('Could not determine recording start time for any MP4 on the SD card.')

    if best_delta_ms > threshold_ms:
        raise RuntimeError(
            f'Best match {best_path.name} has delta {best_delta_ms:.0f}ms, '
            f'which exceeds threshold {threshold_ms:.0f}ms. '
            f'Check that the GoPro clock was synced before recording.'
        )

    log.info(f'Matched {best_path.name} to {start_time.isoformat()} (delta={best_delta_ms:.0f}ms)')
    return best_path
