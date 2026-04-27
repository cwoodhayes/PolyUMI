"""
Minimal GPMF stream parser for GoPro IMU (ACCL, GYRO) and GPS (GPS5) data.

KLV parser logic adapted from gpmf 0.1 (https://pypi.org/project/gpmf/), MIT License.
"""

import json
import logging
import struct
import subprocess
from dataclasses import dataclass
import pathlib

import numpy as np

log = logging.getLogger(__name__)

# GPMF big-endian type codes → numpy dtype strings
_NP_TYPES: dict[str, str] = {
    'd': '>f8',
    'f': '>f4',
    'b': '>i1',
    'B': '>u1',
    's': '>i2',
    'S': '>u2',
    'l': '>i4',
    'L': '>u4',
    'j': '>i8',
    'J': '>u8',
}


def _ceil4(x: int) -> int:
    return (((x - 1) >> 2) + 1) << 2


def _iter_klv(data: bytes) -> list[tuple[str, object]]:
    """Yield (fourcc, payload) pairs from a GPMF binary block."""
    items = []
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos : pos + 4].decode('latin1')
        type_b = data[pos + 4 : pos + 5].decode('latin1')
        size = data[pos + 5]
        repeat = struct.unpack('>H', data[pos + 6 : pos + 8])[0]
        pos += 8
        nbytes = size * repeat
        payload_bytes = data[pos : pos + nbytes]
        pos += _ceil4(nbytes)

        if type_b == '\x00':
            payload: object = _iter_klv(payload_bytes)
        elif type_b == 'c':
            if key == 'UNIT':
                payload = [payload_bytes[i : i + size].rstrip(b'\x00').decode('latin1') for i in range(0, nbytes, size)]
            else:
                payload = payload_bytes.decode('latin1')
        elif type_b == 'U':
            # GPS UTC timestamp string: YYMMDDHHmmss.sss
            s = payload_bytes.decode('latin1')
            payload = f'20{s[:2]}-{s[2:4]}-{s[4:6]}T{s[6:8]}:{s[8:10]}:{s[10:]}Z'
        elif type_b in _NP_TYPES:
            dt = np.dtype(_NP_TYPES[type_b])
            arr = np.frombuffer(payload_bytes, dtype=dt)
            dim1 = size // dt.itemsize
            if arr.size == 1:
                payload = arr[0]
            elif dim1 > 1 and repeat > 1:
                payload = arr.reshape(repeat, dim1)
            else:
                payload = arr
        else:
            payload = payload_bytes

        items.append((key, payload))
    return items


def _walk_streams(data: bytes):
    """Yield each STRM item-dict from all DEVC containers in *data*."""
    for key, payload in _iter_klv(data):
        if key == 'DEVC' and isinstance(payload, list):
            for sub_key, sub_payload in payload:
                if sub_key == 'STRM' and isinstance(sub_payload, list):
                    yield {k: v for k, v in sub_payload}


@dataclass
class ImuStreams:
    """Parsed IMU and GPS arrays from a GoPro GPMF stream."""

    accl: 'np.ndarray | None'  # (N, 3) float32, m/s²  (GoPro axis order: z, x, y)
    gyro: 'np.ndarray | None'  # (N, 3) float32, rad/s (GoPro axis order: z, x, y)
    gps: 'np.ndarray | None'  # (N, 5) float32: lat°, lon°, alt_m, speed2d_m/s, speed3d_m/s


def extract_gpmf_binary(gopro_path: pathlib.Path) -> bytes | None:
    """
    Extract raw GPMF binary from a GoPro MP4 via ffprobe + ffmpeg subprocess.

    Returns None if the file has no GPMF stream or extraction fails.
    """
    try:
        probe_result = subprocess.run(
            [
                'ffprobe',
                '-v',
                'quiet',
                '-print_format',
                'json',
                '-show_streams',
                str(gopro_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(f'ffprobe failed on {gopro_path.name}: {exc}')
        return None

    stream_index: int | None = None
    for s in json.loads(probe_result.stdout).get('streams', []):
        if s.get('codec_tag_string') == 'gpmd':
            stream_index = s['index']
            break

    if stream_index is None:
        log.info(f'No GPMF stream found in {gopro_path.name}')
        return None

    try:
        result = subprocess.run(
            [
                'ffmpeg',
                '-i',
                str(gopro_path),
                '-map',
                f'0:{stream_index}',
                '-codec',
                'copy',
                '-f',
                'rawvideo',
                'pipe:1',
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(f'ffmpeg GPMF extraction failed on {gopro_path.name}: {exc}')
        return None

    return result.stdout


def parse_imu(gpmf_binary: bytes) -> ImuStreams:
    """
    Parse ACCL, GYRO, GPS5 arrays from a raw GPMF byte stream.

    Each GPMF DEVC container covers ~1 s of data (one video keyframe interval).
    Blocks from all containers are concatenated, yielding the full-clip arrays.
    """
    accl_blocks: list[np.ndarray] = []
    gyro_blocks: list[np.ndarray] = []
    gps_blocks: list[np.ndarray] = []

    for block in _walk_streams(gpmf_binary):
        raw_scal = block.get('SCAL', 1)
        # SCAL may be a numpy scalar or a 1-D array (per-axis scaling — rare for IMU)
        scal: float | np.ndarray = float(raw_scal) if np.ndim(raw_scal) == 0 else raw_scal.astype(np.float32)

        for fourcc, dest in (
            ('ACCL', accl_blocks),
            ('GYRO', gyro_blocks),
            ('GPS5', gps_blocks),
        ):
            raw = block.get(fourcc)
            if raw is not None and isinstance(raw, np.ndarray) and raw.ndim == 2:
                dest.append(raw.astype(np.float32) / scal)

    def _stack(blocks: list[np.ndarray]) -> 'np.ndarray | None':
        return np.vstack(blocks) if blocks else None

    return ImuStreams(
        accl=_stack(accl_blocks),
        gyro=_stack(gyro_blocks),
        gps=_stack(gps_blocks),
    )
