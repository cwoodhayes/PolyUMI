"""Parse and write OptiTrack rigid-body pose data into a pzarr store."""

import logging
import pathlib

import numpy as np
import zarr
from numcodecs import Blosc

_BLOSC = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

log = logging.getLogger('pzarr')


def find_optitrack_csv(scene_path: pathlib.Path) -> pathlib.Path | None:
    """Return the first CSV in scene_path whose first line starts with 'Format Version,1.23'."""
    for p in sorted(scene_path.iterdir()):
        if p.suffix.lower() != '.csv':
            continue
        try:
            first_line = p.read_text(errors='replace').splitlines()[0]
        except (OSError, IndexError):
            continue
        if first_line.startswith('Format Version,1.23'):
            return p
    return None


def parse_optitrack_csv(csv_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an OptiTrack rigid-body CSV export into relative timestamps and 6DOF poses.

    Supports CSVs with multiple rigid bodies. When more than one rigid body is
    present the first one whose name contains "PolyUMI" is used; if none match,
    the first rigid body is used (columns 2-7).

    Returns:
        times_s: (N,) float64 seconds since capture start
        poses: (N, 7) float64 [x, y, z, qx, qy, qz, qw] (position metres, quaternion)

    """
    name_row_fields: list[str] = []
    data_start_row = None
    with csv_path.open() as f:
        for i, line in enumerate(f):
            stripped = line.rstrip('\n')
            if stripped.startswith(',Name,'):
                name_row_fields = stripped.split(',')
            if line.startswith('Frame,Time'):
                # check that this is outputting quaternions and not euler angles
                fields = stripped.split(',')
                if fields[2:6] != ['X', 'Y', 'Z', 'W']:
                    raise ValueError(
                        f'Unexpected OptiTrack CSV format in {csv_path}: '
                        f'expected quaternion columns "X,Y,Z,W" but got {fields[2:6]}'
                    )
                data_start_row = i + 1
                break
    if data_start_row is None:
        raise ValueError(f'Could not find data header row in OptiTrack CSV: {csv_path}')

    # Each rigid body occupies 6 data columns (rot X/Y/Z, pos X/Y/Z) starting at col 2.
    # Find the first column belonging to a rigid body named "PolyUMI*".
    rb_col_start = 2  # default: first rigid body
    if name_row_fields:
        for col_idx, name in enumerate(name_row_fields[2:], start=2):
            if 'PolyUMI' in name:
                rb_col_start = col_idx
                break
        else:
            raise ValueError(
                f'No rigid body named "PolyUMI*" found in OptiTrack CSV: {csv_path}'
            )

    data = np.loadtxt(csv_path, delimiter=',', skiprows=data_start_row, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    times_s = data[:, 1]
    rot_quat = data[:, rb_col_start: rb_col_start + 4]
    pos_xyz = data[:, rb_col_start + 4: rb_col_start + 7]

    poses = np.concatenate([pos_xyz, rot_quat], axis=1)
    return times_s, poses


def write_optitrack(root: zarr.Group, csv_path: pathlib.Path, optitrack_start_s: float) -> None:
    """Parse OptiTrack CSV and write pose data to the root zarr group."""
    times_s, poses = parse_optitrack_csv(csv_path)
    abs_timestamps = optitrack_start_s + times_s

    ot_grp = root.require_group('optitrack')
    for name in ('pose', 'timestamps'):
        if name in ot_grp:
            del ot_grp[name]
    ot_grp.create_array('pose', data=poses, compressor=_BLOSC)
    ot_grp.create_array('timestamps', data=abs_timestamps, compressor=_BLOSC)

    duration = float(times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0
    rate = len(times_s) / duration if duration > 0 else 0.0
    log.info(f'  OptiTrack: {len(times_s)} poses @ {rate:.0f} Hz from {csv_path.name}')
