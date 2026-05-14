r"""
Populate an ORB-SLAM3 settings YAML from OpenImuCameraCalibrator output.

Reads the FISHEYE camera calibration JSON and the cam-IMU calibration result
from a calibration dataset directory, then writes the derived values into the
target ORB-SLAM3 settings YAML in-place.

Supported camera model: FISHEYE (OpenImuCameraCalibrator) → KannalaBrandt8 (ORB-SLAM3).
The output YAML must already contain Camera.fx/fy/cx/cy/k1-k4 and Tbc keys.

Usage:
    uv run python ingest/integration/populate_slam_yaml.py \\
        --dataset slam/OpenImuCameraCalibrator/calibration_datasets/gopro-hero-12_polyumi_gripper_2 \\
        [--yaml ingest/config/gopro_hero12_slam.yaml]
"""

import argparse
import json
import pathlib
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

_DEFAULT_YAML = (
    pathlib.Path(__file__).parent.parent / 'config' / 'gopro_hero12_slam.yaml'
)


def _find_json(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matching {pattern!r} in {directory}')
    if len(matches) > 1:
        names = [m.name for m in matches]
        print(f'Warning: multiple matches for {pattern!r}: {names}; using {matches[-1].name}')
    return matches[-1]


def load_fisheye_intrinsics(dataset_dir: pathlib.Path) -> dict:
    """
    Load camera intrinsics from a FISHEYE cam_calib JSON.

    Returns a dict with fx, fy, cx, cy, k1-k4, width, height, fps,
    reproj_error, and source_file.
    """
    json_path = _find_json(dataset_dir / 'cam', 'cam_calib_*_fi_*.json')
    with open(json_path) as f:
        d = json.load(f)
    if d.get('intrinsic_type') != 'FISHEYE':
        raise ValueError(
            f'Expected FISHEYE intrinsics in {json_path.name}, '
            f'got {d.get("intrinsic_type")!r}'
        )
    intr = d['intrinsics']
    fl = intr['focal_length']
    ar = intr['aspect_ratio']
    return {
        'source_file': json_path.name,
        'reproj_error': d['final_reproj_error'],
        'width': d['image_width'],
        'height': d['image_height'],
        'fps': d['fps'],
        'fx': fl,
        'fy': fl * ar,
        'cx': intr['principal_pt_x'],
        'cy': intr['principal_pt_y'],
        'k1': intr['radial_distortion_1'],
        'k2': intr['radial_distortion_2'],
        'k3': intr['radial_distortion_3'],
        'k4': intr['radial_distortion_4'],
    }


def load_cam_imu(dataset_dir: pathlib.Path) -> dict:
    """
    Load camera-IMU extrinsics from a cam_imu_calib_result JSON.

    Builds the 4×4 T_body_camera (Tbc) matrix from q_i_c + t_i_c.
    Returns a dict with Tbc (4×4 ndarray), reproj_error, and source_file.
    """
    json_path = _find_json(dataset_dir / 'cam_imu', 'cam_imu_calib_result_*.json')
    with open(json_path) as f:
        d = json.load(f)
    q = d['q_i_c']
    t = d['t_i_c']
    R = Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    Tbc = np.eye(4)
    Tbc[:3, :3] = R
    Tbc[:3, 3] = [t['x'], t['y'], t['z']]
    return {
        'source_file': json_path.name,
        'reproj_error': d['final_reproj_error'],
        'Tbc': Tbc,
    }


def _set_scalar(content: str, key: str, value: float) -> str:
    pattern = rf'^({re.escape(key)}\s*:)\s*[^\n#]*'
    new_content, n = re.subn(
        pattern, rf'\1 {value:.10f}', content, flags=re.MULTILINE
    )
    if n == 0:
        raise KeyError(f'Key not found in YAML: {key!r}')
    return new_content


def _set_tbc(content: str, Tbc: np.ndarray, source_file: str, reproj: float) -> str:
    rows = Tbc.tolist()
    data_lines = []
    for i, row in enumerate(rows):
        vals = ',  '.join(f'{v:13.10f}' for v in row)
        comma = ',' if i < 3 else ''
        data_lines.append(f'          {vals}{comma}')
    data_block = '\n'.join(data_lines)

    replacement = (
        f'# Source: {source_file} (q_i_c + t_i_c), reproj error {reproj:.2f} px.\n'
        f'Tbc: !!opencv-matrix\n'
        f'   rows: 4\n'
        f'   cols: 4\n'
        f'   dt: f\n'
        f'   data: [{data_block}]'
    )

    # Match from the source comment through the closing ] of the data block
    pattern = r'# Source: cam_imu_calib_result.*?data: \[.*?\]'
    new_content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
    if n == 0:
        raise ValueError('Could not find Tbc block to replace in YAML')
    return new_content


def populate(dataset_dir: pathlib.Path, yaml_path: pathlib.Path) -> None:
    """Read calibration results from dataset_dir and write values into yaml_path."""
    intr = load_fisheye_intrinsics(dataset_dir)
    extr = load_cam_imu(dataset_dir)

    content = yaml_path.read_text()

    content = _set_scalar(content, 'Camera.fx', intr['fx'])
    content = _set_scalar(content, 'Camera.fy', intr['fy'])
    content = _set_scalar(content, 'Camera.cx', intr['cx'])
    content = _set_scalar(content, 'Camera.cy', intr['cy'])
    content = _set_scalar(content, 'Camera.k1', intr['k1'])
    content = _set_scalar(content, 'Camera.k2', intr['k2'])
    content = _set_scalar(content, 'Camera.k3', intr['k3'])
    content = _set_scalar(content, 'Camera.k4', intr['k4'])
    content = _set_scalar(content, 'Camera.width', intr['width'])
    content = _set_scalar(content, 'Camera.height', intr['height'])
    content = _set_tbc(content, extr['Tbc'], extr['source_file'], extr['reproj_error'])

    yaml_path.write_text(content)

    print(f'Updated {yaml_path}')
    print(f'  Camera intrinsics  ({intr["source_file"]}): reproj {intr["reproj_error"]:.3f} px')
    print(f'    fx={intr["fx"]:.4f}  fy={intr["fy"]:.4f}  cx={intr["cx"]:.4f}  cy={intr["cy"]:.4f}')
    print(f'    k1={intr["k1"]:.6f}  k2={intr["k2"]:.6f}  k3={intr["k3"]:.6f}  k4={intr["k4"]:.6f}')
    print(f'  Cam-IMU extrinsics ({extr["source_file"]}): reproj {extr["reproj_error"]:.3f} px')
    if extr['reproj_error'] > 1.0:
        print(f'  Warning: cam-IMU reproj error {extr["reproj_error"]:.2f} px > 1.0 px — '
              f'consider re-recording with more aggressive rotational motion')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=pathlib.Path,
        required=True,
        help='Path to OpenImuCameraCalibrator dataset directory (contains cam/ and cam_imu/).',
    )
    parser.add_argument(
        '--yaml',
        type=pathlib.Path,
        default=_DEFAULT_YAML,
        help=f'ORB-SLAM3 settings YAML to update in-place (default: {_DEFAULT_YAML}).',
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f'Error: dataset directory not found: {args.dataset}', file=sys.stderr)
        sys.exit(1)
    if not args.yaml.exists():
        print(f'Error: YAML not found: {args.yaml}', file=sys.stderr)
        sys.exit(1)

    populate(args.dataset, args.yaml)


if __name__ == '__main__':
    main()
