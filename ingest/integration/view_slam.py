r"""
Launch the ORB-SLAM3 Pangolin viewer for a scene episode.

Exports the episode's frames and IMU to a temp directory, then invokes the
map-builder binary with the viewer enabled and the pre-built atlas loaded so
the 3D map is pre-populated.  The atlas is opened read-only (load, not save),
so it is not modified.

Usage:
    uv run python ingest/integration/view_slam.py \
        recordings/scene_YYYY-MM-DD_hh-mm-ss_XXXX \
        [--episode 0] \
        [--orb-slam3-dir /path/to/ORB_SLAM3] \
        [--bin-subdir Examples/Monocular-Inertial]

Environment variables (used as defaults if flags are omitted):
    ORB_SLAM3_DIR         path to ORB-SLAM3 installation
    ORB_SLAM3_BIN_SUBDIR  subdirectory containing the binaries (default: bin)
"""

import argparse
import logging
import pathlib
import subprocess
import sys
import tempfile

import zarr
from polyumi_ingest.preproc.slam_step import (
    OrbSlam3Step,
    _export_episode,
    _make_temp_settings_yaml,
)
from polyumi_ingest.pzarr.scene_files import SceneFiles
from rich.logging import RichHandler

logging.basicConfig(
    level='INFO',
    format='%(message)s',
    handlers=[RichHandler(show_time=True, show_level=True, show_path=False)],
)
log = logging.getLogger(__name__)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'scene',
        type=pathlib.Path,
        help='Scene directory or scene.zarr path.',
    )
    parser.add_argument(
        '--episode',
        type=int,
        default=0,
        help='Episode index to visualize (default: 0).',
    )
    parser.add_argument(
        '--orb-slam3-dir',
        type=pathlib.Path,
        default=None,
        help='ORB-SLAM3 installation root (overrides ORB_SLAM3_DIR env var).',
    )
    parser.add_argument(
        '--bin-subdir',
        default=None,
        help='Subdirectory of orb-slam3-dir containing binaries (overrides ORB_SLAM3_BIN_SUBDIR).',
    )
    args = parser.parse_args()

    scene_zarr = SceneFiles.resolve_zarr_path(args.scene)
    if not scene_zarr.exists():
        log.error(f'No scene.zarr found at {args.scene}')
        sys.exit(1)

    atlas_path = scene_zarr.parent / f'{scene_zarr.parent.name}.atlas.osa'
    if not atlas_path.exists():
        log.error(
            f'No atlas found at {atlas_path}. '
            f'Run `pingest pp 2` on this scene first to build the map.'
        )
        sys.exit(1)

    kwargs: dict = {}
    if args.orb_slam3_dir:
        kwargs['orb_slam3_dir'] = args.orb_slam3_dir
    if args.bin_subdir:
        kwargs['bin_subdir'] = args.bin_subdir
    step = OrbSlam3Step(**kwargs)

    root = zarr.open_group(str(scene_zarr), mode='r')
    ep_key = f'episode_{args.episode}'
    if ep_key not in root:
        log.error(f'Episode {args.episode} not found in {scene_zarr}')
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix='polyumi_slam_view_') as tmp_str:
        tmp = pathlib.Path(tmp_str)
        log.info(f'Exporting {ep_key} to {tmp}...')
        video_path, json_path, _ = _export_episode(root[ep_key], tmp)

        settings_path = _make_temp_settings_yaml(
            step.settings_yaml, tmp, load_atlas=atlas_path, viewer=True
        )

        cmd = [
            str(step.map_builder_bin),
            str(step._vocab_path),
            str(settings_path),
            str(video_path),
            str(json_path),
        ]
        log.info('Launching viewer (close the Pangolin window to exit)...')
        log.info('  ' + ' '.join(cmd))
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
