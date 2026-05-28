# `pzarr` - PolyUMI's Working Data Format

![pzarr data schema diagram](/docs/polyumi_working_format_schema.svg)

## Purpose

`pzarr` is the *working data format* for the PolyUMI preprocessing pipeline. It sits between raw ingest (mp4s, audio files, metadata json) and the training-ready exports (Northwestern CRB's diffusion policy zarr, LeRobot, MCAP). Pipeline steps modify it in place; once the pipeline is complete for a scene, you can archive the result.

Each `pzarr` corresponds to a single recorded scene, composed of one or many episodes (referred to as "sessions" in the pi app). Sessions are typed: `MAPPING` sessions are used to build the SLAM map; `EPISODE` sessions are the task demonstrations exported for training.

Compared to downstream data formats like LeRobot Dataset or Diffusion Policy's zarr, which are optimized for feeding directly into a training pipeline, `pzarr` is intended as a single source of truth for all scene data that doesn't discard any information. This means that unlike these downstream formats, it:

1. Allows efficient incremental writes from multiple pipeline steps (e.g. SLAM, gripper width extraction) without needing to rewrite the whole episode or scene on each step
2. Preserves the original multi-rate timestamps from each stream, rather than resampling to a common time grid
3. Stores full-fidelity decoded audio and video, rather than pre-encoding into a training codec (like WebM for video) or a heavily downsampled format

`pzarr` is implemented as a zarr `DirectoryStore` with a specific schema. The schema is designed to be flexible and extensible, but the above principles should guide any additions or modifications.


## Library and format version

Use `zarr-python 3.x` with `zarr_format=2` explicitly. zarr-python 3 reads and writes v2 stores cleanly, but the v2 format gives us reliable JpegXl codec support (the v3 codec story for non-spec codecs has interop caveats) and matches the format that downstream tools like forge and CRB's `ReplayBuffer` already expect. If sharding becomes a real pain point as datasets grow, migrate to v3 later via `zarr.copy()`.

The format version is tracked as `pzarr_version` (currently `1`) in the scene root `.zattrs`. Read this from the store in your code rather than hardcoding it, so schema migrations are operational rather than code changes.

## Schema

```
scene.zarr/
├── .zattrs                         scene-level metadata (see below)
├── episode_N/                      one group per session (N = 0, 1, 2, ...)
│   ├── .zattrs                     {session_type: 'MAPPING'|'EPISODE', session_dir: str}
│   ├── finger/
│   │   ├── frames                  (N_finger, H, W, 3) uint8 — RGB frames
│   │   ├── finger_piezo            (N_audio,) float32 — piezo contact mic, normalized [-1, 1]
│   │   └── finger_air              (N_audio,) float32 — air mic, normalized [-1, 1]
│   ├── gopro/
│   │   ├── frames                  (N_gopro, H, W, 3) uint8 — RGB frames
│   │   ├── audio                   (N_gopro_audio,) float32 — mono GoPro mic
│   │   ├── accl                    (N_accl, 3) float64 — [z, x, y] m/s²
│   │   ├── gyro                    (N_gyro, 3) float64 — [z, x, y] rad/s
│   │   ├── gps                     (N_gps, 3) float64 — [lat, lon, alt]
│   │   └── slam_poses              (N_gopro, 7) float64 — [x, y, z, qx, qy, qz, qw], NaN when tracking lost
│   ├── timestamps/
│   │   ├── finger                  (N_finger,) float64 — UTC seconds
│   │   ├── finger_piezo            (N_audio,) float64
│   │   ├── finger_air              (N_audio,) float64
│   │   ├── gopro                   (N_gopro,) float64
│   │   ├── gopro_audio             (N_gopro_audio,) float64
│   │   ├── gopro_accl              (N_accl,) float64
│   │   ├── gopro_gyro              (N_gyro,) float64
│   │   └── gopro_gps               (N_gps,) float64
│   └── annotations/
│       ├── episode_start           scalar float64 — UTC seconds (first finger frame)
│       ├── episode_end             scalar float64 — UTC seconds (last finger frame)
│       ├── sync_chirp_play_time_s  scalar float64 — when the sync chirp was played (set at ingest)
│       ├── time_sync/              populated by step 1
│       │   ├── gopro_to_finger_offset_s   scalar float64 — subtract from GoPro timestamps to align to finger clock
│       │   ├── nominal_start_offset_s     scalar float64
│       │   ├── residual_offset_s          scalar float64
│       │   ├── finger_chirp_onset_s       scalar float64
│       │   ├── gopro_chirp_onset_s        scalar float64
│       │   ├── finger_chirp_peak          scalar float32
│       │   └── gopro_chirp_peak           scalar float32
│       ├── slam/                   populated by step 2
│       │   ├── n_frames_total             scalar int
│       │   ├── n_frames_lost              scalar int
│       │   ├── tracking_ratio             scalar float
│       │   ├── n_relocalization_events    scalar int
│       │   ├── orb_slam3_settings_path    scalar str
│       │   └── atlas_path                 scalar str
│       └── gripper_width/          populated by step 4
│           ├── width_m             (N_gopro,) float32 — meters, interpolated across full frame grid
│           ├── raw_widths_m        (N_detections,) float32 — detections only
│           ├── raw_timestamps_s    (N_detections,) float64
│           ├── finger_corners      (N_gopro, 2, 4, 2) float32 — ArUco corner pixel coords per frame
│           ├── detection_rate      scalar float
│           ├── n_detected          scalar int
│           ├── n_frames            scalar int
│           ├── left_id             scalar int — ArUco marker ID
│           ├── right_id            scalar int — ArUco marker ID
│           ├── marker_size_m       scalar float
│           ├── nominal_z_m         scalar float
│           └── z_tolerance_m       scalar float
└── optitrack/                      scene-level (only if OptiTrack CSVs were found at ingest)
    ├── pose                        (N_optitrack, 7) float64 — [x, y, z, qx, qy, qz, qw]
    └── timestamps                  (N_optitrack,) float64 — UTC seconds
```

## Codecs

- **Video** (finger and GoPro frames): `imagecodecs.numcodecs.Jpegxl(effort=1)`. Per-frame chunking — one frame per chunk — so random-access frame loading at training time doesn't have to decode entire video segments. `effort=1` is perceptually lossless and encodes fast. Decode is slower than raw — mitigate with parallel data loaders.

- **IMU, timestamps, scalar arrays**: `numcodecs.Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)`. Blosc works well for smoothly-varying signals like IMU readings, claiming 4–8× compression and decodes faster than the data loader can consume.

- **Audio**: same Blosc-zstd as a default. This is suboptimal — a real audio codec like FLAC would compress 2–3× better — but the array ergonomics are worth it for `pzarr`. If audio storage becomes a noticeable fraction of dataset size, add a separate FLAC sidecar at archive time.

## Timestamps and shared time

Each stream has its own 1D `float64` timestamp array under `episode_N/timestamps/`, expressing absolute UTC seconds at that stream's native rate. **No resampling at storage time** — preserve raw sample times. To get synchronized data across different timing domains (ie pi+gopro+optitrack), you must select a t0 and interpolate/downsample yourself, synchronizing based on the timing offsets in `annotations/time_sync`. See the existing export scripts in `ingest/polyumi_ingest/export` for examples of this.

The shared-time window for an episode is bracketed by `annotations/episode_start` and `annotations/episode_end`, both stored as zarr scalar arrays.

GoPro's GPMF telemetry contains multiple substreams (accl, gyro, GPS) at differing native rates. Timestamps are synthesized uniformly from the recording start time and actual sample count (`recording_start_s + arange(n) / (n / duration_s)`), so the effective rate is derived from the data rather than assumed.

## Clock alignment (time sync step)

GoPro and Pi finger camera run on separate clocks. Step 1 (`chirp-time-sync`) detects the onset of the sync chirp played at the start of each session in both the finger air mic and GoPro audio tracks. The resulting offset `gopro_to_finger_offset_s` (stored in `annotations/time_sync/`) is subtracted from GoPro timestamps at read time to align all streams to the Pi (finger) clock domain. The full alignment result — nominal offset, fine-tuned residual, chirp onset times, and peak correlation values — is preserved in `annotations/time_sync/` for diagnostics.

## Pipeline steps

Steps are tracked in `preprocessing_steps` (list of completed step numbers) in the scene root `.zattrs`, enabling idempotent re-runs and partial pipeline execution. Run `pingest pp <step> --scene <path>` to execute a step; add `--force` to re-run a completed step.

| Step | Name | Reads | Writes |
|------|------|-------|--------|
| 1 | `chirp-time-sync` | finger_air, gopro/audio, timestamps | `annotations/time_sync/` |
| 2 | `orb-slam3` | gopro/frames, gopro/accl, gopro/gyro, timestamps | `gopro/slam_poses`, `annotations/slam/`, `.osa` atlas sidecar |
| 3 | `slam-optitrack-align` | gopro/slam_poses, optitrack/pose, timestamps | `optitrack_to_slam_transform` in root `.zattrs` |
| 4 | `aruco-gripper-width` | finger/frames, timestamps/finger | `annotations/gripper_width/` |

## SLAM is a swappable step

SLAM is well-isolated: input is GoPro frames + IMU + timestamps from `pzarr`, output is `episode_N/gopro/slam_poses` (N, 7) with NaN for lost frames. The choice of SLAM tool (ORB-SLAM3 fork, DROID-SLAM, MASt3R-SLAM, fiducial+EKF, etc.) is opaque to the working format — it's just a step that fills in `slam_poses`.

If using ORB-SLAM3 specifically, its persistent atlas (keyframes + map points + bag-of-words db) is saved as a binary `.osa` sidecar file alongside the zarr — not inside it. The atlas path is `{scene_name}.atlas.osa`. This is only useful if you want to add new episodes to an existing scene later, or run downstream analysis that benefits from the keyframe database. Other SLAM tools don't generally produce a comparable persistent map artifact, so the sidecar is ORB-SLAM3-specific.

## Pose sources

`slam_poses` is one possible source of gripper trajectory. The schema also supports:

- `optitrack/pose` at the scene level: when external mocap is available for a scene, this is populated at ingest time and aligned to the SLAM frame via step 3
- Future additional pose sources just become new arrays with their own timestamp arrays

You can have multiple sources for the same scene and decide downstream which to use (or fuse them). The DP export step accepts a `pose_source` argument (`'optitrack'` or `'slam'`) to select which to use.

## Gripper width from fiducials

The gripper width for each episode is derived from ArUco fiducial markers (IDs 0 and 1) on the gripper fingers, visible in the **finger camera** footage. Step 4 (`aruco-gripper-width`) detects these markers per frame, computes 6DOF pose via fisheye undistortion + solvePnP, and derives the gripper opening from the x-coordinate difference between fingers. Width is linearly interpolated across the full GoPro frame grid for frames where detection fails, and stored in `annotations/gripper_width/width_m`. Raw per-detection results and diagnostics (detection rate, corner coordinates, marker config) are also preserved.

## Scene-level metadata

The scene `.zattrs` contains:

- Static descriptive fields: `task`, `date`, `n_episodes`, `location`
- Versioning: `pipeline_version`, `git_sha`, `created_at` (ISO 8601), `pzarr_version` — so you can tell which pipeline and schema version produced any given scene
- `alignment_refs`: cross-episode anchor information (e.g. timestamps when a known calibration marker was visible), used to define the shared scene coordinate frame
- `preprocessing_steps`: list of completed step numbers (int), updated by each step on success
- `optitrack_start_time`: ISO 8601 timestamp for the OptiTrack recording start (if present)
- `optitrack_to_slam_transform`: `{translation: [x, y, z], rotation: [qx, qy, qz, qw], rms_pos: float, rms_rot_deg: float}` — populated by step 3
- `gripper_calib`: contents of `gripper_calib.yaml` (transforms between gripper, OptiTrack, GoPro, and world frames; ArUco marker config)

## Sidecar files

These live alongside the zarr, not inside it:

- **Raw `.mp4` originals** (`finger.mp4`, `gopro.mp4` per session directory): keep these. Decoded frames in zarr are post-decode and post-codec, so re-decoding from source is the only way to recover full fidelity. Also useful for debugging and for swapping in different SLAM tools that may want different decoding parameters.
- **SLAM atlas** (`{scene_name}.atlas.osa`): only when using ORB-SLAM3. Placed in the scene directory, not inside the zarr.

## Export targets

`pzarr` is the source of truth; downstream formats are exports produced on demand.

- **Diffusion Policy ReplayBuffer zarr** (`pingest export-dp`): flat zarr layout with `data/{img,state,action,reward,not_done}` arrays resampled to 10 Hz on the GoPro frame grid, plus `meta/episode_ends`. State/action is an 8-vector `[x, y, z, qx, qy, qz, qw, gripper_m]`. Only `EPISODE`-typed sessions are exported; `MAPPING` sessions are skipped.

- **MCAP** (`pingest export-mcap`): one `.mcap` file per episode, with channels for finger image, GoPro image, both audio streams, IMU, GPS, SLAM pose, OptiTrack pose, ArUco annotations, and gripper width. Uses Foxglove JSON schemas; audio is chunked at 4096 samples per message.

- **LeRobot**: not implemented directly. Intended path is pzarr → DP zarr → `forge convert` to LeRobot/RLDS/RoboDM.

## Why not LeRobotDataset v3? (etc)

LeRobotDataset v3 is now the de facto OSS standard for sharing robot learning data, and it's a great fit for that use case — but it's a training/sharing format, not a working format. Its tabular Parquet layout assumes a single time grid per episode, the format is designed to be complete at write time rather than incrementally mutated by pipeline steps, and intermediate artifacts like SLAM atlases have no natural home. We treat it the same as CRB's diffusion policy zarr: an export target downstream of `pzarr`, not a replacement for it.

It is also deliberately *not* the same as CRB's diffusion policy format (`gen_dataset_hitl.py`). That format is downsampled, preprocessed, and single-rate; this one preserves full-rate multi-stream data with per-stream timestamps so SLAM and other steps have everything they need.
