# `pzarr` - PolyUMI's Working Data Format

![pzarr data schema diagram](/docs/polyumi_working_format_schema.svg)

## Purpose

`pzarr` is the *working data format* for the PolyUMI preprocessing pipeline. It sits between raw ingest (mp4s, audio files, metadata json) and the training-ready exports (Northwestern CRB's diffusion policy zarr, LeRobot, MCAP). Pipeline steps modify it in place; once the pipeline is complete for a scene, you can archive the result.

Each `pzarr` corresponds to a single recorded scene, composed of one or many episodes (referred to as "sessions" in the pi app). Sessions are typed: `MAPPING` sessions are used to build the SLAM map; `EPISODE` sessions are the task demonstrations exported for training.

Compared to downstream data formats like LeRobot Dataset or Diffusion Policy's zarr, which are optimized for feeding directly into a training pipeline, `pzarr` is intended as a single source of truth for all scene data that doesn't discard any information. This means that unlike these downstream formats, it:

1. Allows efficient incremental writes from multiple pipeline steps (e.g. SLAM, gripper width extraction) without needing to rewrite the whole episode or scene on each step
2. Preserves the original multi-rate timestamps from each stream, rather than resampling to a common time grid
3. Stores full-fidelity audio and preserves full-fidelity video via the source `gopro.mp4` (frames are decoded from it on demand rather than re-encoded into the store), rather than pre-encoding into a downsampled training codec

`pzarr` is implemented as a zarr `DirectoryStore` with a specific schema. The schema is designed to be flexible and extensible, but the above principles should guide any additions or modifications.


## Library and format version

Use `zarr-python 3.x` with `zarr_format=2` explicitly. zarr-python 3 reads and writes v2 stores cleanly, but the v2 format gives us reliable JpegXl codec support (the v3 codec story for non-spec codecs has interop caveats) and matches the format that downstream tools like forge and CRB's `ReplayBuffer` already expect. If sharding becomes a real pain point as datasets grow, migrate to v3 later via `zarr.copy()`.

The format version is tracked as `pzarr_version` (currently `5`) in the scene root `.zattrs`. Read this from the store in your code rather than hardcoding it, so schema migrations are operational rather than code changes. See `ingest/polyumi_ingest/pzarr/version.py` for the version history.

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
│   ├── gopro/                      NOTE: no `frames` array — GoPro frames are decoded
│   │   │                           on demand from the gopro.mp4 sidecar (see below)
│   │   ├── audio                   (N_gopro_audio,) float32 — mono GoPro mic
│   │   ├── accl                    (N_accl, 3) float64 — [z, x, y] m/s²
│   │   ├── gyro                    (N_gyro, 3) float64 — [z, x, y] rad/s
│   │   ├── gps                     (N_gps, 3) float64 — [lat, lon, alt]
│   │   └── slam_poses              (N_gopro, 7) float64 — [x, y, z, qx, qy, qz, qw] of the GoPro
│   │                                    optical frame (x right, y down, z forward); NaN when lost or never fed
│   ├── eef/                        populated by step 5 — one array per available source
│   │   ├── pose_optitrack          (N_gopro, 7) float64 — [x, y, z, qx, qy, qz, qw] on the
│   │   │                           hand body frame, gopro grid; NaN where unsolved
│   │   └── pose_slam               same shape/frame; NaN where SLAM had no pose, which
│   │                               includes every frame the localizer was never fed
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
│       │   ├── n_frames_lost              scalar int — whole grid; do NOT gate on this
│       │   ├── frame_stride               scalar int — every Nth frame is fed to the localizer
│       │   ├── n_frames_fed               scalar int
│       │   ├── n_frames_fed_tracked       scalar int
│       │   ├── n_frames_fed_post_chirp    scalar int — the window the exporter ships
│       │   ├── n_frames_fed_lost_post_chirp  scalar int — what the usability gate counts
│       │   ├── chirp_gated                scalar bool — False = the two above cover the whole episode
│       │   ├── tracking_ratio             scalar float — over fed frames
│       │   ├── n_relocalization_events    scalar int
│       │   ├── orb_slam3_settings_path    scalar str
│       │   └── atlas_path                 scalar str
│       ├── gripper_width/          populated by step 4
│       │   ├── width_m             (N_gopro,) float32 — meters, interpolated across full frame grid
│       │   ├── raw_widths_m        (N_detections,) float32 — detections only
│       │   ├── raw_timestamps_s    (N_detections,) float64
│       │   ├── finger_corners      (N_gopro, 2, 4, 2) float32 — ArUco corner pixel coords per frame
│       │   ├── detection_rate      scalar float
│       │   ├── n_detected          scalar int
│       │   ├── n_frames            scalar int
│       │   ├── left_id             scalar int — ArUco marker ID
│       │   ├── right_id            scalar int — ArUco marker ID
│       │   ├── marker_size_m       scalar float
│       │   ├── nominal_z_m         scalar float
│       │   └── z_tolerance_m       scalar float
│       └── contact_audio/          populated by step 6
│           ├── frame_blocks              (N_gopro, B) float32 — piezo samples for each GoPro frame
│           ├── frame_block_start_idx     (N_gopro,) int64 — each block's start in finger/finger_piezo
│           ├── logmel                    (N_hops, n_mels) float32 — DIAGNOSTIC ONLY; nothing trains on it
│           ├── logmel_timestamps         (N_hops,) float64 — UTC seconds, hop centres
│           ├── sample_rate_hz            scalar int
│           ├── samples_per_gopro_frame   scalar int — B
│           ├── block_alignment           scalar str — causal | forward
│           ├── nominal_samples_per_frame scalar float — median anchor spacing
│           ├── max_frame_spacing_samples scalar int
│           ├── n_frames                  scalar int
│           ├── n_frame_gaps              scalar int — intervals wider than B, i.e. dropped GoPro frames
│           ├── n_zero_filled_samples     scalar int — blocks reading past the end of the recording
│           ├── coverage                  scalar float — fraction of the piezo array the blocks span
│           └── rms                       scalar float — did the mic record anything at all
└── optitrack/                      scene-level (only if OptiTrack CSVs were found at ingest)
    ├── pose                        (N_optitrack, 7) float64 — [x, y, z, qx, qy, qz, qw]
    └── timestamps                  (N_optitrack,) float64 — UTC seconds
```

## Codecs

- **Video** (finger frames only): `imagecodecs.numcodecs.Jpegxl(effort=1)`. Per-frame chunking — one frame per chunk — so random-access frame loading at training time doesn't have to decode entire video segments. `effort=1` is perceptually lossless and encodes fast. Decode is slower than raw — mitigate with parallel data loaders.

- **GoPro frames are NOT stored in the zarr.** They are decoded on demand straight from the `gopro.mp4` sidecar via `video_helpers.GoproMp4Frames`, which presents a zarr-`Array`-like surface (`len`/`shape`/integer & slice indexing, returning `(H,W,3)` uint8 RGB) on the `timestamps/gopro` grid. Storing a per-frame JpegXl re-encode of frames the mp4 already holds inflated the store ~70× (e.g. ~38 GB of `gopro/frames` vs. a ~0.5 GB `gopro.mp4`) with no fidelity gain — the mp4 is the true source, and JpegXl was a redundant second lossy pass. The reader is forward-sequential (all consumers iterate a contiguous ascending range); it is not thread-safe. `timestamps/gopro` (length = authoritative frame count) is still written at ingest.

- **IMU, timestamps, scalar arrays**: `numcodecs.Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)`. Blosc works well for smoothly-varying signals like IMU readings, claiming 4–8× compression and decodes faster than the data loader can consume.

- **Audio**: same Blosc-zstd as a default. This is suboptimal — a real audio codec like FLAC would compress 2–3× better — but the array ergonomics are worth it for `pzarr`. If audio storage becomes a noticeable fraction of dataset size, add a separate FLAC sidecar at archive time.

## Timestamps and shared time

Each stream has its own 1D `float64` timestamp array under `episode_N/timestamps/`, expressing absolute UTC seconds at that stream's native rate. **No resampling at storage time** — preserve raw sample times. To get synchronized data across different timing domains (ie pi+gopro+optitrack), you must select a t0 and interpolate/downsample yourself, synchronizing based on the timing offsets in `annotations/time_sync`. See the existing export scripts in `ingest/polyumi_ingest/export` for examples of this.

The shared-time window for an episode is bracketed by `annotations/episode_start` and `annotations/episode_end`, both stored as zarr scalar arrays.

**The finger audio anchor is the ADC capture instant.** `timestamps/finger_air` and
`timestamps/finger_piezo` are built from `metadata.json`'s `audio_start_time_ns`, which the Pi
records as the instant the first sample hit the converter — not the instant the block reached
userspace, which is at least one 20 ms callback later. Recordings made before this changed carry
the delivery instant instead, so their finger audio sits ~20 ms late against `timestamps/finger`
(the video, anchored independently on `FrameWallClock`). This does **not** affect anything routed
through the chirp: `annotations/time_sync/gopro_to_finger_offset_s` is measured against the same
timeline, so it absorbs the shift exactly. Only direct finger-audio-to-finger-video comparisons
see it, and only in a dataset mixing recordings from both sides of the change.

GoPro's GPMF telemetry contains multiple substreams (accl, gyro, GPS) at differing native rates. Timestamps are synthesized uniformly from the recording start time and actual sample count (`recording_start_s + arange(n) / (n / duration_s)`), so the effective rate is derived from the data rather than assumed.

## Clock alignment (time sync step)

GoPro and Pi finger camera run on separate clocks. Step 1 (`chirp-time-sync`) detects the onset of the sync chirp played at the start of each session in both the finger air mic and GoPro audio tracks. The resulting offset `gopro_to_finger_offset_s` (stored in `annotations/time_sync/`) is subtracted from GoPro timestamps at read time to align all streams to the Pi (finger) clock domain. The full alignment result — nominal offset, fine-tuned residual, chirp onset times, and peak correlation values — is preserved in `annotations/time_sync/` for diagnostics.

## Pipeline steps

Steps are tracked in `preprocessing_steps` (list of completed step numbers) in the scene root `.zattrs`, enabling idempotent re-runs and partial pipeline execution. Run `pingest pp <step> --scene <path>` to execute a step; add `--force` to re-run a completed step.

| Step | Name | Reads | Writes |
|------|------|-------|--------|
| 1 | `chirp-time-sync` | finger_air, gopro/audio, timestamps | `annotations/time_sync/` |
| 2 | `orb-slam3` | gopro.mp4 (sidecar), gopro/accl, gopro/gyro, timestamps | `gopro/slam_poses`, `annotations/slam/`, `.osa` atlas sidecar |
| 3 | `slam-optitrack-align` | gopro/slam_poses, optitrack/pose, timestamps | `optitrack_to_slam_transform` in root `.zattrs` |
| 4 | `aruco-gripper-width` | gopro frames (from gopro.mp4), timestamps/gopro | `annotations/gripper_width/` |
| 5 | `eef-pose` | optitrack/pose and/or gopro/slam_poses, timestamps, `gripper_calib.yaml` | `eef/pose_optitrack`, `eef/pose_slam` (whichever the scene has) |
| 6 | `contact-audio` | finger/finger_piezo, timestamps/finger_piezo, timestamps/gopro, `annotations/time_sync` | `annotations/contact_audio/` |

## SLAM is a swappable step

SLAM is well-isolated: input is GoPro frames + IMU + timestamps from `pzarr`, output is `episode_N/gopro/slam_poses` (N, 7) with NaN for lost frames. The choice of SLAM tool (ORB-SLAM3 fork, DROID-SLAM, MASt3R-SLAM, fiducial+EKF, etc.) is opaque to the working format — it's just a step that fills in `slam_poses`.

If using ORB-SLAM3 specifically, its persistent atlas (keyframes + map points + bag-of-words db) is saved as a binary `.osa` sidecar file alongside the zarr — not inside it. The atlas path is `{scene_name}.atlas.osa`. This is only useful if you want to add new episodes to an existing scene later, or run downstream analysis that benefits from the keyframe database. Other SLAM tools don't generally produce a comparable persistent map artifact, so the sidecar is ORB-SLAM3-specific.

## Pose sources

`slam_poses` is one possible source of gripper trajectory. The schema also supports:

- `optitrack/pose` at the scene level: when external mocap is available for a scene, this is populated at ingest time and aligned to the SLAM frame via step 3
- Future additional pose sources just become new arrays with their own timestamp arrays

You can have multiple sources for the same scene and decide downstream which to use (or fuse them).

### `eef/pose_<source>` — the canonical trajectory, per source

**Raw pose sources are not directly usable by a policy, and they are not interchangeable.** Each reports a *different body frame*:

| Source | Body frame it reports | Why that's wrong for a policy |
|---|---|---|
| `optitrack/pose` | the marker **rigid body** | Motive puts the origin at the marker centroid — an arbitrary point that moves whenever markers are re-stuck |
| `gopro/slam_poses` | the GoPro **optical frame** | not the point the robot reports at inference |

Step 5 (`eef-pose`) resolves this per source: for every source an episode actually has, it converts that source onto the canonical **hand** body frame, resamples onto the GoPro frame grid, and writes `episode_N/eef/pose_optitrack` and/or `episode_N/eef/pose_slam` — one array per source, not a single winner. (SLAM's array is taken exactly as reported: since pzarr v4 nothing is interpolated, so rows SLAM could not place — including every frame the localizer was never fed under `localization_frame_stride` — stay NaN, and the exporter turns runs of NaN into episode boundaries.) Picking *which* source to train on is deferred to **export time**, not baked in here — see `export.dp.buffer.resolve_pose_source`: it defaults to OptiTrack when present (else SLAM), overridable per session via `scene.json`'s `pose_source_overrides`. This means changing the source is a re-export, not a re-preprocess.

Both sources route through the **GoPro frame**, then take one shared `T_gopro_to_fingertip` hop:

```
slam:      T_s_gp  ─────────────────────────────────►  · T_gp_hand
optitrack: T_o_rb · inv(T_gb_rb) · T_gb_gp  ────────►  · T_gp_hand
```

**The GoPro is the pivot because it is the only body both embodiments share.** `gripper_base` is a mechanical part of the *handheld* gripper that the Franka end-effector does not have, so a frame anchored to it could never be reproduced on the robot — it survives only as an intermediate in the OptiTrack chain, where it is valid because the markers really are mounted on that part. The GoPro-to-fingers geometry is identical across both, so a hand frame defined against the GoPro is reproducible on either.

The **world** frame is deliberately left as each source's own (OptiTrack frame or SLAM frame), recorded per array in its `world_frame` attr. That asymmetry is the whole point:

- A shared **world** frame *cancels* out of the relative pose representation policies train on — `inv(T_0)·T_k` is invariant to a global re-frame. So normalizing it buys nothing.
- A **body** offset does *not* cancel. It conjugates the relative transform: `inv(T_0·X)·(T_k·X) = inv(X)·(inv(T_0)·T_k)·X`, leaking a `(R − I)·x` error into the relative translation. At a 20 cm offset a 30° wrist rotation injects ~11 cm of phantom translation; even at the ~7 cm GoPro-to-fingertip scale it is ~4 cm. That is why this step is not optional.

Each `eef/pose_<source>` array records its own `world_frame`, `body_frame` (`hand`), `grid` (`gopro`), and `n_nan` attrs. The `eef` group itself records `available_sources` (which arrays this episode has) and `default_source` (what export uses absent a `scene.json` override).

> **`T_gopro_to_fingertip` in `config/gripper_calib.yaml` is measured from the PolyUMI CAD assembly**, not from a calibration rig: its origin is the centre of the GoPro lens faceplate plus a 5 mm allowance for the sensor plane, and its target is the midpoint of the closed fingertips on the plane of the finger's upper surface. It is on the critical path for every exported pose, so re-derive it from CAD whenever the mount geometry changes.

> **At inference**, the robot must report this same physical point — the policy compares like with like or not at all. On the FR3 that point is the `polyumi_tcp` frame, defined once in `nuc/tcp_calib.py` and named by both `eef_frame` (observation) and the MoveIt bridge's `eef_link` (command); the stock `fr3_hand_tcp` is a different point in a different axis convention. See [crb-fr3-inference.md](crb-fr3-inference.md).

## Gripper width from fiducials

The gripper width for each episode is derived from ArUco fiducial markers (IDs 0 and 1) on the gripper fingers, visible in the **GoPro** footage (step 4 reads `gopro.mp4`, not the finger camera). Step 4 (`aruco-gripper-width`) detects these markers per frame, computes 6DOF pose via fisheye undistortion + solvePnP, and derives the gripper opening from the x-coordinate difference between fingers. Width is linearly interpolated across the full GoPro frame grid for frames where detection fails, and stored in `annotations/gripper_width/width_m`. Raw per-detection results and diagnostics (detection rate, corner coordinates, marker config) are also preserved.

> **`width_m` is raw tag separation, not jaw opening.** The tags sit on the fingers, so a fully-closed gripper still measures several millimetres. The pzarr deliberately stores the raw measurement — it is calibration-independent, so re-deriving the calibration costs a re-export rather than re-running step 4's per-frame ArUco pass. The subtraction happens in the DP exporter (see below), matching where UMI applies it. Use `raw_widths_m` (the actual detections) rather than `width_m` (resampled onto the GoPro grid with hold-at-edges extrapolation) for anything that cares about the extremes — `pingest calibrate-gripper` does.

## Contact-mic audio on the frame grid

The finger piezo (`finger/finger_piezo`, the left channel; `finger_air` is the air mic that
carries the sync chirp) is recorded at 16 kHz on the Pi's clock, while video is stamped on the
GoPro's. Step 6 reconciles the two and slices the audio into one block per GoPro frame, which
`pingest export --type polyumi` concatenates into the exported `mic_0`.

**Blocks are anchored by timestamp, never by multiplying a rate.** 16000 / 59.94 = 266.93 samples
per frame is not an integer — ManiWAV's 48000 / 60 = 800 was — so a fixed multiply would walk off
the audio over an episode. Each frame's block starts at `searchsorted(piezo_ts, gopro_ts_in_finger_clock)`
for that frame alone, which is also why step 1 is a hard prerequisite: without the chirp offset the
two clocks are unrelated epochs, and an unshifted anchor is wrong by seconds with nothing
downstream able to notice.

**The block width is fixed and slightly generous** (`samples_per_gopro_frame` in
`config/contact_audio.yaml`, currently 268 = `ceil(266.93) + 1`). Because it is at least as large
as the biggest gap between consecutive anchors, block *k* always reaches block *k+1*'s start — so
consecutive blocks abut or overlap and **never leave a hole**. That is the property the exporter
depends on: flattening consecutive `mic_0` rows yields a gapless waveform, which is what ManiWAV's
audio path assumes when it reassembles the signal before computing its spectrogram. The cost is
~0.4% of samples appearing in two adjacent blocks. A genuinely dropped GoPro frame does leave a
hole; `n_frame_gaps` counts them rather than papering over missing audio.

**The `logmel` array is diagnostic only.** It exists so the catalog can show whether the mic
recorded anything; nothing reads it downstream. The spectrogram the policy sees is computed in the
training container from the exported waveform, *after* augmentation that only exists in the
waveform domain — see [maniwav-audio-policy.md](maniwav-audio-policy.md) for why that split is
deliberate and what the `mic_0` contract is.

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

- **Raw `.mp4` originals** (`finger.mp4`, `gopro.mp4` per session directory): keep these. `gopro.mp4` is now **load-bearing** — GoPro frames are decoded from it on demand (there is no `gopro/frames` array), so every GoPro consumer (aruco step 4, DP export, mcap export) resolves it via `scene_files.resolve_gopro_mp4`. `finger.mp4` is still just a convenience encode (finger frames live in the zarr).
- **SLAM atlas** (`{scene_name}.atlas.osa`): only when using ORB-SLAM3. Placed in the scene directory, not inside the zarr.

### Archiving

`pingest archive-scene` produces a **self-contained** `scene.zarr.zip` that bundles `scene.zarr` **plus** each `session_*/gopro.mp4` and the atlas sidecar, with paths relative to the scene directory (so an unzip reproduces `<scene>/scene.zarr` + `<scene>/session_*/gopro.mp4`, exactly what the frame reader resolves against). Because the GoPro video now lives only in the mp4, the mp4s must travel with the archive for it to remain replayable/exportable. `ZIP_STORED` (no re-compress): zarr chunks and the mp4 are already compressed.

## Camera preprocessing contracts

The policy only compares like with like, so the frames the DP exporter bakes into the dataset at training time and the frames the ROS inference node feeds the policy must go through the **same** pixel transforms. Two cameras, two contracts.

### `camera0_rgb` (the GoPro)

- input is an **RGB** `(H, W, 3)` uint8 frame;
- it is **centre-cropped to 4:3**, the GoPro's recording aspect (`crop_to_source_aspect`) — a no-op on a frame already at that aspect;
- output is `(224, 224, 3)` uint8, resized with **`cv2.INTER_AREA`** (the anti-aliased choice for downscaling), squashed to the target;
- any `float32/255` normalization is applied downstream (the training loader / the inference node), not baked into the stored uint8.

### `finger_rgb` (the finger camera)

The gripper mount occludes a fixed strip of the finger camera's view, so this contract is a **crop to given bounds** rather than one derived from the frame:

- input is an **RGB** `(H, W, 3)` uint8 frame — `finger/frames` is already RGB in the pzarr, converted at ingest;
- half-open `[min, max)` bounds from `ingest/config/finger_camera.yaml`, where `null` means the frame's own edge, so the geometry survives a change of camera resolution instead of landing somewhere else. The shipped crop is `x_min: 170` on the **1152x648** recorded frame, giving **982x648**. (1152x648 is the resolution the Pi *stores*; `cam_streamer.py`'s `VIEW_WIDTH`/`VIEW_HEIGHT` of 620x480 size the live preview stream and are not what lands in the pzarr.) 170 is measured rather than guessed: averaged over 9 episodes in 3 scenes, columns 0-130 are saturated blue with frame-to-frame variation ~3 (static hardware), 130-170 is the lens blur across that edge with variation climbing 4→11, and past 170 it plateaus at ~12, which is scene;
- an optional `output_size` resize with `cv2.INTER_AREA`. **Default `null`** — the crop ships at native size and the choice of encoder input size stays with whoever builds the policy;
- bounds that don't fit the frame **raise**. Silently clipping would produce a plausible-looking image that isn't the one the contract names.

The crop **is** the `finger_rgb` contract: a policy trained on one crop cannot be served frames from another, so retuning it invalidates existing checkpoints. The resolved bounds go into the buffer's `meta.attrs` so a checkpoint says which crop it trained under. Nothing on the inference side calls this yet — `policy_client_node` has no finger-camera subscription — but the transform is mirrored now so that wiring is a subscription rather than a second derivation.

### Why it's implemented twice

Once per side, because the two live in separate Python environments and can't share an import: `ingest/polyumi_ingest/camera_preproc.py` (used by `export/dp/`) and `ros2_ws/src/polyumi_ros2/polyumi_ros2/camera_preproc.py` (used by `policy_client_node.py`). **Keep them byte-identical** — the whole file, both contracts.

The split is not incidental and is unlikely to go away. Making `polyumi_ingest` a dependency of the ROS package does not work: it declares `requires-python = ">=3.13"` while the ROS node runs Ubuntu 24.04's `/usr/bin/python3` (3.12), and it would drag `polyumi_pi` (→ `lgpio`), zarr, imagecodecs, mcap and scipy into the inference process for the sake of ~30 lines of numpy and cv2. A third minimal package depending on nothing but numpy+cv2 is the option if more shared contract code ever accumulates.

**Identical source is not the guarantee that matters, though — identical output is.** The two environments run different library majors (measured 2026-08-09: ROS `cv2 4.6.0` / `numpy 1.26.4`, uv workspace `cv2 4.13.0` / `numpy 2.4.3`), so `cv2.resize` is a different C++ implementation on each side no matter how the Python is shared. Both test suites therefore pin **the same sha256 digests**, from the single shared `ingest/test/camera_preproc_golden.py` (the ROS suite loads it by path), in `test_golden_vector_is_stable_across_environments`. For `camera0_rgb` the inputs are a synthetic 1920×1080 frame (crop active) and a synthetic 2704×2028 one (crop a no-op). For `finger_rgb` they are a synthetic 1152×648 frame at the shipped crop, once with `output_size=None` and once resized: the crop-only digest is a pure array slice and so really checks that the two *implementations* still agree, while the resized one pins `INTER_AREA` so `output_size` can be switched on later without anything silently skewing. That catches source drift *and* a library upgrade that changes `INTER_AREA` under one side. The inputs are built with plain uint8 arithmetic rather than a seeded RNG, since numpy guarantees nothing about `default_rng`'s stream across versions. A changed digest means checkpoints already trained are on a transform the inference node no longer reproduces; regenerate the constants only when that is the intent.

### Why the camera0 crop exists

Training frames come from `gopro.mp4` at **2704x2028 (4:3)**. Inference frames come off the Elgato at **1920x1080 (16:9)**. Measured on hardware (2026-08-09): the GoPro's clean-HDMI output **pillarboxes** — the image occupies columns 240..1679 of the 1080p frame, exactly 1440x1080, with pure black bars either side, and the fisheye image circle sits at the same fraction of frame width as in the recording. **The field of view is identical; only the framing differs.**

So without the crop the inference 224² was **25.4% black pixels** with the real content squeezed into three-quarters of the width — a train/inference domain gap on every policy output, and a silent one, since the policy runs perfectly happily on the wrong pixels. The crop recovers precisely the 1440x1080 the camera framed, so both paths squash the same field of view. It is a no-op on 2704x2028, so **no dataset needs re-exporting for this** and old buffers stay valid.

> **Known residual skew (not a bug):** even after the crop, training frames are downscaled from 2704 px wide and inference frames from 1440 px, so the two 224² results differ slightly in resampling detail. Same FOV, same aspect, different source resolution. Closing that fully is out of scope; it is recorded here so it is not mistaken for a defect. (Contact-mic audio has its own alignment requirement — see "Contact-mic audio on the frame grid" — but it is a separate contract from the pixel one, and the default `export --type dp` carries no audio at all.)

## Export targets

`pzarr` is the source of truth; downstream formats are exports produced on demand.

- **UMI ReplayBuffer** (`pingest export`, `--type dp` — the default): a `.zarr.zip` matching `universal_manipulation_interface`'s `ReplayBuffer` so `UmiDataset` reads it directly. `meta/episode_ends` plus `data/` keys `camera0_rgb` (T,224,224,3 uint8, Blosc-zstd; see "Camera preprocessing contracts" above), `robot0_eef_pos` (T,3), `robot0_eef_rot_axis_angle` (T,3, rotvec), `robot0_gripper_width` (T,1, **metres of opening from fully closed** — the exporter subtracts `gripper_calib.yaml`'s `closed_mm` from step 4's raw tag separation and clamps the result at 0, following UMI's `get_gripper_calibration_interpolator`, whose `interp1d(..., fill_value=(x[0], x[-1]))` saturates the same way below the calibrated minimum; the clamp matters because `closed_mm` is a percentile, so ~1% of detections fall under it by construction; the value used is recorded as `meta.attrs['gripper_closed_width_m']` so a buffer is self-describing), and `robot0_demo_start_pose`/`robot0_demo_end_pose` (T,6, the episode's first/last `[pos, rotvec]` broadcast). The key *names* are load-bearing — `UmiDataset` name-matches them. Poses are read from `eef/pose_<source>` (so step 5 must have run) and are on the hand frame; the exporter resolves the source per episode (see the `eef/pose_<source>` section above) and records the choice as provenance — in `meta.attrs['pose_provenance']`/`episode_pose_source` inside the `.zarr.zip`, and in a `<output>.provenance.json` sidecar (or the catalog's `DatasetManifest`). The `action` key is deliberately omitted (the sampler synthesises it from `[eef_pos, eef_rot_axis_angle, gripper_width]`). Steps are the frames SLAM was *fed* — every `localization_frame_stride`-th GoPro frame, so ~29.97 Hz at the current stride of 2, not the 59.94 Hz the camera records at — and the training config sets the observation rate from there via `obs_down_sample_steps`. **Those two knobs are coupled:** halving the stored rate must halve `obs_down_sample_steps`, or the policy trains on a different Δt than it runs at. A single buffer may not mix strides; export refuses to write one that does. A session is split into one episode per contiguous trustworthy run, cutting wherever the pose source drops out, the gripper width is missing, a modality stops covering the span, or the hand teleports further between two adjacent frames than `quality_thresholds.yaml`'s `max_pose_jump_m` (a relocalization glitch — both frames either side of the cut are kept, since it is the *transition* that is wrong, not either pose). Runs shorter than `--min-segment-steps` are discarded; that floor defaults to 90, chosen because UMI's sampler runs with `action_padding: False` and so draws **zero** training samples from an episode under `(action_horizon-1)*obs_down_sample_steps+1` = 46 stored steps, and 90 is the shortest round floor at which no surviving segment contributes fewer than five. It is recorded as `meta.attrs['min_segment_steps']`, and each segment's provenance carries `cut_start`/`cut_end` naming what bounded it (`episode_start`, `episode_end`, `chirp`, `pose_gap`, `pose_jump`, `gripper_gap`, or a modality name). There is **no automatic whole-episode veto** — a session with holes is segmented around them, not discarded; only `scene.json`'s human `unusable_episodes` set drops one outright. `pingest segments <scene>` previews the entire plan, including what got cut and why, without decoding a frame. Only `EPISODE`-typed sessions are exported; `MAPPING` sessions are skipped.

- **PolyUMI ReplayBuffer** (`pingest export --type polyumi`): the same buffer as above plus PolyUMI's extra observation streams. First, `data/mic_0` (T, `stride × B`) float32, the contact mic as **raw 16 kHz waveform**, Blosc-zstd like everything else. Each row is the `stride` per-frame blocks belonging to that step, concatenated, so flattening consecutive rows reconstructs the waveform with no gaps (see "Contact-mic audio on the frame grid"). Whether a row holds the audio before or after its observation instant is `block_alignment` in `config/contact_audio.yaml` — **causal by default**, because ManiWAV's forward convention would give the policy 33 ms of look-ahead at our step rate that does not exist at inference. The geometry is recorded in `meta.attrs` (`mic_0_sample_rate_hz`, `mic_0_samples_per_step`, `mic_0_samples_per_gopro_frame`, `mic_0_block_alignment`, `mic_0_source`) so a checkpoint is self-describing. Raw waveform rather than a spectrogram is a deliberate split; see [maniwav-audio-policy.md](maniwav-audio-policy.md). Needs preprocessing step 6; the default `--type dp` reads none of it and still works on scenes that never ran it.

  Second, `data/finger_rgb` (T, 648, 982, 3) uint8, the gripper's finger camera cropped to the region the mount doesn't occlude (see "Camera preprocessing contracts"). Cropped but **not resized**: the exported frame is the native crop, and choosing the encoder's input size stays with whoever builds the policy. Two properties a training config has to account for:

  - **It is ~13x the bytes of `camera0_rgb`** — 1.91 MB/step against 151 kB (measured on a real scene: 1.23 MB/step after Blosc-zstd, so ~1.55x compression). A 62-episode scene runs to ~18 GB raw for this one key. Setting `output_size` in `config/finger_camera.yaml` cuts it by the area ratio — ~38x at 224².
  - **The camera records at 10 fps against a ~30 Hz step grid** (confirmed on real scenes: 3.05 steps per source frame), so roughly three consecutive rows hold the *same* source frame. Each step takes the frame nearest it in time — nothing is interpolated, because a policy must see images the camera actually produced — and the measured rate is recorded as `meta.attrs['finger_rgb_source_rate_hz']`. A `down_sample_steps` that ignores it makes an observation window fetch the same image twice.

  Geometry and coverage go into `meta.attrs` (`finger_rgb_crop`, `finger_rgb_output_size`, `finger_rgb_shape`, `finger_rgb_source_rate_hz`, `finger_rgb_max_staleness_s`, `finger_rgb_source`), and per-segment staleness into the provenance. Steps the finger stream doesn't cover are **excluded from the export** — trimmed, or split around, exactly as a pose dropout is; `max_staleness_s` is the threshold. This is not a rare path: measured across 111 episodes in 3 scenes, the finger camera stops recording ~0.65 s before the GoPro *every time* (and starts ~1 s after, which the chirp trim already removes), so the last ~10% of steps have no coverage while the median staleness over the rest is a healthy 0.02-0.03 s. Rejecting those episodes would reject the whole corpus; exporting them would pair one frozen frame with moving proprioception, which `nearest_idx`'s clamping makes the default outcome and no shape assertion would catch. An episode left with fewer than `--min-segment-steps` after trimming is skipped, as it already is for pose dropouts. Needs no preprocessing step of its own beyond what the default `--type dp` already requires.

- **MCAP** (`pingest export-mcap`): one `.mcap` file per episode, with channels for finger image, GoPro image, both audio streams, IMU, GPS, SLAM pose, OptiTrack pose, ArUco annotations, and gripper width. Uses Foxglove JSON schemas; audio is chunked at 4096 samples per message.

- **LeRobot**: not implemented directly. Intended path is pzarr → DP zarr → `forge convert` to LeRobot/RLDS/RoboDM.

## Why not LeRobotDataset v3? (etc)

LeRobotDataset v3 is now the de facto OSS standard for sharing robot learning data, and it's a great fit for that use case — but it's a training/sharing format, not a working format. Its tabular Parquet layout assumes a single time grid per episode, the format is designed to be complete at write time rather than incrementally mutated by pipeline steps, and intermediate artifacts like SLAM atlases have no natural home. We treat it the same as CRB's diffusion policy zarr: an export target downstream of `pzarr`, not a replacement for it.

It is also deliberately *not* the same as CRB's diffusion policy format (`gen_dataset_hitl.py`). That format is downsampled, preprocessed, and single-rate; this one preserves full-rate multi-stream data with per-stream timestamps so SLAM and other steps have everything they need.
