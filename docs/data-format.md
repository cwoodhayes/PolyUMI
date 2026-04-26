# `pzarr` - PolyUMI's Working Data Format

![pzarr data schema diagram](/docs/polyumi_working_format_schema.svg)

## Purpose

`pzarr` is the *working data format* for the PolyUMI preprocessing pipeline. It sits between raw ingest (mp4s, audio files, metadata json) and the training-ready exports (Northwestern CRB's diffusion policy zarr, LeRobot, MCAP). Pipeline steps modify it in place; once the pipeline is complete for a scene, you can archive the result.

Each `pzarr` corresponds to a single recorded scene, composed of one or many episodes (referred to as "sessions" in the pi app).

Compared to downstream data formats like LeRobot Dataset or Diffusion Policy's zarr, which are optimized for feeding directly into a training pipeline, `pzarr` is intended as a single source of truth for all scene data that doesn't discard any information. This means that unlike these downstream formats, it:

1. Allows efficient incremental writes from multiple pipeline steps (e.g. SLAM, gripper width extraction) without needing to rewrite the whole episode or scene on each step
2. Preserves the original multi-rate timestamps from each stream, rather than resampling to a common time grid
3. Stores full-fidelity decoded video audio and video, rather than pre-encoding into a training codec (like WebM for video) or a heavily downsampled format

`pzarr` is implemented as a zarr `DirectoryStore` with a specific schema. The schema is designed to be flexible and extensible, but the above principles should guide any additions or modifications.


## Library and format version

Use `zarr-python 3.x` with `zarr_format=2` explicitly. zarr-python 3 reads and writes v2 stores cleanly, but the v2 format gives us reliable JpegXl codec support (the v3 codec story for non-spec codecs has interop caveats) and matches the format that downstream tools like forge and CRB's `ReplayBuffer` already expect. If sharding becomes a real pain point as datasets grow, migrate to v3 later via `zarr.copy()`.

Read the version from store metadata in your code rather than hardcoding it, so the migration is operational rather than a code change.

## Codecs

- **Video** (GoPro, finger camera): `imagecodecs.numcodecs.JpegXl`. Per-frame chunking — one frame per chunk — so random-access frame loading at training time doesn't have to decode entire video segments. JPEG XL gets ~30% better compression than plain JPEG at the same visual quality, supports near-lossless at modest size, and is the best per-frame photographic codec available. Decode is slower than raw — mitigate with parallel data loaders. Benchmark actual compression ratios on your own GoPro footage before committing chunk sizes; published 5–10× ratios assume clean photographic content, and noisy/motion-blurred robot footage typically lands lower.

- **IMU, proprioception, scalar arrays**: `numcodecs.Blosc(cname='zstd', shuffle=Blosc.SHUFFLE)`. Blosc bundles a byte-level shuffle filter (which exposes structure across the bytes of float32 values) with zstd, which is Pareto-optimal for compression ratio vs speed. For float arrays of smoothly-varying signals like IMU readings, this routinely gets 4–8× compression and decodes faster than the data loader can consume. Optionally chain a `numcodecs.Delta` filter ahead of Blosc for IMU specifically — it stores sample-to-sample differences instead of raw values, giving another ~1.5× ratio with no correctness risk.

- **Audio**: same Blosc-zstd as a default. This is suboptimal — a real audio codec like FLAC would compress 2–3× better — but the array ergonomics are worth it for `pzarr`. If audio storage becomes a noticeable fraction of dataset size, add a separate FLAC sidecar at archive time.

## Timestamps and shared time

Each stream has its own 1D `float64` timestamp array under `episode/timestamps/`, expressing absolute UTC seconds at that stream's native rate. **No resampling at storage time** — preserve raw sample times. To get synchronized data across streams, use `np.searchsorted` against the timestamp arrays at read time. The shared-time window for an episode is bracketed by `annotations/episode_start` and `annotations/episode_end`, both stored as zarr scalar arrays (not in `.zattrs` — JSON has subtle precision issues for nanosecond-scale timestamps and rewrites the whole file on any change).

GoPro's GPMF telemetry contains multiple substreams (accel, gyro, GPS, gravity, exposure, etc.) at *differing* native rates — verify the exact rates with `gpmf-parser` against a real recording before committing array shapes.

## SLAM is a swappable step

SLAM is well-isolated: input is GoPro frames + IMU + timestamps from `pzarr`, output is `episode/annotations/slam_poses` (N, 7) with its own timestamp array. The choice of SLAM tool (ORB-SLAM3 fork, DROID-SLAM, MASt3R-SLAM, fiducial+EKF, etc.) is opaque to the working format — it's just a step that fills in `slam_poses`.

If using ORB-SLAM3 specifically, its persistent atlas (keyframes + map points + bag-of-words db) is saved as a binary `.osa` sidecar file alongside the zarr — not inside it. This is only useful if you want to add new episodes to an existing scene later, or run downstream analysis that benefits from the keyframe database. Other SLAM tools don't generally produce a comparable persistent map artifact, so the sidecar is ORB-SLAM3-specific.

## Pose source pluralism

`slam_poses` is one possible source of gripper trajectory. The schema also supports:

- `optitrack_poses`: when external mocap is available for a scene
- Future additional pose sources just become new arrays under `annotations/` with their own timestamps

This is intentional: the format treats pose sources as data, not pipeline outputs. You can have multiple sources for the same episode and decide downstream which to use (or fuse them).

## Gripper width from fiducials

The gripper width for each frame is derived from fiducial markers (ArUco/AprilTag) on the gripper itself, visible in the GoPro footage. This runs as its own pipeline step (after SLAM, before final preprocessing) and writes `annotations/gripper_width` (N,) f32 with its own timestamp array.

## Sidecar files

These live alongside the zarr, not inside it:

- **Raw `.mp4` originals**: keep these. Decoded frames in zarr are post-decode and post-codec, so re-decoding from source is the only way to recover full fidelity. Also useful for debugging and for swapping in different SLAM tools that may want different decoding parameters.
- **SLAM atlas** (e.g. `scene_TASK_DATE.atlas.osa`): only when using ORB-SLAM3.

## Scene-level metadata

The scene `.zattrs` contains:

- Static descriptive fields: `task`, `date`, `n_episodes`, `location`
- Versioning: `pipeline_version`, `git_sha`, `created_at` (ISO 8601) — so you can tell which pipeline produced any given scene
- `alignment_refs`: cross-episode anchor information (e.g. timestamps when a known calibration marker was visible), used to define the shared scene coordinate frame

## Concurrency caveat

Zarr v2's `DirectoryStore` is safe for concurrent chunk writes, but group-level metadata (`.zarray`, `.zgroup`) writes are not atomic. If you parallelize SLAM across episodes writing into the same scene store, **serialize the array-creation parts** (group creation, new array allocation) and only parallelize the chunk writes. Or: have each parallel worker write into a per-episode sidecar zarr and merge into the scene store at the end.

## Forge integration

[Forge](https://github.com/arpitg1304/forge)'s zarr reader expects the flat diffusion-policy layout (`data/*` arrays + `meta/episode_ends`), not our scene/episode subgroup structure. Run the export-to-DP-zarr step first to produce a forge-compatible flat zarr, then `forge convert` to LeRobot, RLDS, or RoboDM as needed.

## At-rest archive

Once the pipeline is complete for a scene, copy the `DirectoryStore` to a `ZipStore` (`scene_TASK_DATE.zarr.zip`). For training, load compressed chunks from the zip directly into a `zarr.MemoryStore()` — this streams compressed bytes from RAM and gets you fast random access without needing the decompressed data to fit in memory. (UMI uses this trick.) If a scene's compressed bytes don't fit in RAM, use the `DirectoryStore` directly during training and rely on the OS page cache.

## Why not LeRobotDataset v3? (etc)

LeRobotDataset v3 is now the de facto OSS standard for sharing robot learning data, and it's a great fit for that use case — but it's a training/sharing format, not a working format. Its tabular Parquet layout assumes a single time grid per episode, the format is designed to be complete at write time rather than incrementally mutated by pipeline steps, and intermediate artifacts like SLAM atlases have no natural home. We treat it the same as CRB's diffusion policy zarr: an export target downstream of `pzarr`, not a replacement for it.

It is also deliberately *not* the same as CRB's diffusion policy format (`gen_dataset_hitl.py`). That format is downsampled, preprocessed, and single-rate; this one preserves full-rate multi-stream data with per-stream timestamps so SLAM and other steps have everything they need.