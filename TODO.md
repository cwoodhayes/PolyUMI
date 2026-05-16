# PolyUMI TODO

## SLAM pipeline — before first run

- [ ] **Build ORB-SLAM3** — confirm which fork and build target to use (UMI fork
  uses a custom `gopro_slam` binary; stock build uses `mono_inertial_gopro_vi`).
  Pass the correct binary names to `OrbSlam3Step(map_builder_bin=..., localizer_bin=...)`.

- [ ] **Verify binary CLI interface** — run the map-builder and localizer binaries
  with no arguments and confirm the exact argument order matches what
  `OrbSlam3Step._build_map` and `_localize_episode` pass:
  ```
  # map builder:
  <bin> <vocab> <settings_yaml> <frames_dir> <imu_csv> <atlas_out>
  # localizer:
  <bin> <vocab> <settings_yaml> <frames_dir> <imu_csv> <atlas_in> <traj_out>
  ```

- [ ] **Verify trajectory CSV format** — confirm the localizer outputs a CSV with
  header `timestamp,tx,ty,tz,qx,qy,qz,qw,is_lost`.  If the binary uses a
  different format (e.g. TUM space-separated, no header, no `is_lost` column),
  update `_parse_trajectory_csv` in `ingest/polyumi_ingest/preproc/slam_step.py`.

- [ ] **Verify IMU timestamp convention** — `slam_step.py` exports absolute UTC
  seconds from zarr.  The `mono_inertial_gopro_vi.cc` example uses CTS
  (relative milliseconds from clip start).  Confirm whether the target binary
  expects absolute or relative timestamps; if relative, subtract `gyro_ts[0]`
  in `_export_imu_csv`.

- [ ] **Run Hero 12 calibration** and fill in `ingest/config/gopro_hero12_slam.yaml`.
  Every `CALIBRATE_ME` value must be replaced; the step refuses to run until
  none remain.  Checklist:
  - Camera intrinsics: `fx`, `fy`, `cx`, `cy`, `xi`, `alpha` (DoubleSphere model)
  - `Camera.width`, `Camera.height`, `Camera.fps` — set to your recording config
  - `Tbc` — camera-to-IMU body extrinsics (Kalibr or OpenImuCameraCalibrator)
  - `IMU.NoiseGyro`, `IMU.NoiseAcc`, `IMU.GyroWalk`, `IMU.AccWalk` — Allan
    deviation from a static 2-hour IMU log

## SLAM pipeline — downstream steps (not yet implemented)

- [ ] **Step 3 — workspace calibration**: transform SLAM poses from the map's
  arbitrary coordinate frame into a physical workspace frame using an ArUco
  marker board or calibration target.  Implement as `PreprocessingStep`
  step number 3.

- [ ] **Scale validation**: VI-SLAM with the Hero 12 IMU should give metric scale,
  but nothing currently checks it.  Add a sanity-check annotation
  (e.g. mean translation norm per second) that step 3 or downstream training
  code can use to flag degenerate trajectories.

- [ ] **Cross-episode consistency check**: each episode is localized independently
  against the shared map.  Add an optional check that overlapping spatial
  regions across episodes produce consistent poses (within some tolerance).

## Recording

- [ ] **Record a MAPPING session** using `polyumi-pi start-scene`.  The first
  button press now records a `MAPPING` session; subsequent presses record
  `EPISODE` sessions.  Verify `metadata.json` contains
  `"session_type": "MAPPING"` for the first session after deploying to the Pi.
