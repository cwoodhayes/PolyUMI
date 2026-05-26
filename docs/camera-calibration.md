# Camera Calibration

The GoPro Hero 12 fisheye calibration is consumed by two parts of the pipeline:

| Consumer | File | Format |
|---|---|---|
| ORB-SLAM3 (preprocessing step 2) | `external/ORB_SLAM3_PolyUMI/Examples/Monocular-Inertial/gopro_hero12_slam.yaml` | OpenCV YAML, `KannalaBrandt8` model |
| ArUco gripper-width (preprocessing step 4) | `ingest/config/gopro_intrinsics.json` | OpenImuCameraCalibrator FISHEYE JSON |

Both are **generated artifacts** — do not edit them by hand. They are produced from a single source of truth: the OpenImuCameraCalibrator calibration dataset under `<dataset>/cam/cam_calib_*_fi_*.json` (and the matching cam-IMU JSON for `Tbc`)--see [OpenImuCameraCalibrator GoPro calibration guide](https://github.com/urbste/OpenImuCameraCalibrator/blob/master/docs/gopro_calibration.md) for how to generate these files.

`cv2.fisheye` (used by the ArUco step) implements the Kannala-Brandt equidistant 4-coefficient model, which is identical to `KannalaBrandt8` in ORB-SLAM3, so the same `k1..k4` coefficients work for both consumers.

## Regenerating

After a new calibration run with OpenImuCameraCalibrator:

```bash
uv run python ingest/integration/populate_slam_yaml.py \
    --dataset slam/OpenImuCameraCalibrator/calibration_datasets/<dataset>
```

This writes the SLAM YAML in-place and mirrors the source FISHEYE JSON to `ingest/config/gopro_intrinsics.json`. Both files end up with identical intrinsics derived from the same calibration run.

See the [OpenImuCameraCalibrator GoPro calibration guide](https://github.com/urbste/OpenImuCameraCalibrator/blob/master/docs/gopro_calibration.md) for how to record and process a calibration dataset.
