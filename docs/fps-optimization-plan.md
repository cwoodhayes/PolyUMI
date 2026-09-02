# Pi camera FPS optimization — measurements and plan

The Pi camera has run at 10 fps (`CameraStreamer.FPS`) since the beginning, and that rate has
propagated into the rest of the system as an assumption: the policy runs at 10 Hz, and
`finger_rgb` lands on a ~30 Hz step grid roughly three steps at a time. The GoPro fires at 60.
This document records what the Pi can actually do, measured on hardware, and what it would take
to raise the rate.

**Bottom line: 10 fps is a configuration choice, not a hardware limit.** The capture path has
roughly 4× headroom, and the exposure rationale written next to the constant does not hold at
lab lighting. 15 fps is available for the cost of changing one constant. 20 fps is available on
the streaming path today, and is within ~2% on the recording path, where a cheap SD card is the
remaining obstacle.

## Test setup

Measured 2026-08-29 on `conorpi` — Raspberry Pi Zero 2 W (4×A53 @ 1 GHz, 416 MB RAM,
VideoCore IV), IMX708 camera, RaspiAudio ULTRA++ HAT, "SD16G" SD card, scene at ~91 lux.
`polyumi-pi.service` stopped for the duration so nothing else held the camera.

Every "full app" number below runs the real `_run_video_streamer` and `_run_audio_streamer`
from `pi/polyumi_pi/main.py` with audio capture live, not a camera loop in isolation — only
`CameraStreamer.FPS` was monkeypatched. Achieved rate is computed from
`video_timestamps.csv`, so it counts frames that actually reached disk.

## 1. The hardware ceiling

Three independent measurements agree on ~52–56 fps at the production capture geometry
(sensor mode 1, 2304×1296 → 1152×648 YUV420):

| Check | Result |
|---|---|
| Sensor mode 1 advertised maximum | 56.03 fps |
| `rpicam-vid`, MJPEG, matched settings, 60 fps cap | ~53 fps |
| picamera2 capture + release, no encode, 60 fps cap | 52.5 fps @ 33% of one core |

The `rpicam-vid` run used the same sensor mode, output size, AWB, EV, and manual lens position
as `configure_camera()`, so it is a like-for-like comparison rather than a vendor best case.
That our own capture loop lands within ~1 fps of it says the Python layer is not the problem.

## 2. Encode paths, isolated

At a 60 fps cap, so the encoder rather than `FrameDurationLimits` is what binds:

| Path | fps | CPU | JPEG size |
|---|---|---|---|
| `capture_file(format='jpeg')` — production today | 40.3 | 120% of one core | 93.7 KiB |
| `simplejpeg.encode_jpeg_yuv_planes` | 42.9 | 122% | 66.1 KiB |
| same, `fastdct=True` | 43.9 | 123% | 66.2 KiB |
| hardware `MJPEGEncoder` (bcm2835-codec) | 45.6 | **64%** | **41.7 KiB** |

The current path is a software encode — YUV420 → RGB → libjpeg — and costs about a full core
more than the hardware encoder for a file nearly twice the size. It is nonetheless fast enough
that it is *not* what caps us at 10 fps.

## 3. The full application

| Configuration | achieved | interval jitter (sd) | frame loss |
|---|---|---|---|
| record → SD, 10 fps, 20 s | 10.06 | 0.0 ms | 0% |
| record → SD, 15 fps, 90 s | 14.97 | 4.5 ms | **0.2%** |
| record → SD, 20 fps, 90 s | 19.61 | 16.5 ms | 2.0% |
| record → SD, 25 fps, 25 s | 24.93 | 2.8 ms | 0.5% |
| record → SD, 30 fps, 25 s | 27.67 | 67.8 ms | 8.6% |
| record → **tmpfs**, 20 fps (twice) | **20.00** | 2.3 ms | 0.2% |
| stream → WiFi ZMQ, 10 fps | 10.13 | p95 108 ms | — |
| **stream → WiFi ZMQ, 20 fps** | **20.00** | p95 68 ms, max 353 ms | 12.9 Mbit/s |

The median interval is exactly nominal in every row — the loop keeps pace. What the loss column
counts is a handful of *long* stalls (up to 1.8 s) that swallow whole runs of frames.

## 4. Exposure — the reason written next to the constant

`FPS = 10` carries the comment "locked to improve exposure in dim lighting (vs 30fps default)".
Raising the rate shrinks the maximum frame duration, so this is the right thing to worry about.
It does not bind here. AE was allowed to settle for 3 s at each cap:

| cap | frame budget | ExposureTime | AnalogueGain | mean luma |
|---|---|---|---|---|
| 10 | 100.0 ms | 34.74 ms | 2.00 | 101.0 |
| 15 | 66.7 ms | 34.74 ms | 2.00 | 101.1 |
| 20 | 50.0 ms | 34.74 ms | 2.00 | 101.1 |
| 30 | 33.3 ms | 32.68 ms | 2.12 | 101.0 |

AE asks for 34.7 ms. A 20 fps cap allows 50 ms, so there is 1.44× of light headroom before the
cap costs anything, and the image at 20 fps is pixel-for-pixel as bright as at 10. Only at 30 fps
does AE start trading exposure for gain. **The crossover is around 63 lux** — below that, a 20 fps
cap starts forcing gain (and therefore noise) that a 10 fps cap would not. This was measured in one
room at 91 lux; a genuinely dim collection environment could still bind, which is the argument for
making the rate a parameter rather than a new hardcoded constant.

## 5. What actually limits the recording path

Not bandwidth, and not CPU. Sustained SD write measured **6.1 MB/s** against the 1.54 MB/s that
20 fps needs, and individual writes land in page cache in 0.67 ms (max 1.0 ms). The losses are
2–4 isolated multi-hundred-millisecond stalls per run, characteristic of dirty-page writeback
throttling on top of a bottom-tier card's internal garbage collection.

The decisive evidence is tmpfs: identical code, identical rate, **20.00 fps and 0.2% loss, twice**.
The card is the whole difference.

Two fixes were tried and **neither worked**:

- **Writer thread**, moving `frame_path.write_bytes()` off the capture loop behind a 120-frame
  queue: 19.61 → 19.80 fps. The likely reason it did not help is that the loop still calls
  `timestamps_fp.flush()` on every frame, which is also a write to the same throttled filesystem.
  That combination is untested.
- **Smoothing kernel writeback** (`vm.dirty_background_bytes` = 8 MB, `vm.dirty_bytes` = 32 MB, so
  writeback runs continuously instead of in bursts): 19.65 fps. No improvement. Settings restored.

## 6. Cost of the change

At the measured 78.6 KiB/frame:

| fps | SD write | per hour | WiFi |
|---|---|---|---|
| 10 | 0.77 MB/s | 2.70 GB | 6.4 Mbit/s |
| 15 | 1.15 MB/s | 4.05 GB | 9.7 Mbit/s |
| 20 | 1.54 MB/s | 5.40 GB | 12.9 Mbit/s |

With ~7 GB free, continuous recording headroom falls from ~150 min to ~75 min at 20 fps. That is
a per-session limit, not a corpus limit, since sessions are fetched off between scenes.

## Plan

### Phase 0 — take 15 fps

Change `CameraStreamer.FPS` from 10 to 15. That is the entire diff; 14.97 fps sustained over
90 s with 0.2% loss and 4.5 ms jitter, on the current SD card, with audio running.

### Phase 1 — 20 fps

The two paths have different blockers, so decide them separately.

**Streaming (inference) is already there.** 20.00 fps sustained over WiFi at 12.9 Mbit/s with the
same one-constant change. If lifting the policy off 10 Hz is the goal, nothing further is needed
on the Pi.

**Recording needs the SD card dealt with.** In order:

1. **Try an A2-rated card.** This is the measured root cause and costs no diff at all. Re-run the
   90 s / 20 fps recording benchmark and compare against the 19.61 baseline and the 20.00 tmpfs
   figure — tmpfs is the target the card is failing to reach.
2. If that is not enough, **switch the record path to the hardware `MJPEGEncoder`**. It attacks the
   pressure directly, halving bytes written (79 → 42 KiB) and CPU (120% → 64%). It is a real
   refactor, not a constant: `MJPEGEncoder` delivers frames through an `Output` callback rather
   than the capture loop, so the per-frame `SensorTimestamp` / `FrameWallClock` the wire contract
   and the sidecar CSV depend on has to be re-established from the callback's timestamp. It also
   changes the stored pixels slightly (a different encoder at nominal quality 85), which is a
   consideration against existing checkpoints.
3. The writer thread is worth one more attempt **combined with** batching the
   `timestamps_fp.flush()`, since neither was tested without the other in the loop.

**2% loss at 20 fps is not a reason to stay at 10.** Coverage strictly improves: worst-case
step-to-frame distance is unchanged at ~100 ms while the typical case halves. The 2% is worth
removing, but it should not gate the decision.

### Phase 2 — make the rate a parameter

`CameraStreamer.FPS` is a class constant read in two places in `main.py`. Turning it into a CLI
option / constructor argument means the rate can be dialled back for a genuinely dim environment
(see the ~63 lux crossover above) without a redeploy, and it retires the incorrect
`record-episode --fps 10` in CLAUDE.md, which documents an option that does not exist.

### Adjacent findings, all pre-existing

Not part of this work, but found while measuring and worth their own issues:

- **`ScalerCrop` is anamorphic.** `compute_scaler_crop()` is called with `VIEW_WIDTH`/`VIEW_HEIGHT`
  (620×480, aspect 1.29) but the result is rendered into the 1152×648 main stream (aspect 1.78) —
  a **1.376× horizontal stretch** on every frame the system has ever recorded. It is baked into the
  `finger_rgb` crop bounds and into every existing checkpoint, so this is a flag, not a proposal.
- **`pi_receiver_node.py:164`** builds `ros_msg.data = list(proto.jpeg_data)` — roughly 79 000
  Python ints per frame. That cost doubles at 20 fps.
- **`CameraFrame.width`/`height` are wrong on the wire**: the streamer sets 620×480 while shipping
  1152×648 JPEGs. Harmless today because the receiver publishes a `CompressedImage` and ignores
  both fields, but it is a trap for the next reader.

### Downstream impact

Low. The exporter derives the finger rate from the data itself
(`1.0 / median(diff(finger_ts))` in `ingest/polyumi_ingest/export/dp/finger_camera.py`) and records
it as `finger_rgb_source_rate_hz`, so it is rate-agnostic. The `~3 consecutive steps` remark there
describes a `np.unique` optimization that is correct at any rate.

The one knob tuned against 10 fps is `max_staleness_s: 0.15` in `ingest/config/finger_camera.yaml`,
documented as "1.5 frame periods". At 20 fps it becomes 3 frame periods — more permissive, so
nothing breaks, but it should be retightened to preserve the guard's strength. Prose in
`docs/data-format.md` and `docs/maniwav-audio-policy.md` states 10 fps and would need updating.

### Not tested

Dimmer lighting; a better SD card; the hardware encoder wired into the app rather than
benchmarked in isolation; runs longer than 90 s; thermal soak (46 °C at idle, `get_throttled=0x0`
throughout, but no sustained-load soak was run).
