# Adding contact-mic audio to the policy (the ManiWAV recipe)

This is a **handoff document**, not a description of code that exists. PolyUMI's ingest side now
exports the finger contact mic; nothing yet consumes it. What follows is the data contract you can
build against, why the spectrogram is deliberately *not* precomputed for you, and — as an
appendix — a checklist of what to port from [ManiWAV](https://github.com/real-stanford/maniwav)
if you want to follow their recipe rather than invent your own.

```
pingest fetch → process-all → pp (steps 1–6) → export-polyumi → TRAIN (your container)
                                                    │
                                                    └── data/mic_0, raw waveform
```

The visuomotor path is unchanged and still lives in
[training-instructions.md](training-instructions.md). `pingest export-dp` produces exactly what it
always did; `export-polyumi` is a second command that adds modalities on top.

---

## 1. The contract: what `export-polyumi` gives you

A `.zarr.zip` identical to `export-dp`'s, plus one key:

```
data/mic_0    (T, samples_per_step) float32    Blosc-zstd
```

`samples_per_step` is `frame_stride × samples_per_gopro_frame` — **536** at the current stride of 2
(16 kHz ÷ 59.94 fps ≈ 267 samples per frame, stored 268 wide). Don't hardcode it; read it back:

| `meta.attrs` key | meaning |
|---|---|
| `mic_0_sample_rate_hz` | 16000. No resampling needed — see §4. |
| `mic_0_samples_per_step` | width of one `mic_0` row |
| `mic_0_samples_per_gopro_frame` | the per-frame block width the rows are built from |
| `mic_0_block_alignment` | `causal` or `forward` — see below |
| `mic_0_source` | `finger/finger_piezo`, i.e. the piezo, not the air mic |

**Raw waveform, not a spectrogram.** That is the whole point of §2.

**One row per step, and consecutive rows join seamlessly.** Each row holds the `stride` per-frame
audio blocks belonging to that step, concatenated. Blocks are anchored on each GoPro frame's own
timestamp with a width no smaller than the largest gap between anchors, so
`mic_0.reshape(-1)` is a **gapless waveform** — which is what makes ManiWAV's
`down_sample_steps: 1` audio path valid on our data, since it flattens the rows back into one
signal before computing features. Two caveats worth knowing:

- **A dropped GoPro frame leaves a real hole.** The pzarr's `annotations/contact_audio` records
  `n_frame_gaps` per episode. We do not interpolate over it; the audio genuinely wasn't recorded.
- **The first step of each episode is zero-padded** under causal alignment, because there is no
  audio before the episode starts. Silence, not a repeat of the block that follows — repeating
  would splice a copy of real audio into the signal where it would read as a contact event.
  ManiWAV's sampler zero-pads audio at episode start for the same reason, where it edge-repeats
  RGB. Your sampler should keep doing that at *window* boundaries too.

**Block alignment is causal by default, and this diverges from ManiWAV.** Their blocks are
`[i·800, (i+1)·800)` — the audio *after* the observation instant. At their 60 fps that is 16.7 ms
of look-ahead; at our ~29.97 Hz step rate it would be **33 ms**, and at inference that audio does
not exist yet. So we default to the `stride` frames *ending* at the step. Flip it with
`block_alignment: forward` in `ingest/config/contact_audio.yaml` and re-export if you want to
match them exactly — it is a re-export, not a re-preprocess. Either way the checkpoint's
`meta.attrs` records which convention it trained under, and a server must reproduce it.

**Only the contact mic is exported.** ManiWAV carries `mic_0` (contact) and `mic_1` (air); their
task config uses only `mic_0` anyway. I can add mic_1 if you wish, however.

---

## 2. Why the log-mel is yours to compute, not ours to ship

The obvious request — "have ingest write spectrograms, they're smaller" — is wrong on four counts,
and the first is decisive.

**Augmentation lives in the waveform domain.** ManiWAV's `NoiseAug` and `RobotNoiseAug` add ESC-50
clips and recorded robot-motor noise at p=0.5, in `__getitem__`, *before* the mel. You cannot add
noise in the log domain. Precomputing spectrograms doesn't merely diverge from their structure; it
makes the augmentation their robustness results lean on impossible to run.

**Mel parameters are hyperparameters.** `n_mels`, hop, window, `f_min`. Freezing them at ingest
means re-running preprocessing across the whole corpus to try 128 bins instead of 64.

**Neither ingest nor the ROS side has torch.** Precomputing would need a numpy port of the mel
transform in `ingest/`, *and* a second copy in `ros2_ws/` for inference, to avoid needing to add torch as a dependency (especially a hassle for the ros environment). Your
container already has torch and torchaudio, so putting the transform there means one
implementation instead of three.

**Bandwidth is not a counter-argument.** At the 30 Hz step grid, `mic_0` costs ~2 kB/step against
`camera0_rgb`'s ~75 kB/step (Blosc-compressed). Precomputing mels would save on the order of 1% of
the transfer. It is not worth any of the above.

The one real cost of the split is throughput: the feature extraction runs per batch instead of
once. If it bites, cache it inside your container — not by changing the archival format.

---

## 3. Where ManiWAV actually computes it

Worth stating plainly, because reading their repo the obvious way finds the wrong code. **They do
not use `torchaudio.transforms.MelSpectrogram` in the path that produced their results.** Their
zarr holds raw waveform; the features are computed *online* in the observation encoder:

```
ManiWAVObsEncoder.forward()      diffusion_policy/model/vision/maniwav_obs_encoder.py
  └─ ASTFeatureExtractor         transformers
       └─ torchaudio.compliance.kaldi.fbank      <- the actual mel computation
```

That is a Kaldi-style log filterbank, not `MelSpectrogram` + `AmplitudeToDB`. The
`MelSpectrogram` you will find in their repo is either commented out in the config (the resnet18
audio path) or in `utils/audio_vis.py`, a matplotlib debug renderer that nothing trains on.

The effective parameter set, for reference:

| | |
|---|---|
| sample rate | 16000 (they resample 48k→16k first; **we don't need to**) |
| mel bins | 64, `low_freq=20`, `high_freq=` Nyquist |
| frame / hop | 25 ms (400 samples) / 10 ms (160 samples) |
| window | hanning (overrides Kaldi's povey default) |
| FFT size | 512 — `round_to_power_of_two` pads the 400-sample window *after* windowing |
| pre-emphasis | 0.97, with per-frame DC offset removal |
| output | **natural log**, floored at `finfo(float32).eps`. No dB conversion. |
| normalisation | `do_normalize=False`, then a global min-max to [-1, 1] |

Two things to know before copying it verbatim:

- **The fbank runs a per-sample Python loop on CPU**, and `.cpu().numpy()` forces a GPU sync
  mid-forward. This is a genuine throughput cost, not a stylistic quibble.
- **The [-1, 1] mapping has no clamp.** It uses global dataset min/max, so an out-of-distribution
  loud transient at eval time escapes the range silently. Add the clamp.

---

## 4. Two deltas our data forces

Not optional, and the easiest things to miss when porting.

**Drop the 48k→16k resample.** PolyUMI records the piezo at 16 kHz already; ManiWAV's GoPro audio
was 48 kHz. Their resample appears in *three* places — the workspace config's
`audio_encoder_cfg.transforms`, the encoder's `forward()` audio branch, and `get_normalizer`'s
duplicate transform pipeline. Watch out: the encoder indexes `key_transform_map[key][0]` and
`[1:]` explicitly, so an empty transform list raises. Apply the whole transform module instead of
keeping a no-op `Resample(16000, 16000)`.

**Re-derive `max_length`.** Theirs is `(horizon // 60) * 100`, which hardcodes 60 fps and 800
samples per frame. The general form is:

```
max_length = round(audio_obs_horizon * samples_per_step / sample_rate * 100)
```

At `samples_per_step = 536` and 16 kHz, an `audio_obs_horizon` of **60** gives
60 × 536 / 16000 = 2.01 s → 199 mel frames → `max_length: 200`. That keeps the AST input exactly
the size ManiWAV's was, over the same 2 s of wall time. Note their horizon of 120 covers 2 s at
*their* step rate; ours is half theirs, so the same wall-clock window needs half the steps.

---

## 5. Selecting your policy at train time

`train_policy.sh` forwards a `DP_CONFIG` environment variable into the container:

```bash
DP_CONFIG=train_diffusion_unet_timm_polyumi_audio_workspace \
DATASET=/abs/path/to/audio.zarr.zip ./train_policy.sh
```

For this to do anything, the fork's `docker/train.sh` must read it:

```bash
--config-name="${DP_CONFIG:-train_diffusion_unet_timm_polyumi_workspace}"
```

Unset, it keeps today's visuomotor default, so the hook is inert until you wire that half. It is
an env var rather than a Hydra override because Hydra cannot override `--config-name` through a
passthrough override.

---

## 6. Inference is blocked, and not on you

Do not wire `mic_0` into `serve_policy.py` expecting it to run on the arm. Two independent
blockers:

1. **A clock-domain bug on the ROS side.** `pi_receiver_node` republishes camera frames stamped
   with the Pi's monotonic `SensorTimestamp` and audio stamped with epoch time *as if they shared
   a clock*. Until that is fixed, `latency.piezo_mic` in
   `ros2_ws/src/polyumi_ros2/config/inference.yaml` cannot be given a meaningful value — it is
   declared and deliberately left at 0. An observation is only as fresh as its slowest signal, so
   adding audio also makes the capture instant the oldest across streams.

   (this should not be an issue to bring up training, but I need to fix it before we train a real model intended for actual inference)
2. **The block-alignment convention** (§1) has to be reproduced exactly at serve time, from a live
   audio stream rather than a stored array.

Worth doing now regardless: make the server raise at load time if a checkpoint expects a modality
it cannot supply, so an audio checkpoint fails loudly instead of being served with a missing
observation.

---

## Appendix — port checklist (optional)

Take it or leave it; this is what following ManiWAV closely would involve. Their source is at
`external/` if you have the submodule, or upstream. Each ported file is worth an attribution
header in the style of `ingest/polyumi_ingest/preproc/_umi_cv_util.py` — source URL, license, and
an explicit list of what you changed.

**Add:**

- `model/common/noise_aug.py` — `NoiseAug`, `RobotNoiseAug`. Three fixes worth making on the way
  in: drop the unused `from audiomentations import *`; guard the `np.random.choice` when the noise
  asset is shorter than the clip (tile-and-crop); make a missing `data/esc-50` disable the
  augmentation with a warning rather than kill the run.
- `model/vision/audio_encoder.py`, `model/vision/maniwav_obs_encoder.py`.
- `dataset/umi_maniwav_dataset.py`, and the `policy/` + `workspace/` counterparts.
- Config: `config/train_diffusion_unet_maniwav_workspace.yaml`, `config/task/umi_maniwav.yaml`.
  Their `mic_0` entry is
  `{shape: [800], horizon: 120, latency_steps: 0, down_sample_steps: 1, type: audio}`;
  ours would be `shape: [536]`, `horizon: 60` (§4). Note `down_sample_steps: 1` for audio while
  RGB uses 3 — audio has to stay contiguous.

**Modify — this is the regression surface,** because these files are shared with the visuomotor
path:

- `common/sampler.py` — audio zero-pads at episode start where RGB edge-repeats. Keep the audio
  key list defaulting to empty so `UmiDataset` stays bit-identical for the visuomotor policy.
- `model/common/normalizer.py` — the `if 'mic' in key` skip (waveforms are not normalised; the
  spectrogram is, later).
- `common/normalize_util.py` — `spec_to_stats`.
- `conda_environment.yaml` / `Dockerfile` — may need `transformers` and `torchaudio`.

Whatever you do here, please train the visuomotor config once afterwards as a control. Those four
shared files are the ones that can regress it silently.
