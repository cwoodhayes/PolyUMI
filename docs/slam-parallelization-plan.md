# Parallelizing per-episode SLAM localization (pp step 2)

> **Status: deferred design.** Not implemented. The validation findings in
> [Part 1](#part-1--validation-the-episodes-really-are-independent) and the two issues flagged in
> [Known issues surfaced](#known-issues-surfaced-while-investigating) are true of the code as it
> stands today and are worth reading even if the parallelization itself is never built.

## Motivation

Preprocessing step 2 (`OrbSlam3Step`) localizes each EPISODE session against a pre-built
ORB-SLAM3 atlas **one episode at a time** — `run_step`'s Phase 2 is a plain `for` loop over
`episode_keys`. At ~75 s median per episode (two tracking passes each, since forward+reverse
landed), a 20-episode scene costs **~22 minutes of wall clock** while using roughly one core
of 22.

The premise: episode localization never updates the map, so episodes are independent and can be
distributed across cores.

---

## Part 1 — Validation: the episodes really are independent

Verified in the code rather than assumed.

| Shared resource | Finding |
|---|---|
| **`scene.zarr`** | **Safe.** Each episode writes only under its own `episode_N/` subtree. Empirically checked: writing `episode_1`'s results creates only that subtree's files and modifies **zero** existing files. The store is `zarr_format=2` (no parent index), there is **no consolidated metadata** anywhere in `ingest/`, and `LocalStore` writes are atomic (uuid4 `.partial` + `rename`). |
| **Atlas (`.osa`)** | **Read-only.** `_make_temp_settings_yaml(load_atlas=…)` injects only `System.LoadAtlasFromFile`; `SaveAtlasToFile` is injected *only* by `_build_map`. `System::Shutdown`'s atlas save is guarded by `if(!mStrSaveAtlasToFile.empty())`, which is empty here. The file is opened via `std::ifstream`. |
| **Temp files** | **Unique per episode** — `tempfile.mkdtemp(prefix=f'polyumi_slam_ep{i}_')` holds the settings YAML, telemetry JSON, and both trajectory outputs. |
| **Logs** | **Unique per episode** — `slam_logs/episode_N_slam.{stdout,stderr}`. |
| **`cwd`** | **Shared** (`_run_subprocess(cwd=log_dir)`) but currently harmless: the only cwd-relative writer in the fork (`System::SaveDebugData`, append-mode `init_*.txt`) has **zero callers**. A latent hazard worth removing. |

**Conclusion: per-episode localization is genuinely independent and safe to run concurrently.**

### But cores are not the limiting resource — memory is

1. **Each process buffers every decoded frame in RAM.** The decode-once change stores frames at
   tracking resolution (1352×1014 CV_8UC3 = 3.92 MiB/frame). A typical 374–470-frame episode is
   **1.4–1.8 GiB**; the largest observed (756 frames) is **2.9 GiB**. The buffer is held across
   *both* passes (declared in `main`, passed by const ref to `RunLocalizationPass`).
2. **`ORB_SLAM3::System` has no destructor.** All its members are raw pointers, and
   `RunLocalizationPass` constructs one `System` per pass — so with the reverse pass enabled (the
   default) each process leaks **two vocabularies and two atlases**. Confirmed in the logs:
   "Vocabulary loaded" and "Atlas loaded" each appear twice per episode.

Estimated peak: **~3–5 GB per process.** The dev laptop has 31.5 GB total, but *available* RAM was
observed swinging from 15.8 GB down to 8.3 GB over a single working session, with swap already at
5.3/8 GB. Unbounded parallelism would swap-thrash the desktop — which is why any worker count must
be computed **at run time**, never hardcoded.

3. **OpenCV thread oversubscription.** `cv::setNumThreads` is never called in
   `mono_inertial_gopro_vi_localize.cc` (sibling binaries do call it), so OpenCV's pthreads pool
   defaults to **22 threads per process** here. N processes × 22-thread `parallel_for_` pools would
   thrash the scheduler regardless of memory.

---

## Part 2 — Proposed implementation

Design decisions settled up front: include the grayscale frame-buffer optimization; derive worker
count from available RAM + cores at run time with an explicit override; throttle with
`nice` + an OpenCV thread cap + a bounded pool (no systemd/cgroup limits).

### Step 0 (first): measure the real per-worker peak

The sizing formula needs a calibrated constant, not an estimate. Before and after the grayscale
change, on the worst case available (`episode_15` of `scene_2026-07-28_20-35-20_f406`, 756 frames):

```bash
/usr/bin/time -v <localizer> <vocab> <settings> <mp4> <telemetry> /tmp/f.txt /tmp/r.txt 2>&1 \
  | grep -E "Maximum resident set size|Elapsed"
```

Record peak RSS for a 756-frame and a ~430-frame episode; that becomes `_DEFAULT_WORKER_MEM_GB`.
Don't proceed to sizing without it.

### Step 1 — C++: grayscale frame buffer + OpenCV thread cap

File: `external/ORB_SLAM3_PolyUMI/Examples/Monocular-Inertial/mono_inertial_gopro_vi_localize.cc`

**(a) Grayscale buffer.** In the decode loop (currently `cv::resize(im, resized, img_size);
frames.push_back(std::move(resized));`), convert after the resize and buffer the 1-channel result:
1.37 MB/frame instead of 4.11, a **3× cut**. It also removes a duplicate BGR→gray conversion
currently paid once per pass.

Safety was verified: **nothing downstream mutates the input image**, so one buffer can safely feed
both passes. `ORBextractor::ComputePyramid` level 0 does `copyMakeBorder(image, temp, …)` — it reads
`image` and writes a freshly allocated `temp`, and `mvImagePyramid[0]` is a view into *`temp`*, not
into our buffer. Descriptors are computed on `mvImagePyramid[level].clone()`. `Frame`'s other uses
(`detectMarkers`, `ComputeImageBounds`) are read-only. Passing 1 channel also makes
`Tracking::GrabImageMonocular` skip its `cvtColor` branch entirely and use our Mat directly.

> ⚠ **Use `cv::COLOR_RGB2GRAY`, not `BGR2GRAY` — deliberately.** See
> [the channel-order mismatch](#1-camerargb-says-rgb-but-videocapture-yields-bgr) below. Preserving
> the existing (technically wrong) conversion keeps this a pure memory optimization; "fixing" it
> here would silently change every SLAM result. Comment the choice at the call site, because once
> we pass 1 channel `mbRGB` no longer affects this path and the decode loop becomes the only thing
> that decides.

**(b) OpenCV thread cap.** Read an env var (e.g. `POLYUMI_SLAM_CV_THREADS`) near the top of `main`
and call `cv::setNumThreads(n)` when set and > 0; leave OpenCV's default alone when unset, so
serial runs keep today's speed. Precedent: `mono_inertial_gopro_localize.cc` calls
`cv::setNumThreads(4)`.

Rebuild (~2 min):
```bash
cmake --build external/ORB_SLAM3_PolyUMI/build --target mono_inertial_gopro_vi_localize -j22
```

### Step 2 — Python: bounded thread pool in `slam_step.py`

Mirror the existing pattern in `ingest/polyumi_ingest/video_helpers.py` (`write_frames_to_zarr`):
`num_workers: int | None = None` convention, `ThreadPoolExecutor(max_workers=n)`, collect futures,
`fut.result()` to re-raise.

Use **`ThreadPoolExecutor`, not `ProcessPoolExecutor`** — the real work is a `subprocess.run` that
releases the GIL, and threads keep `_localize_episode` a **bound method**, so
`test_slam_step.py`'s `mock.patch.object(step, '_localize_episode', …)` keeps working. Refactoring
to a module-level function for `ProcessPoolExecutor` would break that test strategy for no gain.

- **Rewrite Phase 2** as a pool, keeping a verbatim sequential path when `n_workers == 1`.
- **Per-thread zarr handle.** Don't share the parent's `root` `Group` across threads: the submitted
  wrapper should `zarr.open_group(str(scene_zarr), mode='a')` itself and derive
  `grp(root_local, ep_key)`, then call `self._localize_episode(ep_grp, …)`. Cheap, and sidesteps any
  `Group` thread-safety question. The parent's `root` won't observe child writes — fine, since the
  only later parent write is `_mark_preprocessing_step` in `step_base.py`, after join.
- **Schedule longest-first (LPT).** Sort `episode_keys` by `len(timestamps/gopro)` descending.
  Minimizes makespan and starts the biggest memory consumer first, so an over-commit shows up
  immediately rather than 20 minutes in.
- **Error handling.** Wrap each `fut.result()`, collect `(ep_key, exc)`, let the others finish, then
  raise one aggregated `RuntimeError` naming every failed episode. Because `run_step` raises,
  `_mark_preprocessing_step` won't mark step 2 complete, while episodes that *did* succeed keep
  their written poses — so a re-run resumes instead of redoing everything.
- **Legible interleaved logs.** `_localize_episode`'s callees emit indented, context-free lines
  (`'  Trajectory reconciliation: …'`). Add a module-level `contextvars.ContextVar` holding the
  current episode key plus a small `logging.Filter` on this module's logger that prefixes
  `[episode_N]`; each worker sets the var on entry. `ContextVar` is per-thread here, so this stays
  contained and needs no signature changes. (Lighter fallback: demote detail lines to DEBUG and emit
  one summary line per episode.)
- **`nice`/`ionice`.** Prefix the localizer argv with `nice -n 10 ionice -c 3` when parallel.
  **Do not use `preexec_fn=os.nice`** — it's documented as unsafe with threads, and we're in a
  thread pool. Both binaries exist at `/usr/bin`; skip the prefix gracefully if absent.
- **`env=`.** `_run_subprocess` passes no `env` today. Add an `env: dict | None = None` parameter and
  pass `{**os.environ, 'POLYUMI_SLAM_CV_THREADS': '1', 'OMP_NUM_THREADS': '1'}` when parallel.
- **Harden `cwd`.** Change `_run_subprocess(cwd=log_dir)` → `cwd=tmp_dir`; nothing in the localizer
  depends on cwd.

### Step 3 — The worker-count knob and formula

`step_base.py` constructs steps as `step_cls()` with **no arguments**, so there's no generic
plumbing for step-specific options. Follow the precedent already in this class, which reads
`ORB_SLAM3_DIR` / `ORB_SLAM3_BIN_SUBDIR` from `os.environ` in `__init__`:

```
OrbSlam3Step.__init__(..., n_workers: int | None = None,
                           worker_mem_gb: float | None = None,
                           mem_reserve_gb: float | None = None)
```

defaulting to `POLYUMI_SLAM_WORKERS` / `POLYUMI_SLAM_WORKER_GB` /
`POLYUMI_SLAM_MEM_RESERVE_GB`, else auto — zero changes to `step_base.py`. Add `--jobs/-j` to
`pingest pp` (`main.py::preprocessing_pipeline`) which sets `POLYUMI_SLAM_WORKERS` before calling
`run_preprocessing`, with a comment explaining why it routes through the environment.

Formula (`_resolve_worker_count`), every input read at run time:

```
explicit override -> clamp(override, 1, n_episodes)

avail_gb  = MemAvailable from /proc/meminfo      # not psutil: installed, but NOT a declared dep
budget_gb = max(0, avail_gb - mem_reserve_gb)    # reserve ~3 GB for the desktop
by_mem    = int(budget_gb // worker_mem_gb)      # worker_mem_gb from the Step 0 measurement
by_cpu    = max(1, len(os.sched_getaffinity(0)) // _CPU_DIVISOR)   # _CPU_DIVISOR = 3 -> 7 here
workers   = max(1, min(by_mem, by_cpu, n_episodes))
```

`MemAvailable` is the kernel's own estimate of free+reclaimable. Size against the **worst-case**
episode so one long episode can't blow the budget. `_CPU_DIVISOR` is a named constant for tuning;
memory normally binds first, and `nice +10` means interactive work preempts regardless. Log the
decision and its inputs on one line so a surprising worker count explains itself.

---

## Verification

1. **Establish a determinism baseline first.** ORB-SLAM3 runs LocalMapping/LoopClosing threads and
   RANSAC, so results may not be bit-identical run-to-run *even serially*. Run one episode twice
   sequentially and diff `gopro/slam_poses`; that measured spread is the tolerance for everything
   below. Don't assume bitwise equality.
2. **Correctness vs. the serial baseline.** On `scene_2026-07-28_20-35-20_f406` (21 episodes), back
   up the current `slam_poses` + `annotations/slam` attrs, run `pingest pp 2 --force` in parallel,
   and compare per episode: `tracking_ratio` within the step-1 tolerance and median pose delta
   sub-mm. Episodes **7 / 8 / 14** are strong anchors — validated at exactly **100%** tracking with
   known merge stats (0.5–1.1 mm forward/reverse overlap agreement).
3. **Grayscale is behavior-preserving.** Because `RGB2GRAY` is retained, one episode re-run after
   the C++ change should match its pre-change poses within the step-1 tolerance. If it doesn't, the
   channel-order reasoning is wrong — stop and investigate before parallelizing.
4. **Speedup.** Time `pingest pp 2 --scene … --force` before (~22 min for 20 episodes) and after;
   report wall clock and the chosen worker count.
5. **Machine stays responsive.** During the run watch `free -m` (swap must not grow), load average,
   and subjective UI responsiveness. Confirm the OpenCV cap took effect per process
   (e.g. `ps -o nlwp`).
6. **Tests** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest ingest/test -q`):
   - Make existing order-dependent assertions order-insensitive
     (`called_localize[0] == '/episode_1'` → compare `sorted(...)`/a set).
   - New: a `threading.Barrier` in the fake `_localize_episode` proving N episodes really do run
     concurrently (times out if the pool is serial).
   - New: one worker raising → aggregated error surfaces, other episodes still complete, step 2 is
     **not** marked complete.
   - New: `_resolve_worker_count` with monkeypatched memory/affinity — low memory → 1, ample
     memory → CPU-capped, explicit override wins, clamped by `n_episodes`.
   - Confirm `n_workers=1` still takes the sequential path.
7. **Lint**: `unset VIRTUAL_ENV && uvx ruff check ingest/`.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Memory exhaustion / swap thrash** | Size from `MemAvailable` at run time against the *worst-case* episode; reserve ~3 GB; LPT ordering surfaces over-commit immediately; `--jobs 1` is always an escape hatch. |
| **Grayscale silently changes results** | Preserve `RGB2GRAY`; verify with check 3 before parallelizing. |
| **Non-determinism mistaken for a parallelism bug** | Establish the run-to-run tolerance first (check 1). |
| **Scheduler thrash** | OpenCV thread cap per worker + `nice -n 10` + `ionice -c 3` + the CPU-divisor cap. |
| **Partial failure leaves a half-processed scene** | Aggregate failures and raise so step 2 isn't marked complete; successful episodes persist, so re-running resumes. |
| **Leaked double vocab/atlas inflates per-worker memory ~2×** | Out of scope, but the biggest remaining win — see below. |

---

## Known issues surfaced while investigating

These are independent of parallelization and are live in the code today.

### 1. `Camera.RGB` says RGB, but `VideoCapture` yields BGR

`Examples/Monocular-Inertial/gopro_hero12_slam.yaml` sets `Camera.RGB: 1`, so `mbRGB` is true and
`Tracking::GrabImageMonocular` applies **`cvtColor(..., COLOR_RGB2GRAY)`**. But
`cv::VideoCapture::read()` returns **BGR**. The pipeline therefore runs an RGB→gray conversion on
BGR data: the R and B luma weights are swapped. The result is a perfectly valid grayscale image —
just not the intended one — which is why this has always worked and never been noticed.

The YAML's own comment ("Frames are exported as standard JPEGs so this should be 1 (RGB)") is
**stale**: this binary reads `gopro.mp4` through `VideoCapture`, not exported JPEGs.

Fixing it (`Camera.RGB: 0` + `BGR2GRAY`) would change the grayscale input to ORB extraction for
every frame, hence change features, tracking, and all stored poses. It may well improve tracking,
but it needs its own validation pass and should not be bundled into an unrelated change.

### 2. `ORB_SLAM3::System` has no destructor

`grep -n "~System"` finds nothing; every member is a raw pointer (`mpVocabulary`, `mpAtlas`,
`mpKeyFrameDatabase`, `mpTracker`, `mpLocalMapper`, `mpLoopCloser`, and the two thread handles).
`RunLocalizationPass` builds one `System` per pass, so a two-pass run holds two of everything at
peak, and the vocabulary (a 145 MB text file parsed via the slow `loadFromTextFile`) is both parsed
and MD5-checksummed twice per episode.

Addressing it — hoisting the vocabulary load so both passes share one `ORBVocabulary`, or freeing
the forward `System` before constructing the reverse one — would roughly **double the safe worker
count** on top of the grayscale win. It's the largest remaining memory lever, but it means touching
ORB-SLAM3's lifetime management, so it's deliberately out of scope here.

### 3. Stale docstring / missing config path

`OrbSlam3Step`'s docstring says the default settings YAML is the bundled `ingest/config/`
template, but `ingest/config/gopro_hero12_slam.yaml` **does not exist** (that directory holds only
`gopro_intrinsics.json` and `gripper_calib.yaml`). `_DEFAULT_SETTINGS_YAML` actually resolves to
the copy under `external/ORB_SLAM3_PolyUMI/Examples/Monocular-Inertial/`. `CLAUDE.md` references
the same non-existent path.

---

## Out of scope

- Parallelizing **map building** (Phase 1): one MAPPING session, and it must precede localization.
- Parallelizing across *scenes* (`run_preprocessing_on_recordings`) — the same idea one level up;
  do the per-episode win first.
- Fixing the `Camera.RGB` mismatch (results-changing; needs its own validation).
