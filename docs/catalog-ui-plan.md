# PolyUMI Catalog UI — Plan

A localhost web app for browsing, associating, and exporting the sessions / scenes /
tasks / datasets produced by PolyUMI. This document scopes the work, fixes the data
model and on-disk formats, and lays out an implementation in phases.

## 1. Problem & goals

As data collection scales, the flat `~/recordings/` tree of `scene_*/session_*`
directories becomes unwieldy. There is no canonical list of tasks, no first-class
notion of a "dataset" (a training-ready combination of scenes), and the only way to
find anything is to grep `metadata.json` files.

**Primary goal:** while recording, quickly associate new sessions/scenes with a
**task**, and later assemble **datasets** from arbitrary combinations of scenes and
export them for training.

Concretely the app must:

1. Index the whole `~/recordings/` tree without re-parsing every `metadata.json` on
   each startup.
2. Browse the Task → Scene → Episode hierarchy with a detail panel (the 4-column
   Miller-columns layout).
3. Create/rename tasks and assign scenes (and MAPPING vs EPISODE sessions) to them.
4. Define a **Dataset** as a named selection of scenes/episodes, and export it to a
   UMI ReplayBuffer with recorded provenance.

## 2. Non-goals (for now)

- Not a data *visualizer* — trajectory/video inspection stays in Foxglove/Rerun.
- Not multi-user / networked — single user, localhost, no auth.
- Not a replacement for `pingest` — it *drives* the existing pipeline, it does not
  reimplement preprocessing or export.
- No cloud sync, no dataset versioning beyond a provenance manifest.

## 3. Data model

The sketch introduces two entities that do not exist today. Current state:

| Entity | Today | This plan |
| --- | --- | --- |
| **Session** | First-class. `metadata.json` per dir; `session_type` = MAPPING \| EPISODE. | Unchanged. Indexed read-only. |
| **Scene** | Directory + shared `scene_id`, reverse-derived from sessions. No scene-level file. | Gains a **`scene.json`** manifest (task assignment, notes). |
| **Task** | Free-text `task: str` on each session's metadata. | First-class entity. Canonical list lives in the catalog DB + is stamped onto `scene.json`. |
| **Dataset** | Does **not** exist. `export-dp` is one-scene-in → one-`.zarr.zip`-out. | First-class entity: a named set of scene/episode members + export config + a provenance manifest written beside the exported buffer. |

Relationships (matching the ER sketch):

- Task 1—N Scene (a scene belongs to at most one task).
- Scene 1—N Session (already true on disk).
- Dataset N—N Scene (a dataset combines scenes; a scene may appear in several
  datasets). Membership may be scene-level or drill down to individual episodes.
- Task 1—N Dataset (a dataset is scoped to one task).

### 3.1 Source-of-truth principle (the load-bearing decision)

**SQLite is a rebuildable cache/index. On-disk manifests are authoritative.**

Rationale: scenes get archived to standalone zips for at-rest storage
(`pingest archive-scene`). If task/dataset associations lived only in SQLite, they
would be silently lost on archive/restore or any DB rebuild. So:

- Session facts → already authoritative in `metadata.json` (read-only to the app).
- Scene→Task association + scene notes → authoritative in a new **`scene.json`** at
  the scene root.
- Dataset definition + provenance → authoritative in a **dataset manifest** written
  beside the exported `.zarr.zip`.
- The catalog DB can always be dropped and rebuilt by re-scanning disk.

### 3.2 New on-disk formats

**`scene_*/scene.json`** (new; scene currently has no manifest):

```json
{
  "scene_id": "3f9c...-uuid",
  "task": "fold_towel",
  "notes": "left-handed demos, cluttered background",
  "file_version": 1
}
```

`task` here is the canonical scene-level assignment. Session `metadata.json` keeps its
own `task` field; on conflict the catalog surfaces the mismatch (see §6). Writing
`scene.json` is the app's only mutation of the recordings tree.

**Dataset manifest** — written next to each exported buffer as
`<name>.dataset.json`:

```json
{
  "name": "fold_towel_v3",
  "task": "fold_towel",
  "created_at": "2026-07-26T12:00:00Z",
  "polyumi_version": "<git hash at export time>",
  "export_params": { "obs_down_sample_steps": null },
  "members": [
    { "scene_id": "3f9c...", "scene_dir": "scene_2026-07-20_.../", "episodes": "all" },
    { "scene_id": "a71b...", "scene_dir": "scene_2026-07-21_.../", "episodes": [0, 2, 3] }
  ],
  "output": "fold_towel_v3.zarr.zip",
  "n_episodes": 42,
  "file_version": 1
}
```

This answers "which scenes/episodes and which code version produced this training
buffer?" — the question that otherwise becomes unanswerable at scale.

## 4. SQLite schema (the cache)

Populated by a `sync` scan; never the sole home of any fact. SQLModel/SQLAlchemy over
`catalog.db` (default under `~/recordings/.catalog/catalog.db`, gitignored).

```sql
task(
  id INTEGER PK, name TEXT UNIQUE, description TEXT, created_at TEXT
)

scene(
  scene_id TEXT PK,                 -- uuid from metadata
  dir TEXT,                         -- absolute path
  task_id INTEGER FK -> task.id,    -- from scene.json
  notes TEXT,
  archived INTEGER,                 -- scene.zarr.zip present, scene.zarr absent
  created_at TEXT,
  synced_at TEXT
)

session(
  session_id TEXT PK,
  scene_id TEXT FK -> scene.scene_id,
  dir TEXT,
  session_type TEXT,                -- MAPPING | EPISODE
  task_meta TEXT,                   -- metadata.json's own task field (for conflict view)
  robot TEXT,
  duration_s REAL,
  n_video_frames INTEGER,
  video_dropped_frames INTEGER,
  created_at TEXT
)

dataset(
  id INTEGER PK, name TEXT UNIQUE, task_id INTEGER FK,
  manifest_path TEXT, output_path TEXT, n_episodes INTEGER,
  polyumi_version TEXT, created_at TEXT
)

dataset_member(
  dataset_id INTEGER FK, scene_id TEXT FK,
  episodes TEXT                     -- "all" or JSON list of episode indices
)
```

`task_id` on `scene` is a *cache* of what `scene.json` says; the writer updates both
`scene.json` (authoritative) and the row in one operation.

## 5. Architecture

**FastAPI + SQLite (SQLModel) + HTMX/Jinja**, shipped in-repo as a new package.

Why this stack:

- Fits the existing world: `inference_server/` already runs FastAPI/uvicorn; this is
  a `uv` workspace; the dataset export is a direct Python import of
  `export_scene_to_dp`, not a shell-out.
- HTMX (not React/SPA): the 4-column reactive browser is partial-swap-shaped —
  click task → server renders the scenes column, etc. No JS build step, no client
  state store, appropriate for a single-user localhost tool.
- In-repo + version controlled, one language, no extra Docker service.

Proposed layout (a new workspace member, mirroring `ingest/` and `inference_server/`):

```
catalog/
├── pyproject.toml            # fastapi, uvicorn, sqlmodel, jinja2, python-multipart
├── polyumi_catalog/
│   ├── main.py               # typer CLI: `polyumi-catalog serve|sync`
│   ├── db.py                 # engine, session, models
│   ├── models.py             # SQLModel tables (§4)
│   ├── sync.py               # scan recordings -> DB (§6)
│   ├── manifests.py          # read/write scene.json + dataset manifest (§3.2)
│   ├── export.py             # multi-scene dataset export (§7)
│   ├── app.py                # FastAPI app + routes
│   ├── templates/            # Jinja + HTMX partials
│   └── static/
└── test/
```

The dataset-export code is the one piece that reaches into `ingest/` — it depends on
`polyumi_ingest.export.dp`. Decide during Phase 3 whether `catalog` takes a dep on
`ingest` or the reusable export core moves into `ingest` and both import it (preferred:
keep export logic in `ingest`, `catalog` imports it).

## 6. Sync strategy

`polyumi-catalog sync` walks `~/recordings/`:

1. For each `scene_*` dir: read `scene.json` if present (else create a row with no
   task); enumerate `session_*` children, parse each `metadata.json`.
2. Upsert scene/session rows keyed by uuid. Use dir mtime / a stored `synced_at` to
   skip unchanged scenes for speed (this is the "don't re-parse everything on
   startup" requirement).
3. Detect **archived** scenes (`scene.zarr.zip` present, `scene.zarr` absent) and flag
   them rather than failing.
4. Surface **conflicts**: session `metadata.json.task` disagreeing with
   `scene.json.task` → shown in the detail panel, resolvable by writing `scene.json`.

Sync is idempotent and runs: on `serve` startup (fast, mtime-gated), on demand via a
"Rescan" button, and standalone from the CLI.

## 7. Dataset builder & export

The UI's Datasets column + detail panel lets you: name a dataset, pick a task,
multi-select member scenes (whole-scene membership; episode-level drill-down is a
later extension — see §10), then Export.

Export path — **extends** today's one-scene exporter:

- `export_scene_to_dp` ([ingest/.../export/dp/buffer.py](../ingest/polyumi_ingest/export/dp/buffer.py))
  builds one `.zarr.zip` from one scene's EPISODE sessions.
- A dataset spans multiple scenes, so add a multi-scene entry point that appends each
  member scene's selected episodes into a single ReplayBuffer (concatenating
  `episode_ends`). This is a modest refactor of the existing `_export_episode` loop,
  ideally living in `ingest` so the CLI can gain a `pingest export-dataset` too.
- On success: write the dataset manifest (§3.2) beside the buffer, stamp
  `polyumi_version`, and upsert the `dataset` + `dataset_member` rows.

## 8. UI

Matches the second sketch: a top row of four columns + a bottom detail panel.

- **Tasks** — list; select filters Scenes. Create / rename here.
- **Scenes** — scenes in the selected task(s); assign/reassign task via drag or a
  picker (writes `scene.json`).
- **Episodes** — sessions in the selected scene(s); MAPPING vs EPISODE badge,
  dropped-frame/quality indicators.
- **Datasets** — existing datasets + a "New dataset" builder that consumes the current
  scene/episode selection.
- **Detail panel** — full metadata for the focused row; task-conflict warnings; for
  datasets, its members + manifest path + export button/status.

HTMX: each column is a partial re-rendered on selection in the column to its left.

## 9. Phasing

Each phase is independently useful and testable.

- **Phase 0 — Catalog core.** Models (§4), `manifests.py` read/write, `sync.py`, and
  `polyumi-catalog sync`. No web UI. Deliverable: a populated `catalog.db` from a real
  recordings tree; unit tests for sync + manifest round-trips.
- **Phase 1 — Read-only browser.** FastAPI app + the 4-column view and detail panel
  over the synced DB. No mutations. Deliverable: browse tasks/scenes/episodes locally.
- **Phase 2 — Associations.** Create/rename tasks; assign scene→task writing
  `scene.json` (authoritative) + DB; surface metadata/scene task conflicts.
  Deliverable: retag a scene, drop the DB, re-sync, tag survives.
- **Phase 2.5 — Per-episode MCAP convenience.** In the Episodes column / session
  detail panel: an **Export to MCAP** action, and, once a `.mcap` exists, an **Open
  in Foxglove** action. Both reuse existing machinery rather than building anything
  new:
  - `ingest/polyumi_ingest/export/mcap.py::export_scene_to_mcap` already exports one
    pzarr episode to `<scene_dir>/episode_{i}.mcap` (exposed today as
    `pingest export-mcap`); `catalog` calls it directly (same import-not-reimplement
    pattern as the Phase 3 dataset export, §5/§10.2). It requires pzarr
    (`scene.zarr`) to already exist for that scene — the UI must say so (and not
    offer the button) if it doesn't.
  - A session's pzarr **episode index** isn't `session_id` — resolve it by matching
    the session's directory name against each `episode_i` group's
    `attrs['session_dir']` in the scene's `scene.zarr`.
  - "Open in Foxglove" shells out to the locally installed `foxglove-studio <path>`
    (confirmed installed at `/opt/Foxglove/foxglove-studio`; its own `.desktop` entry
    opens files the same way via `Exec=... %U`). This is a step beyond "browser +
    HTML" but is in-scope because the app is explicitly single-user/localhost with
    direct host access (§2 non-goals) — there is no networked/multi-user case where
    shelling out to a local GUI app would be wrong. A saved layout already exists at
    `ingest/foxglove/review_mcap.json`; whether to auto-apply it or leave loading it
    manual (as today) is a decision for the phase, not fixed by this plan.
  Deliverable: from the Episodes column, export an episode's MCAP and open it in
  Foxglove with one click each.
- **Phase 3 — Dataset builder + export.** Multi-scene export core in `ingest`;
  dataset manifest; the Datasets column + export action. Deliverable: build a dataset
  from ≥2 scenes, export a valid `.zarr.zip` + manifest, load it in the trainer.
- **Phase 4 — (optional) polish.** Frame thumbnails decoded on-demand from
  `gopro.mp4`, free-text search/filter, aggregate stats (episode counts, dropped
  frames) surfaced per task/scene.

## 10. Settled decisions

1. **Dataset membership is scene-level to start.** Phase 3 ships whole-scene members
   only; episode-level selection (dropping individual bad demos) is deferred to a
   later step. The schema keeps `dataset_member.episodes` (defaulting to `"all"`) so
   episode granularity is a non-breaking extension — no migration needed when added.
2. **`ingest` owns all preprocessing/export logic; `catalog` only imports it.** The
   multi-scene export core lives in `ingest` and is exposed both as a `catalog`
   import and as a `pingest export-dataset` CLI peer. `catalog` is strictly a UI +
   metadata manager and never reimplements pipeline functionality.
3. **Task assignment is scene-level, and `scene.json` is canonical.** The scene's task
   lives in `scene.json`; per-session `metadata.json.task` is treated as a
   historical/collection-time value and reconciled (not authoritative) in the UI.
```

