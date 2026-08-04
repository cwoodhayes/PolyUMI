#!/usr/bin/env bash
#
# One-off: re-run the full preprocessing pipeline with --force on every scene
# whose scene.json records a given task.
#
# This is the pzarr v3 -> v4 migration run. `pingest pp --force` re-runs every
# registered step, which is what earns the v4 restamp -- a partial run leaves the
# store stamped v3 on purpose, since skipped steps would still hold v3-shaped
# output.
#
# Sequential by design: step 2 (ORB-SLAM3) saturates the CPU, so running scenes
# in parallel would just make each slower and interleave the logs.
#
# Usage:
#   ./rerun_pp_task.sh              # show the plan, ask before starting
#   ./rerun_pp_task.sh -y           # skip the confirmation
#   ./rerun_pp_task.sh -n           # dry run: print what would happen, touch nothing
#   TASK='hand open close' ./rerun_pp_task.sh
#
set -uo pipefail

TASK="${TASK:-red trapezoid in black mug}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDINGS="${RECORDINGS:-$REPO_ROOT/recordings}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/pp_rerun_logs/$(date +%Y-%m-%d_%H-%M-%S)}"

ASSUME_YES=0
DRY_RUN=0
while getopts 'ynh' opt; do
  case "$opt" in
    y) ASSUME_YES=1 ;;
    n) DRY_RUN=1 ;;
    h) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "try -h" >&2; exit 2 ;;
  esac
done

# uv picks the workspace venv itself; an inherited VIRTUAL_ENV pointing at pi/.venv
# makes it try to rebuild lgpio, which needs swig and is Pi-only. See CLAUDE.md.
unset VIRTUAL_ENV

# --- select scenes -----------------------------------------------------------
# scene.json's `task` is the only place the task is recorded -- session
# metadata.json carries None for every scene here -- so a scene without that file
# simply has no task and is correctly skipped.
mapfile -t SCENES < <(
  python3 - "$RECORDINGS" "$TASK" <<'PY'
import json, pathlib, sys
recordings, task = pathlib.Path(sys.argv[1]), sys.argv[2]
for scene_json in sorted(recordings.glob('scene_*/scene.json')):
    try:
        if json.loads(scene_json.read_text()).get('task') == task:
            print(scene_json.parent)
    except (OSError, ValueError) as err:
        print(f'skipping unreadable {scene_json}: {err}', file=sys.stderr)
PY
)

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "No scenes under $RECORDINGS have task '$TASK'." >&2
  exit 1
fi

echo "Task:       $TASK"
echo "Recordings: $RECORDINGS"
echo "Logs:       $LOG_DIR"
echo "Scenes:     ${#SCENES[@]}"
total_sessions=0
for scene in "${SCENES[@]}"; do
  n=$(find "$scene" -maxdepth 1 -type d -name 'session_*' | wc -l)
  total_sessions=$((total_sessions + n))
  ver=$(python3 -c "
import json,sys
try: print(json.load(open(sys.argv[1])).get('pzarr_version','?'))
except Exception: print('?')" "$scene/scene.zarr/.zattrs" 2>/dev/null)
  printf '  %-40s %3s sessions  pzarr_v=%s\n' "$(basename "$scene")" "$n" "$ver"
done
echo "            $total_sessions sessions total -- SLAM runs once per session, so expect hours."

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "Dry run; nothing was touched. Command per scene would be:"
  echo "  uv run pingest pp --force --scene <scene>"
  exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  echo
  read -r -p "Re-run the full pipeline on these ${#SCENES[@]} scenes? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

mkdir -p "$LOG_DIR"

# --- run ---------------------------------------------------------------------
# One scene failing does not stop the rest: the point of a batch is to get through
# it and read the summary, not to lose four hours of finished work to scene nine.
declare -a FAILED=()
run_started=$SECONDS

for i in "${!SCENES[@]}"; do
  scene="${SCENES[$i]}"
  name="$(basename "$scene")"
  log="$LOG_DIR/$name.log"
  echo
  echo "=== [$((i + 1))/${#SCENES[@]}] $name -- started $(date +%H:%M:%S) ==="
  echo "    log: $log"
  scene_started=$SECONDS

  if (cd "$REPO_ROOT" && uv run pingest pp --force --scene "$scene") >"$log" 2>&1; then
    printf '    ok (%dm%02ds)\n' $(((SECONDS - scene_started) / 60)) $(((SECONDS - scene_started) % 60))
  else
    rc=$?
    printf '    FAILED rc=%d (%dm%02ds)\n' "$rc" $(((SECONDS - scene_started) / 60)) $(((SECONDS - scene_started) % 60))
    echo "    last lines:"
    tail -n 15 "$log" | sed 's/^/      /'
    FAILED+=("$name")
  fi
done

# --- summary -----------------------------------------------------------------
echo
printf '=== done in %dh%02dm ===\n' $(((SECONDS - run_started) / 3600)) $((((SECONDS - run_started) % 3600) / 60))
echo "Succeeded: $((${#SCENES[@]} - ${#FAILED[@]}))/${#SCENES[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed:"
  for name in "${FAILED[@]}"; do echo "  $name  ($LOG_DIR/$name.log)"; done
fi

# A scene that "succeeded" can still have dropped individual episodes -- the step
# harness flags a bad episode unusable and carries on rather than failing the scene.
# Those are the lines worth reading before trusting the corpus.
echo
echo "Episodes flagged unusable during the run (scene-level success hides these):"
flagged=$(grep -h "flagged unusable\|marked unusable\|continuing with the rest" "$LOG_DIR"/*.log 2>/dev/null)
if [ -n "$flagged" ]; then
  echo "$flagged" | sed 's/^/  /'
else
  echo "  (none)"
fi

echo
echo "Post-run pzarr versions:"
for scene in "${SCENES[@]}"; do
  ver=$(python3 -c "
import json,sys
try: print(json.load(open(sys.argv[1])).get('pzarr_version','?'))
except Exception: print('?')" "$scene/scene.zarr/.zattrs" 2>/dev/null)
  printf '  %-40s pzarr_v=%s\n' "$(basename "$scene")" "$ver"
done

[ "${#FAILED[@]}" -eq 0 ]
