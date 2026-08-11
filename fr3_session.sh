#!/usr/bin/env bash
# Bring up the whole FR3 inference wall — NUC, Pi, GPU box, laptop — as one tmux session.
#
# Replaces the seven-terminal ssh-and-remember dance in docs/crb-fr3-inference.md with:
#
#     ./fr3_session.sh                # create (or re-attach to) the session
#     SKIP_DEPLOY=1 ./fr3_session.sh  # ...without re-syncing the NUC/Pi source trees first
#     ./fr3_session.sh --kill-local   # tear down the LOCAL session only (remote ones survive)
#     ./fr3_session.sh --kill         # --kill-local, plus stop the remote sessions too
#
# Every fresh start (not a re-attach) also makes each machine run this working copy — builds
# polyumi_ros2 here, rsyncs nuc/ to the NUC, and calls ./deploy.sh for the Pi — so nothing runs
# code you no longer have checked out. See the "Deploy" section below.
#
# WHERE TMUX RUNS, AND WHY IT MATTERS
# The NUC and GPU-box panes run tmux *on the remote host* (`ssh -t host tmux new -A -s ...`),
# not a bare ssh. Two reasons:
#   1. A laptop sleep or wifi blip then costs you nothing — the arm session and the policy
#      server keep running, and re-running this script re-attaches to them.
#   2. It fixes the env trap in docs/crb-fr3-inference.md: a non-interactive shell does not
#      source ~/.bashrc, so `ssh host 'cmd'` comes up without CYCLONEDDS_URI, on the wrong
#      RMW, invisible to everything else. `ssh -t host tmux` gets an interactive shell, so the
#      DDS env and the fr3-* aliases are simply there.
# The Pi is a plain ssh: it is stateless and cheap to restart, so the extra layer buys nothing.
#
# WHERE THE LOGS GO
# The three launches that can crash — NUC bringup, NUC inference, laptop policy client — tee their
# console output to ~/.local/state/polyumi/<name>_<date>.log on the machine that ran them
# ($XDG_STATE_HOME if set). ROS's own ~/.ros/log does not capture this: franka_bringup launches
# with output='screen', and a C++ crash message is raw stderr rather than rcl logging, so the line
# naming the fault reaches the terminal and nowhere else. Files older than REMOTE_LOG_KEEP_DAYS
# are pruned on each fresh start.
#
# WHAT AUTO-RUNS, AND WHAT ONLY GETS TYPED
# Commands that are safe and order-independent are run for you. Commands that need a decision
# (which checkpoint, whether the arm is allowed to move) are *typed into the prompt but not
# executed* — review the line, edit it, press Enter. Nothing in here can move the robot on its
# own. See PRETYPE vs RUN at each pane below.

set -euo pipefail

SESSION="${SESSION:-polyumi}"
# Seconds to let a remote shell finish starting before typing into it. Pre-typing races the
# ssh+tmux+shell startup; if lines land mangled or in the wrong pane, raise this.
SHELL_SETTLE_S="${SHELL_SETTLE_S:-5}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ssh aliases for the three remotes, and where the repo lives on each. All overridable together
# — a host and its repo path travel as a pair, so parameterizing one without the other would let
# you point NUC_REPO somewhere new while still ssh'ing to the old box.
#   NUC_SSH_HOST=otherfranka NUC_REPO=~/src/PolyUMI ./fr3_session.sh
# The repo paths are left unexpanded so the REMOTE shell resolves the tilde against its own
# $HOME — franka on the NUC, xhy7159 on the GPU box.
NUC_SSH_HOST="${NUC_SSH_HOST:-jailfranka}"
NUC_REPO="${NUC_REPO:-~/Documents/PolyUMI}"
SHEEP_SSH_HOST="${SHEEP_SSH_HOST:-sheep}"
SHEEP_REPO="${SHEEP_REPO:-~/repos/PolyUMI}"

# ssh destination for the Pi — the same POLYUMI_PI_HOST that `pingest fetch` and the catalog's
# Fetch button read, so one export in your shell rc covers all three. "polyumi-pi" is the alias
# other users are expected to set up in their own ssh config; override if yours is named
# differently (mine is "conorpi"):
#   POLYUMI_PI_HOST=conorpi ./fr3_session.sh
POLYUMI_PI_HOST="${POLYUMI_PI_HOST:-polyumi-pi}"

INFERENCE_URL="${INFERENCE_URL:-http://sheep.mech.northwestern.edu:8002/predict_cartesian/}"
# The Elgato's 1080p software convert runs ~200ms behind; the 50ms auto default drops every tick.
MAX_IMAGE_AGE_S="${MAX_IMAGE_AGE_S:-0.3}"

if [ "${1:-}" = "--kill-local" ] || [ "${1:-}" = "--kill" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed local session '$SESSION'."
  if [ "${1:-}" = "--kill-local" ]; then
    echo "NOTE: the remote tmux sessions on $NUC_SSH_HOST/$SHEEP_SSH_HOST are still running by design."
    echo "      To stop those too:  ./fr3_session.sh --kill"
    exit 0
  fi

  # --kill: also stop the specific remote sessions this script owns. kill-session, not
  # kill-server — the remote might be running other tmux sessions unrelated to us.
  kill_remote_session() {
    local host="$1" sess="$2"
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "tmux kill-session -t $sess" 2>/dev/null; then
      echo "Killed remote session '$sess' on $host."
    else
      echo "No remote session '$sess' on $host (already gone, or host unreachable)."
    fi
  }
  kill_remote_session "$NUC_SSH_HOST" fr3-bringup
  kill_remote_session "$NUC_SSH_HOST" fr3-inference
  kill_remote_session "$SHEEP_SSH_HOST" polyumi
  exit 0
fi

# Already up? Just re-attach — this is the normal path after a laptop sleep.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  exec tmux attach -t "$SESSION"
fi

# The Pi is on DHCP and its address really does change between sessions, so resolve it from the
# ssh config rather than baking in a literal that is wrong by next week. One source of truth.
#
# Checking this for emptiness would prove nothing: with no matching Host block, `ssh -G foo`
# still exits 0 and echoes back `hostname foo`, so a typo'd alias yields a non-empty PI_HOST
# that is just the typo — and it would then ride all the way into `pi_host:=` on the laptop's
# launch line. Actually connecting is the only check that distinguishes the two, and it catches
# a powered-off Pi at the same time. Non-fatal, matching how the deploy section below treats an
# unreachable machine: one box being down should not block bringing the others up.
PI_HOST="$(ssh -G "$POLYUMI_PI_HOST" 2>/dev/null | awk '/^hostname /{print $2}')" || true
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$POLYUMI_PI_HOST" true 2>/dev/null; then
  echo "Pi resolved to $PI_HOST (ssh alias: $POLYUMI_PI_HOST)"
else
  echo "WARNING: cannot reach the Pi at ssh alias '$POLYUMI_PI_HOST' (resolved: '${PI_HOST:-nothing}')." >&2
  echo "         Either the Pi is off, or the alias is not in your ssh config — in which case" >&2
  echo "         ssh hands back the alias verbatim and pi_host:= below will be wrong." >&2
  echo "         Set it with:  POLYUMI_PI_HOST=conorpi ./fr3_session.sh" >&2
  echo "         Continuing; the Pi panes will just fail to connect." >&2
fi

# ---------------------------------------------------------------------------
# Deploy: bring the NUC and Pi source trees in line with this working copy before anything
# runs against them — the fix for "I edited a launch file locally and the NUC ran the old one".
# SKIP_DEPLOY=1 bypasses both for a fast re-launch once you know they're already current.
# Non-fatal per target: a machine that's unreachable (Pi powered off, say) warns and is
# skipped rather than blocking the machines that ARE up. Sheep is deliberately not included —
# it tracks its own training branch, not this one, so force-syncing it would be wrong.
# ---------------------------------------------------------------------------
if [ "${SKIP_DEPLOY:-0}" = 1 ]; then
  echo "SKIP_DEPLOY=1 — leaving the laptop build and the NUC/Pi source trees as they are."
else
  # The laptop needs this as much as the remotes do, and it is easier to forget: colcon COPIES
  # sources into install/, so an edited node keeps running the old code until you rebuild, with
  # no error and no clue — a stale `eef_frame` default cost an on-arm debugging session once.
  # Only polyumi_ros2: it holds the nodes this session launches, and the two msgs packages are
  # slow to build and change roughly never.
  echo "==> Building polyumi_ros2 (this laptop) ..."
  if ! (unset VIRTUAL_ENV; cd "$REPO_DIR/ros2_ws" \
      && source /opt/ros/kilted/setup.bash && colcon build --packages-select polyumi_ros2); then
    echo "WARNING: colcon build failed — the laptop may run stale polyumi_ros2 code." >&2
  fi

  echo "==> Syncing nuc/ to $NUC_SSH_HOST:$NUC_REPO ..."
  if rsync -a --delete --mkpath --exclude='__pycache__/' --exclude='*.pyc' \
      nuc "${NUC_SSH_HOST}:${NUC_REPO}/"; then
    echo "    done."
  else
    echo "WARNING: rsync to $NUC_SSH_HOST failed — it may be running stale nuc/ code." >&2
  fi

  echo "==> Deploying pi/ to $POLYUMI_PI_HOST via ./deploy.sh ..."
  if ! (cd "$REPO_DIR" && ./deploy.sh "$POLYUMI_PI_HOST"); then
    echo "WARNING: deploy.sh failed (Pi unreachable?) — it may be running stale code." >&2
  fi
fi

# Is a remote session already running? Probed BEFORE any pane is opened, because attaching
# creates it. This is the re-attach case: the laptop went away but the NUC kept working.
remote_session_exists() {
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$1" "tmux has-session -t $2" 2>/dev/null
}

# A live remote session must NOT be typed into. Its shell may be mid-run (bringup), or holding
# a pre-typed line the operator has not pressed Enter on yet — and send-keys appends to that
# readline buffer rather than replacing it, so a second pass concatenates two commands and
# submits the result. Re-attaching is supposed to hand the pane back exactly as it was.
NUC_BRINGUP_FRESH=1; remote_session_exists "$NUC_SSH_HOST" fr3-bringup   && NUC_BRINGUP_FRESH=0
NUC_INFER_FRESH=1;   remote_session_exists "$NUC_SSH_HOST" fr3-inference && NUC_INFER_FRESH=0
SHEEP_FRESH=1;       remote_session_exists "$SHEEP_SSH_HOST" polyumi     && SHEEP_FRESH=0
if [ "$NUC_BRINGUP_FRESH$NUC_INFER_FRESH$SHEEP_FRESH" != "111" ]; then
  echo "Re-attaching to remote sessions that are already running; leaving those panes untouched."
fi

# Type a command into a pane and leave the cursor on it, unexecuted. The point is that the
# operator reads and confirms the line — this is how every robot-moving command gets in.
pretype() { tmux send-keys -t "$1" "$2"; }

# Wrap a launch so its console output also lands on disk. See "WHERE THE LOGS GO" above for why
# ~/.ros/log is not enough. Written to $XDG_STATE_HOME (~/.local/state) — the spec's home for
# state that persists across restarts but is not precious — so this stays out of $HOME and out of
# ROS's tree. Left unexpanded so the REMOTE shell resolves it against its own $HOME.
REMOTE_LOG_DIR='"${XDG_STATE_HOME:-$HOME/.local/state}"/polyumi'
#: Logs older than this are pruned on each fresh start. Scoped by -maxdepth 1 and a name glob to
#: the directory we create and the files we write, so it cannot reach anything else.
REMOTE_LOG_KEEP_DAYS="${REMOTE_LOG_KEEP_DAYS:-14}"

logged() {
  # $1 = short name for the log file, $2 = the command to run.
  #
  # `trap '' INT` in the tee subshell is load-bearing, not tidiness. Ctrl-C goes to the whole
  # foreground process group, so a bare `| tee` would kill tee first; ros2 launch then writes its
  # teardown into a closed pipe. Ignoring INT there lets tee outlive the launch and read until
  # stdout closes on its own — which also means the shutdown sequence, the part that says whether
  # bringup released the FCI cleanly, is the part that actually gets recorded.
  local dir="$REMOTE_LOG_DIR"
  printf 'mkdir -p %s && find %s -maxdepth 1 -name "%s_*.log" -mtime +%s -delete 2>/dev/null; %s 2>&1 | { trap "" INT; tee -a %s/%s_$(date +%%F).log; }' \
    "$dir" "$dir" "$1" "$REMOTE_LOG_KEEP_DAYS" "$2" "$dir" "$1"
}

# Build the command that opens a durable shell on a remote host.
#
# Prefers a tmux ON THE REMOTE, so the pane survives a laptop sleep or a dropped link — but
# degrades to a plain ssh when the host has no tmux or is not answering, rather than handing
# back a pane that opens and then silently does nothing. Degrading beats failing here: one
# machine being down should not block bringing the others up.
remote_shell() {
  local host="$1" session="$2"
  if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" 'command -v tmux >/dev/null' 2>/dev/null; then
    echo "ssh -t $host 'tmux new-session -A -s $session'"
  else
    echo "WARNING: $host has no remote tmux (not installed, or host unreachable)." >&2
    echo "         Using a plain ssh — this pane will NOT survive a disconnect." >&2
    echo "         Fix with:  ssh $host 'sudo apt install tmux'" >&2
    echo "ssh -t $host"
  fi
}

# Every pane below is addressed by the pane ID tmux hands back from -P -F (%0, %7, ...), never
# by a "window.index" string. Indices are not ours to predict: `pane-base-index 1` in the
# operator's ~/.tmux.conf — a common setting — makes every ".0" target here miss, and the script
# would then type robot commands into whatever pane it did find. IDs are assigned by tmux,
# unique for the life of the server, and immune to both that option and to later splits.
#
# Window IDs (@0, @1, ...) are captured alongside the pane IDs where a window gets used as an
# anchor: `new-window -t` and `select-window -t` take a target-WINDOW and reject a pane spec
# outright ("can't specify pane here"), so a pane ID cannot stand in for one.
#
# ---------------------------------------------------------------------------
# Window 1: the NUC — hardware session and inference stack, one pane each.
# Split because bringup is the piece that crashes mid-session and needs restarting on its own
# (docs/crb-fr3-inference.md, "TF lookup fails"), which is also why they are two launch files.
# ---------------------------------------------------------------------------
read -r NUC_BRINGUP_PANE NUC_WINDOW < <(
  tmux new-session -d -P -F '#{pane_id} #{window_id}' -s "$SESSION" -n nuc -c "$REPO_DIR")
tmux send-keys -t "$NUC_BRINGUP_PANE" "$(remote_shell "$NUC_SSH_HOST" fr3-bringup)" C-m
NUC_INFER_PANE="$(tmux split-window -t "$NUC_BRINGUP_PANE" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
tmux send-keys -t "$NUC_INFER_PANE" "$(remote_shell "$NUC_SSH_HOST" fr3-inference)" C-m

# ---------------------------------------------------------------------------
# Window 2: the Pi (camera/audio stream) and the GPU box (policy server).
# ---------------------------------------------------------------------------
# `-a -t "$NUC_WINDOW"` (a WINDOW target), not `-t "$SESSION"` (a session target). tmux resolves
# a bare session target to its CURRENT ACTIVE window and tries to create the new window there —
# and by this point that's the "nuc" window, still active since nothing has switched off it.
# Without -a that fails outright with "index N in use" the moment two windows exist; -a instead
# means "insert right after this window, shifting later ones if needed", which is what we
# actually want and cannot fail this way. Verified empirically: the plain `-t "$SESSION"` form
# intermittently threw exactly that error once >= 2 windows existed.
read -r PI_PANE PI_WINDOW < <(
  tmux new-window -a -t "$NUC_WINDOW" -n polyumi-pi -P -F '#{pane_id} #{window_id}' -c "$REPO_DIR")
tmux send-keys -t "$PI_PANE" "ssh -t $POLYUMI_PI_HOST" C-m
SHEEP_PANE="$(tmux split-window -t "$PI_PANE" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
tmux send-keys -t "$SHEEP_PANE" "$(remote_shell "$SHEEP_SSH_HOST" polyumi)" C-m

# ---------------------------------------------------------------------------
# Window 3: this laptop — DDS env sourced, ready to launch the client.
# ---------------------------------------------------------------------------
LAPTOP_PANE="$(tmux new-window -a -t "$PI_WINDOW" -n laptop -P -F '#{pane_id}' -c "$REPO_DIR")"
tmux send-keys -t "$LAPTOP_PANE" "source setup_franka_env.sh" C-m

# Everything above only opened shells. Let them finish before typing into them.
sleep "$SHELL_SETTLE_S"

# --- NUC, left pane: RUN bringup. Safe (no motion) and everything else waits on it. If FCI is
# --- not enabled on the Desk UI it fails loudly and you re-run it; that is cheap.
if [ "$NUC_BRINGUP_FRESH" = 1 ]; then
  tmux send-keys -t "$NUC_BRINGUP_PANE" \
    "cd $NUC_REPO && $(logged fr3_bringup 'ros2 launch nuc/launch/fr3_bringup.launch.py')" C-m
fi

# --- NUC, right pane: PRETYPE. Carries the execute flags, so it is yours to press Enter on.
if [ "$NUC_INFER_FRESH" = 1 ]; then
  tmux send-keys -t "$NUC_INFER_PANE" "cd $NUC_REPO" C-m
  pretype "$NUC_INFER_PANE" \
    "$(logged fr3_inference 'ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true execute_arm:=true max_velocity_scaling:=1.0')"
fi

# --- Pi: RUN the stream. Stateless, moves nothing, and the laptop warns without it.
tmux send-keys -t "$PI_PANE" "polyumi-pi stream" C-m

# --- Sheep: PRETYPE. The checkpoint changes every training run, so the path is yours to pick.
# --- The five most recent are listed above the prompt to save a hunt through dp_outputs/.
if [ "$SHEEP_FRESH" = 1 ]; then
  tmux send-keys -t "$SHEEP_PANE" "cd $SHEEP_REPO" C-m
  tmux send-keys -t "$SHEEP_PANE" \
    "ls -t data/dp_outputs/*/*/checkpoints/latest.ckpt 2>/dev/null | head -5" C-m
  pretype "$SHEEP_PANE" "CKPT=\$(ls -t $SHEEP_REPO/data/dp_outputs/*/*/checkpoints/latest.ckpt | head -1) ./serve_policy.sh"
fi

# --- Laptop: PRETYPE. Depends on every pane above being live, and there is no readiness gate
# --- here, so this is the one you press Enter on last.
pretype "$LAPTOP_PANE" \
  "$(logged policy_client "ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=$INFERENCE_URL execute_motion:=true max_image_age_s:=$MAX_IMAGE_AGE_S pi_host:=$PI_HOST")"

tmux select-window -t "$NUC_WINDOW"
cat <<EOF

Session '$SESSION' is up. Order to press Enter in:
  1. nuc, left      already running bringup (enable FCI on the Desk UI first if it errors)
  2. nuc, right     the inference stack — check the execute flags on the line before you run it
  3. polyumi-pi, right   the policy server — edit CKPT to the checkpoint you want
  4. laptop         the client, last

Send the arm home (needs pane 2 running; MOVES THE ARM even in plan-only mode):
  ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"

tmux, minimum viable:
  C-b n / C-b p    next / previous window        C-b o     next pane
  C-b d            detach (everything keeps running)
  C-b C-b          send a prefix to the INNER tmux on $NUC_SSH_HOST/$SHEEP_SSH_HOST
Re-attach any time with ./fr3_session.sh — remote panes pick up where they left off.

EOF
exec tmux attach -t "$SESSION"
