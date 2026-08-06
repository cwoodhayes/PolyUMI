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
# Every fresh start (not a re-attach) also pushes this working copy to the machines that run
# it — rsyncs nuc/ to the NUC, and calls ./deploy.sh for the Pi — so what runs remotely always
# matches what you have checked out here. See the "Deploy" section below.
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
# Where the repo lives on each remote (verified on both hosts). Left unexpanded so the REMOTE
# shell resolves the tilde against its own $HOME — franka on the NUC, xhy7159 on the GPU box.
# Override from the environment if these ever move.
NUC_REPO="${NUC_REPO:-~/Documents/PolyUMI}"
SHEEP_REPO="${SHEEP_REPO:-~/repos/PolyUMI}"

# ssh alias for the Pi. "polyumi-pi" is the name other users are expected to set up in their
# own ssh config; override if yours is named differently (mine is "conorpi"):
#   PI_SSH_HOST=conorpi ./fr3_session.sh
PI_SSH_HOST="${PI_SSH_HOST:-polyumi-pi}"

INFERENCE_URL="${INFERENCE_URL:-http://sheep.mech.northwestern.edu:8000/predict_cartesian/}"
# The Elgato's 1080p software convert runs ~200ms behind; the 50ms auto default drops every tick.
MAX_IMAGE_AGE_S="${MAX_IMAGE_AGE_S:-0.3}"

if [ "${1:-}" = "--kill-local" ] || [ "${1:-}" = "--kill" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed local session '$SESSION'."
  if [ "${1:-}" = "--kill-local" ]; then
    echo "NOTE: the remote tmux sessions on jailfranka/sheep are still running by design."
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
  kill_remote_session jailfranka fr3-bringup
  kill_remote_session jailfranka fr3-inference
  kill_remote_session sheep polyumi
  exit 0
fi

# Already up? Just re-attach — this is the normal path after a laptop sleep.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  exec tmux attach -t "$SESSION"
fi

# The Pi is on DHCP and its address really does change between sessions, so resolve it from the
# ssh config rather than baking in a literal that is wrong by next week. One source of truth.
PI_HOST="$(ssh -G "$PI_SSH_HOST" 2>/dev/null | awk '/^hostname /{print $2}')"
if [ -z "$PI_HOST" ]; then
  echo "ERROR: could not resolve '$PI_SSH_HOST' from your ssh config." >&2
  echo "       Set PI_SSH_HOST to your Pi's ssh alias, e.g.: PI_SSH_HOST=conorpi ./fr3_session.sh" >&2
  exit 1
fi
echo "Pi resolved to $PI_HOST (ssh alias: $PI_SSH_HOST)"

# ---------------------------------------------------------------------------
# Deploy: bring the NUC and Pi source trees in line with this working copy before anything
# runs against them — the fix for "I edited a launch file locally and the NUC ran the old one".
# SKIP_DEPLOY=1 bypasses both for a fast re-launch once you know they're already current.
# Non-fatal per target: a machine that's unreachable (Pi powered off, say) warns and is
# skipped rather than blocking the machines that ARE up. Sheep is deliberately not included —
# it tracks its own training branch, not this one, so force-syncing it would be wrong.
# ---------------------------------------------------------------------------
if [ "${SKIP_DEPLOY:-0}" = 1 ]; then
  echo "SKIP_DEPLOY=1 — leaving the NUC and Pi source trees as they are."
else
  echo "==> Syncing nuc/ to jailfranka:$NUC_REPO ..."
  if rsync -a --delete --mkpath --exclude='__pycache__/' --exclude='*.pyc' \
      nuc "jailfranka:${NUC_REPO}/"; then
    echo "    done."
  else
    echo "WARNING: rsync to jailfranka failed — it may be running stale nuc/ code." >&2
  fi

  echo "==> Deploying pi/ to $PI_SSH_HOST via ./deploy.sh ..."
  if ! (cd "$REPO_DIR" && ./deploy.sh "$PI_SSH_HOST"); then
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
NUC_BRINGUP_FRESH=1; remote_session_exists jailfranka fr3-bringup   && NUC_BRINGUP_FRESH=0
NUC_INFER_FRESH=1;   remote_session_exists jailfranka fr3-inference && NUC_INFER_FRESH=0
SHEEP_FRESH=1;       remote_session_exists sheep polyumi            && SHEEP_FRESH=0
if [ "$NUC_BRINGUP_FRESH$NUC_INFER_FRESH$SHEEP_FRESH" != "111" ]; then
  echo "Re-attaching to remote sessions that are already running; leaving those panes untouched."
fi

# Type a command into a pane and leave the cursor on it, unexecuted. The point is that the
# operator reads and confirms the line — this is how every robot-moving command gets in.
pretype() { tmux send-keys -t "$1" "$2"; }

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

# ---------------------------------------------------------------------------
# Window 1: the NUC — hardware session and inference stack, one pane each.
# Split because bringup is the piece that crashes mid-session and needs restarting on its own
# (docs/crb-fr3-inference.md, "TF lookup fails"), which is also why they are two launch files.
# ---------------------------------------------------------------------------
tmux new-session -d -s "$SESSION" -n nuc -c "$REPO_DIR"
tmux send-keys -t "$SESSION:nuc.0" "$(remote_shell jailfranka fr3-bringup)" C-m
tmux split-window -t "$SESSION:nuc" -h -c "$REPO_DIR"
tmux send-keys -t "$SESSION:nuc.1" "$(remote_shell jailfranka fr3-inference)" C-m

# ---------------------------------------------------------------------------
# Window 2: the Pi (camera/audio stream) and the GPU box (policy server).
# ---------------------------------------------------------------------------
# `-a -t "$SESSION:nuc"` (a WINDOW target, by name), not `-t "$SESSION"` (a session target).
# tmux resolves a bare session target to its CURRENT ACTIVE window and tries to create the new
# window there — and by this point that's window 0 ("nuc", still active since nothing has
# switched off it). Without -a that fails outright with "index 0 in use" the moment two windows
# exist; -a instead means "insert right after this window, shifting later ones if needed",
# which is what we actually want and cannot fail this way. Verified empirically: the plain
# `-t "$SESSION"` form intermittently threw exactly that error once >= 2 windows existed.
tmux new-window -a -t "$SESSION:nuc" -n polyumi-pi -c "$REPO_DIR"
tmux send-keys -t "$SESSION:polyumi-pi.0" "ssh -t $PI_SSH_HOST" C-m
tmux split-window -t "$SESSION:polyumi-pi" -h -c "$REPO_DIR"
tmux send-keys -t "$SESSION:polyumi-pi.1" "$(remote_shell sheep polyumi)" C-m

# ---------------------------------------------------------------------------
# Window 3: this laptop — DDS env sourced, ready to launch the client.
# ---------------------------------------------------------------------------
tmux new-window -a -t "$SESSION:polyumi-pi" -n laptop -c "$REPO_DIR"
tmux send-keys -t "$SESSION:laptop.0" "source setup_franka_env.sh" C-m

# Everything above only opened shells. Let them finish before typing into them.
sleep "$SHELL_SETTLE_S"

# --- NUC pane 0: RUN bringup. Safe (no motion) and everything else waits on it. If FCI is not
# --- enabled on the Desk UI it fails loudly and you re-run it; that is cheap.
if [ "$NUC_BRINGUP_FRESH" = 1 ]; then
  tmux send-keys -t "$SESSION:nuc.0" "cd $NUC_REPO && ros2 launch nuc/launch/fr3_bringup.launch.py" C-m
fi

# --- NUC pane 1: PRETYPE. Carries the execute flags, so it is yours to press Enter on. 
if [ "$NUC_INFER_FRESH" = 1 ]; then
  tmux send-keys -t "$SESSION:nuc.1" "cd $NUC_REPO" C-m
  pretype "$SESSION:nuc.1" "ros2 launch nuc/launch/fr3_inference.launch.py execute_gripper:=true execute_arm:=true max_velocity_scaling:=0.2"
fi

# --- Pi: RUN the stream. Stateless, moves nothing, and the laptop warns without it.
tmux send-keys -t "$SESSION:polyumi-pi.0" "polyumi-pi stream" C-m

# --- Sheep: PRETYPE. The checkpoint changes every training run, so the path is yours to pick.
# --- The five most recent are listed above the prompt to save a hunt through dp_outputs/.
if [ "$SHEEP_FRESH" = 1 ]; then
  tmux send-keys -t "$SESSION:polyumi-pi.1" "cd $SHEEP_REPO" C-m
  tmux send-keys -t "$SESSION:polyumi-pi.1" \
    "ls -t data/dp_outputs/*/*/checkpoints/latest.ckpt 2>/dev/null | head -5" C-m
  pretype "$SESSION:polyumi-pi.1" "CKPT=\$(ls -t $SHEEP_REPO/data/dp_outputs/*/*/checkpoints/latest.ckpt | head -1) ./serve_policy.sh"
fi

# --- Laptop: PRETYPE. Depends on every pane above being live, and there is no readiness gate
# --- here, so this is the one you press Enter on last.
pretype "$SESSION:laptop.0" "ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=$INFERENCE_URL execute_motion:=true max_image_age_s:=$MAX_IMAGE_AGE_S pi_host:=$PI_HOST"

tmux select-window -t "$SESSION:nuc"
cat <<EOF

Session '$SESSION' is up. Order to press Enter in:
  1. nuc.0     already running bringup (enable FCI on the Desk UI first if it errors)
  2. nuc.1     the inference stack — check the execute flags on the line before you run it
  3. polyumi-pi.1    the policy server — edit CKPT to the checkpoint you want
  4. laptop.0  the client, last

tmux, minimum viable:
  C-b n / C-b p    next / previous window        C-b o     next pane
  C-b d            detach (everything keeps running)
  C-b C-b          send a prefix to the INNER tmux on jailfranka/sheep
Re-attach any time with ./fr3_session.sh — remote panes pick up where they left off.

EOF
exec tmux attach -t "$SESSION"
