#!/usr/bin/env bash
# Bring up the whole FR3 inference wall — NUC, Pi, lamb — as one tmux session.
#
#     ./fr3_session.sh                # create (or re-attach to) the session
#     SKIP_DEPLOY=1 ./fr3_session.sh  # ...without re-syncing the remote source trees first
#     ./fr3_session.sh --kill-local   # tear down the LOCAL session only (remote ones survive)
#     ./fr3_session.sh --kill         # ...and stop the remote sessions too
#
# This machine is only a terminal. ROS and the policy server both run on lamb, so the inference
# request never leaves that box and the laptop can sleep mid-run without consequence.
#
# WHAT AUTO-RUNS, AND WHAT ONLY GETS TYPED
# Commands that are safe and order-independent are RUN for you. Commands that need a decision
# (which checkpoint, whether the arm is allowed to move) are PRE-TYPED at the prompt but not
# executed — read the line, edit it, press Enter. Nothing here can move the robot on its own.
#
# WHERE TMUX RUNS, AND WHY IT MATTERS
# The remote panes run tmux *on the remote host* (`ssh -t host tmux new -A -s ...`), not a bare
# ssh, so a laptop sleep or a wifi blip costs nothing — re-running this script re-attaches. It
# also gets an interactive shell, so ~/.bashrc's DDS env and fr3-* aliases are simply there;
# `ssh host 'cmd'` comes up without them, on the wrong RMW, invisible to everything else.
# The Pi is a plain ssh: it is stateless and cheap to restart, so the extra layer buys nothing.
#
# WHERE THE LOGS GO
# The launches that can crash tee their console output to ~/.local/state/polyumi/<name>_<date>.log
# on the machine that ran them. ROS's own ~/.ros/log does not capture this: franka_bringup
# launches with output='screen', and a C++ crash message is raw stderr rather than rcl logging,
# so the line naming the fault reaches the terminal and nowhere else.
#
# Every fresh start (not a re-attach) also makes each machine run this working copy. See "Deploy".

set -euo pipefail

ACTION="${1:-}"
SESSION="${SESSION:-polyumi}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Seconds to let a remote shell finish starting before typing into it. Pre-typing races the
# ssh+tmux+shell startup; if lines land mangled or in the wrong pane, raise this.
SHELL_SETTLE_S="${SHELL_SETTLE_S:-5}"

# A host and the repo path on it travel as a pair, so they are overridden as a pair — pointing
# NUC_REPO somewhere new while still ssh'ing to the old box would be worse than not allowing it.
# Repo paths are left unexpanded so the REMOTE shell resolves the tilde against its own $HOME.
NUC_SSH_HOST="${NUC_SSH_HOST:-jailfranka}"
NUC_REPO="${NUC_REPO:-~/Documents/PolyUMI}"
# The NUC's franka_ros2 workspace. ~/franka_ws/src/franka_streaming_impedance_controller is a
# symlink into $NUC_REPO/external (refreshed on every deploy below), so a build there picks up
# whatever the rsync landed.
NUC_FRANKA_WS="${NUC_FRANKA_WS:-~/franka_ws}"

# lamb runs both the ROS client and the policy server. One host, one checkout, one sync.
ROS_SSH_HOST="${ROS_SSH_HOST:-lamb}"
ROS_REPO="${ROS_REPO:-~/repos/PolyUMI}"
# The Elgato's device node is lamb's USB enumeration, not this laptop's (which is /dev/video2,
# the launch file's default). Confirm with `v4l2-ctl --list-devices` if a capture pane stalls.
ROS_VIDEO_DEVICE="${ROS_VIDEO_DEVICE:-/dev/video0}"

# The same POLYUMI_PI_HOST that `pingest fetch` and the catalog's Fetch button read, down to this
# default, so one export in your shell rc covers all three.
POLYUMI_PI_HOST="${POLYUMI_PI_HOST:-polyumi-pi}"

# Client and server are the same machine now, so this is a loopback hop rather than a trip over
# the direct link. Port matches serve_policy.sh's default.
INFERENCE_URL="${INFERENCE_URL:-http://localhost:8002/predict_cartesian/}"
# The Elgato's 1080p software convert runs ~200ms behind; the 50ms auto default drops every tick.
MAX_IMAGE_AGE_S="${MAX_IMAGE_AGE_S:-0.3}"
# Defaults true because the line is only PRE-TYPED, never run, and the NUC has its own
# execute_arm/execute_gripper flags (both default false) in front of the arm. Set false for a
# dry run: the preview topics still show every commanded chunk, but nothing is published.
EXECUTE_MOTION="${EXECUTE_MOTION:-true}"
# Which gripper driver the NUC starts: `faulhaber` (franka_gripper_control over CANopen; needs
# can0 up and a completed calibration) or `hand` (a stock Franka Hand over libfranka, kept working
# but not what we run). Only rides the PRE-TYPED nuc-inference line, so it still takes an Enter
# before anything moves.
GRIPPER="${GRIPPER:-faulhaber}"

# Every remote tmux session this script owns. Also spelled out in the pane table below; kept
# separate because --kill must work without first resolving the Pi or building the table.
REMOTE_SESSIONS=(
  "$NUC_SSH_HOST fr3-bringup"
  "$NUC_SSH_HOST fr3-inference"
  "$ROS_SSH_HOST polyumi"
  "$ROS_SSH_HOST polyumi-ros"
)

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
if [ "$ACTION" = "--kill-local" ] || [ "$ACTION" = "--kill" ]; then
  # Interrupt everything, wait once, then kill. Killing a pane delivers SIGHUP, which none of
  # these clean up on: the Pi leaves the finger LED lit (its `finally:` only runs via
  # KeyboardInterrupt), the GPU box's `docker run` CLI dies without forwarding it so the
  # container and its port survive, and ros2 launch skips the shutdown that reports whether the
  # FCI was released. 6s covers the Pi's worst case: stream() stops both child streamers
  # (2s SIGTERM + 2s SIGKILL join each) before it touches the LED.
  KILL_GRACE_S="${KILL_GRACE_S:-6}"
  NEEDS_GRACE=0

  # The Pi on BOTH paths: that pane is a plain `ssh -t` with no remote tmux to survive into, so
  # it dies either way — this only decides whether it dies cleanly. ^C into the pane rather than
  # `pkill` over ssh, because `polyumi-pi` is a zsh alias and no remote cmdline matches it.
  PI_PANE="$(tmux list-panes -t "$SESSION:polyumi-pi" -F '#{pane_id}' 2>/dev/null | head -1 || true)"
  if [ -n "$PI_PANE" ]; then
    tmux send-keys -t "$PI_PANE" C-c
    echo "Sent C-c to the Pi pane (so 'polyumi-pi stream' turns the LED off on its way out)."
    NEEDS_GRACE=1
  fi

  if [ "$ACTION" = "--kill" ]; then
    # ^C every pane of the session, which spares a pre-typed line and anything started by hand.
    # has-session gates it so the exit status means "interrupted something".
    for hs in "${REMOTE_SESSIONS[@]}"; do
      read -r host sess <<<"$hs"
      if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" \
          "tmux has-session -t $sess 2>/dev/null && tmux list-panes -s -t $sess -F '#{pane_id}' \
           | xargs -r -I{} tmux send-keys -t {} C-c" 2>/dev/null; then
        NEEDS_GRACE=1
      fi
    done
    [ "$NEEDS_GRACE" = 1 ] && echo "Sent C-c to the remote sessions." || true
  fi

  if [ "$NEEDS_GRACE" = 1 ]; then
    echo "Waiting ${KILL_GRACE_S}s for shutdown (KILL_GRACE_S to change) ..."
    sleep "$KILL_GRACE_S"
  fi

  tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed local session '$SESSION'." || true

  if [ "$ACTION" = "--kill-local" ]; then
    echo "NOTE: the remote tmux sessions on $NUC_SSH_HOST/$ROS_SSH_HOST are still running by design."
    echo "      To stop those too:  ./fr3_session.sh --kill"
    exit 0
  fi

  # kill-session, not kill-server — the remotes may run other tmux sessions unrelated to us.
  for hs in "${REMOTE_SESSIONS[@]}"; do
    read -r host sess <<<"$hs"
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "tmux kill-session -t $sess" 2>/dev/null; then
      echo "Killed remote session '$sess' on $host."
    else
      echo "No remote session '$sess' on $host (already gone, or host unreachable)."
    fi
  done
  exit 0
fi

# Already up? Just re-attach — the normal path after a laptop sleep.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' exists; attaching."
  exec tmux attach -t "$SESSION"
fi

# ---------------------------------------------------------------------------
# Resolve the Pi
# ---------------------------------------------------------------------------
# The Pi is on DHCP and its address really does change between sessions, so resolve it from the
# ssh config rather than baking in a literal that is wrong by next week.
#
# Checking this for emptiness would prove nothing: with no matching Host block, `ssh -G foo`
# still exits 0 and echoes back `hostname foo`, so a typo'd alias yields a non-empty PI_HOST that
# is just the typo — and it would ride all the way into `pi_host:=` on the launch line. Actually
# connecting is the only check that distinguishes the two, and it catches a powered-off Pi too.
# Non-fatal: one box being down should not block bringing the others up.
PI_HOST="$(ssh -G "$POLYUMI_PI_HOST" 2>/dev/null | awk '/^hostname /{print $2}')" || true
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$POLYUMI_PI_HOST" true 2>/dev/null; then
  PI_REACHABLE=1
  echo "Pi resolved to $PI_HOST (ssh alias: $POLYUMI_PI_HOST)"
else
  PI_REACHABLE=0
  echo "WARNING: cannot reach the Pi at ssh alias '$POLYUMI_PI_HOST' (resolved: '${PI_HOST:-nothing}')." >&2
  echo "         Either the Pi is off, or the alias is not in your ssh config — in which case" >&2
  echo "         ssh hands back the alias verbatim and pi_host:= below will be wrong." >&2
  echo "         Set it with:  POLYUMI_PI_HOST=conorpi ./fr3_session.sh" >&2
  echo "         Continuing; the Pi panes will just fail to connect." >&2
fi

# ---------------------------------------------------------------------------
# Deploy: bring the remote source trees in line with this working copy before anything runs
# against them — the fix for "I edited a launch file locally and the NUC ran the old one".
# SKIP_DEPLOY=1 bypasses it for a fast re-launch. Non-fatal per target: a machine that is
# unreachable warns and is skipped rather than blocking the machines that are up.
# ---------------------------------------------------------------------------
if [ "${SKIP_DEPLOY:-0}" = 1 ]; then
  echo "SKIP_DEPLOY=1 — leaving the remote source trees as they are."
else
  # The two submodules the NUC builds. Both are transferred with --delete, so an unchecked-out
  # one would wipe the NUC's copy instead of updating it. Check before transferring, not after.
  NUC_SUBMODULES=(external/franka_gripper_control external/franka_streaming_impedance_controller)
  for sub in "${NUC_SUBMODULES[@]}"; do
    if [ -z "$(ls -A "$REPO_DIR/$sub" 2>/dev/null)" ]; then
      echo "ERROR: $sub is empty — the submodule is not checked out." >&2
      echo "       Syncing it would DELETE the NUC's copy. Run:" >&2
      echo "         git submodule update --init $sub" >&2
      exit 1
    fi
  done

  echo "==> Syncing nuc/ + the NUC's submodules to $NUC_SSH_HOST:$NUC_REPO ..."
  # -R (--relative) so each source keeps its path under $NUC_REPO — external/<name> must land at
  # external/<name>, not at the repo root. --delete stays scoped to the transferred directories;
  # the implied external/ is created, never scanned, so the other submodules (which the NUC does
  # not have and does not need) are safe.
  if (cd "$REPO_DIR" && rsync -aR --delete --mkpath --exclude='__pycache__/' --exclude='*.pyc' \
      --exclude='.git/' nuc "${NUC_SUBMODULES[@]}" "${NUC_SSH_HOST}:${NUC_REPO}/"); then
    # fr3_home_service runs straight from the synced tree, but the controller is C++ and
    # franka_gripper_control is an installed ament_python package: rsync only updates the sources
    # that ~/franka_ws/src symlinks at, so without this the NUC keeps running the previously built
    # artifacts — a torque controller, the Franka Hand driver, and the FAULHABER driver.
    # Sourcing is explicit because `ssh host 'cmd'` gets no ~/.bashrc.
    echo "==> Rebuilding franka_streaming_impedance_controller + franka_gripper_control on $NUC_SSH_HOST ..."
    if ssh -o ConnectTimeout=10 "$NUC_SSH_HOST" \
        "ln -sfn $NUC_REPO/external/franka_gripper_control \
             $NUC_FRANKA_WS/src/franka_gripper_control \
         && ln -sfn $NUC_REPO/external/franka_streaming_impedance_controller/franka_streaming_impedance_controller \
             $NUC_FRANKA_WS/src/franka_streaming_impedance_controller \
         && source /opt/ros/humble/setup.bash \
         && source $NUC_FRANKA_WS/install/setup.bash \
         && cd $NUC_FRANKA_WS \
         && colcon build --packages-select franka_streaming_impedance_controller franka_gripper_control \
              --cmake-args -DCMAKE_BUILD_TYPE=Release"; then
      echo "    done. NOTE: the impedance controller is a pluginlib .so that controller_manager"
      echo "    keeps mapped, so it runs the OLD build until fr3_bringup restarts; the gripper"
      echo "    driver needs fr3_inference restarted. Re-attaching to either live session picks"
      echo "    up neither."
    else
      echo "WARNING: colcon build on $NUC_SSH_HOST failed — it may run a stale impedance" >&2
      echo "         controller or a stale gripper driver." >&2
    fi
  else
    echo "WARNING: rsync to $NUC_SSH_HOST failed — it may be running stale nuc/ code." >&2
  fi

  if [ "$PI_REACHABLE" = 1 ]; then
    echo "==> Deploying pi/ to $POLYUMI_PI_HOST via ./deploy.sh ..."
    if ! (cd "$REPO_DIR" && ./deploy.sh "$POLYUMI_PI_HOST"); then
      echo "WARNING: deploy.sh failed — it may be running stale code." >&2
    fi
  else
    echo "==> Skipping Pi deploy — already confirmed unreachable above." >&2
  fi

  # Whole tree, both builds, one script — also runnable by hand for a training-only push.
  echo "==> Deploying to $ROS_SSH_HOST via ./deploy_lamb.sh ..."
  if ! (cd "$REPO_DIR" && ./deploy_lamb.sh "$ROS_SSH_HOST" "$ROS_REPO"); then
    echo "WARNING: deploy_lamb.sh failed — $ROS_SSH_HOST may run stale code, or fail to import" >&2
    echo "         polyumi_inference in policy_client_node." >&2
  fi
fi

# ---------------------------------------------------------------------------
# The pane table
# ---------------------------------------------------------------------------
# Wrap a launch so its console output also lands on disk (see "WHERE THE LOGS GO"). Written to
# $XDG_STATE_HOME (~/.local/state), left unexpanded so the REMOTE shell resolves it.
REMOTE_LOG_DIR='"${XDG_STATE_HOME:-$HOME/.local/state}"/polyumi'
REMOTE_LOG_KEEP_DAYS="${REMOTE_LOG_KEEP_DAYS:-14}"

logged() {
  # $1 = short name for the log file, $2 = the command to run.
  #
  # `trap '' INT` in the tee subshell is load-bearing. Ctrl-C goes to the whole foreground
  # process group, so a bare `| tee` would kill tee first and ros2 launch would then write its
  # teardown into a closed pipe. Ignoring INT there lets tee read until stdout closes on its own,
  # which is how the shutdown sequence — the part that says whether bringup released the FCI —
  # ends up in the file. The find is scoped by -maxdepth 1 and a name glob to files we write.
  #
  # Every step is &&-chained, back into whatever the caller prefixed (a `cd`, a `source`): that
  # prefix is a precondition, and a launch that runs without it fails deep in rmw, naming nothing.
  # Log rotation is the one step allowed to fail — a full disk should not stop a run — so its
  # exit status is swallowed rather than breaking the chain.
  printf 'mkdir -p %s && { find %s -maxdepth 1 -name "%s_*.log" -mtime +%s -delete 2>/dev/null || true; } && %s 2>&1 | { trap "" INT; tee -a %s/%s_$(date +%%F).log; }' \
    "$REMOTE_LOG_DIR" "$REMOTE_LOG_DIR" "$1" "$REMOTE_LOG_KEEP_DAYS" "$2" "$REMOTE_LOG_DIR" "$1"
}

P_NAME=(); P_HOST=(); P_SESS=(); P_LAYOUT=(); P_RUN=(); P_PRE=()
add_pane() {
  # name  ssh_host  remote_tmux_session ("" = plain ssh)  layout(window|split)  run  pretype
  P_NAME+=("$1"); P_HOST+=("$2"); P_SESS+=("$3"); P_LAYOUT+=("$4"); P_RUN+=("$5"); P_PRE+=("$6")
}

# The NUC's two launch files get a pane each. Bringup is the crash-prone, FCI-gated piece and
# must be restartable on its own (docs/crb-fr3-inference.md, "When it doesn't come up").
# RUN: safe, moves nothing, and everything else waits on it. If FCI is not enabled on the Desk
# UI it fails loudly and you re-run it; that is cheap.
add_pane nuc "$NUC_SSH_HOST" fr3-bringup window \
  "cd $NUC_REPO && $(logged fr3_bringup 'ros2 launch nuc/launch/fr3_bringup.launch.py')" ""

# PRETYPE: carries the execute flags, so it is yours to press Enter on.
add_pane nuc-inference "$NUC_SSH_HOST" fr3-inference split "" \
  "cd $NUC_REPO && $(logged fr3_inference "ros2 launch nuc/launch/fr3_inference.launch.py gripper:=$GRIPPER execute_gripper:=true execute_arm:=true")"

# RUN: stateless, moves nothing, and the client warns without it.
#
# Stop polyumi-pi.service first. It runs `start-scene` on boot, and start-scene and stream both
# construct an LEDManager on the same hardware PWM channel; whoever constructs one last wins,
# because HardwarePWM.start(0) zeroes the duty cycle, and the loser goes on believing its LED is
# lit. Restart=on-failure means the service retries all through a stream that holds the camera.
# Nothing in the stream path can defend against this — the other process owns the pin just as
# legitimately. `;` not `&&`, so a Pi without the unit still gets a stream.
add_pane polyumi-pi "$POLYUMI_PI_HOST" "" window \
  "sudo systemctl stop polyumi-pi; polyumi-pi stream" ""

# PRETYPE: the checkpoint changes every training run, so the path is yours to pick. The five most
# recent are listed above the prompt to save a hunt through dp_outputs/.
add_pane policy-server "$ROS_SSH_HOST" polyumi split \
  "cd $ROS_REPO && ls -t data/dp_outputs/*/*/checkpoints/latest.ckpt 2>/dev/null | head -5" \
  "CKPT=\$(ls -t $ROS_REPO/data/dp_outputs/*/*/checkpoints/latest.ckpt | head -1) ./serve_policy.sh"

# PRETYPE: depends on every pane above being live, and there is no readiness gate, so this is the
# one you press Enter on last. It sources setup_franka_env.sh even though the shell already has
# the env, because this is the line that lands in shell history — every later recall of it then
# carries its own DDS env instead of inheriting whatever the shell happened to have. An
# interactive rc exporting its own ROS_DOMAIN_ID silently beats tmux's inherited environment, and
# the only symptom is policy_client_node never seeing fr3_link0.
# Its two lines of output are left visible: they are the only evidence of which DDS config this
# pane got, and a config path missing on this host otherwise surfaces as every node aborting
# inside rmw with nothing naming the file.
add_pane ros-client "$ROS_SSH_HOST" polyumi-ros window "" \
  "cd $ROS_REPO && source setup_franka_env.sh && $(logged policy_client "ros2 launch polyumi_ros2 inference_demo.launch.xml inference_server_url:=$INFERENCE_URL execute_motion:=$EXECUTE_MOTION max_image_age_s:=$MAX_IMAGE_AGE_S pi_host:=$PI_HOST video_device:=$ROS_VIDEO_DEVICE")"

# ---------------------------------------------------------------------------
# Probe, open, type
# ---------------------------------------------------------------------------
# One ssh per remote pane answering both questions at once. Does the host have tmux — if not we
# degrade to a plain ssh rather than hand back a pane that opens and then silently does nothing,
# because one machine being down should not block the others. And is the session already live —
# a live session must NOT be typed into: its shell may be mid-run, or holding a pre-typed line
# nobody has pressed Enter on yet, and send-keys APPENDS to that readline buffer rather than
# replacing it, so a second pass would concatenate two commands and submit the result.
P_FRESH=(); P_CMD=(); P_PANE=()
for i in "${!P_NAME[@]}"; do
  P_FRESH[i]=1
  P_CMD[i]="ssh -t ${P_HOST[i]}"
  if [ -n "${P_SESS[i]}" ]; then
    probe="$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${P_HOST[i]}" \
      "command -v tmux >/dev/null && echo has-tmux; tmux has-session -t ${P_SESS[i]} 2>/dev/null && echo live" \
      2>/dev/null || true)"
    case "$probe" in
      *has-tmux*) P_CMD[i]="ssh -t ${P_HOST[i]} 'tmux new-session -A -s ${P_SESS[i]}'" ;;
      *) echo "WARNING: ${P_HOST[i]} has no remote tmux (not installed, or host unreachable)." >&2
         echo "         Pane '${P_NAME[i]}' will use a plain ssh and will NOT survive a disconnect." >&2
         echo "         Fix with:  ssh ${P_HOST[i]} 'sudo apt install tmux'" >&2 ;;
    esac
    case "$probe" in *live*) P_FRESH[i]=0; echo "Re-attaching to '${P_SESS[i]}' on ${P_HOST[i]}; leaving that pane untouched." ;; esac
  fi
done

# Panes are addressed by the ID tmux hands back from -P -F (%0, %7, ...), never by a
# "window.index" string. Indices are not ours to predict: `pane-base-index 1` in the operator's
# ~/.tmux.conf — a common setting — makes every ".0" target miss, and the script would then type
# robot commands into whatever pane it did find. Window IDs (@0, @1, ...) are captured too where
# a window is used as an anchor: new-window/select-window take a target-WINDOW and reject a pane
# spec outright.
LAST_WINDOW=""; LAST_PANE=""
for i in "${!P_NAME[@]}"; do
  if [ "${P_LAYOUT[i]}" = window ] && [ -z "$LAST_WINDOW" ]; then
    read -r pane win < <(tmux new-session -d -P -F '#{pane_id} #{window_id}' \
      -s "$SESSION" -n "${P_NAME[i]}" -c "$REPO_DIR")
  elif [ "${P_LAYOUT[i]}" = window ]; then
    # `-a -t <window>`, not `-t <session>`: tmux resolves a bare session target to its CURRENT
    # active window and tries to create there, which fails "index N in use" once two windows
    # exist. -a means "insert right after this window, shifting later ones", which cannot.
    read -r pane win < <(tmux new-window -a -t "$LAST_WINDOW" -n "${P_NAME[i]}" \
      -P -F '#{pane_id} #{window_id}' -c "$REPO_DIR")
  else
    pane="$(tmux split-window -t "$LAST_PANE" -h -P -F '#{pane_id}' -c "$REPO_DIR")"
    win="$LAST_WINDOW"
  fi
  P_PANE[i]="$pane"; LAST_WINDOW="$win"; LAST_PANE="$pane"
  tmux send-keys -t "$pane" "${P_CMD[i]}" C-m
done
FIRST_WINDOW="$(tmux list-windows -t "$SESSION" -F '#{window_id}' | head -1)"

# Everything above only opened shells. Let them finish before typing into them.
sleep "$SHELL_SETTLE_S"

for i in "${!P_NAME[@]}"; do
  if [ "${P_FRESH[i]}" = 1 ]; then
    [ -n "${P_RUN[i]}" ] && tmux send-keys -t "${P_PANE[i]}" "${P_RUN[i]}" C-m || true
    # No C-m: the operator reads the line and presses Enter. This is how every robot-moving
    # command gets in.
    [ -n "${P_PRE[i]}" ] && tmux send-keys -t "${P_PANE[i]}" "${P_PRE[i]}" || true
  fi
done

tmux select-window -t "$FIRST_WINDOW"
cat <<EOF

Session '$SESSION' is up ($ROS_SSH_HOST runs both the ROS client and the policy server).
Order to press Enter in:
  1. nuc, left           already running bringup (enable FCI on the Desk UI first if it errors)
  2. nuc, right          the inference stack — check the execute flags on the line before you run it
  3. polyumi-pi, right   the policy server — edit CKPT to the checkpoint you want
  4. ros-client          the client, last

Send the arm home (needs pane 2 running; MOVES THE ARM even in plan-only mode):
  ros2 service call /polyumi/home std_srvs/srv/Trigger "{}"

Gripper only, for a first run — edit pane 2's line down to:
  ros2 launch nuc/launch/fr3_inference.launch.py gripper:=$GRIPPER execute_gripper:=true
Then, from the ros-client pane, the acceptance test (MOVES THE FINGERS):
  ros2 run polyumi_ros2 latency_probe --ros-args -p mode:=gripper_chirp

Running gripper:=faulhaber? Bring the bus up on the NUC and, on a NUC that has never been
calibrated, find the hard stops once (SWEEPS THE FULL STROKE — clear the mechanism first):
  ssh $NUC_SSH_HOST 'sudo ip link set can0 up type can bitrate 500000'
  ros2 service call /faulhaber_gripper/calibrate std_srvs/srv/Trigger "{}"
It tracks nothing until that succeeds. The result persists in ~/.ros/ across launches.

Running gripper:=hand? franka_hand_node logs every move(width, speed) it issues in pane 2, at
0.7-1.7 Hz. That ceiling is the hand, not a fault: docs/crb-fr3-inference.md, "Gripper problems".

tmux, minimum viable:
  C-b n / C-b p    next / previous window        C-b o     next pane
  C-b d            detach (everything keeps running)
  C-b C-b          send a prefix to the INNER tmux on $NUC_SSH_HOST/$ROS_SSH_HOST
Re-attach any time with ./fr3_session.sh — remote panes pick up where they left off.

EOF
exec tmux attach -t "$SESSION"
