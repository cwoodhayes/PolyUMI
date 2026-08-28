#!/usr/bin/env bash
# Smallest thing that fails if fr3_session.sh's pane logic breaks.
#
# Only the failures you would NOT catch by running the script and looking at it. Layout and
# argument values are visibly wrong the moment the session comes up; these three are silent:
#
#   1. A robot-moving command gains a trailing `C-m` and executes on its own. The whole
#      RUN-vs-PRETYPE split exists to stop that, and the arm moving is how you'd otherwise learn.
#   2. A pane whose remote session is already live gets typed into. send-keys APPENDS to the
#      readline buffer, which may hold a half-edited launch line, and submits the concatenation.
#   3. --kill aborts partway when a remote is unreachable, so you believe you tore down and
#      did not.
#
# The script's output is side effects on three machines, so it runs against `tmux` and `ssh`
# shims that record what they were asked to do. tmux's own behaviour is not under test.
#
#   ./test_fr3_session.sh

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/bin"

cat > "$WORK/bin/tmux" <<'SHIM'
#!/usr/bin/env bash
echo "tmux $*" >> "$TMUX_LOG"
case "$1" in
  new-session|new-window|split-window)
    n=$(( $(cat "$PANE_N" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$PANE_N" ;;
esac
case "$1" in
  has-session)  exit "${FAKE_HAS_SESSION:-1}" ;;
  new-session|new-window) echo "%$n @$n" ;;
  split-window) echo "%$n" ;;
  list-windows) echo "@1" ;;
  list-panes)   echo "%99" ;;
  attach)       exit 0 ;;
esac
exit 0
SHIM

cat > "$WORK/bin/ssh" <<'SHIM'
#!/usr/bin/env bash
echo "ssh $*" >> "$SSH_LOG"
last="${!#}"
case "$*" in
  *"-G "*) echo "hostname 10.9.9.9"; exit 0 ;;
esac
[ "${FAKE_SSH_FAIL:-0}" = 1 ] && exit 255
case "$last" in *"command -v tmux"*)
  echo has-tmux
  for s in ${FAKE_LIVE:-}; do case "$last" in *"has-session -t $s "*) echo live ;; esac; done
esac
exit 0
SHIM
chmod +x "$WORK/bin/tmux" "$WORK/bin/ssh"

export TMUX_LOG="$WORK/tmux.log" SSH_LOG="$WORK/ssh.log" PANE_N="$WORK/n"
run() {  # run fr3_session.sh with the shims; extra env in the caller's environment
  : > "$TMUX_LOG"; : > "$SSH_LOG"; echo 0 > "$PANE_N"
  PATH="$WORK/bin:$PATH" SKIP_DEPLOY=1 SHELL_SETTLE_S=0 \
    bash "$HERE/fr3_session.sh" "$@" >"$WORK/out" 2>&1
  echo $? > "$WORK/rc"
}

FAILED=0
ok()   { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1"; FAILED=1; }

# Both helpers require the line to EXIST. A bare "no C-m found" grep passes when the send-keys is
# simply absent, so reordering the pane table would silently retire the safety check it names.
pretyped() {  # label, fragment — must be sent, must NOT end in Enter
  local line; line="$(grep -F -- "$2" "$TMUX_LOG" || true)"
  if   [ -z "$line" ];                          then fail "$1 (no such send-keys — pane table changed?)"
  elif printf '%s' "$line" | grep -q 'C-m$';    then fail "$1 (ends in C-m: it WILL execute)"
  else ok "$1"; fi
}
ran() {       # label, fragment — must be sent, and MUST end in Enter
  local line; line="$(grep -F -- "$2" "$TMUX_LOG" || true)"
  if   [ -z "$line" ];                          then fail "$1 (no such send-keys)"
  elif printf '%s' "$line" | grep -q 'C-m$';    then ok "$1"
  else fail "$1 (missing C-m: it will not run)"; fi
}

echo "== nothing already running: the robot-moving lines must sit unexecuted =="
run
# Control: proves the harness can see a C-m at all, so the three checks below are not passing
# because Enter never reaches the log.
ran      "bringup RUNS (control)"             "send-keys -t %1 cd ~/Documents/PolyUMI"
pretyped "NUC inference (execute_arm:=true)"  "send-keys -t %2 cd ~/Documents/PolyUMI"
pretyped "policy server (picks a checkpoint)" "send-keys -t %4 CKPT="
pretyped "ROS client (execute_motion:=true)"  "send-keys -t %5 cd ~/repos/PolyUMI"

echo "== a remote session is already live: that pane must not be typed into =="
FAKE_LIVE="polyumi-ros" run
ran   "live pane still gets its ssh" "send-keys -t %5 ssh -t lamb 'tmux new-session -A -s polyumi-ros'"
if grep -qF -- "ros2 launch polyumi_ros2 inference_demo.launch.xml" "$TMUX_LOG"; then
  fail "live pane is NOT typed into (would append to its readline buffer)"
else ok "live pane is NOT typed into"; fi

echo "== --kill with every remote unreachable still tears down locally =="
FAKE_SSH_FAIL=1 run --kill
[ "$(cat "$WORK/rc")" = 0 ] && ok "exits 0 (set -e does not abort the teardown)" \
                            || fail "exits 0 (set -e does not abort the teardown); rc=$(cat "$WORK/rc")"
grep -qF -- "kill-session -t polyumi" "$TMUX_LOG" && ok "local session is killed" \
                                                  || fail "local session is killed"

[ "$FAILED" = 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
