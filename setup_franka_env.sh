# Source this (do NOT execute) to configure the laptop for the FR3 NUC.
#
#   source setup_franka_env.sh
#
# Sets the DDS environment so the Kilted laptop interoperates with the Humble
# FR3 NUC, and brings up the static IP on the wired link to the NUC via a
# toggleable NetworkManager profile. Confined to this script so the default /
# old-arm workflow is untouched when not sourced.
# See docs/crb-fr3-inference.md for the full topology and rationale.

# Abort if executed instead of sourced, before nmcli makes any changes — if we let execution
# continue, the NM profile side effect would run but the exported env vars would be discarded
# with the subshell, silently leaving the caller's shell unconfigured.
#   - zsh: `return` at the top level of a script behaves like `exit` (always "succeeds"), so
#     it can't be used to detect sourcing there. Check ZSH_EVAL_CONTEXT instead — it ends in
#     ":file" when sourced, and is exactly "toplevel" when executed directly (verified).
#   - bash/sh: `return` outside a function only succeeds when the script is sourced.
if [ -n "${ZSH_VERSION:-}" ]; then
  case $ZSH_EVAL_CONTEXT in
    *:file) _polyumi_sourced=1 ;;
    *) _polyumi_sourced=0 ;;
  esac
elif (return 0 2>/dev/null); then
  _polyumi_sourced=1
else
  _polyumi_sourced=0
fi
if [ "$_polyumi_sourced" != "1" ]; then
  echo "ERROR: setup_franka_env.sh must be sourced, not executed — its exported env vars" >&2
  echo "  would otherwise be discarded when the script's subshell exits." >&2
  echo "  Run:  source setup_franka_env.sh" >&2
  exit 1
fi
unset _polyumi_sourced

echo "NOTE - SOURCE THIS (do NOT execute; it will not work)"

# --- Resolve repo root (works for bash and zsh) ---
if [ -n "${BASH_SOURCE:-}" ]; then
  _polyumi_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _polyumi_self="${(%):-%x}"
else
  _polyumi_self="$0"
fi
POLYUMI_ROOT="$(cd "$(dirname "$_polyumi_self")" && pwd)"
unset _polyumi_self

# --- Wired link to the NUC ---
# Host-specific: a machine other than this laptop is on a different NIC and usually a different
# subnet entirely. Each such host keeps its values in config/env.<hostname>.sh (see
# config/env.lamb.sh); the defaults below are this laptop's. Exported vars still win over both,
# so a one-off override works without editing anything.
_polyumi_host_env="${POLYUMI_ROOT}/config/env.$(hostname -s).sh"
if [ -f "$_polyumi_host_env" ]; then
  echo "[setup_franka_env] host config: $_polyumi_host_env"
  . "$_polyumi_host_env"
fi
unset _polyumi_host_env

: "${FR3_IFACE:=enp0s31f6}"
: "${FR3_SELF_IP:=10.0.0.1/24}"
: "${FR3_NM_PROFILE:=fr3-link}"

# --- franka_description, for Foxglove's URDF panel ---
# The NUC latches /robot_description, so Foxglove can draw the arm on this laptop — but the
# meshes it references are package:// URIs that foxglove_bridge resolves on ITS machine, i.e.
# here. franka_description is not in /opt/ros/kilted, so point at whatever workspace has it.
# Purely a debugging nicety: a missing workspace costs meshes, not a run.
#
# Extending AMENT_PREFIX_PATH rather than sourcing that workspace's setup.bash is deliberate:
# the ament index is all package:// resolution needs (franka_description is data-only), and
# sourcing ROS setup.bash chains from zsh is the failure documented in CLAUDE.md.
: "${FRANKA_DESCRIPTION_WS:=${HOME}/ws/franka/install}"
if [ -d "${FRANKA_DESCRIPTION_WS}/share/franka_description" ]; then
  export AMENT_PREFIX_PATH="${FRANKA_DESCRIPTION_WS}:${AMENT_PREFIX_PATH}"
  echo "[setup_franka_env] franka_description from ${FRANKA_DESCRIPTION_WS}"
else
  echo "[setup_franka_env] WARNING: no franka_description at ${FRANKA_DESCRIPTION_WS} —"
  echo "  Foxglove will show TF frames but no arm meshes. Export FRANKA_DESCRIPTION_WS to fix."
fi

# --- ROS base + this workspace's overlay ---
# So 'ros2 launch polyumi_ros2 ...' works right after sourcing this script, without depending on
# the caller's shell rc having already done it — true on a laptop set up for ROS dev, not
# guaranteed elsewhere (e.g. a GPU box whose rc sources the distro but never this repo's build).
#
# `cd` into each setup.bash's own directory before sourcing it, then back: the setup chain
# resolves a companion file by relative path, which under zsh falls back to $PWD. See CLAUDE.md,
# "Running colcon build / ros2 from a non-interactive (or zsh) shell".
: "${ROS_DISTRO_DIR:=/opt/ros/kilted}"
_polyumi_pwd="$PWD"
if ! command -v ros2 >/dev/null 2>&1; then
  if [ -f "${ROS_DISTRO_DIR}/setup.bash" ]; then
    cd "${ROS_DISTRO_DIR}" && source setup.bash
  else
    echo "[setup_franka_env] WARNING: no ROS install at ${ROS_DISTRO_DIR} — 'ros2' will not be on PATH."
  fi
fi
case ":${AMENT_PREFIX_PATH:-}:" in
  *":${POLYUMI_ROOT}/ros2_ws/install"*) ;; # already sourced
  *)
    if [ -f "${POLYUMI_ROOT}/ros2_ws/install/setup.bash" ]; then
      cd "${POLYUMI_ROOT}/ros2_ws/install" && source setup.bash
    else
      echo "[setup_franka_env] WARNING: ${POLYUMI_ROOT}/ros2_ws/install not built —"
      echo "  'ros2 launch polyumi_ros2 ...' will fail until colcon build runs there."
    fi
    ;;
esac
cd "$_polyumi_pwd"
unset _polyumi_pwd

# --- DDS: match the NUC (CycloneDDS, domain 0, unicast peers) ---
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
# Interface name AND peer addresses are both host-specific, so this points at a whole config file
# rather than patching one field of a shared one. cyclonedds_laptop.xml is this machine's own;
# other hosts set CYCLONEDDS_CONFIG_FILE in their config/env.<hostname>.sh.
: "${CYCLONEDDS_CONFIG_FILE:=${POLYUMI_ROOT}/ros2_ws/config/cyclonedds_laptop.xml}"
export CYCLONEDDS_URI="file://${CYCLONEDDS_CONFIG_FILE}"

# --- Static IP on the NUC link (toggleable NetworkManager profile) ---
# The NUC's cyclonedds.xml hardcodes peers 10.0.0.1 (laptop) / 10.0.0.2 (NUC),
# so the laptop MUST hold 10.0.0.1 for unicast discovery to work.
#
# We use a named NM profile ($FR3_NM_PROFILE) with autoconnect off, so the port
# still does normal DHCP for other uses and the static IP is only active while
# this profile is up. To revert: `nmcli connection down $FR3_NM_PROFILE`.
if ! command -v nmcli >/dev/null 2>&1; then
  echo "[setup_franka_env] WARNING: nmcli not found; bring up ${FR3_SELF_IP} on ${FR3_IFACE} yourself"
elif nmcli -t -f NAME connection show --active 2>/dev/null | grep -qx "$FR3_NM_PROFILE"; then
  echo "[setup_franka_env] NM profile '$FR3_NM_PROFILE' already active"
else
  if ! nmcli -t -f NAME connection show 2>/dev/null | grep -qx "$FR3_NM_PROFILE"; then
    echo "[setup_franka_env] creating NM profile '$FR3_NM_PROFILE' (${FR3_SELF_IP} on ${FR3_IFACE}, autoconnect off)"
    nmcli connection add type ethernet ifname "$FR3_IFACE" con-name "$FR3_NM_PROFILE" \
      ipv4.method manual ipv4.addresses "$FR3_SELF_IP" connection.autoconnect no
  fi
  echo "[setup_franka_env] bringing up NM profile '$FR3_NM_PROFILE'"
  nmcli connection up "$FR3_NM_PROFILE"
fi

echo "[setup_franka_env] RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[setup_franka_env] CYCLONEDDS_URI=$CYCLONEDDS_URI"
