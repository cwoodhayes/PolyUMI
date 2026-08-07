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
# Override by exporting these before sourcing if your hardware differs.
: "${FR3_IFACE:=enp0s31f6}"
: "${FR3_LAPTOP_IP:=10.0.0.1/24}"
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

# --- DDS: match the NUC (CycloneDDS, domain 0, unicast peers) ---
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export CYCLONEDDS_URI="file://${POLYUMI_ROOT}/ros2_ws/config/cyclonedds_laptop.xml"

# --- Static IP on the NUC link (toggleable NetworkManager profile) ---
# The NUC's cyclonedds.xml hardcodes peers 10.0.0.1 (laptop) / 10.0.0.2 (NUC),
# so the laptop MUST hold 10.0.0.1 for unicast discovery to work.
#
# We use a named NM profile ($FR3_NM_PROFILE) with autoconnect off, so the port
# still does normal DHCP for other uses and the static IP is only active while
# this profile is up. To revert: `nmcli connection down $FR3_NM_PROFILE`.
if ! command -v nmcli >/dev/null 2>&1; then
  echo "[setup_franka_env] WARNING: nmcli not found; bring up ${FR3_LAPTOP_IP} on ${FR3_IFACE} yourself"
elif nmcli -t -f NAME connection show --active 2>/dev/null | grep -qx "$FR3_NM_PROFILE"; then
  echo "[setup_franka_env] NM profile '$FR3_NM_PROFILE' already active"
else
  if ! nmcli -t -f NAME connection show 2>/dev/null | grep -qx "$FR3_NM_PROFILE"; then
    echo "[setup_franka_env] creating NM profile '$FR3_NM_PROFILE' (${FR3_LAPTOP_IP} on ${FR3_IFACE}, autoconnect off)"
    nmcli connection add type ethernet ifname "$FR3_IFACE" con-name "$FR3_NM_PROFILE" \
      ipv4.method manual ipv4.addresses "$FR3_LAPTOP_IP" connection.autoconnect no
  fi
  echo "[setup_franka_env] bringing up NM profile '$FR3_NM_PROFILE'"
  nmcli connection up "$FR3_NM_PROFILE"
fi

echo "[setup_franka_env] RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "[setup_franka_env] CYCLONEDDS_URI=$CYCLONEDDS_URI"
