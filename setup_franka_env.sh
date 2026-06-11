# Source this (do NOT execute) to configure the laptop for the FR3 NUC.
#
#   source setup_franka_env.sh
#
# Sets the DDS environment so the Kilted laptop interoperates with the Humble
# FR3 NUC, and brings up the static IP on the wired link to the NUC via a
# toggleable NetworkManager profile. Confined to this script so the default /
# old-arm workflow is untouched when not sourced.
# See docs/crb-fr3-inference.md for the full topology and rationale.

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
