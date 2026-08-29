# shellcheck shell=sh
# lamb's link to the FR3 NUC. Sourced by setup_franka_env.sh when `hostname` is lamb.
#
# lamb reaches the NUC over its own NetworkManager profile ("direct") bound to enp37s0f1, on a
# 192.168.100.x subnet — not the laptop's fr3-link / enp0s31f6 / 10.0.0.x. Interface name AND peer
# addresses both differ, which is why the DDS side names a whole config file rather than one field.
#
# `:=` throughout, so an exported override still wins over this file.
: "${FR3_NM_PROFILE:=direct}"
: "${FR3_IFACE:=enp37s0f1}"
: "${FR3_SELF_IP:=192.168.100.1/24}"
: "${CYCLONEDDS_CONFIG_FILE:=${POLYUMI_ROOT}/ros2_ws/config/cyclonedds_lamb.xml}"
