# Source this on THIS machine (ebimHP) before running teleop or checking robot topics.
#   source configs/tmr_laptop_env.sh
# Assumes you are already inside the pixi ROS 2 env (pixi shell) with the workspace
# sourced, AND that multicast works on the laptop<->robot Ethernet link (test with
# `ros2 multicast send` / `ros2 multicast receive` - see README).
#
# Native FastDDS, default discovery. No discovery server, no robot-side config.

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# Domain 0 is the teleop domain. Nothing in the robot's shell rc files sets
# ROS_DOMAIN_ID, so the robot stack lands on 0, and the rest of the system already
# assumes 0: labs_integration/tmr_station/docker-compose.yml, fastdds_labs.xml,
# fastdds_laptop_discovery.xml, .devcontainer/docker-compose.yml and the Olive sensors.
# Laptop and robot MUST match; a mismatch makes each host see only local topics, and
# running on 100 silently orphans LABS and the Olive IMU.
# Override per-shell with: TMR_ROS_DOMAIN_ID=<n> source configs/tmr_laptop_env.sh
# The container image exports ROS_DOMAIN_ID=77, so do not inherit that unrelated default.
export ROS_DOMAIN_ID="${TMR_ROS_DOMAIN_ID:-0}"
unset ROS_DISCOVERY_SERVER

# Pin DDS to the Ethernet NIC facing the robot (this host is multi-homed: WiFi + Ethernet).
# Render the profile with the address assigned today; DHCP may change it between sessions -
# a hardcoded address here is exactly the bug class that took down GELLO, both wrist
# cameras and the ZED head camera on 2026-08-29 (a stale IP in a CycloneDDS config file).
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Which subnet DDS should use. The dedicated robot Ethernet (172.16.16.x) is preferred
# because it is a direct link, but falls back to the shared 192.168.50.x LAN if that link
# is down rather than refusing to start.
#
# Override either with TMR_DDS_SUBNET / TMR_DDS_FALLBACK_SUBNET (trailing dot included).
_tmr_pref_subnet="${TMR_DDS_SUBNET:-172.16.16.}"
_tmr_fallback_subnet="${TMR_DDS_FALLBACK_SUBNET:-192.168.50.}"

# The minimal teleop container uses host networking but does not include iproute2.
_tmr_addr_on() {
  local prefix="$1"
  if command -v ip >/dev/null 2>&1; then
    ip -4 -o addr show up scope global \
      | awk -v p="$prefix" 'index($4, p) == 1 {sub(/\/.*/, "", $4); print $4; exit}'
  else
    hostname -I | tr ' ' '\n' | awk -v p="$prefix" 'index($0, p) == 1 {print; exit}'
  fi
}

_tmr_eth_ip="${TMR_DDS_ADDR:-}"
_tmr_link="explicit TMR_DDS_ADDR"
if [[ -z "$_tmr_eth_ip" ]]; then
  _tmr_eth_ip="$(_tmr_addr_on "$_tmr_pref_subnet")"
  _tmr_link="robot Ethernet"
fi
if [[ -z "$_tmr_eth_ip" ]]; then
  _tmr_eth_ip="$(_tmr_addr_on "$_tmr_fallback_subnet")"
  _tmr_link="LAN fallback"
  if [[ -n "$_tmr_eth_ip" ]]; then
    echo "NOTE: no ${_tmr_pref_subnet}x address; using the ${_tmr_fallback_subnet}x LAN for DDS." >&2
    echo "      Teleop tolerates this (the base watchdog is 0.5 s against a 20 Hz stream)," >&2
    echo "      but WiFi jitter is far worse than the wired link - prefer the cable when it works." >&2
  fi
fi
if [[ -z "$_tmr_eth_ip" ]]; then
  echo "No active ${_tmr_pref_subnet}x or ${_tmr_fallback_subnet}x address; cannot configure TMR DDS." >&2
  echo "Check the link, or set TMR_DDS_ADDR=<this host's address> explicitly." >&2
  return 1 2>/dev/null || exit 1
fi

# Match ANY address in the template, not just 172.16.16.x, or the LAN fallback would be
# written into a file that still whitelists an interface this host does not have.
_tmr_fastdds_profile="/tmp/tmr_fastdds_laptop_$(id -u).xml"
sed -E "s#<address>[0-9]+(\.[0-9]+){3}</address>#<address>${_tmr_eth_ip}</address>#g" \
  "$_here/fastdds_laptop_discovery.xml" > "$_tmr_fastdds_profile"
export FASTRTPS_DEFAULT_PROFILES_FILE="$_tmr_fastdds_profile"

# The ros2 CLI daemon caches DDS settings; restart it so it picks up the above.
ros2 daemon stop >/dev/null 2>&1
ros2 daemon start >/dev/null 2>&1

echo "TMR DDS env set (native FastDDS, default discovery, Ethernet-pinned):"
echo "  NOTE: launch the robot stack with the SAME ROS_DOMAIN_ID."
echo "  RMW=$RMW_IMPLEMENTATION  DOMAIN=$ROS_DOMAIN_ID"
echo "  Interface=$_tmr_eth_ip ($_tmr_link)"
echo "  profile=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "Verify multicast first:  (robot) ros2 multicast receive   (laptop) ros2 multicast send"
echo "Then:  ros2 topic list   (should include the robot's /olive/... topics)"
