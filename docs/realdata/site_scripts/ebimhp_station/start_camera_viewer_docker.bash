#!/usr/bin/env bash
# RETIRED as the default (2026-08-30): ./start_camera_viewer.bash is now native (pixi env,
# same conda/RoboStack ROS Humble build as the native cameras). This Docker version's ROS
# install (apt-based Humble) and the native cameras' ROS install (conda/RoboStack) are
# mutually invisible over DDS - confirmed with a trivial talker/listener test: not even
# basic discovery works across that boundary, regardless of any FastDDS profile tuning.
# Kept as reference / for use against the Docker-based camera path if that's ever revived.
#
# Start the read-only three-camera operator viewer in a separate Humble container.
# This script does not start, stop, restart, or modify either teleoperation stack.
set -Eeuo pipefail

IMAGE="${TMR_IMAGE:-teleoperation_devcontainer-gello-ros2:latest}"
CONTAINER="${TMR_CONTAINER:-gello-humble}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

repo_host="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$repo_host" in
  "$HOME"/*) repo_ctr="/workspace/${repo_host#"$HOME"/}" ;;
  *) echo "ERROR: expected the repo under \$HOME ($HOME), got $repo_host" >&2; exit 1 ;;
esac

[ -n "${DISPLAY:-}" ] || {
  echo "ERROR: DISPLAY is not set; run this from a graphical terminal." >&2
  exit 1
}

# Match the DDS interface selection used by the teleoperation stack. An explicit empty
# TMR_DDS_PROFILE keeps the standard all-interface Fast DDS behavior.
dds_host="${TMR_DDS_PROFILE-$(bash "$repo_host/render_dds_profile.sh" 2>/dev/null || true)}"
dds_mount=()
dds_setup=""
if [ -n "$dds_host" ]; then
  [ -f "$dds_host" ] || { echo "ERROR: DDS profile not found: $dds_host" >&2; exit 1; }
  case "$dds_host" in
    "$HOME"/*) dds_ctr="/workspace/${dds_host#"$HOME"/}" ;;
    *)
      dds_ctr="/tmp/camera_viewer_fastdds.xml"
      dds_mount=(-v "$dds_host:$dds_ctr:ro")
      ;;
  esac
  dds_setup="export FASTRTPS_DEFAULT_PROFILES_FILE='$dds_ctr' &&"
fi

if img="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null)" && [ -n "$img" ]; then
  : # Reuse the image of an existing teleoperation container.
else
  img="$IMAGE"
fi
xauth_args=()
if [ -f "$HOME/.Xauthority" ]; then
  xauth_args=(-v "$HOME/.Xauthority:/tmp/.Xauthority:ro" -e XAUTHORITY=/tmp/.Xauthority)
fi
software_gl_args=()
if [ -n "${LIBGL_ALWAYS_SOFTWARE:-}" ]; then
  software_gl_args=(-e "LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE")
fi

echo "Starting read-only camera viewer. Close it with q, Escape, or Ctrl+C."
exec docker run --rm --network host --ipc=host --privileged \
  -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "${xauth_args[@]}" \
  "${dds_mount[@]}" \
  -v /dev:/dev -v "$HOME:/workspace" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "${software_gl_args[@]}" \
  -u "$(id -u):20" -e HOME=/tmp "$img" bash -lc \
  "source /opt/ros/humble/setup.bash && \
   source /opt/ros/humble/franka/setup.bash && \
   cd '$repo_ctr' && source install/setup.bash && \
   $dds_setup exec python3 camera_viewer.py \"\$@\"" bash "$@"
