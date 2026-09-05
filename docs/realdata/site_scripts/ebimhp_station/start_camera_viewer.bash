#!/usr/bin/env bash
# Start the read-only three-camera operator viewer natively, in the SAME conda/RoboStack
# ROS Humble build as the native cameras (station/pixi.toml). This does not start, stop,
# restart, or modify the teleop/camera stacks - it only subscribes.
#
#   ./start_camera_viewer.bash [-- extra ros args passed to camera_viewer.py]
#
# WHY NATIVE, NOT DOCKER
# The retired start_camera_viewer_docker.bash ran camera_viewer.py inside the
# teleoperation_devcontainer-gello-ros2 image, which is built on apt's ROS Humble. That
# install and the native cameras' conda/RoboStack ROS Humble cannot see each other over
# DDS at all - confirmed 2026-08-30 with a trivial talker/listener test across that
# boundary: not even discovery worked, independent of any FastDDS profile tuning. Running
# the viewer in the exact same pixi env as the cameras sidesteps the problem entirely.

set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

source configs/teleop_common.sh
teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "$@"
teleop_source_workspace

echo "Starting read-only camera viewer. Close it with q, Escape, or Ctrl+C."
exec python3 camera_viewer.py "$@"
