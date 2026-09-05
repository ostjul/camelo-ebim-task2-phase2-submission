#!/usr/bin/env bash
# Start ONLY the cameras: head (ZED-M) + both wrist (RealSense D405) cameras.
#
#   ./start_cameras.bash [--restart]
#
# Independent of GELLO and pedals - neither needs this, and stopping this leaves them
# running. ./start_teleop.bash --cameras starts this alongside the rest.
#
# Native pixi env (RoboStack ROS 2 Humble, FastDDS), no Docker, no CycloneDDS interface
# pinning to go stale. The ZED node is zed_open_capture_ros (see its package README for
# why plain V4L2/OpenCV capture does not work on this camera and a vendored library is
# needed).

set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

restart=false
original_args=("$@")

while (( $# > 0 )); do
  case "$1" in
    --restart) restart=true; shift ;;
    *) echo "Usage: $0 [--restart]" >&2; exit 2 ;;
  esac
done

source configs/teleop_common.sh
teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "${original_args[@]+"${original_args[@]}"}"
teleop_source_workspace

camera_nodes='/install(_pixi)?/(realsense2_camera|zed_open_capture_ros)/'
teleop_guard_running "$camera_nodes" "Camera nodes" "$restart" 'start_cameras\.bash|start_teleop\.bash'

trap teleop_stop_children EXIT INT TERM

wait_pids=()

if teleop_start_launch "wrist camera (left)" 2 \
     realsense2_camera rs_launch.py \
     camera_namespace:=wrist_camera_left serial_no:="'260322272197'" \
     enable_color:=true enable_depth:=false enable_infra1:=false enable_infra2:=false \
     depth_module.depth_profile:="'640,480,30'" \
     config_file:=/home/ebim/teleoperation/station/configs/d405_color_640x480.yml; then
  wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
else
  echo "WARNING: left wrist camera failed to start." >&2
fi

if teleop_start_launch "wrist camera (right)" 2 \
     realsense2_camera rs_launch.py \
     camera_namespace:=wrist_camera_right serial_no:="'260322271869'" \
     enable_color:=true enable_depth:=false enable_infra1:=false enable_infra2:=false \
     depth_module.depth_profile:="'640,480,30'" \
     config_file:=/home/ebim/teleoperation/station/configs/d405_color_640x480.yml; then
  wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
else
  echo "WARNING: right wrist camera failed to start." >&2
fi

if teleop_start_launch "head camera (ZED)" 2 \
     zed_open_capture_ros zed_camera.launch.py \
     resolution:=HD720; then
  wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
else
  echo "WARNING: head camera failed to start." >&2
fi

(( ${#wait_pids[@]} > 0 )) || { echo "No camera started." >&2; exit 1; }

cat <<MSG

Cameras running:
  /wrist_camera_left/camera/color/image_raw
  /wrist_camera_right/camera/color/image_raw
  /head_camera/zed_node/rgb/color/rect/image

Press Ctrl+C to stop all cameras.
MSG

set +e
wait -n "${wait_pids[@]}"
status=$?
set -e
(( status == 0 )) && status=1
echo "A camera launch process exited (status $status); stopping the rest." >&2
exit "$status"
