#!/usr/bin/env bash
# Start ALL of station-side teleoperation, in the required order: pedals, then GELLO, then
# (optionally) cameras.
#
#   ./start_teleop.bash [--restart] [--skip-device-checks] [--task-id UUID] [--cameras]
#
# Native pixi env (RoboStack ROS 2 Humble, FastDDS), no Docker. See pixi.toml.
#
# The three halves are independent and can be run on their own, in separate terminals,
# which is usually better - one failing then does not take the others down:
#
#   ./start_pedal.bash    pedals  -> base + spine
#   ./start_gello.bash    GELLO   -> both arms + both grippers
#   ./start_cameras.bash  cameras -> head (ZED) + both wrist (RealSense)
#
# This script is the convenience wrapper that starts all requested halves and stops them
# together. All share configs/teleop_common.sh, so a fix lands in one place.

set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

skip_device_checks=false
restart=false
cameras=false
task_id="${TMR_LABS_TASK_ID:-}"
original_args=("$@")

while (( $# > 0 )); do
  case "$1" in
    --skip-device-checks) skip_device_checks=true; shift ;;
    --restart)            restart=true; shift ;;
    --cameras)            cameras=true; shift ;;
    --task-id)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "--task-id requires a LABS task UUID." >&2
        exit 2
      fi
      task_id="$2"; shift 2 ;;
    *) echo "Usage: $0 [--restart] [--skip-device-checks] [--task-id UUID] [--cameras]" >&2
       exit 2 ;;
  esac
done

source configs/teleop_common.sh
teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "${original_args[@]+"${original_args[@]}"}"
teleop_source_workspace

# strict: this script starts the pedals, and they drive the base.
$skip_device_checks || teleop_preflight_robot strict

pedal_nodes='/install(_pixi)?/(pedal_state_publisher|tmr_pedal_teleop|keyboard_state_publisher)/'
gello_nodes='/install(_pixi)?/franka_gello_state_publisher/'
camera_nodes='/install(_pixi)?/(realsense2_camera|zed_open_capture_ros)/'
wrappers='start_teleop\.bash|start_pedal\.bash|start_gello\.bash|start_cameras\.bash'
teleop_guard_running "$pedal_nodes" "Pedal teleoperation" "$restart" "$wrappers"
teleop_guard_running "$gello_nodes" "GELLO teleoperation" "$restart" "$wrappers"
$cameras && teleop_guard_running "$camera_nodes" "Camera nodes" "$restart" "$wrappers"

if ! $skip_device_checks; then
  teleop_check_gello_devices
  teleop_check_pedal_devices
fi

trap teleop_stop_children EXIT INT TERM

# Pedals first: mobile_teleop shuts itself down if it cannot open both foot switches, and
# you want to know that before the GELLOs (or cameras) are live.
pedal_launch_args=()
if [[ -n "$task_id" ]]; then
  pedal_launch_args+=("task_id:=$task_id")
else
  echo "WARNING: no LABS task ID set; the recording pedal will refuse to start recording." >&2
  echo "         Use --task-id UUID or set TMR_LABS_TASK_ID." >&2
fi

if ! teleop_start_launch "pedal teleoperation" 3 \
     tmr_pedal_teleop mobile_teleop.launch.py "${pedal_launch_args[@]+"${pedal_launch_args[@]}"}"; then
  echo "GELLO was not started." >&2
  exit 1
fi
pedal_pid="$TELEOP_LAST_LAUNCH_PID"

echo "Pedal teleoperation is running. Starting GELLO..."
teleop_start_launch "GELLO" 3 \
  franka_gello_state_publisher main.launch.py config_file:=franka_gello_duo.yaml
gello_pid="$TELEOP_LAST_LAUNCH_PID"

wait_pids=("$pedal_pid" "$gello_pid")
if $cameras; then
  echo "Starting cameras..."
  # Independent nodes, not one launch group: one camera failing should not take the
  # others (or GELLO/pedals, already running) down with it.
  if teleop_start_launch "wrist camera (left)" 2 \
       realsense2_camera rs_launch.py \
       camera_namespace:=wrist_camera_left serial_no:="'260322272197'" \
       enable_color:=true enable_depth:=false enable_infra1:=false enable_infra2:=false \
       depth_module.depth_profile:="'640,480,30'"; then
    wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
  else
    echo "WARNING: left wrist camera failed to start." >&2
  fi
  if teleop_start_launch "wrist camera (right)" 2 \
       realsense2_camera rs_launch.py \
       camera_namespace:=wrist_camera_right serial_no:="'260322271869'" \
       enable_color:=true enable_depth:=false enable_infra1:=false enable_infra2:=false \
       depth_module.depth_profile:="'640,480,30'"; then
    wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
  else
    echo "WARNING: right wrist camera failed to start." >&2
  fi
  if teleop_start_launch "head camera (ZED)" 2 \
       zed_open_capture_ros zed_camera.launch.py; then
    wait_pids+=("$TELEOP_LAST_LAUNCH_PID")
  else
    echo "WARNING: head camera failed to start." >&2
  fi
fi

cat <<MSG

Station teleoperation is running (pedal PID $pedal_pid, GELLO PID $gello_pid).
Base, spine and both grippers are live. The ARMS are not: activation is manual and
requires both arms at the recorded home pose first (RUNBOOK.md section 5).
Press Ctrl+C to stop everything together.
MSG

set +e
wait -n "${wait_pids[@]}"
status=$?
set -e
(( status == 0 )) && status=1
echo "A teleoperation launch process exited (status $status); stopping the rest." >&2
exit "$status"
