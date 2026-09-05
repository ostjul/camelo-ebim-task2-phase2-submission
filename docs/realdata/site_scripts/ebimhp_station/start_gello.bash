#!/usr/bin/env bash
# Start ONLY the GELLO half of laptop-side teleoperation: both arms + both grippers.
#
#   ./start_gello.bash [--restart] [--skip-device-checks]
#
# The pedals are a separate, independent script: ./start_pedal.bash. Neither needs the
# other, and stopping one leaves the other running. ./start_teleop.bash starts both.
#
# This publishes GELLO joint states and gripper percentages. The grippers follow the
# triggers immediately; the ARMS do not move until joint_impedance_controller is activated
# by hand, which stays a supervised step (RUNBOOK.md section 5).

set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

skip_device_checks=false
restart=false
config_file="franka_gello_duo.yaml"
original_args=("$@")

while (( $# > 0 )); do
  case "$1" in
    --skip-device-checks) skip_device_checks=true; shift ;;
    --restart)            restart=true; shift ;;
    --config)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "--config requires a file name from franka_gello_state_publisher/config." >&2
        exit 2
      fi
      config_file="$2"; shift 2 ;;
    *) echo "Usage: $0 [--restart] [--skip-device-checks] [--config FILE.yaml]" >&2; exit 2 ;;
  esac
done

source configs/teleop_common.sh
teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "${original_args[@]+"${original_args[@]}"}"
teleop_source_workspace

# Warn, never fatal. Unlike the pedals, nothing in the GELLO path is clock-sensitive:
# joint_impedance_controller allows a 10 s future_timestamp_tolerance and enforces real
# staleness against its own receive time, and the gripper percent topic has no header at
# all. Refusing to start GELLO over base clock skew would be wrong.
$skip_device_checks || teleop_preflight_robot warn

gello_nodes='/install(_pixi)?/franka_gello_state_publisher/'
teleop_guard_running "$gello_nodes" "GELLO teleoperation" "$restart" 'start_gello\.bash|start_teleop\.bash'

$skip_device_checks || teleop_check_gello_devices

trap teleop_stop_children EXIT INT TERM

teleop_start_launch "GELLO" 3 \
  franka_gello_state_publisher main.launch.py "config_file:=$config_file"
gello_pid="$TELEOP_LAST_LAUNCH_PID"

cat <<MSG

GELLO is running (PID $gello_pid), publishing both arms and both grippers.

The GRIPPERS follow the triggers now. The ARMS do not move yet: activation is manual and
requires both arms at the recorded home pose first - see RUNBOOK.md section 5.

  ros2 control list_controllers -c /left/controller_manager   # want: inactive
  # home both arms, then, hands OFF the GELLOs:
  ros2 control set_controller_state joint_impedance_controller active -c /left/controller_manager

Hands are CROSSED against the namespaces and that is intended: the GELLO in your right
hand drives the 'left' arm and the 'left' gripper.

Pedals (base + spine) are separate: ./start_pedal.bash
Press Ctrl+C to stop GELLO.
MSG

set +e
wait -n "$gello_pid"
status=$?
set -e
(( status == 0 )) && status=1
echo "The GELLO launch exited (status $status)." >&2
exit "$status"
