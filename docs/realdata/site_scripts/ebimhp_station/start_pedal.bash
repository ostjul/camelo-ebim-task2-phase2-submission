#!/usr/bin/env bash
# Start ONLY the foot-pedal half of laptop-side teleoperation: base + spine.
#
#   ./start_pedal.bash [--restart] [--skip-device-checks] [--task-id UUID]
#
# GELLO is a separate, independent script: ./start_gello.bash. Neither needs the other,
# and stopping one leaves the other running. ./start_teleop.bash starts both.
#
# keyboard_state_publisher (the Ctrl+0 DRIVE/RECORD toggle) is NOT started here - it used
# to be bundled into mobile_teleop.launch.py, but a keyboard input node dying took the
# whole pedal stack down with it (ros2 launch treats every node as required). Run it by
# hand when you need the toggle: `ros2 run keyboard_state_publisher keyboard_state_publisher`
# (no terminal focus needed - it reads the operator keyboard directly via evdev).

set -Eeuo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

skip_device_checks=false
restart=false
task_id="${TMR_LABS_TASK_ID:-}"
original_args=("$@")

while (( $# > 0 )); do
  case "$1" in
    --skip-device-checks) skip_device_checks=true; shift ;;
    --restart)            restart=true; shift ;;
    --task-id)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo "--task-id requires a LABS task UUID." >&2
        exit 2
      fi
      task_id="$2"; shift 2 ;;
    *) echo "Usage: $0 [--restart] [--skip-device-checks] [--task-id UUID]" >&2; exit 2 ;;
  esac
done

source configs/teleop_common.sh
teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "${original_args[@]+"${original_args[@]}"}"
teleop_source_workspace

# The pedals drive the base, so clock skew IS fatal here: SwerveDriveController discards
# stale commands silently and a skewed laptop is indistinguishable from a dead base.
$skip_device_checks || teleop_preflight_robot strict

# mobile_teleop.launch.py owns pedal_state_publisher, mode_manager, keyboard_state_publisher,
# base_bridge, spine_bridge, labs_pedal_bridge, mobile_base_state_bridge and
# spine_state_publisher. GELLO is deliberately NOT in this pattern.
pedal_nodes='/install(_pixi)?/(pedal_state_publisher|tmr_pedal_teleop|keyboard_state_publisher)/'
teleop_guard_running "$pedal_nodes" "Pedal teleoperation" "$restart" 'start_pedal\.bash|start_teleop\.bash'

$skip_device_checks || teleop_check_pedal_devices

trap teleop_stop_children EXIT INT TERM

pedal_launch_args=()
if [[ -n "$task_id" ]]; then
  pedal_launch_args+=("task_id:=$task_id")
else
  echo "WARNING: no LABS task ID set; the recording pedal will refuse to start recording." >&2
  echo "         Use --task-id UUID or set TMR_LABS_TASK_ID." >&2
fi

# mobile_teleop shuts itself down if the pedal publisher cannot open both switches, so give
# device opening and launch event handling time before declaring success.
teleop_start_launch "pedal teleoperation" 3 \
  tmr_pedal_teleop mobile_teleop.launch.py "${pedal_launch_args[@]+"${pedal_launch_args[@]}"}"
pedal_pid="$TELEOP_LAST_LAUNCH_PID"

cat <<MSG

Pedal teleoperation is running (PID $pedal_pid). The base and spine are live.

  FS1.a forward   FS2.a backward     spine up:   FS1.a + FS2.c
  FS1.b left      FS2.b right        spine down: FS1.c + FS2.a
  FS1.c rotate CW FS2.c rotate CCW

The base moves at 0.1 m/s, the controller's ceiling, and accelerates at 0.1 m/s^2 - so a
one-second tap is only a few cm. Hold a pedal for several seconds before concluding it is
not working; that mistake has cost a whole debugging session before.
GELLO (arms + grippers) is separate: ./start_gello.bash
Press Ctrl+C to stop the pedal stack.
MSG

set +e
wait -n "$pedal_pid"
status=$?
set -e
(( status == 0 )) && status=1
echo "The pedal launch exited (status $status)." >&2
exit "$status"
