#!/usr/bin/env bash
# Start ONLY the TMR mobile base, for testing base control in isolation.
#
#   ./start_base.bash [--restart]
#
# Run this ON THE ROBOT (the companion). It is the first stage of ./start_robot.bash and
# nothing else: no spine, no arms, no grippers. Use it when you want to prove the base
# half of the chain - pedals -> cmd_vel -> wheels - without four other stacks in the way.
#
# Deliberately does NOT move anything. The base sits with a zero command until something
# publishes to /swerve_drive_controller/cmd_vel.
#
# The laptop half is ./start_pedal.bash. That script also brings up spine_bridge, which
# will log action-server timeouts while the spine is not running here; harmless for a
# base-only test, and the base bridge is unaffected.

set -Eeuo pipefail

# Absolute path to this script, for the purged re-exec below.
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

ws_dir="${TMR_WS:-$HOME/tams_ws}"
# Laptop and robot MUST share a domain. Nothing in the robot's rc files sets one, so 0 is
# what every ROS 2 process here lands on unless something overrides it. Keep this in step
# with configs/tmr_laptop_env.sh, which defaults to the same value.
export ROS_DOMAIN_ID="${TMR_ROS_DOMAIN_ID:-0}"

# Fast DDS: UDP only. The SHM transport on this machine repeatedly fails with
#   [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_portNNNN: open_and_lock_file failed
# and when it does, every controller_manager becomes unreachable while the stacks still
# look alive. Removing the SHM transport removes the failure mode; localhost falls back to
# UDP, the same path that already works cross-host. See ~/fastdds_udp_only.xml.
export FASTRTPS_DEFAULT_PROFILES_FILE="${TMR_DDS_PROFILE:-$HOME/fastdds_udp_only.xml}"

restart=false
for arg in "$@"; do
  case "$arg" in
    # Stop an already-running base bringup instead of refusing to start.
    --restart) restart=true ;;
    *) echo "Usage: $0 [--restart]" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- environment
# NOTE: local_setup.bash, never setup.bash. setup.bash chains to the workspace's
# recorded parent (~/ros2_ws, a stale franka stack) and re-injects it as an overlay;
# colcon prepends only-if-absent, so once it is in front it stays in front and
# franka_bringup resolves to the wrong workspace.
set +u
source /opt/ros/humble/setup.bash
source "$ws_dir/install/local_setup.bash"
set -u

# The failure this catches: an overlay already on AMENT_PREFIX_PATH (typically a
# ~/ros2_ws sourced from .bashrc) stays in front, because colcon's setup files prepend
# only-if-absent. The script inherits that from whatever shell started it, and
# franka_bringup then resolves to a stale stack.
#
# The old advice was "open a genuinely new terminal". That is just a way of purging the
# inherited ROS variables, which the script can do itself - so it retries ONCE with them
# unset before giving up. Failing again means franka_bringup is not in $ws_dir at all.
franka_prefix="$(ros2 pkg prefix franka_bringup 2>/dev/null || true)"
if [[ "$franka_prefix" != "$ws_dir"* ]]; then
  if [[ "${TMR_ENV_PURGED:-0}" != "1" ]]; then
    echo "franka_bringup resolves to ${franka_prefix:-nothing}, not $ws_dir."
    echo "Retrying with the inherited ROS environment purged..."
    exec env \
      -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
      -u AMENT_CURRENT_PREFIX -u ROS_PACKAGE_PATH -u PYTHONPATH \
      -u LD_LIBRARY_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
      TMR_ENV_PURGED=1 bash "$script_path" "$@"
  fi

  echo "franka_bringup does not resolve to $ws_dir, even with a purged environment." >&2
  echo "  resolves to: ${franka_prefix:-<not found at all>}" >&2
  echo >&2
  echo "So this is not an overlay-ordering problem. Check where the package actually is:" >&2
  echo "  ls -d $ws_dir/install/franka_bringup" >&2
  echo "  ros2 pkg prefix franka_bringup" >&2
  echo >&2
  echo "If the workspace lives elsewhere, point the script at it:" >&2
  echo "  TMR_WS=/path/to/workspace $0" >&2
  exit 1
fi

# rviz2 is part of tmrv0_2.launch.py and aborts (SIGABRT) with no X display, e.g. over
# plain ssh. Headless keeps it from dying noisily; it does not affect the base either way.
if [[ -z "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=offscreen
fi

# ------------------------------------------------------------- already running?
# A leftover bringup is not harmless: its controller_manager answers the new launch's
# spawners, which then see "Controller already loaded" and fail to configure.
#
# Narrower than start_robot.bash's pattern on purpose. That script owns every stack and
# sweeps ros2_control_node wholesale; this one must not kill the arm or gripper managers,
# which run the same executable. Orphaned base nodes are reported below instead.
stale_pattern='tmrv0_2\.launch'
existing="$(pgrep -af "$stale_pattern" || true)"
if [[ -n "$existing" ]]; then
  if ! $restart; then
    echo "A base bringup is already running; refusing to start a second one:" >&2
    echo "$existing" >&2
    echo >&2
    echo "Stop it first (Ctrl+C in its terminal), or re-run with --restart to have" >&2
    echo "this script stop it for you." >&2
    exit 1
  fi

  echo "Stopping the running base bringup (--restart)..."
  echo "$existing" >&2
  # SIGINT is what ros2 launch expects: it shuts its own children down in order.
  for pid in $(pgrep -f "$stale_pattern" || true); do
    kill -INT "$pid" 2>/dev/null || true
  done
  for _ in {1..60}; do
    pgrep -f "$stale_pattern" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "  ignored SIGINT; escalating to SIGTERM." >&2
    pkill -TERM -f "$stale_pattern" 2>/dev/null || true
    for _ in {1..30}; do
      pgrep -f "$stale_pattern" >/dev/null 2>&1 || break
      sleep 0.5
    done
  fi
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "  ignored SIGTERM; escalating to SIGKILL." >&2
    pkill -KILL -f "$stale_pattern" 2>/dev/null || true
    sleep 2
  fi
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "Could not stop the existing base bringup:" >&2
    pgrep -af "$stale_pattern" >&2
    exit 1
  fi
  echo "  stopped."
  # The hardware needs a moment to drop the previous base session.
  sleep 3
fi

# An orphaned ros2_control_node - one whose launch parent already exited - still owns the
# base connection, and the new bringup cannot start until it is gone. It is NOT killed
# automatically here because the arms and grippers run the same executable and this script
# has no business touching them.
if pgrep -f 'ros2_control_node' >/dev/null 2>&1 && \
   ! pgrep -f "$stale_pattern" >/dev/null 2>&1; then
  echo "note: ros2_control_node processes are running with no base launch attached:" >&2
  pgrep -af 'ros2_control_node' >&2
  echo "      If these are arm/gripper managers, ignore this. If the base fails to come" >&2
  echo "      up, one of them is an orphan holding the base connection - kill that PID." >&2
fi

# -------------------------------------------------------------------- lifecycle
launch_pid=""

stop_all() {
  trap - EXIT INT TERM
  [[ -z "$launch_pid" ]] && return 0
  kill -TERM -- "-$launch_pid" 2>/dev/null || true
  # The launch leader can exit before ros2_control children; test the group, not the PID.
  for _ in {1..60}; do
    kill -0 -- "-$launch_pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -0 -- "-$launch_pid" 2>/dev/null && kill -KILL -- "-$launch_pid" 2>/dev/null || true
  wait "$launch_pid" 2>/dev/null || true
}
trap stop_all EXIT INT TERM

# ------------------------------------------------------------------- launch
echo "Starting mobile base (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)..."
setsid ros2 launch franka_bringup tmrv0_2.launch.py \
  controller_name:=swerve_drive_controller &
launch_pid=$!
sleep 2
if ! kill -0 "$launch_pid" 2>/dev/null; then
  echo "The base launch exited immediately." >&2
  exit 1
fi

# Non-fatal by design: under `set -e` a non-zero return here would fire the EXIT trap and
# tear down a base that is actually running fine, just slower than expected.
deadline=$(( SECONDS + 60 ))
odom_ok=false
while (( SECONDS < deadline )); do
  if ros2 topic list 2>/dev/null | grep -q '/swerve_drive_controller/odom'; then
    odom_ok=true; break
  fi
  sleep 1
done
if $odom_ok; then
  echo "  ok: base odometry"
else
  echo "  WARNING: timed out waiting for /swerve_drive_controller/odom." >&2
  echo "           Continuing anyway - verify by hand before relying on it." >&2
fi

# The base is useless if this one is not active - it owns the cartesian_velocity command
# interfaces the pedal bridge ultimately drives.
if ! ros2 control list_controllers 2>/dev/null | grep -q 'swerve_drive_controller.*active'; then
  echo "  WARNING: swerve_drive_controller is not active; the base will not move." >&2
fi
if ! ros2 control list_controllers 2>/dev/null | grep -q 'joint_state_broadcaster.*active'; then
  # Cosmetic only (it feeds /dynamic_joint_states and real wheel values), but it is the
  # controller that loses the race against a stale manager, so recover it in place.
  echo "  joint_state_broadcaster inactive; spawning it."
  ros2 run controller_manager spawner joint_state_broadcaster -c /controller_manager || true
fi

cat <<'MSG'

The mobile base is up. Nothing else is running, and nothing will move on its own.

To test WITHOUT the pedals, from this terminal - the base creeps forward ~86 mm:

  python3 base_nudge.py              # or --y 0.1 to strafe, --yaw 0.1 to rotate

Do NOT use `ros2 topic pub` for this. It leaves header.stamp at 0, and the controller
ages every command against its own clock (swerve_drive_controller.cpp:78), discarding
anything older than cmd_vel_timeout 0.5 s. A zero stamp reads as ~1.8 billion seconds
old, so every message is thrown away while the topic still looks healthy at 20 Hz.

To test WITH the pedals, on the laptop: ./start_pedal.bash
Watch what the base actually accepted:

  ros2 topic echo /swerve_drive_controller/cmd_vel_out

The controller clamps to 0.1 m/s and 0.1 rad/s and ramps at 0.1 m/s^2, so a one-second
tap is only a few cm. Hold a pedal for several seconds before concluding it is dead.

Press Ctrl+C to stop the base.
MSG

set +e
wait "$launch_pid"
status=$?
set -e
(( status == 0 )) && status=1
echo "The base launch exited (status $status)." >&2
exit "$status"
