# Shared helpers for the laptop-side teleoperation start scripts.
#
#   ./start_pedal.bash   pedals -> base + spine   (+ LABS recording bridge)
#   ./start_gello.bash   GELLO  -> both arms + both grippers
#   ./start_teleop.bash  both, in the required order
#
# Source this, do not execute it. It is sourced by all three so a fix lands once:
#
#   source configs/teleop_common.sh
#
# Every function assumes the caller runs under `set -Eeuo pipefail` and that the working
# directory is the repository root.

# ---------------------------------------------------------------- container entry
# ROS 2 lives in the teleop_inputs container on the TAMS laptop, where /workspace is this
# repository bind-mounted. Re-exec the calling script inside it, preserving its arguments.
#
#   teleop_enter_ros_env "$(basename "${BASH_SOURCE[0]}")" "$@"
teleop_enter_ros_env() {
  local script_name="$1"; shift

  command -v ros2 >/dev/null 2>&1 && return 0

  if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    source /opt/ros/humble/setup.bash
    set -u
    return 0
  fi

  if command -v docker >/dev/null 2>&1 &&
     [[ "$(docker inspect -f '{{.State.Running}}' teleop_inputs 2>/dev/null || true)" == "true" ]]; then
    local docker_options=()
    # Without a TTY keyboard_state_publisher dies on termios and the `m` mode toggle is
    # unavailable, so pass -it through whenever the caller actually has one.
    if [[ -t 0 && -t 1 ]]; then
      docker_options=(-it)
    fi
    echo "ROS 2 is containerized; entering teleop_inputs..."
    exec docker exec "${docker_options[@]}" teleop_inputs bash -lc \
      "source /opt/ros/humble/setup.bash && cd /workspace && exec ./${script_name} \"\$@\"" \
      bash "$@"
  fi

  echo "ROS 2 is unavailable and the teleop_inputs container is not running." >&2
  echo "Start the ROS 2 teleoperation container, then run ./${script_name} again." >&2
  exit 1
}

# ------------------------------------------------------------------- workspace env
# TMR_WS_INSTALL selects which colcon install tree to source. It exists because the
# laptop has two incompatible ROS 2 environments: the teleop_inputs container (Python
# 3.10) built into install/, and a host pixi env whose Python differs, built into
# install_pixi/. Sourcing the wrong one puts foreign-ABI site-packages on PYTHONPATH and
# the nodes fail on import, so each environment keeps its own tree.
teleop_source_workspace() {
  local install_dir="${TMR_WS_INSTALL:-}"

  # Auto-select when the caller did not. Running inside the pixi env (whether via
  # `pixi run`, `pixi shell`, or a plain `exec bash` from one - CONDA_PREFIX survives
  # exec) means Python 3.12, so install/ (built in the Python 3.10 container) cannot
  # work: its setup.bash chains to the container's /opt/ros/humble and its packages are
  # invisible to ament. That failure reads as "Package 'tmr_pedal_teleop' not found",
  # which sends you looking for a build problem that is not there.
  if [[ -z "$install_dir" ]]; then
    if [[ "${CONDA_PREFIX:-}" == *"/.pixi/envs/"* && -f install_pixi/setup.bash ]]; then
      install_dir="install_pixi"
    else
      install_dir="install"
    fi
  fi
  if [[ ! -f "$install_dir/setup.bash" ]]; then
    echo "Missing $install_dir/setup.bash. Build the workspace before starting teleop." >&2
    exit 1
  fi
  # ROS setup scripts are not guaranteed to be nounset-safe.
  set +u
  source "$install_dir/setup.bash"
  source configs/tmr_laptop_env.sh
  set -u
}

# ------------------------------------------------------------------ robot preflight
# teleop_preflight_robot <strict|warn>
#
# Measures clock skew from the robot's own odom stamps. SwerveDriveController ages every
# command against its OWN clock and discards anything older than cmd_vel_timeout (0.5 s)
# while logging nothing, so a skewed laptop looks exactly like a dead base. Receiving odom
# at all also proves the robot stack is reachable, so this doubles as the link check.
#
#   strict - skew beyond 0.4 s is fatal. Use for the pedals: they drive the base.
#   warn   - report and continue. Use for GELLO: the arm controller's
#            future_timestamp_tolerance is 10 s and the gripper percent topic has no
#            header at all, so neither is sensitive to this.
teleop_preflight_robot() {
  local mode="${1:-warn}"
  local robot_ip="172.16.16.10"
  local clock_skew

  clock_skew="$(timeout 15 python3 <<'PY' 2>/dev/null || true
import time

import rclpy
from nav_msgs.msg import Odometry

rclpy.init()
node = rclpy.create_node("teleop_skew_probe")
samples = []


def on_odom(msg):
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    samples.append(stamp - time.time())


node.create_subscription(Odometry, "/swerve_drive_controller/odom", on_odom, 10)
deadline = time.time() + 6.0
# Take more than a handful: the first samples after a subscription match can be a stale
# burst that reads as several seconds of "skew" when the clocks are in fact aligned.
while time.time() < deadline and len(samples) < 40:
    rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.try_shutdown()
if samples:
    steady = samples[10:] or samples
    print(f"{sum(steady) / len(steady):.3f}")
PY
)"

  if [[ -z "$clock_skew" ]]; then
    if command -v ping >/dev/null 2>&1 && ! ping -c1 -W2 "$robot_ip" >/dev/null 2>&1; then
      echo "WARNING: robot $robot_ip did not answer. Check the Ethernet cable and that" >&2
      echo "         the robot is powered; teleop will start but nothing will move." >&2
    else
      echo "WARNING: no /swerve_drive_controller/odom received, so the clock skew could" >&2
      echo "         not be checked. Is the robot-side bringup running on ROS_DOMAIN_ID=$ROS_DOMAIN_ID?" >&2
    fi
    return 0
  fi

  if python3 -c "import sys; sys.exit(0 if abs(float(sys.argv[1])) > 0.4 else 1)" "$clock_skew"; then
    if [[ "$mode" == "strict" ]]; then
      echo "Clock skew robot-laptop is ${clock_skew}s, beyond the 0.4 s safety margin." >&2
      echo "The base watchdog (cmd_vel_timeout 0.5 s) would silently discard every command." >&2
      echo "Fix with: ./configs/sync_robot_clock.sh   (or sudo systemctl restart systemd-timesyncd)" >&2
      exit 1
    fi
    echo "WARNING: clock skew robot-laptop is ${clock_skew}s. Harmless for GELLO, but the" >&2
    echo "         base would discard pedal commands - fix before running ./start_pedal.bash." >&2
    return 0
  fi

  echo "Robot reachable; clock skew robot-laptop is ${clock_skew}s (limit 0.5 s)."
}

# -------------------------------------------------------------------- device checks
teleop_check_gello_devices() {
  local devices
  shopt -s nullglob
  devices=(/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_*)
  shopt -u nullglob
  if (( ${#devices[@]} < 2 )); then
    echo "Expected two GELLO serial devices; found ${#devices[@]}." >&2
    echo "Check /dev/serial/by-id or use --skip-device-checks for diagnostics." >&2
    exit 1
  fi
}

teleop_check_pedal_devices() {
  local count
  count="$(python3 <<'PY'
import evdev

wanted = {evdev.ecodes.KEY_A, evdev.ecodes.KEY_B, evdev.ecodes.KEY_C}
physical_devices = set()
for path in evdev.list_devices():
    try:
        device = evdev.InputDevice(path)
    except OSError:
        continue
    try:
        keys = set(device.capabilities().get(evdev.ecodes.EV_KEY, []))
        if device.info.vendor == 0x3553 and device.info.product == 0xB001 and wanted <= keys:
            physical_devices.add(device.phys or device.path)
    finally:
        device.close()
print(len(physical_devices))
PY
)"
  if (( count < 2 )); then
    echo "Expected two PCsensor foot switches; found $count." >&2
    echo "Check pedal USB access or use --skip-device-checks for diagnostics." >&2
    exit 1
  fi
}

# ------------------------------------------------------------- already-running guard
# teleop_guard_running <node_regex> <label> <restart:true|false> [wrapper_regex]
#
# The pedal publisher GRABS both foot switches exclusively, so a survivor makes a new run
# silently see no pedals. This must come back empty before anything launches.
teleop_guard_running() {
  local node_pattern="$1" label="$2" restart="$3" wrapper_pattern="${4:-}"
  local existing pid

  existing="$(pgrep -af "$node_pattern" || true)"
  [[ -z "$existing" ]] && return 0

  if [[ "$restart" != "true" ]]; then
    echo "$label is already running; refusing to compete for hardware devices:" >&2
    echo "$existing" >&2
    echo >&2
    echo "Stop it first (Ctrl+C in its terminal), or re-run with --restart to have this" >&2
    echo "script stop it for you." >&2
    exit 1
  fi

  echo "Stopping the running $label (--restart)..."
  # Signal any sibling wrapper first so its own trap can stop its launches in order.
  # Exclude this process and its parent, or the script kills itself.
  if [[ -n "$wrapper_pattern" ]]; then
    for pid in $(pgrep -f "$wrapper_pattern" || true); do
      [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
      kill -TERM "$pid" 2>/dev/null || true
    done
  fi
  for pid in $(pgrep -f "$node_pattern" || true); do
    kill -INT "$pid" 2>/dev/null || true
  done
  for _ in {1..60}; do
    pgrep -f "$node_pattern" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if pgrep -f "$node_pattern" >/dev/null 2>&1; then
    echo "  some processes ignored the stop request; escalating to SIGKILL." >&2
    pkill -KILL -f "$node_pattern" 2>/dev/null || true
    sleep 2
  fi
  if pgrep -f "$node_pattern" >/dev/null 2>&1; then
    echo "Could not stop the existing $label:" >&2
    pgrep -af "$node_pattern" >&2
    exit 1
  fi
  echo "  stopped."

  # The process exiting does not mean the kernel has finished releasing its USB device
  # claim yet. Launching the replacement immediately can race that teardown - seen
  # 2026-08-30 as "failed to claim usb interface... busy" on the first camera to start
  # right after --restart. Harmless (the driver's own retry recovers it a few seconds
  # later), but this settle avoids the race outright.
  sleep 1
}

# ------------------------------------------------------------------- child lifecycle
# Launches are started with setsid so each owns a process group; the trap tears the whole
# group down, because a ros2 launch leader can exit before the nodes that hold the devices.
TELEOP_CHILD_PIDS=()
TELEOP_TRACKED_NODE_PIDS=()

teleop_track_children() {
  local parent_pid="$1" node_pid
  while read -r node_pid; do
    [[ -n "$node_pid" ]] && TELEOP_TRACKED_NODE_PIDS+=("$node_pid")
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
}

teleop_stop_children() {
  trap - EXIT INT TERM
  local pid idx alive

  for (( idx=${#TELEOP_CHILD_PIDS[@]}-1; idx>=0; idx-- )); do
    pid="${TELEOP_CHILD_PIDS[$idx]}"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for pid in "${TELEOP_TRACKED_NODE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done

  for _ in {1..30}; do
    alive=false
    for pid in "${TELEOP_CHILD_PIDS[@]:-}" "${TELEOP_TRACKED_NODE_PIDS[@]:-}"; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=true
    done
    $alive || break
    sleep 0.1
  done

  for pid in "${TELEOP_CHILD_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  for pid in "${TELEOP_TRACKED_NODE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}

# teleop_start_launch <label> <settle_seconds> <ros2 launch args...>
# Returns non-zero (after printing) if the launch did not survive the settle window.
teleop_start_launch() {
  local label="$1" settle="$2"; shift 2
  local pid

  echo "Starting $label..."
  setsid ros2 launch "$@" &
  pid=$!
  TELEOP_CHILD_PIDS+=("$pid")

  sleep "$settle"
  if ! kill -0 "$pid" 2>/dev/null; then
    wait "$pid" 2>/dev/null || true
    echo "$label failed to stay running." >&2
    return 1
  fi
  teleop_track_children "$pid"
  TELEOP_LAST_LAUNCH_PID="$pid"
  return 0
}
