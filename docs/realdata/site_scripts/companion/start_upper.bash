#!/usr/bin/env bash
# Start ONLY the upper body - both FR3 arms and both Robotiq grippers - for GELLO
# teleoperation. This is stages 3 and 4 of ./start_robot.bash and nothing else.
#
#   ./start_upper.bash [--restart] [--skip-grippers]
#
# Run this ON THE ROBOT (the companion). The laptop half is ./start_gello.bash.
#
# The base and spine are NOT started and are NOT required: each arm runs its own
# controller_manager against its own FCI (tmr_duo_config.yaml), so the upper body is
# independent of the mobile base. Use ./start_base.bash if you want the base as well.
#
# Deliberately does NOT move anything. The arm impedance controllers spawn INACTIVE, and
# sending the arms to the home pose stays a manual, supervised step - see the closing
# message and RUNBOOK.md S5.

set -Eeuo pipefail

# Absolute path to this script, for the purged re-exec below.
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

ws_dir="${TMR_WS:-$HOME/tams_ws}"
# Laptop and robot MUST share a domain. Keep this in step with configs/tmr_laptop_env.sh.
export ROS_DOMAIN_ID="${TMR_ROS_DOMAIN_ID:-0}"

# Fast DDS: UDP only. The SHM transport on this machine repeatedly fails with
#   [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_portNNNN: open_and_lock_file failed
# and when it does, every controller_manager becomes unreachable while the stacks still
# look alive. Removing the SHM transport removes the failure mode; localhost falls back to
# UDP, the same path that already works cross-host. See ~/fastdds_udp_only.xml.
export FASTRTPS_DEFAULT_PROFILES_FILE="${TMR_DDS_PROFILE:-$HOME/fastdds_udp_only.xml}"

restart=false
skip_grippers=false
for arg in "$@"; do
  case "$arg" in
    # Stop an already-running upper-body bringup instead of refusing to start.
    --restart)        restart=true ;;
    --skip-grippers)  skip_grippers=true ;;
    *) echo "Usage: $0 [--restart] [--skip-grippers]" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- environment
# NOTE: local_setup.bash, never setup.bash. setup.bash chains to the workspace's
# recorded parent (~/ros2_ws, a stale franka stack) and re-injects it as an overlay;
# colcon prepends only-if-absent, so once it is in front it stays in front.
set +u
source /opt/ros/humble/setup.bash
source "$ws_dir/install/local_setup.bash"
set -u

# An overlay already on AMENT_PREFIX_PATH stays in front, and franka_fr3_arm_controllers
# then resolves to a stale stack. "Open a new terminal" is just a way of purging the
# inherited ROS variables, so do that here and retry ONCE.
arm_prefix="$(ros2 pkg prefix franka_fr3_arm_controllers 2>/dev/null || true)"
if [[ "$arm_prefix" != "$ws_dir"* ]]; then
  if [[ "${TMR_ENV_PURGED:-0}" != "1" ]]; then
    echo "franka_fr3_arm_controllers resolves to ${arm_prefix:-nothing}, not $ws_dir."
    echo "Retrying with the inherited ROS environment purged..."
    exec env \
      -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
      -u AMENT_CURRENT_PREFIX -u ROS_PACKAGE_PATH -u PYTHONPATH \
      -u LD_LIBRARY_PATH -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
      TMR_ENV_PURGED=1 bash "$script_path" "$@"
  fi

  echo "franka_fr3_arm_controllers does not resolve to $ws_dir, even with a purged environment." >&2
  echo "  resolves to: ${arm_prefix:-<not found at all>}" >&2
  echo >&2
  echo "So this is not an overlay-ordering problem. Check where the package actually is:" >&2
  echo "  ls -d $ws_dir/install/franka_fr3_arm_controllers" >&2
  echo "  ros2 pkg prefix franka_fr3_arm_controllers" >&2
  echo >&2
  echo "If the workspace lives elsewhere:  TMR_WS=/path/to/workspace $0" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=offscreen
fi

# ------------------------------------------------------------- already running?
# A leftover bringup is not harmless: its controller_manager answers the new launch's
# spawners, which then see "Controller already loaded" and fail to configure.
#
# Narrower than start_robot.bash's pattern on purpose. That script owns every stack and
# sweeps ros2_control_node wholesale; this one must not kill the BASE's manager, which
# runs the same executable. Orphans are reported below instead.
stale_pattern='franka_fr3_arm_controllers\.launch|robotiq_gripper_controller_client\.launch'
existing="$(pgrep -af "$stale_pattern" || true)"
if [[ -n "$existing" ]]; then
  if ! $restart; then
    echo "An upper-body bringup is already running; refusing to start a second one:" >&2
    echo "$existing" >&2
    echo >&2
    echo "Stop it first (Ctrl+C in its terminal), or re-run with --restart to have" >&2
    echo "this script stop it for you." >&2
    exit 1
  fi

  echo "Stopping the running upper-body bringup (--restart)..."
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
    echo "Could not stop the existing upper-body bringup:" >&2
    pgrep -af "$stale_pattern" >&2
    exit 1
  fi
  echo "  stopped."
  # The arms need a moment to drop the previous FCI session.
  sleep 3
fi

# An orphaned ros2_control_node - one whose launch parent already exited (PPid 1) - still
# owns an arm's FCI. The new bringup then starts a SECOND node on the same node names,
# every spawner dies with exit 1, and the arm sits unconfigured while the orphan logs
# update errors at 1 kHz. Observed on 2026-08-23 with both arms.
#
# Matched by NAMESPACE, never by executable: the base's node is __ns:=/ and the grippers'
# are __ns:=/left/gripper, so only a bare /left or /right can be an arm orphan. That makes
# it safe to kill here, which the generic ros2_control_node pattern would not be.
arm_orphan_pids() {
  local pid ns ppid
  for pid in $(pgrep -f ros2_control_node 2>/dev/null || true); do
    [[ -r "/proc/$pid/cmdline" ]] || continue
    ns="$(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '__ns:=/(left|right)( |$)' | head -1)"
    [[ -n "$ns" ]] || continue
    ppid="$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null)"
    [[ "$ppid" == "1" ]] && echo "$pid"
  done
}

orphans="$(arm_orphan_pids)"
if [[ -n "$orphans" ]]; then
  if ! $restart; then
    echo "Orphaned arm ros2_control_node(s) are holding the FCI (parent already exited):" >&2
    for pid in $orphans; do
      echo "  pid $pid $(tr '\0' ' ' < "/proc/$pid/cmdline" | grep -oE -- '__ns:=[^ ]+')" >&2
    done
    echo >&2
    echo "Starting now would put a second node on the same names and every spawner would" >&2
    echo "die with exit 1. Re-run with --restart, or: kill -INT $(echo $orphans | tr '\n' ' ')" >&2
    exit 1
  fi

  echo "Stopping orphaned arm ros2_control_node(s) (--restart): $(echo $orphans | tr '\n' ' ')"
  for pid in $orphans; do kill -INT "$pid" 2>/dev/null || true; done
  for _ in {1..40}; do
    [[ -z "$(arm_orphan_pids)" ]] && break
    sleep 0.5
  done
  if [[ -n "$(arm_orphan_pids)" ]]; then
    echo "  ignored SIGINT; escalating." >&2
    for pid in $(arm_orphan_pids); do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 3
    for pid in $(arm_orphan_pids); do kill -KILL "$pid" 2>/dev/null || true; done
    sleep 1
  fi
  if [[ -n "$(arm_orphan_pids)" ]]; then
    echo "Could not stop the orphaned arm nodes: $(arm_orphan_pids)" >&2
    exit 1
  fi
  echo "  stopped."
  # The arms need a moment to drop the previous FCI session.
  sleep 3
fi

# -------------------------------------------------------------------- lifecycle
launch_pids=()

stop_all() {
  trap - EXIT INT TERM
  for pid in "${launch_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  # The launch leader can exit before ros2_control children; test the group, not the PID.
  for _ in {1..60}; do
    local alive=false
    for pid in "${launch_pids[@]:-}"; do
      [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null && alive=true
    done
    $alive || break
    sleep 0.25
  done
  for pid in "${launch_pids[@]:-}"; do
    [[ -n "$pid" ]] && kill -0 -- "-$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap stop_all EXIT INT TERM

start_stage() {
  local name="$1"; shift
  echo "Starting $name..."
  setsid "$@" &
  launch_pids+=($!)
  sleep 2
  if ! kill -0 "${launch_pids[-1]}" 2>/dev/null; then
    echo "$name exited immediately." >&2
    return 1
  fi
}

# Controller state via the list_controllers SERVICE, not `ros2 control`. The CLI
# (ros2controlcli) is not guaranteed to be installed, and its output format varies
# between releases; the service is available wherever a controller_manager runs.
#
# A healthy controller_manager answers in well under a second, so a short per-call
# timeout keeps a wedged manager from burning the whole budget one 20 s call at a time.
cm_call_timeout="${TMR_CM_CALL_TIMEOUT:-5}"

controller_state() {
  local ns="$1" name="$2"
  timeout "$cm_call_timeout" ros2 service call "/$ns/controller_manager/list_controllers" \
      controller_manager_msgs/srv/ListControllers 2>/dev/null \
    | grep -o "name='$name', state='[a-z]*'" | grep -o "state='[a-z]*'" | cut -d"'" -f2
}

# Waits until the controller is loaded AND settled (active/inactive) - not merely until
# the controller_manager's services exist, which happens almost immediately.
# NEVER fatal: a detection failure must not tear down healthy arms via the EXIT trap.
wait_for_controller() {
  local ns="$1" name="$2" timeout="$3" state=""
  local deadline=$(( SECONDS + timeout ))
  while (( SECONDS < deadline )); do
    state="$(controller_state "$ns" "$name" || true)"
    case "$state" in
      active|inactive) echo "  ok: $ns/$name ($state)"; return 0 ;;
      unconfigured)
        # A spawner whose load_controller call timed out retries, hits "already loaded"
        # and dies FATAL, leaving the controller loaded but unconfigured. Configuring is
        # a state transition only: it claims no command interfaces and moves nothing.
        echo "  $ns/$name is unconfigured (spawner race); configuring it."
        timeout "$cm_call_timeout" ros2 service call "/$ns/controller_manager/configure_controller" \
          controller_manager_msgs/srv/ConfigureController "{name: '$name'}" >/dev/null 2>&1 || true
        ;;
    esac
    sleep 3
  done
  echo "  WARNING: $ns/$name did not settle within ${timeout}s (last state: ${state:-unknown})." >&2
  echo "           Continuing anyway - check with: ros2 control list_controllers -c /$ns/controller_manager" >&2
  return 0
}

# wait_for <kind> <pattern> <timeout_s> <description>
wait_for() {
  local kind="$1" pattern="$2" timeout="$3" what="$4"
  local deadline=$(( SECONDS + timeout ))
  while (( SECONDS < deadline )); do
    if ros2 "$kind" list 2>/dev/null | grep -q -- "$pattern"; then
      echo "  ok: $what"
      return 0
    fi
    sleep 1
  done
  # Non-fatal by design: under `set -e` a non-zero return here would fire the EXIT trap
  # and tear down arms that are actually running fine, just slower than expected.
  echo "  WARNING: timed out waiting for $what ($kind matching '$pattern')." >&2
  echo "           Continuing anyway - verify by hand before relying on it." >&2
  return 0
}

# ------------------------------------------------------------------- 1. arms
start_stage "both arms" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
    robot_config_file:=tmr_duo_config.yaml

# joint_impedance_controller is the one GELLO teleop activates. It spawns --inactive by
# design, so "inactive" here is success, not a failure.
#
# The budget only has to cover the spawners: on healthy arms they load and configure
# within a few seconds, so 30 s is generous. Raise with TMR_CM_SETTLE_TIMEOUT if a
# spawner ever legitimately needs longer.
cm_settle_timeout="${TMR_CM_SETTLE_TIMEOUT:-30}"
wait_for_controller left  joint_impedance_controller "$cm_settle_timeout"
wait_for_controller right joint_impedance_controller "$cm_settle_timeout"

# franka_robot_state_broadcaster asks for 'fr3/robot_state' while this robot exports
# 'left_fr3v2/robot_state' / 'right_fr3v2/robot_state' - its arm_id is not being set from
# tmr_duo_config.yaml's arm_id (fr3v2), and its param file is missing the /left and
# /right namespaces. It therefore stays inactive and its topics never publish.
# Teleoperation does not use them: joint_impedance_controller reads ros2_control state
# interfaces directly. Reported, not repaired.
for side in left right; do
  if [[ "$(controller_state "$side" franka_robot_state_broadcaster)" != "active" ]]; then
    echo "  note: $side/franka_robot_state_broadcaster not active (arm_id mismatch); harmless for teleop."
  fi
done

# --------------------------------------------------------------- 2. grippers
if $skip_grippers; then
  echo
  echo "Both arms are up, INACTIVE (--skip-grippers)."
else
  # Let the arm managers go quiet before the gripper launch adds two more
  # controller_managers and six spawners.
  sleep 5

  start_stage "both grippers" \
    ros2 launch franka_gripper_manager robotiq_gripper_controller_client.launch.py \
      config_file:=tmr_duo_config_robotiq.yaml
  wait_for topic '/left/gripper/gripper_client/target_gripper_width_percent' 60 "left gripper client"
  wait_for topic '/right/gripper/gripper_client/target_gripper_width_percent' 60 "right gripper client"
fi

cat <<'MSG'

Upper body is up: arms (INACTIVE) and grippers. The base and spine are NOT running.

Nothing will move yet. On the laptop run ./start_gello.bash, then, one arm at a time:

  1. Send the arm to the recorded home pose (configs/teleop_home_pose.yaml, RUNBOOK.md S5).
     The GELLO->arm map is a DELTA from the pose captured at activation, so activating
     away from home offsets the whole correspondence.
  2. Hands OFF the GELLOs, then:
       ros2 control set_controller_state joint_impedance_controller active -c /left/controller_manager
       ros2 control set_controller_state joint_impedance_controller active -c /right/controller_manager

Press Ctrl+C to stop the arms and grippers. Shut the LAPTOP side down first, then this.
MSG

set +e
wait -n "${launch_pids[@]}"
status=$?
set -e
(( status == 0 )) && status=1
echo "An upper-body launch process exited (status $status); stopping the rest." >&2
exit "$status"
