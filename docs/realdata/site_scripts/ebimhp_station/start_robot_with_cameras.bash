#!/usr/bin/env bash
# Start ALL robot-side stacks for TMR teleoperation, in the required order:
#   base -> spine -> arms -> grippers
#
# Run this ON THE ROBOT after a reboot. The laptop half is ./start_teleop.bash.
#
# Deliberately does NOT move anything: the arm impedance controllers spawn inactive,
# and sending the arms to the home pose stays a manual, supervised step.
#
# start_robot_with_cameras.bash - derived from start_robot.bash on 2026-09-01. ONLY
# difference: start_sensors() also brings up all three cameras (both wrist D405s + the head
# ZED) over SSH on the station, since they now run there natively rather than on companion -
# see that function for why. Everything else is identical to start_robot.bash; keep the two
# in sync by hand if start_robot.bash changes.

set -Eeuo pipefail

# Absolute path to this script, for the purged re-exec below.
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

ws_dir="${TMR_WS:-$HOME/tams_ws}"
spine_ip="${TMR_SPINE_IP:-172.16.16.10}"
# Laptop and robot MUST share a domain. Nothing in the robot's rc files sets one, so 0 is
# what every ROS 2 process here lands on unless something overrides it - and 0 is what the
# rest of the system already assumes: labs_integration/tmr_station/docker-compose.yml,
# fastdds_labs.xml, configs/fastdds_laptop_discovery.xml, .devcontainer/docker-compose.yml
# and the Olive sensors are all on 0. Keep this in step with configs/tmr_laptop_env.sh.
export ROS_DOMAIN_ID="${TMR_ROS_DOMAIN_ID:-0}"

# Keep every byte of DDS on WiFi so the wired 172.16.16.x link carries ONLY the three
# 1 kHz Franka FCI streams. Sharing it is what aborted the base and BOTH arms with
# ["communication_constraints_violation"] on 2026-08-23, within 10 ms of each other.
# See ~/fastdds_wifi.xml for the full reasoning.
#
# Escape hatch: TMR_DDS_PROFILE="" falls back to default all-interface discovery,
# TMR_DDS_PROFILE=/path/to.xml uses another profile.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
# DEFAULT IS EMPTY: default all-interface discovery, which is what actually works.
#
# ~/fastdds_wifi.xml is OPT-IN and currently BROKEN: its initialPeersList destroys the
# robot's own LOCAL discovery, because unicast initial peers probe only a small range of
# participant indices per address and this robot runs well over a dozen participants. The
# symptom is spawners unable to contact their own controller_manager:
#   [spawner-3] Could not contact service /left/gripper/controller_manager/list_controllers
#
# Reserving the wired link for FCI is still the right idea - see docs/RUNBOOK.md S9 - but
# do not enable this until the profile is fixed and tested on throwaway nodes:
#   TMR_DDS_PROFILE=$HOME/fastdds_wifi.xml ~/start_robot.bash --restart
# UDP-ONLY IS THE DEFAULT since 2026-08-24. Fast DDS prefers shared memory between
# same-host participants, and the SHM transport on this machine repeatedly fails:
#   [RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_portNNNN: open_and_lock_file failed
# When it does, EVERY controller_manager becomes unreachable while the stacks still look
# alive - recover.py reports "controller_manager not reachable" for both arms and the base
# and only the spine (HTTP, not DDS) answers. Clearing /dev/shm/fastrtps_* fixes it until
# the segments accumulate again. Dropping the SHM transport removes the failure mode;
# localhost then uses UDP, the same path that already works cross-host.
#
#   TMR_DDS_PROFILE=none        -> old behaviour (default transports, SHM + UDP)
#   TMR_DDS_PROFILE=<path>      -> some other profile
export FASTRTPS_DEFAULT_PROFILES_FILE="${TMR_DDS_PROFILE-$HOME/fastdds_udp_only.xml}"
if [ "$FASTRTPS_DEFAULT_PROFILES_FILE" = none ] || [ -z "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  echo "DDS: default transports (SHM + UDP) - SHM has failed on this machine before."
elif [ ! -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
  echo "ERROR: DDS profile not found: $FASTRTPS_DEFAULT_PROFILES_FILE" >&2
  exit 1
else
  # The label used to say "WiFi only" because the only profile that ever existed was
  # fastdds_wifi.xml. The default is now fastdds_udp_only.xml, which restricts the
  # TRANSPORT (no SHM) and not the interfaces - describing it as WiFi-only is wrong
  # and would send someone hunting a link problem that does not exist.
  echo "DDS: $FASTRTPS_DEFAULT_PROFILES_FILE"
fi
# The ros2 CLI daemon caches DDS settings, so a stale one would keep using the old
# transports and report a graph that does not match what the nodes actually see.
ros2 daemon stop >/dev/null 2>&1 || true

skip_arms=false
home_pose=true
sensors=true
activate=true
home_file="${TMR_HOME_POSE:-$HOME/teleop_home_pose.yaml}"
restart=false
for arg in "$@"; do
  case "$arg" in
    --skip-arms) skip_arms=true ;;
    # Stop any already-running robot stacks instead of refusing to start.
    --restart)   restart=true ;;
    --no-home)   home_pose=false ;;
    --no-sensors) sensors=false ;;
    --no-activate) activate=false ;;
    *) echo "Usage: $0 [--skip-arms] [--restart] [--no-home] [--no-activate] [--no-sensors]" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- environment
# NOTE: local_setup.bash, never setup.bash. setup.bash chains to the workspace's
# recorded parent (~/ros2_ws, a ~4-month-stale franka stack) and re-injects it as an
# overlay; colcon prepends only-if-absent, so once it is in front it stays in front
# and franka_bringup resolves to the wrong workspace.
set +u
source /opt/ros/humble/setup.bash
source "$ws_dir/install/local_setup.bash"
set -u

# An overlay already on AMENT_PREFIX_PATH stays in front, because colcon's setup files
# prepend only-if-absent - so a shell that once sourced ~/ros2_ws keeps resolving
# franka_bringup there no matter what this script sources. The old advice was "open a
# genuinely new terminal", which is only a way of purging the inherited ROS variables.
# Do that here instead, and retry ONCE. Failing again means franka_bringup really is not
# in $ws_dir, which is a different problem and says so.
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
  echo "So this is not an overlay-ordering problem. Check:" >&2
  echo "  ls -d $ws_dir/install/franka_bringup" >&2
  echo "  ros2 pkg prefix franka_bringup" >&2
  exit 1
fi

# rviz2 is part of the base launch and aborts (SIGABRT) with no X display, e.g. over
# plain ssh. Headless keeps it from dying noisily; it does not affect teleop either way.
if [[ -z "${DISPLAY:-}" ]]; then
  export QT_QPA_PLATFORM=offscreen
fi

# ------------------------------------------------------------- already running?
# A leftover bringup is not harmless: its controller_manager answers the new launch's
# spawners, which then see "Controller already loaded" and fail to configure. That is
# exactly how joint_state_broadcaster died on 2026-08-20.
stale_pattern='ros2_control_node|tmrv0_2.launch|spine.launch|franka_fr3_arm_controllers.launch|robotiq_gripper_controller_client.launch'

# The stale_pattern above matches LAUNCH processes and ros2_control_node, but the NODES
# those launches spawn have their own executable names and match none of it. Kill a launch
# (or lose its terminal) and those nodes are reparented to init, where they survive every
# later --restart. Observed 2026-08-23: six orphaned robotiq_gripper_client and one
# franka_spine_server, the oldest 8 h old, plus five `timeout ... ros2 service call` pairs
# leaked by wait_for_controller below.
#
# They are not just idle CPU. A second spine_action_server serves the SAME action name,
# and every orphaned gripper client subscribes to the same command topics and contends for
# the gripper serial ports. The load they add is also what starves the base's 1 kHz FCI
# loop into a communication_constraints_violation reflex.
#
# Deliberately narrow: PPID 1 (genuinely orphaned) AND owned by this user AND matching a
# node this script starts. A process whose launch parent is still alive is never touched.
#
# tmr_health.py belongs here for the same reason, added 2026-08-27. It is spawned with
# `setsid` below, so it outlives the script and is reparented to init - and it matched
# NEITHER pattern, so every --restart left another one behind. Observed that day: 31
# daemons, the oldest 14 h old, driving the companion to load 30 on 12 cores. Each one
# polls both controller_managers over service calls, so they are exactly the "too little
# headroom" that shows up as communication_constraints_violation and made a bringup fail
# with the arm stacks never coming up.
#
# Safe to sweep even though the run spawns one: sweep_orphans only kills PPID 1, and the
# daemon this run starts keeps the live script as its parent until the script exits. The
# sweep also runs long before that spawn.
orphan_pattern='tams_ws/install/(franka_gripper_manager|franka_spine_server)|robotiq_gripper_client|spine_action_server|/(robot_state_publisher|joint_state_publisher|rviz2)|ros2 service call .*controller_manager|tmr_health\.py|controller_manager[/ ]spawner'

list_orphans() {
  # `|| true` throughout: grep exits 1 on no-match and `set -o pipefail` would abort.
  ps -eo pid,ppid,user:24,args --no-headers 2>/dev/null \
    | awk -v me="$(id -un)" '$2==1 && $3==me' \
    | grep -E "$orphan_pattern" \
    | awk '{print $1}' || true
}

sweep_orphans() {
  local pids sig p
  for sig in INT TERM KILL; do
    pids="$(list_orphans)"
    [ -z "$pids" ] && return 0
    echo "  orphaned nodes, sending SIG$sig: $(echo $pids | tr '\n' ' ')"
    for p in $pids; do kill "-$sig" "$p" 2>/dev/null || true; done
    # Reaping is not instant; the first sweep of 2026-08-23 looked like a failure purely
    # because it re-checked after 2 s.
    for _ in $(seq 1 16); do
      [ -z "$(list_orphans)" ] && break
      sleep 0.5
    done
  done
  pids="$(list_orphans)"
  [ -n "$pids" ] && echo "  WARNING: orphans survived SIGKILL: $(echo $pids | tr '\n' ' ')" >&2
  return 0
}
existing="$(pgrep -af "$stale_pattern" || true)"
if [[ -n "$existing" ]]; then
  if ! $restart; then
    echo "Robot stacks are already running; refusing to start a second set:" >&2
    echo "$existing" >&2
    echo >&2
    echo "Stop them first (Ctrl+C in their terminal), or re-run with --restart to have" >&2
    echo "this script stop them for you." >&2
    exit 1
  fi

  echo "Stopping the running robot stacks (--restart)..."
  echo "$existing" >&2
  # SIGINT is what ros2 launch expects: it shuts its own children down in order.
  # Signal launch processes first so they can clean up their ros2_control_node.
  for pid in $(pgrep -f 'tmrv0_2.launch|spine.launch|franka_fr3_arm_controllers.launch|robotiq_gripper_controller_client.launch' || true); do
    kill -INT "$pid" 2>/dev/null || true
  done
  for _ in {1..60}; do
    pgrep -f "$stale_pattern" >/dev/null 2>&1 || break
    sleep 0.5
  done
  # Anything still holding on after 30 s is an orphan (typically a ros2_control_node
  # whose launch parent already exited). It owns the FCI/base connection, so the new
  # bringup cannot start until it is gone.
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "  some processes ignored SIGINT; escalating to SIGTERM." >&2
    pkill -TERM -f "$stale_pattern" 2>/dev/null || true
    for _ in {1..30}; do
      pgrep -f "$stale_pattern" >/dev/null 2>&1 || break
      sleep 0.5
    done
  fi
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "  some orphaned processes ignored SIGTERM; escalating to SIGKILL." >&2
    pkill -KILL -f "$stale_pattern" 2>/dev/null || true
    for _ in {1..20}; do
      pgrep -f "$stale_pattern" >/dev/null 2>&1 || break
      sleep 0.25
    done
  fi
  if pgrep -f "$stale_pattern" >/dev/null 2>&1; then
    echo "Could not stop the existing robot stacks even with SIGKILL:" >&2
    pgrep -af "$stale_pattern" >&2
    exit 1
  fi
  # Reparented nodes the stale_pattern cannot see.
  sweep_orphans
  echo "  stopped."
  # The hardware needs a moment to drop the previous FCI/base session.
  sleep 3
elif $restart; then
  # --restart must reset EVERY leftover, not only those stale_pattern can see. Once a
  # previous bringup has died on its own (or been killed), `existing` is empty and the
  # whole block above is skipped - yet reparented orphans survive, and leaked tmr_health
  # daemons accumulate exactly in that state, one per --restart. Without this the machine
  # keeps the load that made the previous bringup fail, so the retry fails the same way.
  echo "No running stacks; sweeping leftover orphans (--restart)..."
  sweep_orphans
fi

# -------------------------------------------------------------------- lifecycle
launch_pids=()
launch_names=()
# Sensors go here instead of launch_pids. They are stopped on exit like everything else, but
# their liveness is NOT monitored: on 2026-08-23 a ZED that could not find its camera exited,
# wait_any_launch saw the stage die, and the EXIT trap tore down the ENTIRE robot - base,
# arms, grippers and spine - because one camera was unplugged. A sensor must never do that.
aux_pids=()
aux_names=()

stop_all() {
  trap - EXIT INT TERM
  # Sensors first: they are pure publishers, and stopping them early quiets the graph while
  # the control stacks shut down.
  for pid in "${aux_pids[@]:-}"; do
    [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for pid in "${launch_pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  # The launch leader can exit before ros2_control children; test the process group, not only its PID.
  for _ in {1..60}; do
    local alive=false
    for pid in "${launch_pids[@]}"; do
      kill -0 -- "-$pid" 2>/dev/null && alive=true
    done
    $alive || break
    sleep 0.25
  done
  for pid in "${launch_pids[@]}"; do
    kill -0 -- "-$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap stop_all EXIT INT TERM

# Wait until any launch stage exits, and say WHICH one.
#
# `wait -n "${launch_pids[@]}"` fails with "no such job" once bash has reaped a pid it no
# longer tracks. Under `set -e` plus the EXIT trap that stops every stack, that failure
# tore the whole robot down right after a successful bringup on 2026-08-23, reporting only
# a bare pid. A stage really had died - the message just made it look like a bash bug.
wait_any_launch() {
  local i pid rc live
  while :; do
    live=()
    for i in "${!launch_pids[@]}"; do
      pid="${launch_pids[$i]}"
      if kill -0 "$pid" 2>/dev/null; then
        live+=("$pid")
      else
        echo "The '\''${launch_names[$i]}'\'' stage (pid $pid) is gone." >&2
        return 1
      fi
    done
    (( ${#live[@]} )) || { echo "No launch stages are running." >&2; return 1; }
    set +e
    wait -n "${live[@]}"
    rc=$?
    set -e
    # 127 = "no such job": the pid vanished between the check above and the wait. Loop and
    # re-check rather than reporting it as a stage failure.
    (( rc == 127 )) && { sleep 1; continue; }
    return "$rc"
  done
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
  # and tear down a robot that is actually running fine, just slower than expected.
  echo "  WARNING: timed out waiting for $what ($kind matching '$pattern')." >&2
  echo "           Continuing anyway - verify by hand before relying on it." >&2
  return 0
}

# Like start_stage, but the process is not monitored for liveness - see aux_pids.
start_aux_stage() {
  local name="$1"; shift
  echo "Starting $name..."
  setsid "$@" &
  aux_pids+=($!)
  aux_names+=("$name")
  sleep 2
  if ! kill -0 "${aux_pids[-1]}" 2>/dev/null; then
    echo "  WARNING: $name exited immediately; continuing without it." >&2
    return 0
  fi
}

start_stage() {
  local name="$1"; shift
  echo "Starting $name..."
  setsid "$@" &
  launch_pids+=($!)
  launch_names+=("$name")
  sleep 2
  if ! kill -0 "${launch_pids[-1]}" 2>/dev/null; then
    echo "$name exited immediately." >&2
    return 1
  fi
}

# ------------------------------------------------------- stages, as functions
# The base is started LAST when the arms are involved. Why: on 2026-08-23 the base came up
# healthy, ran fine for 14 s, and then died 2.4 s after the arm stacks activated their own
# two 1 kHz FCI loops:
#
#   t=875.9  swerve_drive_controller configured and activated
#   t=887.4  left+right FrankaHardwareInterface activate
#   t=890.1  BASE: libfranka Move command aborted by reflex
#            ["communication_constraints_violation"]
#
# ros2_control then demotes TmrHardware to `unconfigured`, its vx/vy/wz cartesian_velocity
# interfaces go [unavailable], and swerve_drive_controller never claims them - while still
# reporting `active`. The pedals publish correct cmd_vel and the base ignores every message
# with NOTHING logged on the laptop. Verify with ./base_health.sh: want [claimed].
#
# The reflex fires inside the arm bringup burst, not during steady-state teleop, so bring
# the base up after that burst has passed rather than making it survive it.
#
# Do NOT instead lower the base controller_manager update_rate (1000 Hz, in
# franka_ros2/franka_bringup/config/controllers.yaml): the base IS the 1 kHz consumer here,
# so slowing it makes the deadline misses worse, not better.
# ---------------------------------------------------------------- health daemon
# Started HERE, with the rest of the stack, and never mid-session. That is the whole point:
# every `ros2` CLI check creates a new DDS participant, and participant creation is a
# 15-25 s discovery burst that can make a live 1 kHz FCI loop miss its deadline - so
# polling for trouble during teleop CAUSES the trouble. This daemon polls over service
# calls on its own long-lived participant and writes a plain file, so checking costs no
# DDS at all:
#
#   cat /tmp/tmr_health.json     <- safe at any time, including mid-episode
#
# It also logs the moment a unit is lost or recovers, which is otherwise invisible: the
# controller keeps reporting `active` with healthy odom long after the hardware has gone.
if [ -f "$HOME/tmr_health.py" ]; then
  setsid python3 "$HOME/tmr_health.py" > /tmp/tmr_health.log 2>&1 < /dev/null &
  echo "  health daemon started; check with: cat /tmp/tmr_health.json"
fi

start_base() {
  start_stage "mobile base" \
    ros2 launch franka_bringup tmrv0_2.launch.py controller_name:=swerve_drive_controller
  wait_for topic '/swerve_drive_controller/odom' 30 "base odometry"

  # joint_state_broadcaster FIRST, deliberately. It is cosmetic (real wheel values into
  # /dynamic_joint_states), but spawning it is a controller SWITCH, and a switch is exactly
  # what faults the hardware (see below). Do it before the repair loop, so any damage it
  # causes is repaired rather than left behind.
  # BOTH calls below are bounded, added 2026-08-27. Neither was, and `spawner` does not
  # give up: with no controller_manager on / it polls list_controllers every 10 s FOREVER,
  # so this line never returns and `|| true` cannot save it - the failure is a hang, not a
  # non-zero exit. Observed that day after a mid-bringup Ctrl+C killed the launches:
  # start_robot.bash sat here for minutes, holding a spawner child, while every launch
  # below it was already dead and the log repeated
  #   [spawner_joint_state_broadcaster]: waiting for service /controller_manager/list_controllers
  # once per 10 s with nothing left alive that could ever answer it.
  # `ros2 control list_controllers` blocks on the same missing service, so it is bounded too.
  if ! timeout "${TMR_CM_QUERY_TIMEOUT:-15}" \
        ros2 control list_controllers 2>/dev/null | grep -q 'joint_state_broadcaster.*active'; then
    echo "  joint_state_broadcaster inactive; spawning it."
    # -k: the `ros2 run` wrapper does not always pass SIGTERM to the spawner it exec's, so
    # follow up with SIGKILL. Any child that still escapes is caught by the orphan sweep on
    # the next --restart, which is why the spawner is in orphan_pattern.
    timeout -k 5 "${TMR_SPAWNER_TIMEOUT:-40}" \
      ros2 run controller_manager spawner joint_state_broadcaster -c /controller_manager \
      || echo "  joint_state_broadcaster did not spawn (timeout or error); continuing." >&2
  fi

  # --------------------------------------------------- make the base COMMANDABLE
  # This bringup faults itself. Measured on 2026-08-24, from this script's own launch:
  #
  #   t=201.4  Successful 'activate' of hardware 'TmrHardware'
  #   t=207.5  Loading controller 'swerve_drive_controller'
  #   t=211.0  Loading controller 'joint_state_broadcaster'
  #   t=213.3  spawner died, exit code 1     <- TmrHardware is `unconfigured` from here
  #
  # A controller switch stalls the 1 kHz RT loop past libfranka's deadline, the robot
  # faults, and ros2_control demotes TmrHardware. Left ALONE the hardware holds `active`
  # indefinitely (verified 60 s untouched), so this is a startup race, not decay - which
  # is why a one-shot repair here is enough and no watchdog is needed.
  #
  # NEVER trust list_controllers for this: swerve_drive_controller keeps reporting
  # `active` and /swerve_drive_controller/odom keeps publishing a healthy 50 Hz while the
  # base ignores every command. The only honest signal is whether vx/cartesian_velocity is
  # [claimed].
  #
  # ORDER: hardware first, THEN cycle the controller. Activating the controller alone does
  # nothing - it is already `active`, so on_activate() never re-runs and never re-binds.
  # Repair from ONE DDS participant, via recover.py --base. This must NOT be done with
  # `ros2 service call` / `ros2 control`: each invocation creates a NEW DDS participant,
  # participant creation is a 15-25 s discovery burst on this network, and such a burst is
  # exactly what makes the 1 kHz FCI loop miss its deadline. An earlier version of this
  # block looped three `ros2 service call` repairs and made things strictly worse - it
  # fired a dozen participants at a control loop that was already failing.
  #
  # recover.py --base touches nothing but the base, so it is safe to run here even while
  # GELLO is publishing: it never activates an arm controller.
  # Report from recover.py's EXIT STATUS, never from `ros2 control`. Under bringup load
  # that CLI returns empty output (or a traceback), so a grep-based check reads "not
  # claimed" on a base that is perfectly healthy - it printed exactly that false warning
  # on 2026-08-24 while recover.py reported hardware=active claimed=6 in the same second.
  recover_py="${TMR_RECOVER:-$HOME/recover.py}"
  if [ ! -f "$recover_py" ]; then
    echo "  WARNING: $recover_py not found; cannot verify or repair the base." >&2
    echo "           Check by hand:  python3 recover.py --check" >&2
  elif python3 "$recover_py" --base; then
    echo "  ok: base commandable"
  else
    echo "  WARNING: the base is NOT commandable." >&2
    echo "           Pedals will publish correct cmd_vel and the base will ignore it," >&2
    echo "           silently. Re-check with:  python3 $recover_py --check" >&2
    echo "           If the hardware is stuck, open Desk (https://172.16.16.10/) and" >&2
    echo "           confirm the base is powered on." >&2
  fi
}

# Drive both arms to the agreed teleop home pose (configs/teleop_home_pose.yaml on the
# laptop, copied to $home_file here). Skip with --no-home.
#
# THIS MOVES THE ARMS. The rest of this script deliberately moves nothing, so the countdown
# below is the chance to Ctrl+C if the workspace is not clear.
#
# Why it belongs here, BEFORE start_base: homing puts each arm's FCI into Move mode, and
# that traffic is exactly what has been tripping the base's own 1 kHz loop into a
# communication_constraints_violation. Homing while the base is still down costs nothing.
#
# joint_impedance_controller holds the command interfaces, so it must be inactive first; it
# spawns inactive, so this is normally a no-op and matters only on a re-run.
#
# One arm at a time, per the README: confirm the left/right mapping before two 7-DOF arms
# share a workspace.
home_arms() {
  # Delegates to ~/home_arms.py: ONE DDS participant for both arms.
  #
  # This used to shell out to `ros2 action send_goal` once per arm. Each call creates a new
  # participant, and that discovery burst repeatedly destroyed the other arm's goal
  # response ("Failed to send goal response ... client will not receive response") on
  # 2026-08-23. Same failure as two `ros2 control` calls killing the first arm; same fix.
  #
  # It also made Ctrl+C useless: a hung `timeout 120` call swallowed the interrupt and the
  # bash retry loop just moved to the next attempt. One python process is interruptible.
  local script="${TMR_HOME_SCRIPT:-$HOME/home_arms.py}"
  if [ ! -f "$script" ]; then
    echo "  WARNING: $script not found; skipping homing." >&2
    return 0
  fi
  if [ ! -f "$home_file" ]; then
    echo "  WARNING: home pose file not found ($home_file); skipping homing." >&2
    return 0
  fi

  echo
  echo "=============================================================="
  echo " ABOUT TO MOVE BOTH ARMS to the teleop home pose."
  echo " Clear the workspace; hands off the arms and the GELLOs."
  echo " Ctrl+C now to skip (--no-home disables this permanently)."
  echo "=============================================================="
  for i in 3 2 1; do printf "\r  starting in %ss... " "$i"; sleep 1; done
  echo

  # Bounded so a wedged action server can never freeze the bringup: worst case is roughly
  # discovery (15 s) + two arms * (accept 10 s + motion 30 s).
  timeout 150 python3 "$script" --file "$home_file" || {
    echo "  WARNING: homing did not complete cleanly; verify both arm poses by hand" >&2
    echo "           before activating impedance control." >&2
  }
  echo
}

# ZED head camera + both SICK nanoScan2 lidars.
#
# Deliberately FIRST, before any FCI loop is running. Bringing a driver up is a 15-25 s DDS
# discovery burst on this network, and such a burst aborts control loops that are already
# running - that is what has been tripping communication_constraints_violation all along.
# With the sensors up front, their discovery is finished before the arms or base exist.
#
# Bandwidth is fine despite the ZED being uncompressed: head_camera_zed_params.yaml sets
# pub_downscale_factor 2.0 and pub_frame_rate 15 with depth off, so it is ~640x360x15,
# roughly 10 MB/s - not the ~80 MB/s an untuned HD720@30 stream would be.
#
# start_cameras:=false because default_sensor_suite.yaml declares four D455s and only three
# are plugged in; the camera launch would fail on the missing one.
#
# NEW IN THIS COPY (start_robot_with_cameras.bash): all three cameras - both wrist D405s AND
# the head ZED - now run natively (pixi, no Docker) on the STATION, not companion; see
# station/start_cameras.bash. The old local "$HOME/start_zed.bash" file this function used to
# fall back to no longer exists on companion, so that branch always warned and skipped the
# head camera. Replaced with an SSH call to the station instead, using the SAME
# start_aux_stage/aux_pids contract as everything else in this function: a camera failing to
# start must never take down the base/spine/arms bringup (see the aux_pids comment above).
#
# `ssh -tt` (not -t): this runs inside `setsid ... &`, which has no local controlling
# terminal, so a single -t would refuse to allocate one. -tt forces it unconditionally, which
# is what lets killing the local ssh process (in stop_all(), via the aux_pids process-group
# kill) actually propagate as a hangup to the remote station side and stop
# start_cameras.bash's own children instead of leaving them running detached.
#
# Needs companion -> station passwordless SSH (the `station` host alias in ~/.ssh/config) and
# the station's ~/teleoperation checkout - both set up 2026-09-01 alongside this script.
start_sensors() {
  start_aux_stage "cameras (station: both wrists + ZED)" \
    ssh -tt station 'cd ~/teleoperation/station && exec ~/.pixi/bin/pixi run cameras'
  # Topic names verified against the native pixi camera pipeline (2026-08-30/31), NOT the
  # retired Docker build's names - see station/record_bag.bash and camera_viewer.py for the
  # same image_rect_raw vs image_raw distinction.
  wait_for topic '/head_camera/zed_node/rgb/color/rect/image' 40 "ZED rgb"
  wait_for topic '/wrist_camera_left/camera/color/image_rect_raw' 40 "left wrist camera"
  wait_for topic '/wrist_camera_right/camera/color/image_rect_raw' 40 "right wrist camera"

  start_aux_stage "base lidars" \
    ros2 launch franka_mobile_sensors franka_mobile_sensors.launch.py \
      start_cameras:=false start_lidars:=true start_rviz:=false
  # Topic names come from the namespaces in default_sensor_suite.yaml. wait_for is
  # non-fatal, so a wrong guess warns rather than tearing the robot down.
  wait_for topic '/lidar_front/scan' 40 "front lidar"
  wait_for topic '/lidar_rear/scan' 40 "rear lidar"
}

start_spine() {
  start_stage "spine" \
    ros2 launch franka_spine_server spine.launch.py spine_ip:="$spine_ip"
  wait_for service '/franka_spine_node/get_state' 30 "spine services"
}

if $skip_arms; then
  # Unchanged ordering, deliberately: with no arm stacks there is no bringup storm for the
  # base to survive, and base-then-spine is the path verified working on 2026-08-23.
  start_base
  start_spine
  echo
  echo "Base and spine are up (--skip-arms). Pedal teleoperation is ready."
  echo "Press Ctrl+C to stop everything."
  wait_any_launch
  exit $?
fi

# ---------------------------------------------------------------- 1. sensors
if $sensors; then
  start_sensors
fi

# ------------------------------------------------------------------ 2. spine
start_spine

# Controller state via the list_controllers SERVICE, not `ros2 control`. The CLI
# (ros2controlcli) is not guaranteed to be installed, and its output format varies
# between releases; the service is available wherever a controller_manager runs.
#
# A healthy controller_manager answers list_controllers in well under a second, so a
# short per-call timeout costs nothing and keeps a wedged manager from burning the whole
# budget one 20 s call at a time. Override with TMR_CM_CALL_TIMEOUT if a slow companion
# ever needs more.
cm_call_timeout="${TMR_CM_CALL_TIMEOUT:-3}"

controller_state() {
  local ns="$1" name="$2"
  timeout -k 2 "$cm_call_timeout" ros2 service call "/$ns/controller_manager/list_controllers" \
      controller_manager_msgs/srv/ListControllers 2>/dev/null \
    | grep -o "name='$name', state='[a-z]*'" | grep -o "state='[a-z]*'" | cut -d"'" -f2
}

# Waits until the controller is loaded AND settled (active/inactive) - not merely until
# the controller_manager's services exist, which happens almost immediately.
# NEVER fatal: a detection failure must not tear down a healthy robot via the EXIT trap.
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
        timeout -k 2 "$cm_call_timeout" ros2 service call "/$ns/controller_manager/configure_controller" \
          controller_manager_msgs/srv/ConfigureController "{name: '$name'}" >/dev/null 2>&1 || true
        ;;
    esac
    sleep 3
  done
  echo "  WARNING: $ns/$name did not settle within ${timeout}s (last state: ${state:-unknown})." >&2
  echo "           Continuing anyway - check with: ros2 control list_controllers -c /$ns/controller_manager" >&2
  return 0
}

# ------------------------------------------------------------------- 3. arms
start_stage "both arms" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
    robot_config_file:=tmr_duo_config.yaml

# joint_impedance_controller is the one teleop activates. It spawns --inactive by
# design, so "inactive" here is success, not a failure.
#
# The budget only has to cover the spawners: on a healthy robot they load and configure
# within a few seconds, so 30 s is generous. It used to be 90 s per arm, which is 3
# minutes of dead wait whenever the manager is unresponsive - and the outcome in that
# case is a warning either way, so the extra minutes buy nothing. Raise with
# TMR_CM_SETTLE_TIMEOUT if a spawner ever legitimately needs longer.
cm_settle_timeout="${TMR_CM_SETTLE_TIMEOUT:-15}"
wait_for_controller left  joint_impedance_controller "$cm_settle_timeout"
wait_for_controller right joint_impedance_controller "$cm_settle_timeout"

# franka_robot_state_broadcaster asks for 'fr3/robot_state' while this robot exports
# 'left_fr3v2/robot_state' / 'right_fr3v2/robot_state' - its arm_id is not being set
# from tmr_duo_config.yaml's arm_id (fr3v2), and its param file is missing the /left
# and /right namespaces. It therefore stays inactive and its
# /<side>/franka_robot_state_broadcaster/* topics never publish. Teleoperation does not
# use them: joint_impedance_controller reads ros2_control state interfaces directly.
# Reported, not repaired - the fix belongs in the robot's launch config, not here.
# franka_robot_state_broadcaster publishes 6 of the 28 topics a LeRobot episode needs -
# 20 of the 62 state dimensions. Harmless to LOSE for teleop (joint_impedance_controller
# reads ros2_control state interfaces directly), fatal for recording.
#
# It used to fail on an arm_id mismatch, asking for 'fr3/robot_state' while the hardware
# exports 'left_fr3v2/robot_state'. That is FIXED in franka.launch.py (robot_type +
# arm_prefix, written at the controller-name level) - do not go looking there again.
#
# What actually fails now is the spawner: the logs show 'Configuring controller
# franka_robot_state_broadcaster' and then no 'Activating' and no error, with the spawner
# dead at exit 1. It times out against a controller_manager that is too slow to answer
# during bringup - the same timeout behind every 'failed to send response to
# list_controllers' warning in this stage. The controller is left INACTIVE, publishing
# nothing.
#
# Activating it here is safe: it is a broadcaster, so it claims only STATE interfaces and
# no command interfaces, and cannot move the arm.
for side in left right; do
  state="$(controller_state "$side" franka_robot_state_broadcaster || true)"
  if [[ "$state" == "active" ]]; then
    continue
  fi
  if [[ "$state" == "unconfigured" || -z "$state" ]]; then
    timeout "$cm_call_timeout" ros2 service call "/$side/controller_manager/configure_controller" \
      controller_manager_msgs/srv/ConfigureController "{name: 'franka_robot_state_broadcaster'}" \
      >/dev/null 2>&1 || true
    sleep 1
  fi
  echo "  $side/franka_robot_state_broadcaster is ${state:-unknown}; activating it (spawner race)."
  timeout "$cm_call_timeout" ros2 service call "/$side/controller_manager/switch_controller" \
    controller_manager_msgs/srv/SwitchController \
    "{activate_controllers: ['franka_robot_state_broadcaster'], strictness: 1}" \
    >/dev/null 2>&1 || true
  sleep 2
  state="$(controller_state "$side" franka_robot_state_broadcaster || true)"
  if [[ "$state" == "active" ]]; then
    echo "  ok: $side/franka_robot_state_broadcaster (active)"
  else
    echo "  WARNING: $side/franka_robot_state_broadcaster is '${state:-unknown}'." >&2
    echo "           Teleop is unaffected, but a recording made now loses 10 of the 62" >&2
    echo "           state dimensions for this arm and LABS will fail the episode." >&2
    echo "           Verify with:  python3 verify_topics.py" >&2
  fi
done

# Let the arm managers go quiet before the gripper launch adds two more
# controller_managers and six spawners.
sleep 3

# --------------------------------------------------------------- 4. grippers
start_stage "both grippers" \
  ros2 launch franka_gripper_manager robotiq_gripper_controller_client.launch.py \
    config_file:=tmr_duo_config_robotiq.yaml
wait_for topic '/left/gripper/gripper_client/target_gripper_width_percent' 30 "left gripper client"
wait_for topic '/right/gripper/gripper_client/target_gripper_width_percent' 30 "right gripper client"

# ------------------------------------------------------------- 5. home the arms
if $home_pose; then
  home_arms
fi

# --------------------------------------------------------------- 6. base (LAST)
# Deliberately after the arms and grippers; see start_base above.
start_base

# ----------------------------------------------------------- 7. activate the arms
# Runs LAST, after the base, because that is the order proven to work by hand. Activation
# puts each arm's FCI into Move mode, and doing it before the base is up has repeatedly
# aborted whichever loop was already running.
#
# ~/activate_arms.py uses ONE DDS participant for both arms and discovers both services
# before switching either. Two separate `ros2 control` calls kill the first arm - the
# second call's discovery burst aborts it about 3 s in. Never replace this with two calls.
#
# THIS MAKES THE ARMS LIVE: they begin following the GELLOs immediately.
if $activate; then
  activate_script="${TMR_ACTIVATE_SCRIPT:-$HOME/activate_arms.py}"
  if [ ! -f "$activate_script" ]; then
    echo "  WARNING: $activate_script not found; activate the arms by hand." >&2
  else
    echo
    echo "=============================================================="
    echo " ABOUT TO ACTIVATE BOTH ARMS - they will follow the GELLOs."
    echo " Hands OFF the GELLOs: the GELLO/arm pose delta is captured at"
    echo " this instant, and movement now becomes an approach target."
    echo " Ctrl+C to skip (--no-activate disables this permanently)."
    echo "=============================================================="
    for i in 3 2 1; do printf "\r  activating in %ss... " "$i"; sleep 1; done
    echo
    # Bounded: discovery 15 s + two switch calls at 10 s, plus slack.
    timeout 60 python3 "$activate_script" || {
      echo "  WARNING: activation did not complete; run it by hand:" >&2
      echo "           python3 $activate_script" >&2
    }
  fi
fi

cat <<'MSG'

Robot side is up: base, spine, arms (INACTIVE), grippers.

Nothing will move yet. On the laptop run ./start_teleop.bash, then, one arm at a time:

  1. Send the arm to the recorded home pose (configs/teleop_home_pose.yaml, RUNBOOK.md S5).
     The GELLO->arm map is a DELTA from the pose captured at activation, so activating
     away from home offsets the whole correspondence.
  2. Hands OFF the GELLOs, then:
       ros2 control set_controller_state joint_impedance_controller active -c /left/controller_manager
       ros2 control set_controller_state joint_impedance_controller active -c /right/controller_manager

Press Ctrl+C to stop every robot stack. Stop THIS first, then the laptop.
MSG

set +e
wait_any_launch
status=$?
set -e
(( status == 0 )) && status=1
echo "A robot launch process exited (status $status); stopping the rest." >&2
exit "$status"
