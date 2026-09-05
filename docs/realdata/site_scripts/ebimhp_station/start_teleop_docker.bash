#!/usr/bin/env bash
# RETIRED as the default entry point (2026-08-30): ./start_teleop.bash is now the native
# pixi/FastDDS path (see pixi.toml, start_teleop.bash, start_gello.bash, start_pedal.bash,
# start_cameras.bash). Kept as a fallback / reference - the Docker+CycloneDDS containers
# this script manages were the source of repeated failures (stale /dev in a long-lived
# container, a hardcoded CycloneDDS interface address going stale on DHCP renewal) that
# motivated the move to a native environment.
#
# Start the LAPTOP half of TMR teleoperation: GELLO leaders + foot pedals.
# The robot half is ~/start_robot.bash ON THE ROBOT. Start the robot first.
#
# WHY THIS RUNS IN A CONTAINER
# ----------------------------
# This laptop runs ROS 2 Kilted; the robot runs Humble. ROS 2 does not support
# cross-distro communication: from the Kilted host, `ros2 node list` comes back EMPTY and
# every controller_manager service call times out, even though topics partially discover
# (differing type hashes / service ABI). The same calls succeed from the Humble image, so
# the laptop-side nodes run there, on the host network.
#
# Do NOT "fix" this by switching the laptop to CycloneDDS because it connects faster.
# The robot speaks Fast DDS; mixing vendors breaks services and actions, which is exactly
# what the spine bridge needs. Fast DDS taking 15-25 s to create a node on this machine is
# a slow start, not a failure - be patient rather than changing RMW.
#
# DEVICE PERMISSIONS
#   * GELLO leaders are /dev/ttyACM* (root:dialout 0660). A privileged container gets a
#     FRESH /dev, so the host ACL granting your user access does NOT carry over. That is
#     why the exec user is <uid>:20 - primary group dialout - and not <uid>:<gid>.
#   * The foot switches need configs/99-pcsensor-footswitch.rules (MODE 0666) installed.
#     The shipped rule pins ID_PATH to one machine's USB ports; pedal_state_publisher
#     autodetects by vendor:product, so only the 0666 mode actually matters.
#
# CLOCK SKEW IS THE #1 CAUSE OF "IT STARTED BUT NOTHING MOVES"
#   SwerveDriveController silently discards cmd_vel older than 0.5 s (no error, anywhere),
#   and JointImpedanceController rejects stale GELLO samples. This script warns; fix with
#   ./configs/sync_robot_clock.sh.
#
# USAGE
#   ./start_teleop.bash                       # restart both stacks, motion only
#   ./start_teleop.bash --record --task-id <uuid>
#   ./start_teleop.bash --no-pedals           # foot switches are on ANOTHER host;
#                                             # run the bridges here and let that
#                                             # host publish /pedal/state
#   ./start_teleop.bash --pedal-fg            # pedal stack in the FOREGROUND, so
#                                             # keyboard_state_publisher gets a TTY and
#                                             # 'm' / w,a,s,d,q,e work
#   ./start_teleop.bash --cameras              # also start the head (ZED) + wrist (2x
#                                             # RealSense) cameras, on this same domain 0 -
#                                             # plain `docker run`, no LABS/compose stack.
#                                             # New DDS participants risk the same
#                                             # discovery-burst hit as everything else below,
#                                             # so this stays opt-in rather than default.
#   ./start_teleop.bash -d                    # detach and return to the prompt
#   ./start_teleop.bash viewer [cmd]          # GUI tool in a throwaway container with X11.
#                                             # Default rqt_image_view; try `viewer rviz2`.
#                                             # rviz2 prints libGL/radeonsi errors and then
#                                             # falls back to software rendering - harmless.
#                                             # LIBGL_ALWAYS_SOFTWARE=1 skips the failed probes.
#   ./start_teleop.bash shell                 # a shell INSIDE the Humble container, with
#                                             # ROS + the workspace sourced. This is where
#                                             # ros2/rviz2/rqt run - the host is Kilted and
#                                             # cannot talk to the robot or to a Humble bag.
#   ./start_teleop.bash stop | status | logs [gello|pedal]
#
# By default it stays in the FOREGROUND following both logs; Ctrl+C stops both stacks.
set -Eeuo pipefail

IMAGE="${TMR_IMAGE:-teleoperation_devcontainer-gello-ros2:latest}"
CONTAINER="${TMR_CONTAINER:-gello-humble}"
GELLO_CFG="${TMR_GELLO_CFG:-franka_gello_duo.yaml}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

repo_host="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Mirror .devcontainer/docker-compose.yml, which mounts the repo's GRANDPARENT at
# /workspace. Deriving the container path keeps this working if the repo is renamed.
case "$repo_host" in
  "$HOME"/*) repo_ctr="/workspace/${repo_host#"$HOME"/}" ;;
  *) echo "ERROR: expected the repo under \$HOME ($HOME), got $repo_host" >&2; exit 1 ;;
esac

# Host directory used for raw rosbag output. A per-machine setting avoids baking a
# removable-disk path into the repository; TMR_BAG_ROOT remains an explicit override.
bag_root_config="${XDG_CONFIG_HOME:-$HOME/.config}/teleoperation/bag_root"
if [ -n "${TMR_BAG_ROOT:-}" ]; then
  BAG_ROOT="$TMR_BAG_ROOT"
elif [ -r "$bag_root_config" ]; then
  IFS= read -r BAG_ROOT < "$bag_root_config"
else
  BAG_ROOT="$HOME/teleop_bags"
fi
[ -n "$BAG_ROOT" ] || { echo "ERROR: empty bag root in $bag_root_config" >&2; exit 1; }

cmd=start; record=false; task_id=""; pedal_fg=false; log_which=""; detach=false; viewer_cmd=""
cameras=false
# TMR_LOCAL_PEDALS=false makes --no-pedals the default for a host that never has them.
local_pedals="${TMR_LOCAL_PEDALS:-true}"
while [ $# -gt 0 ]; do
  case "$1" in
    start|stop|status|shell) cmd="$1"; shift ;;
    viewer) cmd=viewer; shift; viewer_cmd="$*"; set -- ;;
    logs) cmd=logs; shift; case "${1:-}" in gello|pedal) log_which="$1"; shift ;; esac ;;
    --record)   record=true; shift ;;
    # The foot switches live on another host (the TAMS laptop). Run the bridges
    # here and let that host publish /pedal/state over DDS.
    --no-pedals) local_pedals=false; shift ;;
    --task-id)  task_id="$2"; record=true; shift 2 ;;
    --pedal-fg) pedal_fg=true; shift ;;
    --cameras)  cameras=true; shift ;;
    -d|--detach) detach=true; shift ;;
    -h|--help)  sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Keep every byte of DDS on WiFi so the robot's wired 172.16.16.x link carries ONLY its
# three 1 kHz Franka FCI streams. Sharing that link aborted the base and BOTH arms with
# ["communication_constraints_violation"] within 10 ms of each other on 2026-08-23, after
# which ros2_control demotes the hardware to `unconfigured` while the controllers still
# report `active` - GELLO and pedal commands then go nowhere, silently.
#
# The robot must run its matching ~/fastdds_wifi.xml (start_robot.bash exports it) or DDS
# still reaches the wired NIC from that side and the problem remains.
#
# Escape hatch: TMR_DDS_PROFILE="" for default all-interface discovery, or a path to
# another profile. It must live under $HOME to be visible inside the container.
# DEFAULT IS EMPTY: all-interface discovery, matching the robot side (fixed in 5a78c47).
#
# This used to default to configs/fastdds_laptop_wifi.xml, which pins DDS to the WiFi
# address ONLY. That silently partitioned the laptop: teleop on 192.168.50.117 could
# not see the wrist cameras, which pin to the wired 172.16.16.140 - so no single
# `ros2 bag record` could ever capture both. Reserving the wired link for FCI is still
# the right idea, but it needs every participant on a matching profile, which is not
# the case today. Opt in with TMR_DDS_PROFILE=<path> once that is sorted.
# DEFAULT: a whitelist profile rendered from the LIVE wired address (see
# render_dds_profile.sh). Every ROS peer is on the wired 172.16.16.0/24 now - the laptop
# pins to its own wired address, the Jetson discovers over the wire - so the "needs every
# participant on a matching profile" precondition above is finally met. Measured before
# this: 25-46 MB/s of DDS image traffic on WiFi during recording. Rendering at each start
# means DHCP churn cannot strand the profile with a stale address.
#   TMR_DDS_PROFILE=""       -> old behaviour (all-interface discovery)
#   TMR_DDS_PROFILE=<path>   -> some other profile
dds_host="${TMR_DDS_PROFILE-$(bash "$HOME/teleoperation/render_dds_profile.sh" 2>/dev/null || true)}"
dds_env=""
if [ -n "$dds_host" ]; then
  [ -f "$dds_host" ] || { echo "ERROR: DDS profile not found: $dds_host" >&2; exit 1; }
  case "$dds_host" in
    "$HOME"/*) dds_ctr="/workspace/${dds_host#"$HOME"/}" ;;
    *) echo "ERROR: DDS profile must be under \$HOME to be visible in the container" >&2; exit 1 ;;
  esac
  dds_env="&& export FASTRTPS_DEFAULT_PROFILES_FILE='$dds_ctr'"
fi

# Env prelude shared by every exec. local_setup vs setup matters less here than on the
# robot, but /opt/ros/humble/franka carries libfranka + franka_msgs and must come first.
prelude="source /opt/ros/humble/setup.bash \
  && source /opt/ros/humble/franka/setup.bash \
  && cd '$repo_ctr' && source install/setup.bash \
  && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_fastrtps_cpp PYTHONUNBUFFERED=1 $dds_env"

dex()  { docker exec -u "$(id -u):20" -e HOME=/tmp "$CONTAINER" bash -lc "$1"; }
dexd() { docker exec -d -u "$(id -u):20" -e HOME=/tmp "$CONTAINER" bash -lc "$1"; }

container_running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ]; }

# Reuses the already-built LABS camera images directly (plain `docker run`, no compose, no
# dependency on ~/labs/*) so cameras live entirely under this script's lifecycle, on the
# SAME domain as GELLO/pedals. Config files are this repo's own copy under configs/.
CAM_ZED_IMAGE="${TMR_ZED_IMAGE:-registry.localhost/labs/zed-camera:latest}"
CAM_RS_IMAGE="${TMR_RS_IMAGE:-registry.localhost/labs/realsense-camera:latest}"
CAM_ZED_NAME="zed-camera-head"
CAM_RS_NAME="realsense-camera-wrist"

# Renders the CycloneDDS interface pin from the LIVE wired address, same reasoning as
# render_dds_profile.sh: a hardcoded address goes stale on the next DHCP renewal and every
# CycloneDDS node then fails "does not match an available interface" - which is exactly
# what took down the ZED, both wrist cameras AND the GELLO publisher earlier today.
render_cyclonedds_profile() {
  local addr
  addr="$(ip -4 -o addr show up scope global 2>/dev/null \
          | awk '$4 ~ /^172\.16\.16\./ {sub(/\/.*/, "", $4); print $4; exit}')"
  [ -n "$addr" ] || addr="127.0.0.1"
  local out="$HOME/.tmr_cyclonedds_cameras.xml"
  cat > "$out" <<XML
<?xml version="1.0" encoding="UTF-8" ?>
<!-- RENDERED by start_teleop.bash - do not edit; edits are overwritten. -->
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="false" address="${addr}"/>
      </Interfaces>
      <MaxMessageSize>65500B</MaxMessageSize>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>10000</MaxAutoParticipantIndex>
    </Discovery>
    <Internal>
      <SocketReceiveBufferSize min="10MB" />
    </Internal>
  </Domain>
</CycloneDDS>
XML
  echo "$out"
}

start_cameras() {
  local dds_file cam_x11=()
  dds_file="$(render_cyclonedds_profile)"
  if [ -d /tmp/.X11-unix ] && [ -n "${DISPLAY:-}" ]; then
    cam_x11=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix)
    [ -f "$HOME/.Xauthority" ] && cam_x11+=(-v "$HOME/.Xauthority:/tmp/.Xauthority:ro" -e XAUTHORITY=/tmp/.Xauthority)
  fi

  echo "Starting head camera (ZED)..."
  docker run -d --name "$CAM_ZED_NAME" --privileged --network host --ipc=host --pid=host \
    -v /dev:/dev -v /etc/localtime:/etc/localtime:ro \
    -v "$dds_file:/workspace/cyclonedds.xml:ro" \
    -v "$repo_host/configs/config_zed_camera.yml:/workspace/config_zed_camera.yml:ro" \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///workspace/cyclonedds.xml \
    "$CAM_ZED_IMAGE" >/dev/null

  echo "Starting wrist cameras (RealSense x2)..."
  docker run -d --name "$CAM_RS_NAME" --privileged --network host --ipc=host --pid=host \
    --shm-size=1gb \
    -v /dev:/dev -v /etc/localtime:/etc/localtime:ro \
    "${cam_x11[@]}" \
    -v "$dds_file:/workspace/cyclonedds.xml:ro" \
    -v "$repo_host/configs/config_realsense_camera.yml:/workspace/config_realsense_camera.yml:ro" \
    -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///workspace/cyclonedds.xml \
    "$CAM_RS_IMAGE" >/dev/null
}

stop_stacks() {
  # Cameras are separate, single-purpose containers (not launches inside $CONTAINER), so
  # tear them down unconditionally rather than gating on the main container's state.
  # Recreating fresh on every start also sidesteps the stale-/dev bug a long-lived camera
  # container hit earlier today (a device replugged after container start stays invisible
  # to it - see 2026-08-29 GELLO serial-port incident).
  docker rm -f "$CAM_ZED_NAME" "$CAM_RS_NAME" >/dev/null 2>&1 || true

  container_running || return 0
  # SIGINT to the launch leaders first, so they shut their own children down in order.
  docker exec -u "$(id -u):20" "$CONTAINER" bash -lc '
    for p in $(pgrep -f "ros2 launch" 2>/dev/null); do kill -INT $p 2>/dev/null || true; done
    for _ in $(seq 1 20); do pgrep -f "ros2 launch" >/dev/null 2>&1 || break; sleep 0.5; done
    # Orphaned nodes keep the serial ports and evdev grabs, so the next start fails with
    # "Could not open port" / "no candidate device found". Escalate rather than leave them.
    pkill -KILL -f "gello_publisher|pedal_state_publisher|base_bridge|spine_bridge|mode_manager|labs_pedal_bridge|mobile_base_state_bridge|spine_state_publisher|keyboard_state_publisher" 2>/dev/null || true
    exit 0' >/dev/null 2>&1 || true
}

case "$cmd" in
  stop)
    stop_stacks; echo "Laptop teleop stacks stopped (container '$CONTAINER' left running)."; exit 0 ;;
  viewer)
    # A SEPARATE, throwaway container so GUI work never disturbs a running teleop stack, and
    # so it works even when the main container predates the X11 passthrough. --privileged +
    # /dev because rviz2 needs /dev/dri for GL; without it you get
    # "libGL error: glx: failed to create dri3 screen".
    [ -n "${DISPLAY:-}" ] || { echo "ERROR: no DISPLAY set - nothing to draw on." >&2; exit 1; }
    img="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null || echo "$IMAGE")"
    vcmd="${viewer_cmd:-ros2 run rqt_image_view rqt_image_view}"
    echo "Viewer container: $vcmd    (close the window to exit)"
    exec docker run --rm --network host --privileged \
      -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      ${XAUTH_ARGS:+$XAUTH_ARGS} \
      $([ -f "$HOME/.Xauthority" ] && echo "-v $HOME/.Xauthority:/tmp/.Xauthority:ro -e XAUTHORITY=/tmp/.Xauthority") \
      -v /dev:/dev -v "$HOME:/workspace" \
      -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
      ${LIBGL_ALWAYS_SOFTWARE:+-e LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE} \
      -u "$(id -u):20" -e HOME=/tmp "$img" bash -lc \
      "source /opt/ros/humble/setup.bash && cd '$repo_ctr' && source install/setup.bash 2>/dev/null; exec $vcmd"
    ;;
  shell)
    container_running || { echo "ERROR: container '$CONTAINER' is not running. Run ./start_teleop.bash first." >&2; exit 1; }
    if ! docker exec "$CONTAINER" bash -lc '[ -d /tmp/.X11-unix ]' 2>/dev/null; then
      echo "NOTE: this container has no X11 socket, so GUI tools (rviz2, rqt) cannot display." >&2
      echo "      Recreate it with:  ./start_teleop.bash stop && docker rm -f $CONTAINER && ./start_teleop.bash" >&2
      echo >&2
    fi
    exec docker exec -it -u "$(id -u):20" -e HOME=/tmp "$CONTAINER" bash -lc \
      "$prelude && echo 'Humble container. ros2/rviz2/rqt available; bags are under /workspace/teleop_bags.' && exec bash"
    ;;
  status)
    container_running || { echo "container '$CONTAINER': NOT running"; exit 1; }
    echo "container '$CONTAINER': running"
    # grep -v "bash -lc" drops this command's own wrapper shell, which contains the
    # pattern string and would otherwise always match.
    dex 'pgrep -af "gello_publisher|pedal_state_publisher|base_bridge|spine_bridge|mode_manager|labs_pedal_bridge" \
         | grep -v "bash -lc" | sed "s/ --ros-args.*//;s|.*/||" | sed "s/^/  /" \
         | grep . || echo "  (no teleop nodes running)"'
    for n in "$CAM_ZED_NAME" "$CAM_RS_NAME"; do
      if [ "$(docker inspect -f '{{.State.Running}}' "$n" 2>/dev/null)" = true ]; then
        echo "  $n: running"
      else
        echo "  $n: not running"
      fi
    done
    exit 0 ;;
  logs)
    case "$log_which" in
      gello) dex 'tail -n 40 -f /tmp/gello.log' ;;
      pedal) dex 'tail -n 40 -f /tmp/pedal.log' ;;
      *)     dex 'tail -n 20 /tmp/gello.log /tmp/pedal.log | cat -v | cut -c1-160' ;;
    esac
    exit 0 ;;
esac

# ------------------------------------------------------------------ container
# ALWAYS recreated, even if one is already running: a long-lived container's /dev view goes
# stale the moment a GELLO board (or anything else under /dev/serial/by-id) gets replugged or
# power-cycled after the container started - "Could not open port ... No supported devices
# were detected" even though the host sees it fine. Hit twice in two days (2026-08-29,
# 2026-08-30), fixed both times by a manual `docker restart`. Recreating fresh every start
# costs a few seconds (see the mcap/pip-deps checks below, which now always re-run) in
# exchange for never hitting that surprise silently.
echo "Recreating container '$CONTAINER' (always fresh, so a replugged GELLO/device is never stale)..."
bag_mount_args=()
if [ -d "$BAG_ROOT" ]; then
  bag_mount_args=(-v "$BAG_ROOT:/bags")
else
  echo "WARNING: bag root is unavailable: $BAG_ROOT" >&2
  echo "         Starting teleoperation without a /bags mount. Raw recording remains disabled" >&2
  echo "         until a disk is mounted or record_bag.bash --bag-root DIR is used." >&2
fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# privileged + host network: privileged for /dev (GELLO serial, evdev foot switches),
# host network because DDS discovery to the robot must not be NATed.
# X11 passthrough so rviz2 / rqt_image_view can display on the host. Harmless when there
# is no display (the vars are simply empty). GL works because this container is
# privileged with /dev mounted, so it reaches /dev/dri; without that, rviz2 connects to
# the X server and then fails at "libGL error: glx: failed to create dri3 screen".
x11_args=()
if [ -d /tmp/.X11-unix ] && [ -n "${DISPLAY:-}" ]; then
  x11_args=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix)
  [ -f "$HOME/.Xauthority" ] && x11_args+=(-v "$HOME/.Xauthority:/tmp/.Xauthority:ro" -e XAUTHORITY=/tmp/.Xauthority)
fi
# --ipc=host is REQUIRED. Fast DDS prefers shared memory between participants on the
# same host, and SHM segments live in /dev/shm, which is private per IPC namespace.
# With the default (private) namespace this container and the realsense camera
# container each get their own /dev/shm: discovery still succeeds over UDP, so
# `ros2 topic list` shows every camera topic, but NO DATA crosses and each image
# topic reads SILENT - including in record_bag.bash. Diagnosed 2026-08-24; the
# camera compose already declared `ipc: host`, this side did not.
docker run -d --name "$CONTAINER" --privileged --network host --ipc=host --init \
  "${x11_args[@]}" \
  -v "$HOME:/workspace" \
  "${bag_mount_args[@]}" \
  -v /dev/serial/by-id:/dev/serial/by-id \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  "$IMAGE" sleep infinity >/dev/null
# Container is brand new every time now, so wait for dockerd to actually hand it a PID
# before the mcap/pip-deps/workspace checks below start execing into it.
for _ in $(seq 1 20); do container_running && break; sleep 0.25; done

# These are pip-installed into the running container, so they do NOT survive `docker rm`.
# rosbag2's mcap storage plugin, needed by record_bag.bash: LABS' lerobot_mcap_reader
# consumes MCAP, and the sqlite3 default would need a re-encode before conversion. Like the
# pip deps below, an apt install into the running container does NOT survive `docker rm`.
if ! dex 'ls /opt/ros/humble/share | grep -q rosbag2_storage_mcap' >/dev/null 2>&1; then
  echo "Installing rosbag2 mcap storage into the container..."
  docker exec -u 0 "$CONTAINER" bash -lc \
    "apt-get update -qq && apt-get install -y -qq ros-humble-rosbag2-storage-mcap" >/dev/null 2>&1 || true
fi

if ! dex 'python3 -c "import evdev, dynamixel_sdk, requests"' >/dev/null 2>&1; then
  echo "Installing missing Python deps into the container..."
  docker exec -u 0 "$CONTAINER" bash -lc \
    "pip3 install -q -r '$repo_ctr/requirements.txt' requests" >/dev/null 2>&1 || true
fi

if ! dex "[ -f '$repo_ctr/install/setup.bash' ]" 2>/dev/null; then
  echo "Workspace not built for Humble; building..."
  # Build as your uid, NOT root: root builds through the bind mount are what leave
  # build/ install/ log/ owned by root and break the next host-side colcon build.
  dex "source /opt/ros/humble/setup.bash && source /opt/ros/humble/franka/setup.bash \
       && cd '$repo_ctr' \
       && colcon build --symlink-install --packages-skip franka_fr3_arm_controllers franka_gripper_manager" \
    | tail -3
fi

# -------------------------------------------------------------------- clock
if [ -x "$repo_host/configs/sync_robot_clock.sh" ] \
   && ssh -o BatchMode=yes -o ConnectTimeout=5 "${TMR_HOST:-companion}" true 2>/dev/null; then
  skew_line="$("$repo_host/configs/sync_robot_clock.sh" --check 2>/dev/null | grep robot || true)"
  [ -n "$skew_line" ] && echo "Clock:$skew_line"
  case "$skew_line" in
    *NEEDS\ CORRECTION*)
      # auto-sync: skew > 0.5 s makes the base silently discard cmd_vel and the arms reject
      # GELLO samples, with nothing logged. Correcting it is not optional, so do not make
      # the operator remember it. Passwordless via /usr/local/sbin/tmr-set-clock on the
      # robot; without that sudoers rule this falls back to prompting, so it is bounded.
      echo "  correcting..."
      if timeout 90 "$repo_host/configs/sync_robot_clock.sh" </dev/null 2>&1 | grep -E "robot is|Clock synced"; then
        :
      else
        echo "  -> automatic sync failed. Run ./configs/sync_robot_clock.sh by hand," >&2
        echo "     or the base will ignore cmd_vel silently." >&2
      fi
      ;;
  esac
else
  echo "Clock: skipped (no passwordless ssh to ${TMR_HOST:-companion})."
  echo "  Set it up once with:  ssh-copy-id ${TMR_HOST:-companion}"
  echo "  Until then check by hand: ./configs/sync_robot_clock.sh --check"
fi

# ------------------------------------------------------------------- restart
stop_stacks

if [ -n "$dds_host" ]; then echo "DDS: $dds_host "; \
else echo "DDS: default discovery (all interfaces)"; fi
echo "Starting GELLO leaders ($GELLO_CFG)..."
dexd "$prelude && exec ros2 launch franka_gello_state_publisher main.launch.py \
      config_file:=$GELLO_CFG > /tmp/gello.log 2>&1"

pedal_args="record:=$($record && echo true || echo false) pedals:=$local_pedals"
$local_pedals || echo "Pedals: NOT read here - another host must publish /pedal/state."
[ -n "$task_id" ] && pedal_args="$pedal_args task_id:=$task_id"

if $pedal_fg; then
  echo "Waiting for GELLO to come up (Fast DDS start is slow here)..."
  sleep 20
  dex "grep -aiE 'error|died' /tmp/gello.log | tail -5" || true
  echo
  echo "Starting pedal stack in the FOREGROUND. This terminal must keep focus for 'm'"
  echo "and w/a/s/d/q/e to reach keyboard_state_publisher. Ctrl+C stops the pedal stack"
  echo "(GELLO keeps running; use './start_teleop.bash stop' for both)."
  echo
  exec docker exec -it -u "$(id -u):20" -e HOME=/tmp "$CONTAINER" bash -lc \
    "$prelude && exec ros2 launch tmr_pedal_teleop mobile_teleop.launch.py $pedal_args"
fi

echo "Starting pedal stack ($pedal_args)..."
dexd "$prelude && exec ros2 launch tmr_pedal_teleop mobile_teleop.launch.py $pedal_args > /tmp/pedal.log 2>&1"

if $cameras; then
  start_cameras || echo "WARNING: camera startup failed; continuing without cameras (check: docker logs $CAM_ZED_NAME / $CAM_RS_NAME)" >&2
fi

# Poll instead of sleeping a fixed 25 s: Fast DDS node creation is slow here, but how slow
# varies, and waiting the worst case every time wastes most of a minute per restart.
# Returns as soon as both stacks report ready; caps so a genuine failure still surfaces.
ready_wait() {
  local deadline=$(( SECONDS + ${TMR_READY_TIMEOUT:-30} ))
  while (( SECONDS < deadline )); do
    if dex 'grep -aq "Pedal publisher started" /tmp/pedal.log && grep -aq "gripper=" /tmp/gello.log' 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "  (still not ready after ${TMR_READY_TIMEOUT:-30}s - showing the logs anyway)" >&2
  return 1
}
echo "Waiting for both stacks..."
ready_wait || true

echo
echo "=== GELLO ==="
dex 'grep -aiE "error|died" /tmp/gello.log | tail -5 || true; tail -2 /tmp/gello.log | cut -c1-100'
echo "=== PEDAL ==="
dex 'grep -aE "Pedal publisher started|Foot switch [12]:|base_bridge ->|PEDAL MODE|spine_bridge ready" /tmp/pedal.log | tail -8'
if $cameras; then
  echo "=== CAMERAS ==="
  for n in "$CAM_ZED_NAME" "$CAM_RS_NAME"; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$n" 2>/dev/null)" = true ]; then
      echo "  $n: running"
    else
      echo "  $n: FAILED to start - check: docker logs $n"
    fi
  done
fi
echo
echo "keyboard_state_publisher dies without a TTY - expected when not using --pedal-fg."
echo
echo "Arms stay INACTIVE by design. With both arms at the home pose and hands OFF the"
echo "GELLOs, activate impedance control from inside the container:"
echo "  ros2 control set_controller_state joint_impedance_controller active -c /left/controller_manager"
echo "  ros2 control set_controller_state joint_impedance_controller active -c /right/controller_manager"

if $detach; then
  echo
  echo "Detached. Logs: ./start_teleop.bash logs [gello|pedal]   Stop: ./start_teleop.bash stop"
  exit 0
fi

# Foreground by default. Detached start left no visible log and no way to Ctrl+C, so the
# only way to stop was a second, separate invocation. Follow both logs here and make Ctrl+C
# a clean shutdown of BOTH stacks.
echo
echo "=============================================================="
echo " Following both logs. Ctrl+C stops BOTH stacks."
echo " (-d starts detached; --pedal-fg gives the pedal stack a real"
echo "  TTY so 'm' and w/a/s/d/q/e reach keyboard_state_publisher.)"
echo "=============================================================="
echo
cleanup_fg() {
  trap - INT TERM
  echo
  echo "Stopping teleop stacks..."
  # The follower runs INSIDE the container and would outlive this script otherwise.
  docker exec "$CONTAINER" bash -lc 'pkill -f "tail -n 20 -F /tmp/gello.log" || true' >/dev/null 2>&1 || true
  stop_stacks
  echo "stopped."
  exit 0
}
trap cleanup_fg INT TERM
dex 'tail -n 20 -F /tmp/gello.log /tmp/pedal.log' || true
cleanup_fg
