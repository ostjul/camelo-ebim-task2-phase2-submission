#!/usr/bin/env bash
# Record a raw rosbag of everything a LeRobot dataset needs.
#
# A sibling of start_teleop.bash: runs in the Humble container, foreground, Ctrl+C to stop.
#
# WHY A RAW BAG FIRST
# LABS owns the real recording pipeline, but this proves the topics exist and are gapless
# BEFORE committing to it, and an MCAP bag is convertible either way. The topic list below
# is exactly LABS' manifest (labs_integration/tmr_station/config_data_recorder.yml) plus the
# two base lidars, so what this captures is what LABS would capture.
#
# THE RULE THIS SCRIPT EXISTS TO ENFORCE
# LABS' TemporalSynchronizer takes the LATEST first-message time and the EARLIEST last-message
# time across all configured topics, and fails the episode if either is more than 1 s from
# the episode bounds. One silent publisher therefore fails the WHOLE conversion, not just its
# own column. `--check` refuses to record when a required topic has no publisher, so you find
# out in a second rather than after a ten-minute episode.
#
# ROS DOMAIN
# Everything must be on ONE domain - a single `ros2 bag record` sees exactly one. Domain 0 is
# the target: teleop, LABS docker-compose and (since 2026-08-23) start_zed.bash all use it.
# If a camera or the ZED is missing from --check, suspect a stale ROS_DOMAIN_ID=100 first.
#
# USAGE
#   ./record_bag.bash --check              # what is publishing? changes nothing
#   ./record_bag.bash --info               # inspect the most recent bag
#   ./record_bag.bash                      # record everything, Ctrl+C to stop
#   ./record_bag.bash --no-video           # state/action/lidar only (low bandwidth)
#   ./record_bag.bash --bag-root DIR       # save on another disk/directory
#   ./record_bag.bash --out DIR            # exact bag path (must be under bag root)
#   ./record_bag.bash --name my_episode    # bag name suffix
#
# COLLECTING A DATASET
#   ./record_bag.bash --task pick_place    # auto-numbered episode for this task
#   ./record_bag.bash --status             # episode counts for every task
#   ./record_bag.bash --task X --target 50 # per-task goal (default 200)
#
# --task keeps each task in its own directory and numbers episodes by how many are
# already COMPLETE, so the name never depends on the order you happen to run things:
#
#   ~/teleop_bags/pick_place/ep001_20260824-150312/
#   ~/teleop_bags/pick_place/ep002_20260824-150501/
#
# "Complete" means the directory has a metadata.yaml. A recorder that was killed before
# finalising leaves a directory without one; counting those would inflate your progress
# and silently reuse an episode number.
set -Eeuo pipefail

CONTAINER="${TMR_CONTAINER:-gello-humble}"
IMAGE="${TMR_IMAGE:-teleoperation_devcontainer-gello-ros2:latest}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

repo_host="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$repo_host" in
  "$HOME"/*) repo_ctr="/workspace/${repo_host#"$HOME"/}" ;;
  *) echo "ERROR: expected the repo under \$HOME ($HOME), got $repo_host" >&2; exit 1 ;;
esac

# Same native-vs-container detection as configs/teleop_common.sh's teleop_source_workspace:
# CONDA_PREFIX survives `pixi run`, `pixi shell`, and a plain `exec bash` from one.
# Native mode records directly on the host against install_pixi/ - no docker exec, no
# bind-mount indirection, no helper container.
NATIVE=false
if [[ "${CONDA_PREFIX:-}" == *"/.pixi/envs/"* && -f "$repo_host/install_pixi/setup.bash" ]]; then
  NATIVE=true
fi

# Keep the host path and the container path separate. The launcher bind-mounts the
# configured host directory at /bags so removable storage works without symlink tricks.
bag_root_config="${XDG_CONFIG_HOME:-$HOME/.config}/teleoperation/bag_root"
if [ -n "${TMR_BAG_ROOT:-}" ]; then
  BAG_ROOT="$TMR_BAG_ROOT"
elif [ -r "$bag_root_config" ]; then
  IFS= read -r BAG_ROOT < "$bag_root_config"
else
  BAG_ROOT="$HOME/teleop_bags"
fi
[ -n "$BAG_ROOT" ] || { echo "ERROR: empty bag root in $bag_root_config" >&2; exit 1; }

# ---------------------------------------------------------------- topic manifest
# Observation - robot state. These 20 dims are the arms; see modality.json for the layout.
STATE_TOPICS=(
  /left/franka_robot_state_broadcaster/measured_joint_states
  /left/franka_robot_state_broadcaster/external_joint_torques
  /left/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame
  # NOT gripper_joint_states: robotiq_controllers.yaml nests `publish_topic` inside the
  # controller_manager ros__parameters block, where the broadcaster never reads it, so the
  # name never takes effect. Verified on hardware 2026-08-23 - the real topic is
  # joint_states. LABS' config_data_recorder.yml still says gripper_joint_states and would
  # therefore silently drop both gripper state columns.
  /left/gripper/joint_states
  /right/franka_robot_state_broadcaster/measured_joint_states
  /right/franka_robot_state_broadcaster/external_joint_torques
  /right/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame
  /right/gripper/joint_states
  /mobile_base/pose
  /mobile_base/twist
  /swerve_drive_controller/cmd_vel_out
  /spine/joint_states
)
# Actions - the follower topic of each teleop device.
ACTION_TOPICS=(
  /left/gello/joint_states
  /right/gello/joint_states
  /left/gripper/gripper_client/target_gripper_width_percent
  /right/gripper/gripper_client/target_gripper_width_percent
  /swerve_drive_controller/cmd_vel
  /spine/target_height
)
# Video. UNCOMPRESSED sensor_msgs/Image only - LABS rejects CompressedImage outright.
VIDEO_TOPICS=(
  # NOT image_raw: the D405's color sensor is one of its stereo pair, so the native pixi
  # realsense2_camera (v4.55.1) treats it as inherently rectified and names the topic
  # image_rect_raw instead - confirmed 2026-08-30, differs from the retired Docker setup's
  # older realsense2_camera which used image_raw. See camera_viewer.py's identical note.
  /wrist_camera_left/camera/color/image_rect_raw
  /wrist_camera_right/camera/color/image_rect_raw
  # NOT rgb/image_rect_color. This zed_wrapper build advertises the rectified colour image
  # as rgb/color/rect/image - verified against the running node 2026-08-23. LABS'
  # config_data_recorder.yml and config_station.yml still carry the old name and would
  # silently record no head camera at all.
  /head_camera/zed_node/rgb/color/rect/image
)
# Not in the LABS manifest; recorded for our own use. Names inferred from the namespaces in
# franka_mobile_sensors' default_sensor_suite.yaml - verify with --check.
LIDAR_TOPICS=(
  /lidar_front/scan
  /lidar_rear/scan
)
# Archived: useful context, dropped at dataset-build time.
EXTRA_TOPICS=(
  /tf
  /tf_static
  /pedal/state
  /teleop/pedal_mode
  # 'odom', not 'odometry' - verified on hardware 2026-08-23.
  /swerve_drive_controller/odom
)

mode=record; want_video=true; out=""; name=""
task=""; target="${TMR_EPISODE_TARGET:-200}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check)    mode=check; shift ;;
    --info)     mode=info; shift ;;
    --no-video) want_video=false; shift ;;
    --bag-root|--save-dir)
                 [ $# -ge 2 ] || { echo "ERROR: $1 requires a directory" >&2; exit 2; }
                 BAG_ROOT="$2"; shift 2 ;;
    --out)      out="$2"; shift 2 ;;
    --name)     name="$2"; shift 2 ;;
    --task)     task="$2"; shift 2 ;;
    --target)   target="$2"; shift 2 ;;
    --status)   mode=status; shift ;;
    -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$mode" != check ]; then
  case "$BAG_ROOT" in
    /*) ;;
    *) echo "ERROR: bag root must be an absolute path: $BAG_ROOT" >&2; exit 2 ;;
  esac
  if [ ! -d "$BAG_ROOT" ]; then
    echo "ERROR: bag root does not exist: $BAG_ROOT" >&2
    echo "       Mount the disk and create this directory first." >&2
    exit 1
  fi
  BAG_ROOT="$(realpath -e -- "$BAG_ROOT")"
  [ "$BAG_ROOT" != "/" ] || { echo "ERROR: refusing to use / as the bag root" >&2; exit 2; }
fi

topics=("${STATE_TOPICS[@]}" "${ACTION_TOPICS[@]}" "${LIDAR_TOPICS[@]}" "${EXTRA_TOPICS[@]}")
$want_video && topics+=("${VIDEO_TOPICS[@]}")

# Required = anything whose absence breaks the LeRobot conversion. The extras are not.
required=("${STATE_TOPICS[@]}" "${ACTION_TOPICS[@]}")
$want_video && required+=("${VIDEO_TOPICS[@]}")

# DEFAULT TRANSPORTS, deliberately - everything must share one transport world.
#
# The wrist cameras used to run with useBuiltinTransports=false and UDP whitelisted to the
# wired 172.16.16.140. A default-transport participant then could not discover them AT ALL:
# their topics were simply absent, with no error. Pinning the recorder to match found the
# cameras but lost the teleop nodes instead, and adding SHM did not bridge the two - SHM
# cannot cross a container boundary, and the cameras run in their own container with their
# own /dev/shm. The fix was on the camera side: drop its custom profile so it uses the same
# default transports as the robot stack and the teleop nodes.
#
# Opt into a profile with TMR_DDS_PROFILE=<path>, but check what it does to discovery first.
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
  if $NATIVE; then
    dds_ctr="$dds_host"
  else
    case "$dds_host" in
      "$HOME"/*) dds_ctr="/workspace/${dds_host#"$HOME"/}" ;;
      *) echo "ERROR: DDS profile must live under \$HOME to be visible in the container" >&2; exit 1 ;;
    esac
  fi
  dds_env="&& export FASTRTPS_DEFAULT_PROFILES_FILE='$dds_ctr' FASTDDS_DEFAULT_PROFILES_FILE='$dds_ctr'"
fi

EXEC_CONTAINER="$CONTAINER"
HELPER_CONTAINER=""
if $NATIVE; then
  # No franka overlay natively: install_pixi/ (station/pixi.toml's build-teleop task)
  # deliberately excludes libfranka-dependent packages, which only run ON the robot. That
  # is fine here - recording only needs each topic's message-type package for CDR
  # (de)serialization, not the franka control stack itself.
  prelude="cd '$repo_host' && source install_pixi/setup.bash \
    && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_fastrtps_cpp PYTHONUNBUFFERED=1 $dds_env"
  dex() { bash -lc "$1"; }
else
  prelude="source /opt/ros/humble/setup.bash \
    && source /opt/ros/humble/franka/setup.bash \
    && cd '$repo_ctr' && source install/setup.bash \
    && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=rmw_fastrtps_cpp PYTHONUNBUFFERED=1 $dds_env"
  dex()  { docker exec -u "$(id -u):20" -e HOME=/tmp "$EXEC_CONTAINER" bash -lc "$1"; }

  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ] || {
    echo "ERROR: container '$CONTAINER' is not running. Start it with ./start_teleop.bash" >&2
    exit 1
  }
fi

# A bag counts as an episode only once rosbag2 has written metadata.yaml. Anything else
# is a recorder that was interrupted - it is not convertible and must not consume a number.
episode_count() {
  local dir="$1" n=0 b
  [ -d "$dir" ] || { echo 0; return; }
  for b in "$dir"/*/; do
    [ -f "${b}metadata.yaml" ] && n=$((n + 1))
  done
  echo "$n"
}

# --status touches neither the ROS graph nor the robot; safe at any time.
if [ "$mode" = status ]; then
  root="$BAG_ROOT"
  [ -d "$root" ] || { echo "No bags yet in $root"; exit 0; }
  printf "%-24s %8s %8s   %s\n" TASK DONE TARGET PROGRESS
  found=0
  for d in "$root"/*/; do
    [ -d "$d" ] || continue
    # A task directory holds episode directories; a bare bag directory has metadata.yaml
    # of its own. Skip the latter so old flat-layout bags are not reported as tasks.
    [ -f "${d}metadata.yaml" ] && continue
    n="$(episode_count "$d")"
    [ "$n" -eq 0 ] && continue
    found=1
    pct=$(( n * 100 / target ))
    [ "$pct" -gt 100 ] && pct=100
    # Build the bar by string slicing, not `printf FMT $(seq 1 0)` - seq emits nothing at
    # zero and printf then prints the format once, drawing a filled block at 0%.
    bars=$(( pct / 5 ))
    full="####################"
    empty="...................."
    printf "%-24s %8s %8s   [%s%s] %s%%\n" \
      "$(basename "$d")" "$n" "$target" \
      "${full:0:$bars}" "${empty:0:$(( 20 - bars ))}" "$pct"
  done
  [ "$found" -eq 1 ] || echo "(no task directories yet - record one with --task NAME)"
  exit 0
fi

# A running container cannot acquire a new bind mount. If the selected bag root differs
# from the teleoperation container's /bags mount, create a recorder-only helper container.
# It shares ROS networking and Fast DDS SHM but never starts/stops teleoperation nodes.
# Native mode has no mount indirection at all - BAG_ROOT is used directly - so this whole
# block is docker-only.
if ! $NATIVE && [ "$mode" != check ]; then
  mounted_root="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/bags"}}{{.Source}}{{end}}{{end}}' "$CONTAINER" 2>/dev/null || true)"
  if [ "$mounted_root" != "$BAG_ROOT" ]; then
    if img="$(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null)" && [ -n "$img" ]; then
      :
    else
      img="$IMAGE"
    fi
    HELPER_CONTAINER="bag-recorder-$(id -u)-$$"
    cleanup_helper() {
      [ -z "$HELPER_CONTAINER" ] || docker rm -f "$HELPER_CONTAINER" >/dev/null 2>&1 || true
    }
    trap cleanup_helper EXIT
    echo "Using recorder-only container for bag root: $BAG_ROOT"
    docker run -d --rm --name "$HELPER_CONTAINER" --network host --ipc=host --init \
      -v "$HOME:/workspace" -v "$BAG_ROOT:/bags" \
      -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
      "$img" sleep infinity >/dev/null
    EXEC_CONTAINER="$HELPER_CONTAINER"
    if ! docker exec "$EXEC_CONTAINER" bash -lc \
      'test -d /opt/ros/humble/share/rosbag2_storage_mcap' >/dev/null 2>&1; then
      echo "Installing MCAP storage in the recorder-only container..."
      docker exec -u 0 "$EXEC_CONTAINER" bash -lc \
        'apt-get update -qq && apt-get install -y -qq ros-humble-rosbag2-storage-mcap' \
        >/dev/null
    fi
  fi
fi

# --info inspects a bag on disk and touches the ROS graph not at all, so it is safe to run
# while the robot is live - unlike --check, which creates a participant.
if [ "$mode" = info ]; then
  # Look one level deeper than the old flat layout: --task nests bags inside a task
  # directory, so `ls -dt ~/teleop_bags/*/` would return the TASK dir, not a bag.
  # Match on metadata.yaml so an interrupted recording is never picked as "the latest".
  bag="${out:-$(find "$BAG_ROOT" -maxdepth 3 -name metadata.yaml -printf '%T@ %h\n' 2>/dev/null \
                 | sort -rn | head -1 | cut -d' ' -f2-)}"
  [ -n "$bag" ] || { echo "No bags found in $BAG_ROOT" >&2; exit 1; }
  bag="${bag%/}"
  echo "Bag: $bag  ($(du -sh "$bag" 2>/dev/null | cut -f1))"
  [ -f "$bag/metadata.yaml" ] || echo "  WARNING: no metadata.yaml - the recorder was killed before finalising." >&2
  if $NATIVE; then
    bag_ctr="$bag"
  else
    case "$bag" in
      "$BAG_ROOT"/*) bag_ctr="/bags/${bag#"$BAG_ROOT"/}" ;;
      *) echo "ERROR: bag is outside the configured bag root: $BAG_ROOT" >&2; exit 1 ;;
    esac
  fi
  if $NATIVE; then
    info="$(dex "source '$repo_host/install_pixi/setup.bash' && ros2 bag info '$bag_ctr' 2>&1" || true)"
  else
    info="$(dex "source /opt/ros/humble/setup.bash && ros2 bag info '$bag_ctr' 2>&1" || true)"
  fi
  echo "$info" | grep -E "Duration|Start|End|Messages|Bag size|Storage"
  echo
  echo "Topics by message count:"
  echo "$info" | grep -oE "Topic: [^ ]+ \| Type: [^ ]+ \| Count: [0-9]+" \
    | sed -E 's/Topic: ([^ ]+) \| Type: [^ ]+ \| Count: ([0-9]+)/\2 \1/' \
    | sort -rn | awk '{printf "  %10s  %s\n", $1, $2}'
  empty="$(echo "$info" | grep -oE "Topic: [^ ]+ \| Type: [^ ]+ \| Count: 0 " | awk '{print $2}' || true)"
  if [ -n "$empty" ]; then
    echo
    echo "  ZERO-MESSAGE TOPICS - these fail the LeRobot conversion:" >&2
    printf '    %s\n' $empty >&2
    exit 1
  fi
  echo
  echo "  No empty topics."
  exit 0
fi

# ------------------------------------------------------------------- check
# One `ros2 topic list` for the whole manifest: a topic list per topic would be a DDS
# discovery burst per call, and those bursts abort running FCI control loops.
echo "Discovering topics on domain $ROS_DOMAIN_ID (spinning ${TMR_SPIN_TIME:-25}s; Fast DDS is slow here)..."
# `ros2 topic list -v`, not plain list: a bare list includes topics that only have a
# SUBSCRIBER, so a robot-side controller listening for /left/gello/joint_states made the
# topic look healthy while nothing published it. Only the "Published topics:" section counts.
# --spin-time is essential, not a nicety. `ros2 topic list` spins ~1 s by default and
# then reports, but Fast DDS discovery on this machine takes 15-25 s, so the default
# reports a half-discovered graph: nodes that were definitely publishing showed up as
# MISSING while a latched topic happened to arrive in time.
spin="${TMR_SPIN_TIME:-25}"
raw="$(dex "$prelude && timeout $((spin + 45)) ros2 topic list -v --spin-time $spin --no-daemon 2>/dev/null" || true)"
live="$(awk '/^Published topics:/{p=1;next} /^Subscribed topics:/{p=0} p && /^ \* /{print $2}' <<<"$raw")"
if [ -z "$live" ]; then
  echo "ERROR: no topics discovered at all. Is anything running?" >&2
  exit 1
fi

missing=(); present=0
printf '\n%-64s %s\n' "TOPIC" "STATUS"
for t in "${topics[@]}"; do
  if grep -qxF "$t" <<<"$live"; then
    printf '%-64s %s\n' "$t" "ok"; present=$((present+1))
  else
    printf '%-64s %s\n' "$t" "MISSING"
    for r in "${required[@]}"; do [ "$r" = "$t" ] && missing+=("$t"); done
  fi
done
printf '\n%d/%d topics present\n' "$present" "${#topics[@]}"

if [ "${#missing[@]}" -gt 0 ]; then
  echo
  echo "${#missing[@]} REQUIRED topic(s) missing:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo >&2
  echo "Recording now would produce an episode that cannot be converted: LABS fails the" >&2
  echo "WHOLE episode if any configured topic is more than 1 s short of the bounds." >&2
  echo "Start the missing publisher, or use --no-video if only cameras are missing." >&2
  [ "$mode" = check ] || exit 1
fi
[ "$mode" = check ] && exit 0

# ------------------------------------------------------------------ record
stamp="$(date +%Y%m%d-%H%M%S)"
if [ -n "$task" ]; then
  # Task-scoped, auto-numbered. The number is derived from what is already on disk, so
  # two operators recording the same task on the same machine cannot collide, and a
  # deleted bad episode frees its number for reuse.
  task_dir="$BAG_ROOT/$task"
  mkdir -p "$task_dir"
  done_n="$(episode_count "$task_dir")"
  next_n=$(( done_n + 1 ))
  printf -v ep "ep%03d" "$next_n"
  [ -n "$name" ] && ep="${ep}_${name}"
  out="${out:-$task_dir/${ep}_${stamp}}"
  echo
  echo "Task '$task': recording EPISODE $next_n   (complete so far: $done_n / $target)"
  if [ "$done_n" -ge "$target" ]; then
    echo "  NOTE: the target of $target is already met; this episode is extra."
  else
    echo "  $(( target - done_n )) more needed after this one."
  fi
  echo
else
  [ -n "$name" ] && stamp="${stamp}_${name}"
  out="${out:-$BAG_ROOT/$stamp}"
fi
if $NATIVE; then
  out_ctr="$out"
else
  case "$out" in
    "$BAG_ROOT"/*) out_ctr="/bags/${out#"$BAG_ROOT"/}" ;;
    *) echo "ERROR: --out must be under the configured bag root: $BAG_ROOT" >&2; exit 1 ;;
  esac
fi
[ -d "$BAG_ROOT" ] || {
  echo "ERROR: bag root is unavailable: $BAG_ROOT" >&2
  echo "       Mount the recording disk before recording." >&2
  exit 1
}
if ! $NATIVE; then
  mounted_root="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/bags"}}{{.Source}}{{end}}{{end}}' "$EXEC_CONTAINER" 2>/dev/null || true)"
  [ "$mounted_root" = "$BAG_ROOT" ] || {
    echo "ERROR: recorder container '$EXEC_CONTAINER' does not mount $BAG_ROOT at /bags." >&2
    exit 1
  }
fi
mkdir -p "$(dirname "$out")"

echo
echo "=============================================================="
echo " Recording -> $out"
echo " ${#topics[@]} topics, video=$want_video"
echo " Ctrl+C to stop."
echo "=============================================================="
echo

# MCAP because that is what LABS' lerobot_mcap_reader consumes; sqlite3 bags would need a
# re-encode before conversion.
storage="-s mcap"
# 1 GiB write cache (default ~100 MiB). Peak topic throughput approaches 30 MB/s of
# images plus ~7,500 state/control messages a second; the default cache holds only a few
# seconds of that, so any disk or scheduler hiccup backs up into the DDS subscribers and
# reads as camera frame drops in the bag. Disk is not the bottleneck (measured ~1.3%
# utilisation while recording) - the cache just has to absorb the bursts.
cache="--max-cache-size 1073741824"
if ! dex "$prelude && ros2 bag record --help 2>&1 | grep -q mcap" >/dev/null 2>&1; then
  echo "NOTE: rosbag2_storage_mcap not found; falling back to the sqlite3 default." >&2
  echo "      Install with: sudo apt install ros-humble-rosbag2-storage-mcap" >&2
  storage=""
fi

# `docker exec` does NOT forward signals to the process inside the container. Killing this
# script therefore leaves `ros2 bag record` running and the bag WITHOUT metadata.yaml, which
# makes it unreadable - observed 2026-08-23: a 124 MB mcap with no metadata, and the recorder
# still running. SIGINT must be delivered inside the container so rosbag2 finalises.
# Native mode has the same problem for a different reason: the recorder is backgrounded
# (see below) so bash's own SIGINT does not reach it either - it must be signalled directly.
cleanup() {
  trap - INT TERM
  echo; echo "stopping the recorder..."
  if $NATIVE; then
    pkill -INT -f "ros2 bag record" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      pgrep -f "ros2 bag record" >/dev/null 2>&1 || break
      sleep 0.5
    done
  else
    docker exec "$EXEC_CONTAINER" bash -lc 'pkill -INT -f "ros2 bag record"' >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      docker exec "$EXEC_CONTAINER" bash -lc 'pgrep -f "ros2 bag record" >/dev/null' 2>/dev/null || break
      sleep 0.5
    done
  fi
}
trap cleanup INT TERM
# NO -it, deliberately. Backgrounding a command redirects its stdin to /dev/null, so
# `docker exec -it ... &` fails with "cannot attach stdin to a TTY-enabled container because
# stdin is not a terminal" EVEN FROM an interactive shell - the -t check on the parent shell
# says nothing about the backgrounded child. And the exec must be backgrounded, because bash
# defers traps until the foreground command returns and `docker exec` never returns.
#
# A TTY is not needed anyway: Ctrl+C is handled by cleanup(), which sends SIGINT to the
# recorder INSIDE the container. Output still streams without -t.
# Background + `wait`, NOT a plain foreground call. Bash defers a trap until the current
# foreground command returns, and `docker exec` never returns on its own - so Ctrl+C was
# deferred forever and the recorder kept running with the bag left unfinalised. `wait` is
# interruptible, which lets cleanup() actually run.
if $NATIVE; then
  bash -lc "$prelude && exec ros2 bag record $storage $cache -o '$out_ctr' ${topics[*]}" &
else
  docker exec -u "$(id -u):20" -e HOME=/tmp "$EXEC_CONTAINER" bash -lc \
    "$prelude && exec ros2 bag record $storage $cache -o '$out_ctr' ${topics[*]}" &
fi
rec_pid=$!
wait "$rec_pid" 2>/dev/null || true

echo
cleanup
if [ -d "$out" ]; then
  echo "Bag: $out  ($(du -sh "$out" 2>/dev/null | cut -f1))"
  if [ ! -f "$out/metadata.yaml" ]; then
    echo "  WARNING: no metadata.yaml - the bag is unreadable. The recorder was killed" >&2
    echo "           before it could finalise." >&2
  fi
  # A topic can have a PUBLISHER and still carry zero messages - an inactive controller
  # advertises without ever publishing. --check cannot see that; only the counts can.
  info="$(dex "source /opt/ros/humble/setup.bash && ros2 bag info '$out_ctr' 2>/dev/null" || true)"
  empty="$(grep -oE "Topic: [^ ]+ \| Type: [^ ]+ \| Count: 0 " <<<"$info" | awk '{print $2}' || true)"
  if [ -n "$empty" ]; then
    echo
    echo "  TOPICS WITH ZERO MESSAGES - these will fail the LeRobot conversion:" >&2
    printf '    %s\n' $empty >&2
  fi
  echo "Inspect with:"
  echo "  ./record_bag.bash --bag-root '$BAG_ROOT' --info --out '$out'"
else
  echo "No bag was written." >&2
fi
