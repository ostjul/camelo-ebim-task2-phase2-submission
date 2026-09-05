#!/usr/bin/env bash
# Replay a rig bag OFF the rig, into the local camelo-ros container's ROS graph.
#
# WHY THIS WORKS WITHOUT THE RIG'S PACKAGES
# tools/rig_probes/rig_bag_topics.py records only stock sensor_msgs /
# geometry_msgs / nav_msgs / std_msgs / tf2_msgs types, so the Jazzy image can
# create publishers for every topic in the bag with no franka, robotiq or LABS
# typesupport installed. Record something outside that list (/pedal/state,
# /teleop/pedal_mode) and playback dies on the missing type instead.
#
# WHY THE STALENESS GUARDS DO NOT FIRE ON A REPLAY
# The U-23 image guard and the U-27 joint-state guard both age messages by their
# WALL RECEIPT time - time.monotonic() inside our own callback - never by
# header.stamp (camelo/control/image_age.py, state_age.py; the U-27 incident was
# itself a clock-skew rejection, so an alarm that trusted the publisher's clock
# would be reasoning with the broken quantity). Replayed messages therefore
# arrive "fresh" however old their stamps are. The visible consequence is the
# other way round: check_obs's clock-stamp column will read the age of the bag.
#
# USAGE
#   tools/rig_probes/play_rig_bag.sh outputs/rig/bags/rigbag_...        # loops until Ctrl+C
#   tools/rig_probes/play_rig_bag.sh <bag> --once                       # play through once, stop
#   tools/rig_probes/play_rig_bag.sh <bag> --rate 0.25                  # slow motion (still loops)
#   tools/rig_probes/play_rig_bag.sh <bag> --list                       # contents, no publish
#
# Then, in a SECOND terminal, point camelo's real-world path at it:
#   ./scripts/ros_compose.sh run --rm camelo-ros \
#       python3 scripts/check_obs.py --world real --seconds 10
# Looping is the default (not opt-in) because that second terminal is the point
# of this script: check_obs needs a stream that outlives its own 10 s window,
# and a bag played once usually doesn't. --once is for when you want playback
# to end on its own, e.g. timing something against the bag's own length.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

bag=""; extra_args=(); list=false; loop=true
while [ $# -gt 0 ]; do
  case "$1" in
    --list)  list=true; shift ;;
    --loop)  loop=true; shift ;;   # default already; accepted for explicitness
    --once|--no-loop) loop=false; shift ;;
    --rate)  extra_args+=(--rate "$2"); shift 2 ;;
    --start-offset) extra_args+=(--start-offset "$2"); shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    -*) extra_args+=("$1"); shift ;;
    *)  bag="$1"; shift ;;
  esac
done
[ -n "$bag" ] || { echo "usage: $0 <bag dir> [--once] [--rate R] [--list]" >&2; exit 2; }

[ -e "$bag" ] || { echo "ERROR: no such path: $bag" >&2; exit 1; }
# A path to the .mcap FILE (not its bag directory) is an easy mistake after
# downloading just the folder and tab-completing into it - accept it too.
case "$bag" in
  *.mcap) [ -f "$bag" ] && bag="$(dirname -- "$bag")" ;;
esac
[ -d "$bag" ] || { echo "ERROR: $bag is not a bag directory" >&2; exit 1; }
# `cd`+`pwd -P`, NOT `realpath -e --`: GNU-only flags (macOS ships BSD realpath,
# which has no -e at all - "illegal option -- e" is that mismatch). This works
# identically on both.
bag_abs="$(cd "$bag" && pwd -P)"
[ -f "$bag_abs/metadata.yaml" ] || {
  echo "ERROR: $bag_abs has no metadata.yaml - the recorder never finalised it," >&2
  echo "       and rosbag2 cannot read it. Re-record and let the recorder finalise." >&2
  exit 1
}

# The compose service bind-mounts the repo root at /workspace/camelo-ebim, so the
# bag has to live inside the repo for the container to see it at all.
case "$bag_abs" in
  "$ROOT"/*) bag_ctr="/workspace/camelo-ebim/${bag_abs#"$ROOT"/}" ;;
  *)
    echo "ERROR: $bag_abs is outside $ROOT - that is the only host path the" >&2
    echo "       camelo-ros container mounts, so it can't see the bag there." >&2
    echo "       Move it in, e.g.:" >&2
    echo "         mkdir -p outputs/rig/bags && mv '$bag_abs' outputs/rig/bags/" >&2
    exit 1 ;;
esac

# The host may still carry a rig-shaped FASTRTPS_DEFAULT_PROFILES_FILE, which
# compose passes through. Off-rig that profile whitelists 172.16.16.118 - an
# interface this box does not have - and the player and check_obs then never
# discover each other, with no error at either end.
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
  echo "WARNING: FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE is set." >&2
  echo "         Off-rig, unset it - the rig profile's interface whitelist silently" >&2
  echo "         prevents local discovery." >&2
fi

if $list; then
  exec ./scripts/ros_compose.sh run --rm camelo-ros \
    bash -lc "ros2 bag info '$bag_ctr'"
fi

play_args=()
$loop && play_args+=(--loop)
# `${extra_args[@]+"${extra_args[@]}"}`, not a bare `"${extra_args[@]}"`: bash
# < 4.4 (macOS's stock /bin/bash is 3.2) throws "unbound variable" on an EMPTY
# array under `set -u`, even though it was declared - fixed upstream in 4.4,
# still live on every Mac. Reproduced and the guard verified against actual
# bash 3.2 and 4.3 (`docker run --rm bash:3.2 ...`) while writing this.
play_args+=(${extra_args[@]+"${extra_args[@]}"})

echo "Playing $bag_ctr  (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}, host network$($loop && echo ", looping — --once to play once and stop"))"
echo "Second terminal:  ./scripts/ros_compose.sh run --rm camelo-ros \\"
echo "                      python3 scripts/check_obs.py --world real --seconds 10"
echo
# Same guard, applied to the interpolated string this time (play_args is empty
# whenever --once is passed with no --rate/--start-offset).
play_args_str="${play_args[*]+"${play_args[*]}"}"
exec ./scripts/ros_compose.sh run --rm camelo-ros \
  bash -lc "ros2 bag play '$bag_ctr' $play_args_str"
