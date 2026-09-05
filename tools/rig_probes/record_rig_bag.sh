#!/usr/bin/env bash
# Record a SMALL, self-finalising rosbag on ebimHP for off-rig debugging.
#
# WHY NOT ./record_bag.bash
# That script is the TMR/LABS dataset recorder and, run from ~/camelo/camelo-ebim
# (which has no install_pixi/), it takes its DOCKER path and needs the
# `gello-humble` container. A camelo CONTAINER participant on ebimHP next to live
# station nodes is exactly what C-3 bans pending T2 steps 3-4 (R-22/R-23: two
# global kernel OOMs, 96 GB in 60 s, arms dropped). A NATIVE pixi participant is
# proven safe beside the station cameras - 24 min, flat RSS (R-43/R-45) - so this
# script runs natively and REFUSES to run any other way.
#
# WHAT IT RECORDS
# camelo's own real-world contract (tools/rig_probes/rig_bag_topics.py, derived
# from camelo/contracts.py), not the LABS manifest. Every message type is a stock
# ROS type, so the bag plays back off-rig with no franka/robotiq/LABS packages -
# see tools/rig_probes/play_rig_bag.sh.
#
# USAGE (from set-up C, doc 16 section 3.1 block A - see PRECONDITIONS below)
#   tools/rig_probes/record_rig_bag.sh --check                 # what is publishing?
#   tools/rig_probes/record_rig_bag.sh --seconds 10            # all 3 cameras (~1.6 GB)
#   tools/rig_probes/record_rig_bag.sh --seconds 20 --cameras head
#   tools/rig_probes/record_rig_bag.sh --seconds 30 --cameras none   # state+action, ~MBs
#   tools/rig_probes/record_rig_bag.sh --info <bag dir>
#
# OTHER FLAGS
#   --spin SECONDS     override the pre-record topic-discovery wait (default 25,
#                       same as TMR_SPIN_TIME below). Discovery is a health-check
#                       table only - the topics actually recorded don't depend on
#                       it - so a shorter value just trades the accuracy of the
#                       printed present/MISSING table for a faster start.
#   --bag-root DIR      write bags under DIR instead of outputs/rig/bags, e.g. an
#                       external disk: --bag-root /media/usb/bags
#
# STDOUT CONTRACT: once recording has actually begun (rosbag2's output directory
# exists), this script prints one line: `RECORDING_STARTED pid=<pid> out=<dir>`.
# A launcher script (e.g. one that starts a policy alongside the recording) can
# tail this script's stdout and start its own process on that line.
#
# PRECONDITIONS (doc 16 section 3.1 block A, in this order, one per block):
#   cd ~/teleoperation/station && pixi shell
#   source configs/tmr_laptop_env.sh
#   cd ~/camelo/camelo-ebim && make dds-profile PY=python
#   export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/outputs/rig/fastdds_camelo.xml
# and run it from a tmux started in an SSH login, never the desktop (C-2/R-23).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BAG_ROOT="${CAMELO_BAG_ROOT:-$ROOT/outputs/rig/bags}"
# Match the Makefile convention (PY ?= python3): on the rig pixi shell `python`
# is what resolves to .pixi/envs/default (3.11); elsewhere python3 is safer.
PY="${PY:-python3}"
seconds=10
cameras="head,wrist_left,wrist_right"
groups="state,action,video,extra"
name=""
mode=record
info_target=""
allow_site_profile=false
spin="${TMR_SPIN_TIME:-25}"

while [ $# -gt 0 ]; do
  case "$1" in
    --check)     mode=check; shift ;;
    --info)      mode=info; info_target="${2:-}"; [ -n "$info_target" ] && shift; shift ;;
    --seconds)   seconds="$2"; shift 2 ;;
    --cameras)   cameras="$2"; shift 2 ;;
    --groups)    groups="$2"; shift 2 ;;
    --name)      name="$2"; shift 2 ;;
    --bag-root)  BAG_ROOT="$2"; shift 2 ;;
    --spin)      spin="$2"; shift 2 ;;
    --allow-site-profile) allow_site_profile=true; shift ;;
    -h|--help)   sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$cameras" = "none" ] && groups="${groups//video/}"

# ------------------------------------------------------------------ guards
# Each of these has a rig incident behind it. None is cosmetic.
if [ "$mode" != info ]; then
  # C-3 (R-22/R-23): container participants only. Native pixi is the cleared path.
  if [ -f /.dockerenv ]; then
    echo "REFUSING: this is a container. C-3 bans a camelo container participant on" >&2
    echo "  ebimHP while station nodes run (R-22/R-23). Use the native pixi shell." >&2
    exit 2
  fi
  case "${CONDA_PREFIX:-}" in
    */.pixi/envs/*) ;;
    *) echo "REFUSING: not in the station pixi env (CONDA_PREFIX=${CONDA_PREFIX:-unset})." >&2
       echo "  Run: cd ~/teleoperation/station && pixi shell   (then the block in --help)" >&2
       exit 2 ;;
  esac
  # C-14 (R-48/R-49): the site's rendered profile sets no socket buffer sizes and a
  # 208 KB net.core.rmem_default drops most of every fragmented Image sample under our
  # participant - the bag would silently carry 10-15 Hz cameras instead of 30.
  prof="${FASTRTPS_DEFAULT_PROFILES_FILE:-}"
  if [ -z "$prof" ] || [ ! -f "$prof" ]; then
    echo "REFUSING: FASTRTPS_DEFAULT_PROFILES_FILE is unset or missing ($prof)." >&2
    echo "  make dds-profile PY=python && export FASTRTPS_DEFAULT_PROFILES_FILE=\$PWD/outputs/rig/fastdds_camelo.xml" >&2
    exit 2
  fi
  if ! grep -q receiveBufferSize "$prof" && ! $allow_site_profile; then
    echo "REFUSING: $prof has no receiveBufferSize - that is the SITE profile, not" >&2
    echo "  camelo's (C-14/R-49). Cameras would record at 10-15 Hz, not 30." >&2
    echo "  Re-run 'make dds-profile PY=python' AFTER sourcing tmr_laptop_env.sh," >&2
    echo "  or pass --allow-site-profile if you deliberately want the site profile." >&2
    exit 2
  fi
  [ "${ROS_DOMAIN_ID:-}" = "0" ] || {
    echo "REFUSING: ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}, must be 0 (R-01/the station domain)." >&2
    exit 2
  }
fi

# ------------------------------------------------------------------- info
if [ "$mode" = info ]; then
  bag="${info_target:-$(find "$BAG_ROOT" -maxdepth 2 -name metadata.yaml -printf '%T@ %h\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)}"
  [ -n "$bag" ] || { echo "No finalised bag under $BAG_ROOT" >&2; exit 1; }
  echo "Bag: $bag  ($(du -sh "$bag" 2>/dev/null | cut -f1))"
  ros2 bag info "$bag"
  exit 0
fi

mapfile -t topics < <("$PY" "$ROOT/tools/rig_probes/rig_bag_topics.py" \
                        --groups "$groups" --cameras "$cameras")
[ "${#topics[@]}" -gt 0 ] || { echo "ERROR: empty topic list" >&2; exit 2; }

# ------------------------------------------------------------------ check
# ONE `ros2 topic list` for the whole manifest. A call per topic is a DDS
# discovery burst per call, and ~20 CLI participants in four minutes is the
# correlate of OOM #1 (R-23). --spin-time because Fast DDS discovery here takes
# 15-25 s and the 1 s default reports a half-discovered graph as MISSING.
echo "Discovering topics on domain $ROS_DOMAIN_ID (spinning ${spin}s)..."
raw="$(timeout $((spin + 45)) ros2 topic list -v --spin-time "$spin" --no-daemon 2>/dev/null || true)"
# "Published topics:" only - a bare `ros2 topic list` also lists topics that have
# nothing but a SUBSCRIBER, which once made a dead GELLO topic look healthy.
live="$(awk '/^Published topics:/{p=1;next} /^Subscribed topics:/{p=0} p && /^ \* /{print $2}' <<<"$raw")"
[ -n "$live" ] || { echo "ERROR: no topics discovered at all. Is the station stack up?" >&2; exit 1; }

missing=(); present=0
printf '\n%-64s %s\n' TOPIC STATUS
for t in "${topics[@]}"; do
  if grep -qxF "$t" <<<"$live"; then printf '%-64s ok\n' "$t"; present=$((present+1))
  else printf '%-64s MISSING\n' "$t"; missing+=("$t"); fi
done
printf '\n%d/%d present\n' "$present" "${#topics[@]}"
if [ "${#missing[@]}" -gt 0 ]; then
  echo
  echo "${#missing[@]} topic(s) have no publisher; they will be recorded EMPTY:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "(spine/base topics are expected missing on the arms-only start_upper.bash stack)" >&2
fi
[ "$mode" = check ] && exit 0

# ----------------------------------------------------------------- record
mkdir -p "$BAG_ROOT"
est_bps="$("$PY" "$ROOT/tools/rig_probes/rig_bag_topics.py" --cameras "$cameras" --bytes-per-second)"
est_mb=$(( est_bps * ${seconds%.*} / 1000000 ))
avail_mb="$(df -Pm "$BAG_ROOT" | awk 'NR==2{print $4}')"
echo
echo "Estimated size: ~${est_mb} MB for ${seconds}s of ${cameras}  (free here: ${avail_mb} MB)"
echo "  NOTE: contract camera shapes. The wrists actually stream 848x480, not the"
echo "  contract's 640x480 (U-21 open), so the real bag runs ~15% larger."
if [ "$avail_mb" -lt $(( est_mb * 2 )) ]; then
  echo "REFUSING: less than 2x the estimate free on $BAG_ROOT." >&2
  exit 1
fi

out="$BAG_ROOT/rigbag_$(date +%Y%m%d-%H%M%S)${name:+_$name}"
echo
echo "Recording -> $out"
echo "  ${#topics[@]} topics, ${seconds}s, bounded by timeout --signal=TERM (finalises itself)"
echo

# Baselines. R-22/R-23's OOM took the arms down with it, so memory is watched, and
# R-48's UdpRcvbufErrors is the acceptance number for the DDS profile actually working.
mem_before="$(free -m | awk '/^Mem:/{print $3}')"
# nstat is not guaranteed to exist (it does not in the camelo-ros container),
# and under `set -e` a missing-command 127 inside a $(...) pipeline aborts the
# whole script right before recording starts - silently, since stderr is
# swallowed. Guard it explicitly; the UdpRcvbufErrors delta (R-48/R-49's DDS
# acceptance number) is a nice-to-have on top of the recording, never a gate.
udp_before=""
command -v nstat >/dev/null 2>&1 && \
  udp_before="$(nstat -az UdpRcvbufErrors 2>/dev/null | awk '/UdpRcvbufErrors/{print $2}')"

# `ros2 bag record` has NO stop-after-duration flag - `-d/--max-bag-duration` only
# SPLITS the bag and keeps recording (checked against the CLI, not assumed). So the
# timer is ours: background the recorder and SIGINT it. That signal is what makes
# rosbag2 finalise; a recorder killed any other way leaves the bag WITHOUT
# metadata.yaml and rosbag2 then cannot read it at all (observed on this rig: a
# 124 MB mcap with no metadata).
# Background + `wait`, not a foreground call: bash defers a trap until the current
# foreground command returns, so Ctrl+C would be deferred until the recorder ended
# by itself - which it never does.
# --max-cache-size 1 GiB (default ~100 MiB): three raw cameras are ~156 MB/s, which
# the default cache holds for well under a second, so any disk hiccup backs up into
# the DDS subscribers and reads as camera frame drops in the bag.
# `timeout --signal=TERM`, not a manual SIGINT via pkill/kill. rosbag2's recorder
# is a python process (rclpy in-process, no exec to a C++ binary), and measured
# here: SIGINT delivered to it by `pkill -INT` was NOT acted on within 10+ s
# (still alive, still recording) while plain SIGTERM finalised cleanly in under a
# second every time - `metadata.yaml` written, clean exit. `--preserve-status`
# makes $? rosbag2's own exit code, not 124. `timeout` relays a signal it
# receives itself to its child, so an early Ctrl+C forwarded to $rec_pid below
# stops the recording the same way the deadline does.
timeout --signal=TERM --preserve-status "$seconds" \
  ros2 bag record -s mcap --max-cache-size 1073741824 -o "$out" "${topics[@]}" &
rec_pid=$!
trap 'echo; echo "stopping early..."; kill -TERM "$rec_pid" 2>/dev/null || true' INT TERM

# rosbag2 creates its output directory as soon as it starts, well before the
# first message - so its appearance is a reliable "recording has begun" signal.
# Print one greppable line so a launcher script (e.g. one that starts a policy
# alongside the recording) can tail this script's stdout and act on it.
started=false
for _ in $(seq 1 100); do
  if [ -d "$out" ]; then
    echo "RECORDING_STARTED pid=$rec_pid out=$out"
    started=true
    break
  fi
  kill -0 "$rec_pid" 2>/dev/null || break
  sleep 0.1
done
$started || echo "WARNING: recorder exited before $out was created - see above output" >&2

wait "$rec_pid" 2>/dev/null || true
trap - INT TERM

mem_after="$(free -m | awk '/^Mem:/{print $3}')"
udp_after=""
command -v nstat >/dev/null 2>&1 && \
  udp_after="$(nstat -az UdpRcvbufErrors 2>/dev/null | awk '/UdpRcvbufErrors/{print $2}')"
echo
echo "MEM used: ${mem_before} -> ${mem_after} MB   (R-22 baseline is 5.7-6.4 GB and flat)"
if [ -n "$udp_before" ] && [ -n "$udp_after" ]; then
  echo "UdpRcvbufErrors delta: $(( udp_after - udp_before ))  (R-49 Run B saw +781 with none of it on the camera path)"
else
  echo "UdpRcvbufErrors: nstat unavailable, not measured"
fi
echo
[ -f "$out/metadata.yaml" ] || { echo "ERROR: no metadata.yaml - the bag is unreadable." >&2; exit 1; }
echo "Bag: $out  ($(du -sh "$out" | cut -f1))"
ros2 bag info "$out"
echo
echo "Zero-message topics (if any) are the ones to explain before trusting this bag."
echo "Copy it off with:"
echo "  rsync -av --info=progress2 ebim@192.168.0.5:${out#"$HOME"/} <local>/"
