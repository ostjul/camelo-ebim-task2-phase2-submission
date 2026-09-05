#!/usr/bin/env bash
# Bring up task2 Isaac Sim + helpers (foreground — use tmux).
#   ./scripts/launch_sim.sh           # DGX (default)
#   ./scripts/launch_sim.sh a10       # A10/A10G headless
#   ./scripts/launch_sim.sh a10 webrtc  # A10 + WebRTC viewer
#   ROBOT_X=4.4 ROBOT_Y=2.0 ROBOT_YAW=-180 ./scripts/launch_sim.sh
#     # override the robot spawn pose (x, y metres; yaw degrees, CCW+ —
#     # matches Makefile's ROBOT_X/ROBOT_Y/ROBOT_YAW, see there). Because
#     # the spawn transform IS the reset pose, this survives every episode
#     # reset. Default (all unset) = the scene's own task preset
#     # (4.4, 2.6, -90 — C.TASK2_SPAWN_XY_YAW).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-dgx}"
VIEWER="${2:-}"
EBIM_BENCHMARK_ROOT="${EBIM_BENCHMARK_ROOT:-../ebim-benchmark}"
# Robot spawn override, forwarded to `make sim-scene` below — same
# ROBOT_X/ROBOT_Y/ROBOT_YAW variables the Makefile already defines for this
# purpose (empty = the scene's own task preset, see Makefile).
ROBOT_X="${ROBOT_X:-}"
ROBOT_Y="${ROBOT_Y:-}"
ROBOT_YAW="${ROBOT_YAW:-}"

# Compose defaults REPUBLISHER_GRIPPER_INVERT=false when .env is missing;
# camelo follows .env.example (true). Seed once so the helper stack matches.
TASK2_ENV="${EBIM_BENCHMARK_ROOT}/task2_isaacsim/.env"
TASK2_ENV_EXAMPLE="${EBIM_BENCHMARK_ROOT}/task2_isaacsim/.env.example"
if [[ ! -f "$TASK2_ENV" && -f "$TASK2_ENV_EXAMPLE" ]]; then
  cp "$TASK2_ENV_EXAMPLE" "$TASK2_ENV"
  echo "launch_sim: created $TASK2_ENV from .env.example"
fi

usage() { echo "usage: $0 [dgx|a10] [webrtc]" >&2; exit 2; }

public_ip() {
  if [[ -n "${PUBLIC_IP:-}" ]]; then
    echo "$PUBLIC_IP"
    return
  fi
  local token ip
  token="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    ip="$(curl -fsS -H "X-aws-ec2-metadata-token: $token" \
      http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  else
    ip="$(curl -fsS --connect-timeout 2 \
      http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
  fi
  echo "$ip"
}

case "$MODE" in
  dgx)
    exec make sim-scene SCENE="${SCENE:-room}" \
      ROBOT_X="$ROBOT_X" ROBOT_Y="$ROBOT_Y" ROBOT_YAW="$ROBOT_YAW"
    ;;
  a10)
    if [[ -f "$EBIM_BENCHMARK_ROOT/task1_isaacsim/docker-compose.yml" ]]; then
      docker compose -f "$EBIM_BENCHMARK_ROOT/task1_isaacsim/docker-compose.yml" down >/dev/null 2>&1 || true
    fi
    if [[ "$VIEWER" == "webrtc" ]]; then
      IP="$(public_ip)"
      if [[ -z "$IP" ]]; then
        echo "launch_sim: set PUBLIC_IP=<eip> (or run on EC2 with instance metadata)" >&2
        exit 2
      fi
      cat <<EOF
WebRTC viewer → connect client to: $IP
  (plain IP only — not IP:port; client hardcodes TCP 49100 / UDP 47998)

SG inbound from your IP (both required; gray viewport = UDP media blocked):
  Custom TCP  49100
  Custom UDP  47998

Client: Isaac Sim WebRTC Streaming Client
  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/manual_livestream_clients.html

If still gray: on the instance run
  ss -ulnp | grep 47998
If that prints nothing, Kit never bound the media port (Isaac 5.1 streamsdk
quirk) — temporarily open UDP 7000-7500 from your IP as a test, or see
docs/setup/SETUP.md §6.
EOF
      # Default room (has the table); SCENE=barebone if the LFS room USD is missing.
      exec make sim-scene SCENE="${SCENE:-room}" LIVESTREAM=1 PUBLIC_IP="$IP" \
        ROBOT_X="$ROBOT_X" ROBOT_Y="$ROBOT_Y" ROBOT_YAW="$ROBOT_YAW"
    elif [[ -n "$VIEWER" ]]; then
      usage
    else
      exec make sim-scene SCENE="${SCENE:-room}" HEADLESS=1 \
        ROBOT_X="$ROBOT_X" ROBOT_Y="$ROBOT_Y" ROBOT_YAW="$ROBOT_YAW"
    fi
    ;;
  -h|--help) usage ;;
  *) usage ;;
esac
