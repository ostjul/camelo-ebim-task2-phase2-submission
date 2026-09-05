#!/usr/bin/env bash
# docker compose for camelo-ros, with GPU overlay only when a device is visible.
#   ./scripts/ros_compose.sh run --rm camelo-ros make check-obs
#   CAMELO_ROS_GPU=0 ./scripts/ros_compose.sh run --rm camelo-ros …  # force CPU
#   CAMELO_ROS_GPU=1 ./scripts/ros_compose.sh run --rm camelo-ros …  # force GPU
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

files=(-f docker/compose.yaml)

want_gpu() {
  case "${CAMELO_ROS_GPU:-auto}" in
    0|false|FALSE|no|NO) return 1 ;;
    1|true|TRUE|yes|YES) return 0 ;;
    auto|"") nvidia-smi -L >/dev/null 2>&1 ;;
    *)
      echo "ros_compose: CAMELO_ROS_GPU must be 0, 1, or auto (got ${CAMELO_ROS_GPU})" >&2
      exit 2
      ;;
  esac
}

# Query mode for callers that just want the GPU-visibility check (e.g.
# launch_policy.sh's docker_gpu_flags()) without running docker compose.
if [[ "${1:-}" == "--want-gpu" ]]; then
  want_gpu
  exit $?
fi

if want_gpu; then
  files+=(-f docker/compose.gpu.yaml)
fi

exec docker compose "${files[@]}" "$@"
