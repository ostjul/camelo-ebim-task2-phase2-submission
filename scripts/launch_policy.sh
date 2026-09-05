#!/usr/bin/env bash
# Serve a policy and drive the live robot/sim (needs launch_sim.sh for sim).
#   ./scripts/launch_policy.sh              # DGX (default): host serve + one rollout
#   ./scripts/launch_policy.sh a10          # A10: camelo-policy + camelo-ros containers
#   ./scripts/launch_policy.sh eval         # DGX + scored eval (needs eval-stack-up)
#   ./scripts/launch_policy.sh a10 eval     # A10 + scored eval
#   ./scripts/launch_policy.sh --world real # real robot: TMR topics from record_bag.bash
#   ./scripts/launch_policy.sh a10 real
#   ADAPTER=replay ./scripts/launch_policy.sh [dgx|a10]  # open-loop demo chunks (ep 89, no HF)
#   CORRECT_POSES=True N=20 ADAPTER=replay ./scripts/launch_policy.sh a10 eval
#   ADAPTER=heuristic ./scripts/launch_policy.sh [dgx|a10]  # demo trajectory + IK (default ON)
#   ADAPTER=lerobot CKPT=<run-or-pretrained_model> ACTION_LAYOUT=canonical STATE_LAYOUT=model16 \
#     ./scripts/launch_policy.sh [dgx|a10] [eval]   # local fine-tune (LeRobotAdapter)
#   RUN_ARGS="--approach perception" ./scripts/launch_policy.sh [dgx|a10]
#     # perception-based desk navigation (camelo/cli.py --approach); passed
#     # through to run_policy.py, ignored by `eval` (use EVAL_ARGS there)
#   START_XY_YAW=4.4,2.6,-1.5708 RUN_ARGS="--approach perception" \
#     ./scripts/launch_policy.sh [dgx|a10]
#     # shorthand for RUN_ARGS="... --approach-start-xy-yaw 4.4,2.6,-1.5708";
#     # rough (x, y, yaw rad) that seeds the perception approach's pose
#     # filter (Task 2 sim spawn shown); composes with RUN_ARGS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HW="dgx"
ACTION="policy"
WORLD="${WORLD:-${CAMELO_WORLD:-sim}}"
ADAPTER="${ADAPTER:-gr00t}"
# Not SECONDS — that is a bash builtin that auto-increments (elapsed wall time).
# Make's variable stays SECONDS; only this shell binding needs a distinct name.
ROLLOUT_S="${ROLLOUT_S:-45}"
EVAL_ROLLOUT_S="${EVAL_ROLLOUT_S:-45}"
# Scored-eval episode count. Override with N=… (e.g. N=1 for a quick smoke).
N="${N:-1}"
SERVER="${SERVER:-ws://127.0.0.1:8765}"
SERVE_PID=""
SERVE_LOG="${SERVE_LOG:-/tmp/camelo-serve.log}"
COMPOSE=("$ROOT/scripts/ros_compose.sh")
REPLAY_EP="${REPLAY_EP:-89}"
HEURISTIC_EP="${HEURISTIC_EP:-89}"
# Whether the caller set CORRECT_POSES at all: replay defaults IK off,
# heuristic defaults IK ON, so an unset variable means different things.
CORRECT_POSES_SET="${CORRECT_POSES+x}"
CORRECT_POSES="${CORRECT_POSES:-False}"
SERVED_IN_DOCKER=0
# Same names as Makefile: CKPT -> --checkpoint, plus layout flags (F-45/F-63).
CKPT="${CKPT:-}"
ACTION_LAYOUT="${ACTION_LAYOUT:-}"
STATE_LAYOUT="${STATE_LAYOUT:-}"
RUN_ARGS="${RUN_ARGS:-}"
START_XY_YAW="${START_XY_YAW:-}"
SERVE_FLAGS=()

usage() {
  echo "usage: $0 [dgx|a10] [eval] [--world sim|real]" >&2
  echo "  policy (default): run-policy with ROLLOUT_S=${ROLLOUT_S} s" >&2
  echo "  eval: make eval with N=${N} episodes, EVAL_ROLLOUT_S=${EVAL_ROLLOUT_S} sim-s + summary.json (needs eval-stack-up)" >&2
  echo "  --world real: TMR station topics (record_bag.bash); skips sim approach + eval" >&2
  echo "  ADAPTER=replay: open-loop chunks from hermanprawiro/task2_fixpos_200 (any HW; bundled REPLAY_EP=89, no HF_TOKEN)" >&2
  echo "  CORRECT_POSES=True: full demo replay + right-arm residual IK to GT TCP" >&2
  echo "  ADAPTER=heuristic: demo trajectory + IK, on by default (bundled HEURISTIC_EP=89; CORRECT_POSES=False disables IK)" >&2
  echo "  CKPT=<run-or-pretrained_model>: --checkpoint override. Local dirs need ACTION_LAYOUT + STATE_LAYOUT." >&2
  echo "  RUN_ARGS=<extra run_policy.py flags>: e.g. RUN_ARGS=\"--approach perception\" (policy action only)" >&2
  echo "  START_XY_YAW=x,y,yaw: shorthand for --approach-start-xy-yaw (rad); seeds the" >&2
  echo "    perception approach's pose filter; composes with RUN_ARGS (policy action only)" >&2
  exit 2
}

is_truthy() {
  case "${1:-}" in
    True|true|1|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

# --gpus all only when a device is visible (CPU boxes die on CDI otherwise).
# Heuristic/replay never request a GPU — they are numpy-only.
docker_gpu_flags() {
  if "$ROOT/scripts/ros_compose.sh" --want-gpu; then
    echo --gpus
    echo all
  fi
}

# Hub id unchanged. A train run dir becomes its latest checkpoints/*/pretrained_model.
# Paths under the repo are made relative so the A10 bind-mount sees them.
resolve_ckpt() {
  local path="$1" latest="" d abs
  [[ -n "$path" ]] || return 0
  if [[ -d "$path" && ! -f "$path/config.json" ]]; then
    for d in "$path"/checkpoints/*/pretrained_model; do
      [[ -f "$d/config.json" ]] && latest="$d"
    done
    if [[ -z "$latest" ]]; then
      echo "launch_policy: CKPT=$path has no config.json or checkpoints/*/pretrained_model" >&2
      exit 1
    fi
    path="$latest"
    echo "launch_policy: resolved run dir to $path" >&2
  fi
  if [[ -d "$path" ]]; then
    abs="$(cd "$path" && pwd)"
    case "$abs" in
      "$ROOT"/*) path="${abs#"$ROOT"/}" ;;
      *)
        if [[ "$HW" == "a10" ]]; then
          echo "launch_policy: CKPT $abs is outside the repo; camelo-policy only mounts $ROOT" >&2
          exit 1
        fi
        path="$abs"
        ;;
    esac
  elif [[ "$path" == /* || "$path" == ./* || "$path" == ../* ]]; then
    echo "launch_policy: CKPT=$1 does not exist" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world)
      [[ $# -ge 2 ]] || usage
      WORLD="$2"
      shift 2
      ;;
    --world=*)
      WORLD="${1#*=}"
      shift
      ;;
    sim|real)
      WORLD="$1"
      shift
      ;;
    dgx|a10)
      HW="$1"
      shift
      ;;
    eval)
      ACTION="eval"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "launch_policy: unknown argument: $1" >&2
      usage
      ;;
  esac
done
[[ "$WORLD" == "sim" || "$WORLD" == "real" ]] || usage
[[ "$HW" == "dgx" || "$HW" == "a10" ]] || usage
[[ "$ACTION" == "policy" || "$ACTION" == "eval" ]] || usage
if [[ "$WORLD" == "real" && "$ACTION" == "eval" ]]; then
  echo "launch_policy: --world real has no eval-camera scorer; use policy (not eval)" >&2
  exit 2
fi
echo "launch_policy: hw=$HW action=$ACTION world=$WORLD"

cleanup() {
  if [[ -n "$SERVE_PID" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ensure_image() {
  local image="$1" service="$2"
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "launch_policy: building $service…"
    "${COMPOSE[@]}" build "$service"
  fi
}

# Stop leftover policy serve containers from a previous aborted launch.
stop_stale_policy() {
  local ids
  ids="$(docker ps -q --filter ancestor=camelo-ebim/policy:latest 2>/dev/null || true)"
  if [[ -n "$ids" ]]; then
    echo "launch_policy: stopping stale camelo-policy container(s)"
    # shellcheck disable=SC2086
    docker stop $ids >/dev/null 2>&1 || true
  fi
}

replay_actions_path() {
  echo "data/replay/task2_fixpos_200/ep$(printf '%03d' "$REPLAY_EP")_actions.npy"
}

replay_gt_path() {
  echo "data/replay/task2_fixpos_200/ep$(printf '%03d' "$REPLAY_EP")_gt_traj.npz"
}

heuristic_actions_path() {
  echo "data/heuristic/task2_fixpos_200/ep$(printf '%03d' "$HEURISTIC_EP")_actions.npy"
}

heuristic_gt_path() {
  echo "data/heuristic/task2_fixpos_200/ep$(printf '%03d' "$HEURISTIC_EP")_gt_traj.npz"
}

# Materialise one episode's action column + GT TCP npz for ADAPTER=replay
# (videos skipped). Episode 89 is bundled in data/replay/ — no hub fetch.
# Other episodes run in camelo-policy so we get huggingface_hub + pandas
# without a host venv. Does not mount ebim-benchmark.
require_hf_token_for_prepare() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "launch_policy: episode $REPLAY_EP is not bundled; export HF_TOKEN or use the default REPLAY_EP=89" >&2
    exit 1
  fi
}

run_prepare_replay() {
  ensure_image camelo-ebim/policy:latest camelo-policy
  # HF cache mount: prepare downloads the gated dataset into the host cache.
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  docker run --rm \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -v "$ROOT:/workspace/camelo-ebim" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -w /workspace/camelo-ebim \
    camelo-ebim/policy:latest \
    python -u scripts/prepare_replay_actions.py --episode "$REPLAY_EP" "$@"
}

ensure_replay_actions() {
  local out gt
  out="$(replay_actions_path)"
  gt="$(replay_gt_path)"
  if [[ -f "$out" ]]; then
    if is_truthy "$CORRECT_POSES" && [[ ! -f "$gt" ]]; then
      echo "launch_policy: CORRECT_POSES set, GT npz missing — re-running prepare --skip-download"
      run_prepare_replay --skip-download
    else
      echo "launch_policy: replay actions ready ($out)"
    fi
    return
  fi
  echo "launch_policy: preparing replay actions (episode $REPLAY_EP)…"
  require_hf_token_for_prepare
  run_prepare_replay
}

# Heuristic twin of the replay prepare path. Single-episode fetch only;
# episode 89 is bundled in data/heuristic/ — no hub fetch.
require_hf_token_for_heuristic() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "launch_policy: episode $HEURISTIC_EP is not bundled; export HF_TOKEN or use the default HEURISTIC_EP=89" >&2
    exit 1
  fi
}

run_prepare_heuristic() {
  ensure_image camelo-ebim/policy:latest camelo-policy
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  docker run --rm \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -v "$ROOT:/workspace/camelo-ebim" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -w /workspace/camelo-ebim \
    camelo-ebim/policy:latest \
    python -u scripts/prepare_heuristic_actions.py --episode "$HEURISTIC_EP" "$@"
}

ensure_heuristic_actions() {
  local out gt
  out="$(heuristic_actions_path)"
  gt="$(heuristic_gt_path)"
  if [[ -f "$out" ]]; then
    # IK is the heuristic default, so the GT npz is required unless
    # CORRECT_POSES is explicitly falsey.
    if [[ ! -f "$gt" ]] && ! { [[ -n "${CORRECT_POSES_SET:-}" ]] && ! is_truthy "$CORRECT_POSES"; }; then
      echo "launch_policy: GT npz missing — re-running prepare --skip-download"
      run_prepare_heuristic --skip-download
    else
      echo "launch_policy: heuristic actions ready ($out)"
    fi
    return
  fi
  echo "launch_policy: preparing heuristic actions (episode $HEURISTIC_EP)…"
  require_hf_token_for_heuristic
  run_prepare_heuristic
}

serve_policy() {
  : >"$SERVE_LOG"

  # Heuristic is HW-agnostic: serve from camelo-policy + prepared npy.
  # IK pose correction is the adapter default; CORRECT_POSES=False disables.
  if [[ "$ADAPTER" == "heuristic" ]]; then
    ensure_image camelo-ebim/policy:latest camelo-policy
    stop_stale_policy
    echo "launch_policy: serving heuristic via camelo-policy (log: $SERVE_LOG)"
    heuristic_flags=()
    if [[ -n "$CORRECT_POSES_SET" ]] && ! is_truthy "$CORRECT_POSES"; then
      echo "launch_policy: CORRECT_POSES off — plain open-loop demo chunks"
      heuristic_flags+=(--no-correct-poses)
    fi
    docker run --rm --network host \
      -v "$ROOT:/workspace/camelo-ebim" \
      -w /workspace/camelo-ebim \
      camelo-ebim/policy:latest \
      python -u scripts/serve_policy.py --adapter heuristic \
        --checkpoint "$(heuristic_actions_path)" \
        "${heuristic_flags[@]}" \
      >"$SERVE_LOG" 2>&1 &
    SERVE_PID=$!
    SERVED_IN_DOCKER=1
    return
  fi

  # Replay is HW-agnostic: always serve from camelo-policy + prepared npy.
  if [[ "$ADAPTER" == "replay" ]]; then
    ensure_image camelo-ebim/policy:latest camelo-policy
    stop_stale_policy
    echo "launch_policy: serving replay via camelo-policy (log: $SERVE_LOG)"
    replay_flags=()
    if is_truthy "$CORRECT_POSES"; then
      echo "launch_policy: CORRECT_POSES on — full demo + right-arm residual IK"
      replay_flags+=(--correct-poses)
    fi
    docker run --rm --network host \
      -v "$ROOT:/workspace/camelo-ebim" \
      -w /workspace/camelo-ebim \
      camelo-ebim/policy:latest \
      python -u scripts/serve_policy.py --adapter replay \
        --checkpoint "$(replay_actions_path)" \
        "${replay_flags[@]}" \
      >"$SERVE_LOG" 2>&1 &
    SERVE_PID=$!
    SERVED_IN_DOCKER=1
    return
  fi

  if [[ "$HW" == "a10" ]]; then
    ensure_image camelo-ebim/policy:latest camelo-policy
    stop_stale_policy
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "launch_policy: warning: HF_TOKEN unset — container will use ~/.cache/huggingface/token" >&2
    fi
    echo "launch_policy: serving $ADAPTER via camelo-policy (log: $SERVE_LOG)"
    # docker run --network host (compose run has no --network on this Docker).
    # Pass HF_TOKEN: gate access is per-token; host curl with $HF_TOKEN can
    # succeed while the mounted cache token still 403s on Cosmos.
    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
    gpu_flags=()
    while IFS= read -r line; do
      gpu_flags+=("$line")
    done < <(docker_gpu_flags)
    docker run --rm --network host ${gpu_flags[@]+"${gpu_flags[@]}"} \
      -e HF_TOKEN \
      -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
      -v "$ROOT:/workspace/camelo-ebim" \
      -v "$HF_CACHE:/root/.cache/huggingface" \
      -w /workspace/camelo-ebim \
      camelo-ebim/policy:latest \
      python -u scripts/serve_policy.py --adapter "$ADAPTER" \
        "${SERVE_FLAGS[@]}" \
      >"$SERVE_LOG" 2>&1 &
    SERVE_PID=$!
    SERVED_IN_DOCKER=1
    return
  fi
  # DGX: policy deps on the host (camelo-policy is x86-only).
  if [[ -x .venv/bin/python ]] && .venv/bin/python -c "import lerobot" 2>/dev/null; then
    .venv/bin/python -u scripts/serve_policy.py --adapter "$ADAPTER" \
      "${SERVE_FLAGS[@]}" >"$SERVE_LOG" 2>&1 &
  elif python3 -c "import lerobot" 2>/dev/null; then
    python3 -u scripts/serve_policy.py --adapter "$ADAPTER" \
      "${SERVE_FLAGS[@]}" >"$SERVE_LOG" 2>&1 &
  else
    make serve ADAPTER="$ADAPTER" CKPT="$CKPT" >"$SERVE_LOG" 2>&1 &
  fi
  SERVE_PID=$!
}

wait_server() {
  local i
  # 300 × 2s = 10 min: pi0.5 LoRA load from cache is minutes, not seconds.
  for i in $(seq 1 300); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "launch_policy: policy server exited early; last log:" >&2
      tail -n 80 "$SERVE_LOG" >&2 || true
      exit 1
    fi
    if ss -ltn 2>/dev/null | grep -q ':8765 '; then
      echo "launch_policy: server up on $SERVER"
      return
    fi
    sleep 2
  done
  echo "launch_policy: timed out waiting for :8765; last log:" >&2
  tail -n 80 "$SERVE_LOG" >&2 || true
  exit 1
}

# START_XY_YAW is sugar over RUN_ARGS, not a separate mechanism: run_policy.py
# only knows --approach-start-xy-yaw (camelo/cli.py), so this just appends it.
if [[ -n "$START_XY_YAW" ]]; then
  RUN_ARGS="$RUN_ARGS --approach-start-xy-yaw $START_XY_YAW"
fi

ensure_image camelo-ebim/ros:latest camelo-ros
if [[ "$ADAPTER" == "replay" ]]; then
  ensure_replay_actions
elif [[ "$ADAPTER" == "heuristic" ]]; then
  ensure_heuristic_actions
elif [[ -n "$CKPT" ]]; then
  CKPT="$(resolve_ckpt "$CKPT")"
  if [[ -d "$CKPT" && ( -z "$ACTION_LAYOUT" || -z "$STATE_LAYOUT" ) ]]; then
    echo "launch_policy: local CKPT needs ACTION_LAYOUT and STATE_LAYOUT (F-45/F-63)" >&2
    echo "  derive them: python scripts/eval_recipe.py $CKPT" >&2
    echo "  pi0.5 on our contract: ACTION_LAYOUT=canonical STATE_LAYOUT=model16" >&2
    exit 1
  fi
  echo "launch_policy: checkpoint $CKPT"
fi
[[ -n "$CKPT" ]] && SERVE_FLAGS+=(--checkpoint "$CKPT")
[[ -n "$ACTION_LAYOUT" ]] && SERVE_FLAGS+=(--action-layout "$ACTION_LAYOUT")
[[ -n "$STATE_LAYOUT" ]] && SERVE_FLAGS+=(--state-layout "$STATE_LAYOUT")
serve_policy
wait_server
if [[ "$ACTION" == "eval" ]]; then
  echo "launch_policy: eval — ${N} episode(s), rollout ${EVAL_ROLLOUT_S} sim-s (override: EVAL_ROLLOUT_S=… N=…)"
  "${COMPOSE[@]}" run --rm -T -e CAMELO_WORLD="$WORLD" camelo-ros \
    make eval ADAPTER="$ADAPTER" BACKEND=remote SERVER="$SERVER" N="$N" \
      SECONDS="$EVAL_ROLLOUT_S" WORLD="$WORLD"
else
  echo "launch_policy: policy — ${ROLLOUT_S} s rollout, world=$WORLD"
  "${COMPOSE[@]}" run --rm -T -e CAMELO_WORLD="$WORLD" camelo-ros \
    make run-policy ADAPTER="$ADAPTER" BACKEND=remote SERVER="$SERVER" \
      SECONDS="$ROLLOUT_S" WORLD="$WORLD" "RUN_ARGS=$RUN_ARGS"
fi
# docker-run client kill can leave the policy container; reap explicitly.
stop_stale_policy
if [[ "$SERVED_IN_DOCKER" -eq 0 ]] && [[ -n "$SERVE_PID" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
  SERVE_PID=""
fi
