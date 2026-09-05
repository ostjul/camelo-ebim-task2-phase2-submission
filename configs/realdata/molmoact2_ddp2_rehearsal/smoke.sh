#!/bin/bash
# The rehearsal itself -- this directory has no separate long-run script.
# 2xH100 DDP, gpu-test, 00:30:00. CPUS/MEM scale the documented 4-GPU
# ratio (16 cores + 64G per GPU, docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md
# §6) down to 2 GPUs -- not independently spelled out in the docs for the
# 2-GPU case, flagged as inferred rather than sourced.
# `--multi_gpu` is LOAD-BEARING, not decoration (added 2026-08-31 after the
# 2-GPU rehearsal, job 3983176, died in 11 s with
#   ImportError: DeepSpeed is not installed => run `pip3 install deepspeed`).
# `accelerate launch` reads $HF_HOME/accelerate/default_config.yaml when no
# distributed flag is given, and the one on this account is a leftover from an
# unrelated project: `distributed_type: DEEPSPEED`, pointing at a ZeRO-2 JSON
# in a different workspace. accelerate/commands/launch.py:1215-1225 only
# adopts that file's distributed_type when NONE of --multi_gpu/--cpu/--tpu/
# --use_deepspeed/--use_fsdp was passed; passing --multi_gpu skips the block
# entirely, leaving use_deepspeed False so :1397 dispatches to
# multi_gpu_launcher. Explicitly asking for DDP is also simply more honest
# than inheriting whatever a global config happens to say.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/molmoact2_ddp2_rehearsal/train.yaml' DATA="$DATA" \
EXTRA="$EPISODES" \
LAUNCHER='accelerate launch --multi_gpu --num_processes=2 --num_cpu_threads_per_process=4' \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:2 --cpus-per-task=32 --mem=128G --job-name=camelo-smoke-molmoact2-2gpu \
    slurm/train.slurm
