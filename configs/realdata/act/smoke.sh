#!/bin/bash
# 300-step gpu-test smoke: exit 0 + checkpoint written, trainable-param
# count, mem_gb, updt_s, and (for ACT specifically) whether a dataloader
# Resize is needed for the 720p head. gpu-test carries no MaxWall of its
# own -- 00:30:00 is this task's own budget, comfortably inside the
# ~60-minute window the historical-jobs probe measured
# (docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md "the four live
# findings" #1) -- NOT `make train-smoke`'s hardcoded 00:10:00.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/act/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-act \
    slurm/train.slurm
