#!/bin/bash
# 300-step gpu-test smoke -- checks 27/15 fits the 32-dim caps cleanly,
# updt_s, and VRAM for the 3.77B Qwen2.5-VL-3B-init backbone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/eo1/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-eo1 \
    slurm/train.slurm
