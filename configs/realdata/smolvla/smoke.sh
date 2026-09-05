#!/bin/bash
# 300-step gpu-test smoke -- checks the rebuilt state-27 base loads
# cleanly and load_vlm_weights=true actually initialises from the
# pretrained SmolVLA weights (not a randomly-initialised VLM).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/smolvla/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-smolvla \
    slurm/train.slurm
