#!/bin/bash
# 300-step gpu-test smoke -- same checklist as ../pi05_lora_15k/smoke.sh
# (trainable ~=14.1M, mem_gb~=20, updt_s~=1.6, no F-81/F-85 signatures).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/pi05_lora_30k/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-pi05-30k \
    slurm/train.slurm
