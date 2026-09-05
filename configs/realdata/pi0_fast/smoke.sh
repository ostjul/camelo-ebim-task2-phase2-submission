#!/bin/bash
# 300-step gpu-test smoke -- checks the hand-authored PEFT regex actually
# matches (an empty match silently trains 0 params), and whether updt_s on
# 720p AV1 frames fits at all within a sane 25,000-step budget.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/pi0_fast/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-pi0fast \
    slurm/train.slurm
