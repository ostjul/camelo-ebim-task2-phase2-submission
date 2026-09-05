#!/bin/bash
# 300-step gpu-test smoke -- checks reinit_modules actually produces 27/15
# projections (not a silent shape mismatch), the 3-vs-2 camera question
# (see train.yaml caveat), updt_s, and inference latency against the
# 0.35 s (chunk_size=7 @ 20 Hz) replan budget the executor needs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/vla_jepa/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-vla-jepa \
    slurm/train.slurm
