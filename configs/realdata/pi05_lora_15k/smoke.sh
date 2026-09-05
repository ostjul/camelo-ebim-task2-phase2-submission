#!/bin/bash
# 300-step gpu-test smoke. Checklist (docs/realdata/00_PROTOCOL.md §2.1):
# exit 0 + checkpoint written; `grep -i "could not load state dict"` empty
# (F-81 -- a silently randomly-initialised LoRA checkpoint); trainable
# params ~= 14.1 M, NOT 1.3 M (the unset-lora_alpha / wrong-regex failure
# mode); mem_gb ~= 20, updt_s ~= 1.6; loss falling from ~0.5, not 1e11
# (F-85, the collapsed-quantile signature).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/pi05_lora_15k/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --job-name=camelo-smoke-pi05-15k \
    slurm/train.slurm
