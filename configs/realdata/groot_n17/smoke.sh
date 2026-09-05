#!/bin/bash
# 300-step gpu-test smoke -- the load-bearing one for this rung: measures
# real updt_s (decides whether 20,000 steps can be raised toward 30,000,
# still under the 24h requested-walltime cliff), confirms the Resize
# transform prevents np.stack() failing on mixed camera resolutions,
# watches host RSS (GR00T has OOM'd on host RAM before, not just VRAM),
# and confirms warmup comes out ~1,000 (policy.max_steps=300 here, so
# warmup=ceil(300*0.05)=15 -- expected for a smoke, not a bug).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"
mkdir -p outputs/slurm

CONFIG='configs/realdata/groot_n17/train.yaml' DATA="$DATA" \
EXTRA="--steps=300 --save_freq=300 --log_freq=50 --policy.max_steps=300 $EPISODES" \
sbatch --export=ALL --qos=gpu-test --time=00:30:00 \
    --gres=gpu:1 --cpus-per-task=8 --mem=128G --job-name=camelo-smoke-groot \
    slurm/train.slurm
