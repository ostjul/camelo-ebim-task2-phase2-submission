#!/bin/bash
# Full pi0-FAST run. 1 GPU, opportunistic -- launch only if a slot is
# idle and everything ranked above it is already running/evaluated.
# Prerequisite: `make stage RUNG=pi0fast` must have built
# outputs/models/pi0fast-base-lerobot-tokenizer first.
# WALLTIME RAISED AT LAUNCH (2026-08-31) from 16:00:00 to 21:00:00.
# Smoke 3983183 measured updt_s 2.048 / data_s 0.037 -> 2.085 s/step at
# batch 32, mem_gb 31.46. That is 25,000 x 2.085 = 14.5 h of training
# alone, and this rung keeps `eval_steps: 1000`, which walks the ENTIRE
# eval split (10,522 frames = 329 batches at batch 32) every 1,000 steps
# -- ~5.5 min x 25 evals = ~2.3 h more. 16.8 h against a 16:00:00 request
# is a job that dies at the wall with nothing to show; 21:00:00 gives 25%
# margin and is still under the 24 h cliff.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/pi0_fast/train.yaml \
    DATA="$DATA" \
    TIME=21:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
