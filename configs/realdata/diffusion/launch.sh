#!/bin/bash
# Full Diffusion Policy run. 1 GPU. No measurement of any kind exists for
# this rung on any corpus (docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md
# §2.4) -- 8 h is budget, not a derived floor. Read the real number off
# the smoke.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/diffusion/train.yaml \
    DATA="$DATA" \
    TIME=08:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
