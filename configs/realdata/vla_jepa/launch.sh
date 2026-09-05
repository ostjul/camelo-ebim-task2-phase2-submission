#!/bin/bash
# Full VLA-JEPA run. 1 GPU. ESTIMATE only (0.8-1.5 s/step, unmeasured on
# any corpus) -- read the real number off the smoke before trusting
# the 7-13 h band.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/vla_jepa/train.yaml \
    DATA="$DATA" \
    TIME=16:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
