#!/bin/bash
# Full SmolVLA run. 1 GPU, opportunistic. Floor 2.1 h (40,000 x 0.19 s/step
# derived from a sim-corpus measurement, 16/85 samples/s at batch 16) --
# the real corpus's larger/AV1 frames are expected slower; read the smoke.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/smolvla/train.yaml \
    DATA="$DATA" \
    TIME=08:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
