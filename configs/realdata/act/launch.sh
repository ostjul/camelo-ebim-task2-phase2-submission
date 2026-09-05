#!/bin/bash
# Full ACT run on the real-robot corpus. 1 GPU, requested walltime well
# under the 24 h cliff (docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md
# "the four live findings" #2). Floor at 0.159 s/step (sim corpus) x
# 100,000 steps = 4.4 h; this corpus's AV1-decoded 720p head frames are
# expected to be slower -- read the real number off the smoke before
# trusting the floor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/act/train.yaml \
    DATA="$DATA" \
    TIME=12:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
