#!/bin/bash
# Full GR00T N1.7 run. 1 GPU. Requested walltime 22:00:00 -- a margin
# under the hard 24h cliff on top of the 15% step-count margin already
# baked into train.yaml's steps=20000 (see that file's derivation
# comment). Also request generous host RAM: a 3-camera N1 run has died
# with a host-RAM MemoryError in the dataloader before, not CUDA OOM.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/groot_n17/train.yaml \
    DATA="$DATA" \
    TIME=22:00:00 CPUS=16 MEM=256G \
    EXTRA="$EPISODES"
