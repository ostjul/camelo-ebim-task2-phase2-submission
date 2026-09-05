#!/bin/bash
# Full MolmoAct2 run, 4xH100 single-node DDP -- the only multi-GPU shape
# on this cluster (gpu:h200:8 sits in the DOWN "all" partition, confirmed
# twice; docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md "the four
# live findings" #4). Requires slurm/train.slurm's A0 fix (divides
# --num_workers by num_processes under a LAUNCHER) to already be landed --
# it is, per this repo's Makefile train-slurm-4gpu target/comment.
#
# STAGE FIRST, on the login node (compute nodes have no DNS):
#   hf download allenai/MolmoAct2
#   hf download allenai/MolmoAct2-FAST-Tokenizer   # only if action_mode uses discrete
#   checkquota   # budget 43.6 GB: MolmoAct2-DROID is pulled transitively
#                # at model-load time from the checkpoint's own config.json
#
# batch_size=8 in train.yaml x 4 processes = global 32, per the fixed
# global-batch contract (see train.yaml comment).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

# NOT `make train-slurm-4gpu`: that target hard-codes its LAUNCHER, and this
# rung needs one extra flag on it (M-2, 2026-08-31).
#   --num_cpu_threads_per_process=4
#     `slurm/train.slurm` exports OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK (=64
#     here) and `accelerate launch` then propagates that value into EVERY
#     rank (accelerate/utils/launch.py:195), and each rank's 16 spawned
#     dataloader workers inherit it -- 4 ranks x 64 OMP threads plus 64
#     worker processes on 64 cores. Passing the flag makes accelerate export
#     OMP_NUM_THREADS=4 per rank instead. Measured on one H100 (batch 16,
#     16 cores, 16 workers): data_s 0.42 -> 0.275 s/step, total step
#     4.33 -> 4.17 s. Small, because this rung is GPU-bound, but free.
# TIME=20:00:00, not 16:00:00: the measured per-rank rate at the 4-GPU shape
# (batch 8) is 2.20 s/step, so 25,000 steps is 15.3 h of pure stepping before
# DDP sync, the ~6 min startup and the eval passes. 16 h had no margin at all.
# `sbatch --test-only` on 2026-08-31 returned the SAME estimated start for
# 16/20/22 h at gpu:4, so the longer request costs nothing in queue position.
# Still under the 24 h cliff, and a 19:00 start lands ~15:00 on 09-01, before
# the evening T0.
make train-slurm \
    GPUS=4 CPUS=64 MEM=256G \
    LAUNCHER="accelerate launch --multi_gpu --num_processes=4 --num_cpu_threads_per_process=4" \
    CONFIG=configs/realdata/molmoact2/train.yaml \
    DATA="$DATA" \
    TIME=20:00:00 \
    EXTRA="$EPISODES"
