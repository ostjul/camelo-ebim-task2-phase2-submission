#!/bin/bash
# Full EO-1 run. 4xH100 single-node DDP, 30,000 steps, 20:00:00 requested.
#
# WHY 4 GPUs. On one H100 this rung measures 5.73-6.63 s/step at batch 16
# (smokes 3984855 / 3986305), which buys only ~8,000 steps = 1.28 epochs
# inside a 20 h request -- a throughput ceiling, not a choice: three cameras
# at EO-1's released 128x28x28 pixel budget cost ~1,400 visual tokens per
# sample, so one H100 sees ~2.6 samples/s. The 4-GPU shape was MEASURED
# rather than projected (smoke 3986485, batch 4/rank = the same global 16):
#     updt_s 1.850   data_s 0.011   mem_gb 43.19/rank
#     wall clock step:25 -> step:50 = 47 s / 25 = 1.88 s/step  (3.5x)
# 30,000 x 1.88 = 15.7 h; 18.0 h with the ladder's 15% margin; 20:00:00
# requested. 4.82 epochs instead of 1.28, at an unchanged global batch.
#
# `sbatch --test-only` at 14:24 on 2026-08-31 put a 20:00:00 gpu:4 job at a
# 14:34 start and returned the SAME estimate for 16 h, 20 h and 22 h -- four
# GPUs were not scarce at submission time and the longer request cost
# nothing in queue position. That is what made this option live; it was not
# true earlier the same day (see 13_LAUNCH_LOG.md §6).
#
# NOT `make train-slurm-4gpu`: that target hard-codes its LAUNCHER, and this
# one has to carry `--num_cpu_threads_per_process=4`. Without it,
# `slurm/train.slurm` exports OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK (=64) and
# `accelerate` propagates that into EVERY rank
# (accelerate/utils/launch.py:195), giving 4 ranks x 64 OMP threads on 64
# cores. On the MolmoAct2 rung that pathology cost 6.5 s/step of data_s (see
# the M-2 update in 13_LAUNCH_LOG.md §6). EO-1 is compute-bound so it is far
# less exposed -- and the flag is why we know that: 3986485's data_s came
# back at 0.011, so the absence is measured, not assumed. Check the job's
# `launcher='...'` echo line if a rate ever looks wrong.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    GPUS=4 CPUS=64 MEM=256G \
    LAUNCHER="accelerate launch --multi_gpu --num_processes=4 --num_cpu_threads_per_process=4" \
    CONFIG=configs/realdata/eo1/train.yaml \
    DATA="$DATA" \
    TIME=20:00:00 \
    EXTRA="$EPISODES"
