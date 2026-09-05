#!/bin/bash
# Full pi0.5 LoRA v2 run, 15,000-step budget. 1 GPU. Measured 1.636 s/step
# (job 3872346) => 15000 x 1.636 = 6.8 h. Request well under the 24 h cliff.
#
# WALLTIME RAISED AT LAUNCH (2026-08-31), from 08:00:00 to 11:00:00. The original
# budget counted TRAINING steps only. It does not survive the eval pass:
# `eval_steps: 1000` + `dataset.eval_split: 0.1` makes the trainer walk the
# ENTIRE eval split every 1,000 steps (lerobot_train.py:653-666 -- it is a
# `for eval_batch in eval_dataloader` over the whole loader, not a fixed
# number of batches). Eval split = 20 episodes of the 195 = ~11,200 frames
# = ~351 batches at batch_size=32; at roughly half a training step per
# forward-only batch that is ~4.7 min per eval, x 15 evals = ~1.2 h on top of
# 6.8 h of training. The old request left no margin at all (6.8 + 1.2 = 8.0 h). Requested walltime is
# still far under the 24:00:00 cliff.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source configs/realdata/_lib.sh

EPISODES="$(train_episodes_flag)"

make train-slurm \
    CONFIG=configs/realdata/pi05_lora_15k/train.yaml \
    DATA="$DATA" \
    TIME=11:00:00 CPUS=16 MEM=128G \
    EXTRA="$EPISODES"
