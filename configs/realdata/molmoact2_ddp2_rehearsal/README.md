# MolmoAct2 2-GPU DDP rehearsal — parallel wave, de-risking smoke

**Purpose.** Prove the `accelerate launch --num_processes=N` path and the `slurm/train.slurm` A0 `--num_workers` division fix at 2-GPU/negligible cost, before `../molmoact2/`'s 4-GPU run sits in a volatile multi-hour queue. Runs in parallel with the first wave, not gated behind it.
**Init/state/action/gripper/normalization/cameras/horizon/trainable:** identical to `../molmoact2/` — see that README.
**Batch:** global 32 = **16/GPU × 2 GPUs** (not 4).
**Steps:** **300, baked into train.yaml** — this directory's whole purpose is the rehearsal, not a scaled-down long run.
**GPUs/walltime:** 2×H100, `gpu-test`, 00:30:00.
**Launch = smoke:** `configs/realdata/molmoact2_ddp2_rehearsal/smoke.sh` (no separate long-run script).
**Expect / selection probe:** none — launcher-plumbing check, not a checkpoint candidate. Success = exit 0, both GPUs show non-trivial utilization, `--num_workers` correctly divided by 2, updt_s/VRAM recorded for comparison against 1-GPU/4-GPU numbers.
