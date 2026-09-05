# MolmoAct2 — parallel wave (4×H100 DDP)

**Purpose.** Best pretraining match on paper (`yam_dual_molmoact2`: bimanual, absolute joint pose, 3 cameras, >720h) but the worst integration profile: 43.6GB unstaged, the cluster's first multi-GPU job, a volatile 4-GPU queue, an unmeasured VRAM estimate. Stage + rehearse (`../molmoact2_ddp2_rehearsal/`) before committing this run.
**Init:** `checkpoint_path=allenai/MolmoAct2` (never `--policy.path` — a raw HF repo, not a lerobot checkpoint), ~21.8GB + a transitive ~21.8GB `MolmoAct2-DROID` pull at load time (budget 43.6GB).
**State/action:** 27/15; no state cap (256-way token/dim); action padded to 32 (`expected_max_action_dim=32`, hard error otherwise).
**Gripper:** `normalize_gripper=false` — passthrough, found by the substring "gripper" in the feature name (corpus's rename to `franka_robot_right_gripper_open_fraction` is load-bearing). No flip.
**Normalization:** QUANTILES + clamp, against recomputed corpus stats.
**Cameras / image op:** `image_keys` explicit (head/wrist_left/wrist_right); own shipped squash-plus-tile pipeline unmodified (U11).
**Horizon:** `chunk_size=n_action_steps=20` (1.0s at 20Hz).
**Trainable:** `enable_lora_vlm=true`, `enable_lora_action_expert=false` (LoRA on VLM/ViT, fully-trainable action expert — AI2's own ablation calls action-expert-only the clearest failure mode).
**Batch/steps/LR:** global batch **32 fixed** (8/GPU × 4 GPUs — no gradient accumulation in lerobot 0.6.1), 25,000 steps (8.03 epochs on this corpus; AI2 quote 5.5 "epochs" at their own real-world scale), `scheduler_decay_steps=25000` explicit (default 100k parks LR high). LoRA LR hard-coded 5e-5 internally. `save_freq=2083`.
**Eval:** `eval_steps=1000` **with `max_eval_samples=512`** — the cap is load-bearing, see below.
**GPUs/walltime:** 4×H100, **20:00:00** requested (was 16:00:00; see the throughput section).
**Launch:** `configs/realdata/molmoact2/launch.sh` — `make train-slurm GPUS=4` with an explicit LAUNCHER, **not** `make train-slurm-4gpu`, because that target hard-codes its launcher and this rung needs `--num_cpu_threads_per_process=4` on it (M-2). Requires rehearsal + staging first.
**Smoke:** `configs/realdata/molmoact2/smoke.sh` (300 steps, 4×H100, `gpu-test`, 00:55:00, `log_freq=25`, `eval_steps=0` — submit early, queue wait is volatile)
**Expect / selection probe:** M1 gripper-close timing, only if the run lands and a rig slot is free — explicitly opportunistic. `setup_type`/`control_mode` in train.yaml are this task's best-effort assumption, not sourced — verify first.

---

**Launched 2026-08-31 — 4×H100, job 3986557** (25,000 steps × global batch 32 / 99,654 frames = **8.03 epochs**), 20:00:00 requested (raised from 16:00:00).

Three attempts to get here, and the first two diagnoses were both wrong in instructive ways — the full story is `docs/realdata/13_LAUNCH_LOG.md` §6 and its M-2 update. In order: 2-GPU rehearsal 3983176 died in 11 s on a stale global accelerate config forcing DeepSpeed (`--multi_gpu` now on every DDP path); 3983334 emitted no step line at all and was read as "≥ 38.6 s/step, CPU-bound" — actually an `eval_steps: 50` artifact against `log_freq: 200`, with the "0 % GPU" samples taken during worker spawn; 3986257 then ran all 300 steps at **4.19 s/step** on 2 GPUs once `eval_steps: 0` and `--num_cpu_threads_per_process=4` were set.

**The gate was a 4-GPU number, and it passed.** Smoke 3986484 ran to completion — `COMPLETED` 00:19:15, exit 0, 300/300 steps, checkpoint written: **2.316 s/step** over steps 50→300 (579 s for 250 steps; the first window alone reads 2.40, and §7 note 2 says not to trust one window), `updt_s` 2.12–2.30, `data_s` **0.162–0.185**, `mem_gb` 24.12/rank, loss 3.438 → 1.590. The M-2 update's thresholds were `data_s ≤ 0.6` and `s/step ≤ 2.88`; both hold with room. 25,000 × 2.316 = 16.1 h, plus ~0.25 h of startup and capped evals ≈ **16.3 h** against 20:00:00 — which is why the request is 20 h and not the originally booked 16 h, and why 20 h still leaves 22 % headroom (it tolerates 2.83 s/step).

`--num_cpu_threads_per_process=4` is load-bearing: without it `accelerate` propagates `OMP_NUM_THREADS=64` into all four ranks and `data_s` goes to 6.5 s/step (measured, job 3983482). `save_freq` was cut 2083 → **6250** (four 14 GB checkpoints instead of twelve) for the shared scratch fileset, 13 §6b.

---

**RESUMED 2026-09-01 — job 3989764**, after job 3986557 was OOM-killed by SLURM at step ~7,000. Full diagnosis in `docs/realdata/13_LAUNCH_LOG.md` §6c; launch with `configs/realdata/molmoact2/resume.sh`.

**Same host-OOM as EO-1, not a per-model fault.** `sacct` on the `.0` row: `State=OUT_OF_MEMORY`, `MaxRSS 255.97G` against `ReqMem 256G`, `Detected 2 oom_kill events`. Rank 1 SIGKILLed at 19:21:20; ranks 2 and 3 died first with `DataLoader worker … killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory.` raised at `lerobot_train.py:659` — inside `for eval_batch in eval_dataloader`. `/dev/shm` is charged to the job's memory cgroup, so the bus error and the OOM kill are one cause, not two. `mem_gb` read **24.30 of 80**: the GPUs were never close. Last training loss 0.774 (`discrete_ce_loss` 0.766, `action_flow_loss` 0.007).

**Why the eval pass is what tips it:** lerobot builds a *second* DataLoader for the held-out split (`lerobot_train.py:528`) with the same `num_workers` and the same `persistent_workers=True`. From the first eval onward each rank holds 16 training + 16 eval workers, permanently — 4 × 32 = **128 `spawn`ed processes**, none sharing a copy-on-write page with its parent.

**Resumes from 006250**, which is complete (14 GB, weights + `training_state/`).

**What changed (config only), identical to EO-1:** `--mem` 256G → **512G**, `--num_workers` 16 → **8**/rank, `--prefetch_factor` 4 → **2**. This rung has less dataloader slack than EO-1 (`data_s` 0.157 against `updt_s` 2.11, ~7 % of the step) but still ~2× headroom at 8 workers. `eval_steps` stays 1000: against `save_freq: 6250` it already collides only at 25000, the final step.

**`TRAIN_MODULE=camelo.train.resume_train` is LOAD-BEARING on this rung** — it cannot resume through the stock `lerobot-train` at all. Found by the R-2 smoke (job 3989706), which died before step 1 with `KeyError: Override keys ['normalizer_processor'] do not match any step in the saved configuration`. `lerobot_train.py:355` injects a hard-coded `normalizer_processor` override whenever `policy.pretrained_path` is set — and that field is `None` for a *fresh* MolmoAct2 run, because this rung initialises through `checkpoint_path`, not `policy.path`. Resume is exactly what sets it, so the override branch fires for the first time and hits a pipeline whose steps are named `molmoact2_masked_normalizer` / `molmoact2_masked_unnormalizer`. So the rung's resumability depended on *how it was initialised*, 200 lines away. `camelo/train/resume_train.py` patches lerobot in-process (never site-packages, the `pi05_train.py` contract) and drops override keys the saved pipeline does not define — correct rather than a workaround, since lerobot itself treats the checkpoint's processor state as authoritative on resume. It **refuses** to drop an override carrying a `stats` payload, so a future lerobot that pushes real statistics through this path fails loudly instead of silently training on the wrong normalization.

**Budget:** 18,750 × 2.26 s (in-flight measured) = 11.8 h, ≈ 12.0 h with the ~11 min startup for 43.6 GB of weights and 19 capped evals, against **16:00:00** requested.
