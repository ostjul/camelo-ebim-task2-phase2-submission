# EO-1 — second wave

**Purpose.** DROID is in the pretraining mixture, and the model card carries the strongest verbatim external number on the ladder ("Franka Pick-and-Place: 0.935" — authors' own number, demo count unknown). MIT licence.
**Init:** `policy.path=outputs/checkpoints/eo1_3b_lerobot` — a derived directory built from **`lerobot/eo1-base`**, upstream's own lossless key-layout conversion of `IPEC-COMMUNITY/EO-1-3B` at the same revision (`fe0a015d…`). Rebuild with `python scripts/stage_assets.py --rung eo1`. **Not** `vlm_base=IPEC-COMMUNITY/EO-1-3B` (that binds 0/836 tensors) and **not** the raw repo via `--policy.path` (raw `transformers` config, no lerobot `type` tag). **Verified, not assumed:** through lerobot's own load path on CPU, `missing_keys=0, unexpected_keys=0` over 836 tensors — the Qwen backbone *and* EO-1's flow-matching head, nothing freshly initialised.
**State/action:** 27/15, fixed linear projection, `max_state_dim=max_action_dim=32`.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** MEAN_STD (class default) — unspecified/ESTIMATE beyond dims/horizon in the protocol docs.
**Cameras / image op:** no `rename_map` — Qwen2.5-VL consumes generic multi-image input, not fixed slots.
**Horizon:** `chunk_size=n_action_steps=8` (0.4s at 20Hz) — tight replan, like VLA-JEPA's.
**Trainable:** full fine-tune (no PEFT field on this policy).
**Batch/steps/LR:** batch 16, **8,000 steps (1.28 epochs)** — re-derived from the measured 5.73–6.63 s/step, see train.yaml; 30,000 would be 55.3 h on one H100. AdamW lr 1e-4 cosine (preset). `save_freq=2000` (4 saves — an EO-1 checkpoint is **22 GB**), `eval_steps=2500`.
**GPUs/walltime:** 1×H100, 20:00:00 requested, **≈16.0 h projected** (MEASURED rate, not an estimate). `gradient_checkpointing: true` is **required** — without it, OOM at 79.14/79.17 GiB before step 1 (job 3986246).
**Launch:** `configs/realdata/eo1/launch.sh`
**Smoke:** `configs/realdata/eo1/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. No own-corpus prior exists; rank purely on the probe.

---

**Launched 2026-08-31 — 4×H100, job 3986556** (30,000 steps × global batch 16 / 99,654 frames = **4.82 epochs**), 20:00:00 requested.

Was BLOCKED on the morning pass (0 of 836 checkpoint tensors bound), unblocked by `lerobot/eo1-base` + `make stage RUNG=eo1` → `outputs/checkpoints/eo1_3b_lerobot`; see `train.yaml` and `docs/realdata/13_LAUNCH_LOG.md` §4 E-1 + its E-1 update.

**Why 4 GPUs, decided on a measurement not a projection.** One H100 at batch 16 measures 5.73–6.63 s/step (smokes 3984855 / 3986305) — a throughput ceiling that buys only ~8,000 steps / **1.28 epochs** in 20 h. A 4-GPU smoke at the *same global batch 16* (3986485, batch 4/rank) measured **1.88 s/step** — 47 s per 25 steps over three consecutive windows, `updt_s` 1.851–1.864, `data_s` **0.011**, `mem_gb` 43.19/rank. That is 3.5×, i.e. near-linear, so 30,000 steps = 15.7 h (18.0 h with the ladder's 15% margin) and **4.82 epochs**, level with `vla_jepa` and `pi05_lora_15k`.

The 1-GPU/8,000 fallback was live and safe; it lost because four GPUs turned out not to be scarce (`sbatch --test-only` at 14:24 gave a 14:34 start for a 20 h `gpu:4` job, and the same estimate for 16/20/22 h), and 1.28 epochs would have made this rung hard to read against the rest of the ladder.

`gradient_checkpointing: true` is mandatory, not defensive: without it job 3986246 dies with `torch.OutOfMemoryError` at 79.14 of 79.17 GiB before step 1. `save_freq: 7500` gives four 22 GB checkpoints instead of twelve — the shared scratch fileset was at 39.3 of 40 TiB when this was launched (13 §6b). `max_eval_samples: 512` caps each eval pass, which lerobot 0.6.1 otherwise runs over the entire 10,522-frame split.

---

**RESUMED 2026-09-01 — job 3989703**, after job 3986556 was OOM-killed by SLURM at step 15,000. Full diagnosis in `docs/realdata/13_LAUNCH_LOG.md` §6c; launch with `configs/realdata/eo1/resume.sh`.

**It was HOST memory, not VRAM.** `sacct` on the `.0` row: `State=OUT_OF_MEMORY`, `MaxRSS 255.97G` against `ReqMem 256G`, `Detected 1 oom_kill event`. Rank 0 was SIGKILLed *inside* `save_checkpoint` at step 15000 — which is a multiple of both `eval_steps: 2500` and `save_freq: 7500`, so the eval pass and rank 0 materialising 22 GB of model + AdamW state on the host landed on the same step. lerobot's `mem_gb` read **43.29 of 80** at the time: every GPU number was healthy, and the resource that ran out was never in the log.

**Resumes from 007500, not 015000.** The kill landed mid-save, so `checkpoints/015000/` holds only a 7.6 KB `config.json` — no weights, no `training_state/` — and `checkpoints/last` correctly still points at 007500. That makes the resume **22,500 steps**, not the 15,000 the directory listing suggests. Nothing was deleted.

**What changed (config only):** `--mem` 256G → **512G** (the node is ~1002 GiB and the partition sets `MaxMemPerNode=UNLIMITED`); `--num_workers` 16 → **8** per rank and `--prefetch_factor` 4 → **2**, which cuts the standing worker count from 128 to 64 — free here, since `data_s` was 0.011 against `updt_s` 1.874, i.e. 0.6 % of the step; and `eval_steps` 2500 → **2000** so eval and save stop colliding at every save. `slurm/resume.slurm` also samples the job's memory cgroup every 60 s (`MEMWATCH`), so the number that killed the first attempt is now visible in the log.

The resume runs `lerobot-train` **directly**, not `camelo.train.train`: this rung's `train.yaml` carries `policy.path`, and lerobot checks that *before* `resume`, so the wrapper would have restarted from the base checkpoint with a resumed step counter and nothing on the command line would have looked wrong. `--config_path=<ckpt>/pretrained_model/train_config.json --resume=true` with no `--policy.path` restores step, optimizer, scheduler and the original `output_dir` — which also keeps the two sidecars beside that dir valid (F-63).

**Budget:** 22,500 × 1.885 s (the in-flight measured rate, not the smoke's) = 11.8 h, ≈ 11.9 h with startup and 12 capped evals, against **16:00:00** requested — the same `gpu-short` QOS a 20 h request gets, so the shorter ask costs nothing and buys nothing in priority.
