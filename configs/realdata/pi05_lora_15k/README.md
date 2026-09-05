# pi0.5 LoRA v2 @ 15k — first wave, guaranteed deliverable

**Purpose.** The best-measured recipe on this cluster, sized to land before T0 with margin. Peer run: `../pi05_lora_30k/` (same recipe, 30k steps).
**Init:** `lerobot/pi05_base` (cached, 3.6B params).
**State/action:** 27/15; state rides the prompt (no cap), action zero-padded to 32 internally.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction (1.0=open), no flip.
**Normalization:** QUANTILES against the recomputed corpus stats (mandatory — shipped stats clip 40–55% of values).
**Cameras / image op:** `rename_map` to pi0.5's DROID slots; letterbox (`resize_with_pad_torch`, 224×126 content in 224×224) unchanged (U11 — matches DROID's own 16:9 pretraining letterbox).
**Horizon:** `chunk_size=n_action_steps=50` (2.5s at 20Hz).
**Trainable:** LoRA r=32, α=64 (**explicit** — unset defaults to PEFT's α=8, halving every update), 130-module regex → 14,059,520 params.
**Batch/steps/LR:** batch 32, 15,000 steps (4.82 epochs), preset LR 2.5e-5→2.5e-6 cosine, warmup 1000, `scheduler_decay_steps=15000` (must equal `steps`). `save_freq=1250`.
**GPUs/walltime:** 1×H100, 08:00:00 requested. Est. 6.8h (measured 1.636s/step, job 3872346).
**Launch:** `configs/realdata/pi05_lora_15k/launch.sh`
**Smoke:** `configs/realdata/pi05_lora_15k/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes. This recipe's worst sim-lineage record was ABSENT (0 executed closes/1,120 chunks, channel pinned open) and image-*indifferent* — the opposite of vision-timed; treat a repeat as a real finding, not noise.

---

**Launched 2026-08-31** (15,000 steps x batch 32 / 99,654 frames = **4.82 epochs**) — smoke **3983179 PASS** -- trainable **14,059,520** (the expected 14.1 M, not 1.3 M), updt_s 1.640 (vs 1.636 on job 3872346, a different corpus), mem_gb 20.1, loss 0.108->0.051, no F-81 and no F-85 signature -> long run **3983315**. Walltime raised 08:00:00 -> **11:00:00** for the eval passes (13_LAUNCH_LOG.md W-1).

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
