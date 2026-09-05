# pi0.5 LoRA v2 @ 30k — first wave, better checkpoint if it lands

**Purpose.** Same LoRA recipe as `../pi05_lora_15k/`, doubled step budget (6.62 vs 3.31 epochs). Do not substitute this run's step-15000 checkpoint for the dedicated 15k run — mid-cosine LR there is a worse object than an annealed one.
**Init:** `lerobot/pi05_base` (cached).
**State/action:** 27/15; state rides the prompt, action zero-padded to 32.
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** QUANTILES against recomputed corpus stats.
**Cameras / image op:** `rename_map` to DROID slots; letterbox unchanged (U11).
**Horizon:** `chunk_size=n_action_steps=50` (2.5s).
**Trainable:** LoRA r=32, α=64 explicit, 130-module regex → 14,059,520 params.
**Batch/steps/LR:** batch 32, 30,000 steps (9.63 epochs), preset LR 2.5e-5→2.5e-6 cosine, warmup 1000, `scheduler_decay_steps=30000`. `save_freq=2500`.
**GPUs/walltime:** 1×H100, 16:00:00 requested. Est. 13.6h (measured 1.636–1.64s/step).
**Launch:** `configs/realdata/pi05_lora_30k/launch.sh`
**Smoke:** `configs/realdata/pi05_lora_30k/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes — same caveat as 15k: prior record is ABSENT/image-indifferent, not vision-timed.

---

**Launched 2026-08-31** (30,000 steps x batch 32 / 99,654 frames = **9.63 epochs**) — covered by smoke **3983179** (same config, different step budget) -> long run **3983316**. Walltime raised 16:00:00 -> **20:00:00** for the eval passes (13_LAUNCH_LOG.md W-1): 13.7 h of training + ~2.4 h of eval did not fit 16 h.

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
