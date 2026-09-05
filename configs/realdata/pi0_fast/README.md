# pi0-FAST — opportunistic

**Purpose.** Discrete-head control vs pi0.5's flow-matching head, matched LoRA recipe. A **science** run, not a rig candidate: at a matched budget on the sim corpus it measured FAIL/MISPLACED on the exact gripper-timing instrument checkpoint selection runs on (near 0.171/far 0.000, vs ACT's 0.958). Launch only if a GPU slot is idle.
**Init:** `outputs/models/pi0fast-base-lerobot-tokenizer` (local derived base — `make stage RUNG=pi0fast` first; never the raw `lerobot/pi0fast-base` hub id, its tokenizer config can't load in this transformers version).
**State/action:** 27/15, fixed 32-wide linear projection (pads, not truncates).
**Gripper:** state dim 14 open fraction, action dim 14 width fraction, no flip.
**Normalization:** MEAN_STD (class default).
**Cameras / image op:** `rename_map` to DROID slots, same letterbox as pi0.5.
**Horizon:** `chunk_size=n_action_steps=50` (2.5s).
**Trainable:** LoRA r=32, α=64, hand-authored regex targeting every LM layer's self-attn+MLP (PI0FastPolicy has no built-in PEFT default).
**Batch/steps/LR:** batch 32, **25,000 steps (8.03 epochs), cut from the sim corpus's 34,854-step convention** — this corpus's 720p AV1 frames plausibly slower per step, risked the 24h cliff, preset LR 2.5e-5→2.5e-6 cosine, `scheduler_decay_steps=25000`. `save_freq=2083`.
**GPUs/walltime:** 1×H100, 16:00:00 requested. Est. up to ~14h by analogy to the sim corpus's measured 20h18m/34,854-step run — trust the smoke.
**Launch:** `configs/realdata/pi0_fast/launch.sh`
**Smoke:** `configs/realdata/pi0_fast/smoke.sh` (300 steps, `gpu-test`, 00:30:00)
**Expect / selection probe:** M1 gripper-close timing on held-out episodes — expect MISPLACED or worse; a repeat is a real finding about FAST tokenization's chunk-boundary sensitivity. Don't spend rig rollouts here while anything ranked above is untested.

---

**Launched 2026-08-31** (25,000 steps x batch 32 / 99,654 frames = **8.03 epochs**) — smoke **3983183** -- updt_s 2.048, data_s 0.037 -> 2.085 s/step, mem_gb 31.46, ce_loss 11.295->7.240 (PEFT regex matches: 39,223,296 trainable, not 0). Walltime raised 16:00:00 -> **21:00:00**: 25,000 x 2.085 = 14.5 h plus ~2.3 h of eval passes does not fit 16 h.

_Epoch figures on this page are `steps x batch / 99,654` -- the frames the trainer actually sees. The corpus has 121,828 frames in 217 episodes; 195 are on the train list, and lerobot's `dataset.eval_split: 0.1` then holds back the last 20 of those, leaving **175 episodes / 99,654 frames** to train on (10,522 in the eval split). Earlier epoch numbers on this page were against a larger count._
