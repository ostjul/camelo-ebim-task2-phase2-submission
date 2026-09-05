# Real-robot (`task2_munich_s27a15_hires`) training configs

Per-policy configs for the confirmed plan in
[docs/realdata/00_PROTOCOL.md §2](../../docs/realdata/00_PROTOCOL.md) and
[docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md](../../docs/realdata/08_MODEL_SELECTION_AND_RIG_SCHEDULE.md)
(reviewer blockquotes override doc text where they disagree). All
policies use the canonical **27-dim `observation.state` / 15-dim
`action`** (source indices and names in `00_PROTOCOL.md` §0.2), 3 cameras
(`head`/`wrist_left`/`wrist_right` — **except `vla_jepa`, which its own
checkpoint restricts to 2**, see the wave table), 20 Hz, on
`outputs/datasets/ebim_task2_realrobotdata/task2_munich_s27a15_hires/` — the
`--drop-lowres` build (217 episodes, the 21 native-336x188 head episodes
removed rather than re-encoded). The root is named in exactly one
functional place, `_lib.sh`'s `DATA` default. Every `launch.sh`/`smoke.sh`
reads `$DATA/meta/train_episodes.txt` for the train-episode list at
submission time, not at config-authoring time — so switching corpora is the
one-line `DATA` edit and nothing else.

`PARSE_CHECK.md` records the pre-launch parse check, and it remains true as
far as it goes — but read 13_LAUNCH_LOG.md §4 next to it. **Five of the
eleven configs parsed cleanly and still failed on GPU**, every one of them
in the gap PARSE_CHECK.md names in its own "What this check does *not*
verify" section: weights that do not bind, camera keys that do not match,
tensors that cannot be stacked, a postprocessor step aimed at the wrong
joint, a launcher inheriting a global config. Parsing is not loading.

## Conventions

- **Train config:** `<policy>/train.yaml`, draccus-loadable, following the
  repo's existing pattern (`configs/pi05_ft_lora_v2.yaml`, `configs/act.yaml`,
  `configs/groot_ft.yaml`, ...). Every one carries, at the top level:
  `tolerance_s: 1.01` (never `dataset.tolerance_s` — that field does not
  exist on `DatasetConfig`), `dataset.video_backend: pyav`,
  `dataset.eval_split: 0.1`, `eval_steps: 1000`, and `policy.push_to_hub: false`.
  Three rungs (`diffusion`, `groot_n17`, `vla_jepa`) carry `eval_steps: 0`
  instead — not an oversight: lerobot builds the eval dataset **without**
  `dataset.image_transforms` (`datasets/factory.py:209`), and those three
  need that Resize to stack their cameras at all, so an eval pass would
  crash at step 1000. `eval_split` still stays 0.1 on all eleven so every
  rung trains on the identical 175 episodes (13_LAUNCH_LOG.md §D-2).
  `--dataset.episodes` is deliberately **absent** from every `train.yaml`
  — it depends on the corpus's frozen train/held-out split and is injected
  by `launch.sh`/`smoke.sh` at submission time (`_lib.sh`'s
  `train_episodes_flag()`).
- **Launch:** `<policy>/launch.sh` — the full run, via
  `make train-slurm`/`make train-slurm-4gpu` (1 GPU unless noted; ≤24:00:00
  requested walltime, always — the cluster has a **hard cliff** at 24h of
  *requested* walltime, not a priority gradient: a job asking one second
  more starts nine days later, docs/realdata/08 "the four live findings" #2).
- **Smoke:** `<policy>/smoke.sh` — 300 steps, `gpu-test` QOS, **00:30:00**
  requested (this task's own budget; `make train-smoke`'s Makefile
  default hardcodes 00:10:00, so these call `sbatch`/`slurm/train.slurm`
  directly with the same env-var convention instead of going through that
  target — `gpu-test` itself carries no `MaxWall`, the ~60-minute figure
  comes from historical-jobs inspection, so 00:30:00 has real margin).
  Run every smoke before its corresponding long run.
- **Resume:** `<policy>/resume.sh` (+ `resume_smoke.sh`) for the two 4-GPU
  rungs, via `slurm/resume.slurm`. These drive `lerobot-train` **directly**,
  not `camelo/train/train.py`: the wrapper re-emits the yaml, and a
  `policy.path` in it takes priority over `--resume` in lerobot's
  `_resolve_pretrained_from_cli`, so the run would restart from the base
  checkpoint with a resumed step counter and look fine. It also stamps a new
  job-ID `output_dir`, which orphans the sidecars keyed to the old one.
  MolmoAct2 additionally needs `TRAIN_MODULE=camelo.train.resume_train` —
  it cannot resume through stock `lerobot-train` at all
  (13_LAUNCH_LOG.md §6c). **The memory settings live here, not in
  `train.yaml`**: `--mem=512G`, `--num_workers=8`/rank,
  `--prefetch_factor=2`, after both 4-GPU runs were OOM-killed on the host
  at `--mem=256G` with 16 workers/rank.
- **Output dirs / job-ID keying (AGENTS.md F-100):** handled automatically
  by `camelo/train/train.py` — `output_dir`/`job_name` default to
  `outputs/runs/<config_stem>_<timestamp>_<$SLURM_JOB_ID or pid>`, so no
  config or script here needs its own job-ID logic.
- **Verification:** every `train.yaml` is parse-checked against the
  *installed* lerobot 0.6.1 dataclasses (draccus `TrainPipelineConfig` +
  `cfg.validate()`, the same two-step resolution the real `lerobot-train`
  entrypoint performs) without training, without the corpus, and without
  any uncached checkpoint — see `PARSE_CHECK.md`, which also documents two
  real bugs the check found and fixed (EO-1's init mechanism, VLA-JEPA's
  `reinit_modules` serialization).

## Wave table — launched 2026-08-31

Job IDs, measured rates and every launch-time fix are in
[docs/realdata/13_LAUNCH_LOG.md](../../docs/realdata/13_LAUNCH_LOG.md).
Walltimes marked **raised** were increased at launch because the original
budget counted training steps only and did not fit the eval passes
(13 §W-1); every one is still under the 24 h cliff. `eval_steps: 0` on
three rungs is deliberate and explained in 13 §D-2.

| wave | policy dir | GPUs | walltime | steps | epochs | smoke | long run | notes |
|---|---|---|---|---|---|---|---|---|
| **First** | `act` | 1 | 12:00:00 | 100,000 | 8.03 | **3983177 PASS** | **3983190** RUNNING | 0.090 s/step measured; ~3.7 h |
| | `diffusion` | 1 | 08:00:00 | 40,000 | 3.21 | 3983178, 3983235 FAIL → **3983369 PASS** | **3983411 ✅ COMPLETED** (00:49:26, loss→0.009) | needed a 240×320 dataset Resize *and* an explicit `input_features`; `eval_steps: 0` |
| | `pi05_lora_15k` | 1 | **11:00:00** raised | 15,000 | 4.82 | **3983179 PASS** | **3983315** RUNNING | 14,059,520 trainable, 1.640 s/step — every §2.1 number hit |
| | `pi05_lora_30k` | 1 | **20:00:00** raised | 30,000 | 9.63 | covered by 3983179 | **3983316** RUNNING | |
| **Second** | `vla_jepa` | 1 | 16:00:00 | 30,000 | 4.82 | 3983180, 3983302 FAIL → **3983402 PASS** | **3983466** RUNNING | four fixes: `action_decoder` reinit, 2-camera `rename_map`, gripper binarizer, 224×224 Resize; `eval_steps: 0` |
| | `groot_n17` | 1 | 22:00:00 | **30,000** raised | 9.63 | **3983181 PASS** | **3983337** RUNNING | 0.907 s/step measured (est. band was 1.5–3.0), so the 20,000 provisional cut was reversed; `eval_steps: 0` |
| | `eo1` | **4** | 20:00:00 | **30,000** | **4.82** | 3983182 FAIL → 3984855/3986305 (1-GPU) PASS → **3986485 (4-GPU) PASS** | **3986556** OOM-killed at step 15,000 → **RESUMED as 3989703** from `checkpoints/007500` | was BLOCKED (0 of 836 tensors bound); unblocked via `lerobot/eo1-base` + `make stage RUNG=eo1`. 4×H100 at global batch 16 measures **1.88 s/step** vs 5.73–6.63 on one — 3.5×, so 4.82 epochs instead of 1.28. 13 §1.3 |
| **Parallel** | `molmoact2_ddp2_rehearsal` | 2 | 00:30:00 (`gpu-test`) | 300 | — | 3983176 FAIL → 3983334 TIMEOUT → **3986257 PASS** | — | it earned its place: found the stale global accelerate config launching DeepSpeed (13 §M-1), then an unreadable run that turned out to be an `eval_steps: 50` artifact rather than a step rate (13 M-2 update). Clean at **4.19 s/step** once `eval_steps: 0` and the thread flag were set |
| | `molmoact2` | 4 | **20:00:00** raised | 25,000 | 8.03 | 3983482 (pre-fix) → **3986484 PASS** | **3986557** OOM-killed at step ~7,000 → **RESUMED as 3989764** from `checkpoints/006250` | the §6 "≥ 38.6 s/step, CPU-bound" reading was an `eval_steps: 50` artifact — see the M-2 update. Fixed with `--num_cpu_threads_per_process=4` + `eval_steps: 0` in the rehearsal: **2.40 s/step** on 4×H100, `data_s` 0.185. 13 §1.3 |
| **Opportunistic** | `pi0_fast` | 1 | **21:00:00** raised | 25,000 | 8.03 | **3983183 PASS** | **3983467** RUNNING | 2.085 s/step measured — 25,000 steps + eval does not fit the old 16 h |
| | `smolvla` | 1 | 08:00:00 | 40,000 | 6.42 | **3983184 PASS** | **3983370 ✅ COMPLETED** (02:23:58, final loss 0.023) | `scheduler_decay_steps` raised to 40,000 to match `steps` (13 §S-1) |
| **Skipped** | — X-VLA, RDT2 — | | | | | | | no config dirs (`10_RDT2_ASSESSMENT.md`, `00_PROTOCOL.md` §2.8/§2.10) |

Epochs are `steps × batch / 99,654` — the frames the trainer actually sees.
The corpus holds 121,828 frames in 217 episodes; 195 are on the train list,
and `dataset.eval_split: 0.1` holds back the last 20 of those, leaving 175
episodes / 99,654 frames for training and 10,522 for eval.

Every smoke is `gpu-test`, 00:30:00, 300 steps, 1 GPU unless noted.

## Selection

Every rung is judged the same way: the **M1 gripper-close timing verdict**
(TIMED / MISPLACED / UNTIMED / ABSENT) on held-out episodes — see each
`README.md`'s **Expect / selection probe** line, and
`docs/realdata/00_PROTOCOL.md` §5.2 for the full taxonomy. Training loss
is not a selection signal (repo MEMORY).
