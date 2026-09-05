"""`lerobot-train`, with the resume-time processor-override mismatch filtered out.

    accelerate launch --module camelo.train.resume_train --config_path=... --resume=true

WHY THIS EXISTS (found by the R-2 resume smoke, job 3989706, 2026-09-01;
13_LAUNCH_LOG.md §8). Resuming MolmoAct2 dies before step 1 with

    KeyError: Override keys ['normalizer_processor'] do not match any step in
    the saved configuration. Available step keys: [...,
    'molmoact2_masked_normalizer', ...]

and the cause is a genuine lerobot 0.6.1 interaction between two features,
not anything wrong with the checkpoint:

  * `lerobot_train.py:355` builds `preprocessor_overrides` — including a
    hard-coded `normalizer_processor` / `unnormalizer_processor` entry —
    whenever `active_cfg.pretrained_path is not None`.
  * That field is **None for a fresh MolmoAct2 run**, because this rung
    initialises through `policy.checkpoint_path=allenai/MolmoAct2` rather
    than `policy.path` (a raw HF repo, not a lerobot checkpoint). So the
    whole override branch was skipped and the pipeline was built fresh from
    the policy class, which registers MolmoAct2's OWN step names.
  * On resume, `TrainPipelineConfig._resolve_resume_checkpoint` sets
    `policy.pretrained_path` to the checkpoint. The branch now fires, and
    `PolicyProcessorPipeline._validate_overrides_used` (processor/pipeline.py
    :1282) raises because MolmoAct2's pipeline has no step by either generic
    name — it has `molmoact2_masked_normalizer` and
    `molmoact2_masked_unnormalizer`.

So the rung is only resumable if those two override keys are dropped, and
dropping them is CORRECT rather than a workaround: lerobot itself says the
checkpoint's saved processor state is authoritative on resume, which is why
the same block guards `stats` behind `if not cfg.resume`. What remains in the
override payload is `features` and `norm_map`, and the checkpoint's own
normalizer step already carries both.

That reasoning is also the safety boundary, so it is ENFORCED rather than
trusted: `drop_unmatched_overrides` refuses to drop an override that carries
a `stats` payload. If a future lerobot starts pushing real statistics through
this path, this module fails loudly instead of silently training on the wrong
normalization — the F-63 family of failure, where the weights look fine and
the checkpoint is quietly unevaluable.

EO-1 does not need this (its pipeline uses the generic `normalizer_processor`
name, so every override matches) and the filter is a no-op there; it is wired
per-rung in `configs/realdata/*/resume.sh` rather than applied globally.

Nothing is patched in site-packages — same contract as
`camelo/train/pi05_train.py`, which this module mirrors. lerobot is imported
inside functions only, so `make test` still runs with no lerobot installed
(AGENTS.md layering rule 3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The two filenames lerobot writes into a checkpoint's pretrained_model/.
# Hard-coded rather than imported so this module stays importable without
# lerobot; `patch()` cross-checks them against lerobot's own constants.
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"


def available_step_keys(config: dict) -> list[str]:
    """The override keys a saved processor config actually accepts.

    Mirrors `PolicyProcessorPipeline._validate_overrides_used`
    (lerobot/processor/pipeline.py:1280) exactly — registry name when present,
    otherwise the final component of the dotted class path. Kept in step with
    that function on purpose: a filter that computed a *different* key set
    would drop overrides lerobot would have accepted.
    """
    keys = []
    for step in config.get("steps", []):
        name = step.get("registry_name")
        if not name and "class" in step:
            name = str(step["class"]).rsplit(".", 1)[-1]
        if name:
            keys.append(name)
    return keys


def drop_unmatched_overrides(
    overrides: dict | None, available: list[str]
) -> tuple[dict, list[str]]:
    """Return (kept, dropped_names) for overrides against a saved pipeline.

    Raises ValueError if an unmatched override carries a `stats` payload —
    see the module docstring. Dropping `features`/`norm_map` for a step the
    checkpoint defines itself is a no-op; dropping statistics would silently
    change what the model is trained against.
    """
    if not overrides:
        return {}, []
    kept, dropped = {}, []
    for key, value in overrides.items():
        if key in available:
            kept[key] = value
            continue
        if isinstance(value, dict) and "stats" in value:
            raise ValueError(
                f"refusing to drop processor override {key!r}: it carries a "
                f"'stats' payload, but the saved pipeline has no such step "
                f"(it has {available}). Silently discarding normalization "
                f"statistics would train against the wrong normalization with "
                f"no error and no visible symptom. Investigate before resuming."
            )
        dropped.append(key)
    return kept, dropped


def _filter_for(pretrained_path, filename: str, overrides: dict | None) -> dict:
    """Filter `overrides` against the processor config saved at `pretrained_path`.

    A non-local path (a Hub repo id) is left untouched: there is nothing to
    read, and lerobot's own error is then the right one to surface.
    """
    if not pretrained_path:
        return overrides or {}
    config_file = Path(str(pretrained_path)) / filename
    if not config_file.is_file():
        return overrides or {}
    try:
        config = json.loads(config_file.read_text())
    except (OSError, ValueError):
        return overrides or {}
    kept, dropped = drop_unmatched_overrides(overrides, available_step_keys(config))
    if dropped:
        print(
            f"camelo.train.resume_train: dropped processor override(s) "
            f"{dropped} — absent from {config_file.name} "
            f"(steps: {available_step_keys(config)}). The checkpoint's own "
            f"pipeline is authoritative on resume.",
            file=sys.stderr,
            flush=True,
        )
    return kept


def patch() -> None:
    """Wrap `make_pre_post_processors` where `lerobot_train` looks it up.

    `lerobot/scripts/lerobot_train.py:59` does `from lerobot.policies import
    ... make_pre_post_processors`, binding the function into ITS module
    namespace — so patching `lerobot.policies.factory` would have no effect on
    the call at line 394. The name must be rebound on `lerobot_train` itself.
    """
    from lerobot.scripts import lerobot_train

    original = lerobot_train.make_pre_post_processors
    if getattr(original, "_camelo_resume_patched", False):
        return

    def make_pre_post_processors(*args, **kwargs):
        path = kwargs.get("pretrained_path")
        pre_name = kwargs.get("preprocessor_config_filename", PREPROCESSOR_CONFIG)
        post_name = kwargs.get("postprocessor_config_filename", POSTPROCESSOR_CONFIG)
        if "preprocessor_overrides" in kwargs:
            kwargs["preprocessor_overrides"] = _filter_for(
                path, pre_name, kwargs["preprocessor_overrides"]
            )
        if "postprocessor_overrides" in kwargs:
            kwargs["postprocessor_overrides"] = _filter_for(
                path, post_name, kwargs["postprocessor_overrides"]
            )
        return original(*args, **kwargs)

    make_pre_post_processors._camelo_resume_patched = True
    lerobot_train.make_pre_post_processors = make_pre_post_processors


def main() -> int:
    patch()
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
