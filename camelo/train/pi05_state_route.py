"""Three ways to feed pi0.5 its proprioceptive state, as one switchable patch.

    CAMELO_STATE_ROUTE=blind python -m camelo.train.pi05_train ...

Why this exists. lerobot's pi0.5 has exactly ONE state route: normalize,
digitize into 256 bins, splice the digits into the language prompt
(`processor_pi05.py:66-74`). openpi has two — `Pi0Config.discrete_state_input`
defaults to `pi05` but is a separate, overridable flag, and PI's own
`pi05_libero` config sets it **False**, feeding state continuously through a
`state_proj` linear instead. lerobot never ported that branch, so the choice
does not exist here. This module adds it, plus a no-state ablation.

    digits      (default, = stock lerobot) state -> 256 bins -> prompt text
    blind       state is DROPPED entirely; the policy sees images + task only
    continuous  state -> nn.Linear -> one token prepended to the action suffix,
                and the digits are removed from the prompt (openpi pi05_libero)

## Read this before trusting a `continuous` run

`pi05_base` ships **no `state_proj` tensor** — it is not merely unused, it
was never trained, because pi0.5's whole design point is that state rides in
the prompt (docs/research/PI05_STATE_ACTION_SUBTASKS.md section 3.2). So `continuous`
trains a **randomly initialised** projection from scratch on 199 episodes
while everything around it is frozen or LoRA'd. It is therefore NOT a
one-variable comparison against lora_v2, and a bad result is at least as
likely to mean "a fresh 32x1024 layer did not converge on 199 episodes" as
"the continuous route is worse". `blind` IS one-variable and should be read
first.

## Train/eval parity is the whole risk here (F-63)

Every route changes what the policy is fed at INFERENCE too. A checkpoint
trained under one route and evaluated under another is exactly the failure
F-63 documents: no error, no shape mismatch, just a wrong number. So the
route is written into a `camelo_state_route.json` sidecar beside the run and
must be re-applied at eval via `apply_for_checkpoint()`. Never apply a route
by hand on one side only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROUTES = ("digits", "blind", "continuous")
SIDECAR = "camelo_state_route.json"
ENV_VAR = "CAMELO_STATE_ROUTE"

_applied: str | None = None


def route_from_env(default: str = "digits") -> str:
    route = os.environ.get(ENV_VAR, default) or default
    if route not in ROUTES:
        raise ValueError(f"{ENV_VAR}={route!r} is not one of {ROUTES}")
    return route


def sidecar_path(run_dir: Path) -> Path:
    """A SIBLING file, deliberately not `run_dir/SIDECAR`.

    lerobot refuses to start when its output dir already exists ("already
    exists and resume is False"), so creating the run dir early to hold the
    sidecar makes every run fail before step 0. Writing beside the dir keeps
    the marker available from the moment the job launches -- including for a
    run that dies mid-training -- without touching the path lerobot owns.
    """
    return run_dir.parent / f"{run_dir.name}.{SIDECAR}"


def write_sidecar(run_dir: Path, route: str) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path(run_dir).write_text(json.dumps({"state_route": route}, indent=2) + "\n")


def read_sidecar(checkpoint: Path) -> str:
    """Walk up from a checkpoint dir to the run root looking for the sidecar.

    Absent sidecar means a stock run, i.e. `digits` — that keeps every
    checkpoint trained before this module existed evaluating exactly as it
    did before.
    """
    for parent in [checkpoint, *checkpoint.parents]:
        for candidate in (parent / SIDECAR, sidecar_path(parent)):
            if candidate.is_file():
                return json.loads(candidate.read_text())["state_route"]
    return "digits"


def apply_for_checkpoint(checkpoint: str | Path) -> str:
    route = read_sidecar(Path(checkpoint))
    apply(route)
    return route


def apply(route: str) -> None:
    """Monkeypatch lerobot's pi0.5 in place. Idempotent per route."""
    global _applied
    if route not in ROUTES:
        raise ValueError(f"state route {route!r} is not one of {ROUTES}")
    if _applied == route:
        return
    if _applied is not None and _applied != route:
        raise RuntimeError(
            f"state route already patched as {_applied!r}; cannot switch to "
            f"{route!r} in one process (the patches are not reversible)"
        )
    if route == "digits":
        _applied = route
        return

    _patch_prompt_drops_state()
    if route == "continuous":
        _patch_continuous_state()
    _applied = route
    print(f"[camelo] pi0.5 state route = {route}")


# ---------------------------------------------------------------------------
# 1. Take the digits out of the prompt (both `blind` and `continuous`)
# ---------------------------------------------------------------------------
def _patch_prompt_drops_state() -> None:
    from lerobot.policies.pi05 import processor_pi05 as P
    from lerobot.lerobot_types import TransitionKey

    step_cls = P.Pi05PrepareStateTokenizerProcessorStep

    def __call__(self, transition):
        transition = transition.copy()
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        # Same cleaning as upstream (processor_pi05.py:72), same trailing
        # "Action: " marker -- ONLY the ", State: <digits>" span is removed,
        # so the prompt the model sees differs in exactly one respect.
        prompts = [f"Task: {t.strip().replace('_', ' ').replace(chr(10), ' ')};\nAction: " for t in tasks]
        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = prompts
        return transition

    step_cls.__call__ = __call__


# ---------------------------------------------------------------------------
# 2. Continuous state: a fresh projection into the action-expert suffix
# ---------------------------------------------------------------------------
def _patch_continuous_state() -> None:
    import torch
    import torch.nn as nn
    from lerobot.policies.common.vla_utils import pad_vector
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
    from lerobot.utils.constants import OBS_STATE

    orig_init = PI05Pytorch.__init__
    orig_embed_suffix = PI05Pytorch.embed_suffix

    def __init__(self, config, *a, **kw):
        orig_init(self, config, *a, **kw)
        # Width must match the ACTION EXPERT stream, which is what the suffix
        # carries; action_in_proj already maps into exactly that width.
        width = self.action_in_proj.out_features
        self.state_proj = nn.Linear(config.max_state_dim, width)
        self._camelo_state = None

    def embed_suffix(self, noisy_actions, timestep):
        action_emb, pad_masks, att_masks, adarms_cond = orig_embed_suffix(self, noisy_actions, timestep)
        state = getattr(self, "_camelo_state", None)
        if state is None:
            raise RuntimeError(
                "continuous state route: embed_suffix reached with no state staged. "
                "The policy-level patch must set _camelo_state around every model call "
                "-- a silent fallback here would train on a stale batch's proprio."
            )
        state_emb = self.state_proj(state.to(action_emb.dtype))[:, None, :]
        bsize = action_emb.shape[0]
        emb = torch.cat([state_emb, action_emb], dim=1)
        pad = torch.cat(
            [torch.ones(bsize, 1, dtype=pad_masks.dtype, device=pad_masks.device), pad_masks], dim=1
        )
        # att_mask 1 opens a NEW block for the state token, so it attends to
        # the prefix only; the action block that follows (also opening with 1)
        # then attends to prefix + state + itself. `make_att_2d_masks` is
        # cumsum-based, so this is the documented block-causal layout, and
        # `suffix_out[:, -chunk_size:]` still slices the state token away.
        one = torch.ones(bsize, 1, dtype=att_masks.dtype, device=att_masks.device)
        att = torch.cat([one, att_masks], dim=1)
        return emb, pad, att, adarms_cond

    PI05Pytorch.__init__ = __init__
    PI05Pytorch.embed_suffix = embed_suffix

    # `pi05_base` has no state_proj, so a strict load of it into a model that
    # now HAS one raises -- and `from_pretrained` swallows that in a bare
    # except (modeling_pi05.py:859-862), printing a one-line warning and
    # handing back a RANDOMLY INITIALISED 4B model. Measured: the first smoke
    # of this route hit exactly that and looked completely healthy (loss fell,
    # 14M trainable params reported). So seed the checkpoint's state dict with
    # THIS model's freshly initialised state_proj and keep strict=True for
    # every other key, rather than relaxing strictness globally.
    orig_load = PI05Policy.load_state_dict

    def load_state_dict(self, state_dict, strict=True, *a, **kw):
        own = self.state_dict()
        missing = {k: v for k, v in own.items() if ".state_proj." in f".{k}." and k not in state_dict}
        if missing:
            state_dict = {**state_dict, **missing}
            print(f"[camelo] seeded {sorted(missing)} from fresh init (absent from pi05_base)")
        return orig_load(self, state_dict, strict=strict, *a, **kw)

    PI05Policy.load_state_dict = load_state_dict

    def _stage(policy, batch):
        state = batch.get(OBS_STATE)
        if state is None:
            raise ValueError("continuous state route requires observation.state in the batch")
        return pad_vector(state, policy.config.max_state_dim)

    for name in ("forward", "predict_action_chunk"):
        orig = getattr(PI05Policy, name)

        def wrapper(self, batch, *a, __orig=orig, **kw):
            # Live only for the duration of the call: a leaked value would be
            # silently reused by the next batch (F-63-shaped bug).
            self.model._camelo_state = _stage(self, batch)
            try:
                return __orig(self, batch, *a, **kw)
            finally:
                self.model._camelo_state = None

        setattr(PI05Policy, name, wrapper)
