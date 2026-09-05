"""Adapter registry: ``make_adapter("gr00t")``, ``make_adapter("lerobot:<path>")`` …

All VLA presets route through the LeRobot policy stack (GR00T N1.7, X-VLA,
pi0, and smolVLA are all lerobot-native as of lerobot >= 0.6) — one
framework for zero-shot, fine-tuning, and our own trained checkpoints.
``gr00t-isaac`` keeps the Isaac-GR00T native stack as an alternative
(TensorRT deployment path). ``replay`` is the model-free open-loop baseline
that plays prepared demo action chunks (see prepare_replay_actions.py).
``heuristic`` is its submission-grade sibling: demo-sampled reference
trajectory + right-arm residual IK by default (see
prepare_heuristic_actions.py).
"""

from __future__ import annotations

from camelo.policy.base import PolicyAdapter

# Zero-shot default checkpoints; override with --checkpoint.
# gr00t: the DROID post-train is the closest embodiment head to ours —
# DROID is Franka + Robotiq 2F-85 (our per-arm hardware exactly).
DEFAULT_CHECKPOINTS = {
    "gr00t": "nvidia/GR00T-N1.7-DROID",
    "gr00t-base": "nvidia/GR00T-N1.7-3B",
    "xvla": "lerobot/xvla-base",
    # lerobot/pi0 is deprecated (pre-0.6 config format, redirects to pi0_old
    # and fails PreTrainedConfig decoding — DGX_FINDINGS.md F-07/F-21).
    "pi0": "lerobot/pi05_base",
    "smolvla": "lerobot/smolvla_base",
    # Single-instance arm-splitter rungs (F-58; ZERO_SHOT_RUNBOOK.md P0/P1).
    "molmoact2": "lerobot/MolmoAct2-DROID-LeRobot",
    "pi05droid": "DAVIAN-Robotics/pi05_droid_jointpos",
}

_LEROBOT_SPECS = (
    "gr00t",
    "gr00t-base",
    "xvla",
    "pi0",
    "smolvla",
    "lerobot",
    "molmoact2",
    "pi05droid",
)

# Declared action semantics per preset (F-45: a 20-dim output is NOT
# evidence of the canonical contract — X-VLA emits 20-dim absolute EEF).
# EEF layouts refuse to load until the differential-IK work item lands.
# Unlisted presets and lerobot:<ckpt> specs default to "model16";
# checkpoints WE trained on the recorded contract pass
# --action-layout canonical explicitly. The droid8 presets route through
# the arm-splitter (right arm active, left latched — F-58): molmoact2 is
# absolute joints, pi05droid is joint DELTAS + absolute gripper.
PRESET_ACTION_LAYOUTS = {
    "xvla": "eef_abs",
    "molmoact2": "droid8_right",
    "pi05droid": "droid8_right_delta",
}


def make_adapter(
    spec: str, checkpoint: str | None = None, device: str = "cuda", **kwargs
) -> PolicyAdapter:
    """spec: dummy | replay | heuristic | gr00t | … | lerobot:<ckpt> | gr00t-isaac."""
    name, _, inline = spec.partition(":")
    checkpoint = checkpoint or inline or DEFAULT_CHECKPOINTS.get(name)

    if name == "dummy":
        from camelo.policy.adapters.dummy import DummyAdapter

        return DummyAdapter(**kwargs)
    if name == "replay":
        from camelo.policy.adapters.replay import ReplayAdapter

        # checkpoint / replay:<path> overrides the prepared npy location.
        return ReplayAdapter(actions_path=checkpoint or None, **kwargs)
    if name == "heuristic":
        from camelo.policy.adapters.heuristic import HeuristicAdapter

        # checkpoint / heuristic:<path> overrides the prepared npy location.
        return HeuristicAdapter(actions_path=checkpoint or None, **kwargs)
    if name in _LEROBOT_SPECS:
        from camelo.policy.adapters.lerobot_generic import LeRobotAdapter

        if not checkpoint:
            raise ValueError("lerobot adapter needs a checkpoint: lerobot:<repo_or_path>")
        kwargs.setdefault("action_layout", PRESET_ACTION_LAYOUTS.get(name, "model16"))
        return LeRobotAdapter(checkpoint=checkpoint, device=device, name=name, **kwargs)
    if name == "gr00t-isaac":
        from camelo.policy.adapters.gr00t import Gr00tAdapter

        return Gr00tAdapter(
            checkpoint=checkpoint or DEFAULT_CHECKPOINTS["gr00t-base"], device=device, **kwargs
        )
    raise ValueError(
        f"unknown adapter spec: {spec!r} "
        "(dummy|replay|heuristic|gr00t|gr00t-base|xvla|pi0|smolvla|lerobot:<ckpt>|gr00t-isaac)"
    )
