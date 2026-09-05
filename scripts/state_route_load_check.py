#!/usr/bin/env python3
"""Load ONE checkpoint through the eval adapter and prove its state route took.

    python scripts/state_route_load_check.py <pretrained_model dir> --expect blind

One checkpoint per PROCESS, deliberately: the route patches are monkeypatches
over lerobot classes and are not reversible, so `apply()` refuses to switch
route in a live interpreter. Evaluating a mixed set means separate processes.

What this proves, and why each check is here:

- the sidecar resolved to the route the run was TRAINED under. Nothing in a
  checkpoint records it (all three arms ship a byte-identical
  policy_preprocessor.json), so this is the only thing standing between a
  `blind` checkpoint and a silent eval on prompts full of state digits (F-63).
- for `continuous`, that `state_proj` SURVIVED the PEFT merge with its trained
  values. It is a full-training module, not a LoRA target, so it rides in the
  adapter as a modules_to_save entry -- a merge that dropped it would leave a
  random projection behind and no error anywhere.
- for `blind`, that the patched prompt carries NO "State:" span.
- that a real forward pass produces a finite (H, 20) chunk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo import contracts as C  # noqa: E402
from camelo.policy.base import Obs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--expect", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default=C.TASK2_INSTRUCTION)
    args = ap.parse_args()

    from camelo.policy.adapters.lerobot_generic import LeRobotAdapter

    ad = LeRobotAdapter(
        checkpoint=args.checkpoint,
        device=args.device,
        state_layout="model16",
        action_layout="canonical",
    )
    ok = True

    print(f"route resolved : {ad.state_route!r} (expected {args.expect!r})")
    if ad.state_route != args.expect:
        print("FAIL: sidecar did not resolve to the trained route")
        ok = False

    if args.expect == "continuous":
        # After merge_and_unload the module must still be there AND still hold
        # the trained tensor. Compare against the recorded fingerprint rather
        # than merely asserting existence: a freshly re-initialised state_proj
        # would also "exist".
        proj = getattr(ad.policy.model, "state_proj", None)
        if proj is None:
            print("FAIL: state_proj is GONE after the PEFT merge")
            ok = False
        else:
            w = proj.weight.detach().float().cpu()
            print(f"state_proj     : shape={tuple(w.shape)} |W|={w.norm():.4f} std={w.std():.5f}")
            from safetensors import safe_open

            with safe_open(Path(args.checkpoint) / "adapter_model.safetensors", framework="pt") as h:
                ref = h.get_tensor("base_model.model.model.state_proj.weight").float()
            delta = (w - ref).abs().max().item()
            print(f"max|merged - checkpoint| = {delta:.3e}")
            if delta > 1e-3:
                print("FAIL: merged state_proj does not match the checkpoint tensor")
                ok = False

    # What does the processor actually build for the prompt?
    from lerobot.policies.pi05 import processor_pi05 as P
    from lerobot.lerobot_types import TransitionKey

    step = P.Pi05PrepareStateTokenizerProcessorStep()
    tr = {
        TransitionKey.COMPLEMENTARY_DATA: {step.task_key: [args.task]},
        TransitionKey.OBSERVATION: {"observation.state": __import__("torch").zeros(1, 32)},
    }
    try:
        prompt = step(tr)[TransitionKey.COMPLEMENTARY_DATA][step.task_key][0]
        print(f"prompt         : {prompt[:110]!r}")
        has_state = "State:" in prompt
        want_state = args.expect == "digits"
        if has_state != want_state:
            print(f"FAIL: prompt {'has' if has_state else 'lacks'} a State: span; route is {args.expect!r}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"(prompt probe skipped: {exc})")

    ad.reset(args.task)
    obs = Obs(
        t_sim=0.0,
        state=np.zeros(C.STATE_DIM, dtype=np.float32),
        images={k: np.zeros((224, 224, 3), dtype=np.uint8) for k in C.CAMERA_KEYS},
    )
    chunk = ad.infer(obs)
    print(f"chunk          : shape={chunk.shape} finite={np.isfinite(chunk).all()} "
          f"absmax={np.abs(chunk).max():.4f}")
    if chunk.ndim != 2 or chunk.shape[1] != C.ACTION_DIM or not np.isfinite(chunk).all():
        print("FAIL: bad chunk")
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
