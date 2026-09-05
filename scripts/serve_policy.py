#!/usr/bin/env python3
"""M1.7 — policy server: adapter + model near the GPU, no ROS needed.

    python scripts/serve_policy.py --adapter pi0 --port 8765
    python scripts/serve_policy.py --adapter gr00t --checkpoint nvidia/GR00T-N1.5-3B

Run on the H100 (docker/Dockerfile.policy) or the GB10 for small models.
The ROS-side client is scripts/run_policy.py --backend remote.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camelo.cli import add_policy_args, make_adapter_from_args, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_policy_args(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed pinned before the rollout's first forward pass "
        "(docs/realdata/15_RIG_WINDOW_RUNBOOK.md §1.2/§2.4). Seeds "
        "random/numpy/torch (CPU + all CUDA devices) once at server start "
        "and again on every reset request, so every rollout starts from "
        "the same RNG state. Default: None = unseeded (unchanged "
        "behaviour)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging()

    from camelo.policy.server import main as serve
    from camelo.policy.server import seed_everything

    if args.seed is not None:
        # Pinned before the pre-serve warm-up reset below too, so that
        # call (unlike wire-protocol resets, not itself repeated per
        # rollout) also runs on the pinned RNG state.
        seed_everything(args.seed)

    adapter = make_adapter_from_args(args)
    adapter.reset(args.task)
    serve(adapter, host=args.host, port=args.port, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
