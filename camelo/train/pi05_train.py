"""`lerobot-train`, with the pi0.5 state route patched in first.

    CAMELO_STATE_ROUTE=blind .venv/bin/python -m camelo.train.pi05_train <draccus args...>

`camelo.train.train` execs the `lerobot-train` console script as a
subprocess, so a monkeypatch applied in the launcher's process would never
reach the trainer. This module is a drop-in replacement for that console
script: it applies the route, then calls lerobot's own `main()` unchanged.

Everything about the run other than the state route is stock lerobot — the
same argv, the same config, the same trainer. See
`camelo/train/pi05_state_route.py` for what each route does and why
`continuous` is not a one-variable comparison.
"""

from __future__ import annotations

import sys

from camelo.train.pi05_state_route import apply, route_from_env


def main() -> int:
    apply(route_from_env())
    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
