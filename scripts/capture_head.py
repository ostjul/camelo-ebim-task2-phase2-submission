#!/usr/bin/env python3
"""Capture one live head-camera frame + the measured arm joints on the real robot.

Rig C-15 (docs/realdata/16 R-58..R-61): the mobile base and the table are placed
by overlaying this frame on the corpus episode's frame 0
(`tools/rig_probes/overlay_head.py`). Run natively in the station's pixi shell
with the DDS env sourced and camelo's profile exported (16 §3.1 A):

    python -u scripts/capture_head.py [--out outputs/rig/t5] [--camera head]

Must be a FILE, not stdin: the camera worker is spawned as a fresh process that
re-imports this module by path. Nothing is commanded; the arms are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/rig/t5")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    from camelo.contracts import S_LEFT_ARM, S_RIGHT_ARM, topics_for
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topics = topics_for("real")
    with ros_session("camelo_capture", world="real") as node:
        collector = ObsCollector(node, camera_keys=[args.camera], topics=topics)
        obs = wait_for_obs(collector, timeout_s=args.timeout_s, require_images=True)
        time.sleep(1.0)
        img = collector.images[args.camera]
        ts = time.strftime("%H%M%S")
        png = out / f"live_{args.camera}_{ts}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(png)
        meta = {
            "left": obs.state[S_LEFT_ARM].tolist(),
            "right": obs.state[S_RIGHT_ARM].tolist(),
            "shape": list(img.shape),
            "camera": args.camera,
        }
        (out / f"live_{args.camera}_{ts}.json").write_text(json.dumps(meta, indent=1))
        print("saved", png, tuple(img.shape), "right", np.round(obs.state[S_RIGHT_ARM], 3).tolist())
        collector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
