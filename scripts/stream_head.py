#!/usr/bin/env python3
"""Stream live head-camera frames to a single overwritten file, for the
Mac-side live visualizer (`tools/rig_probes/live_overlay.py`).

Rig C-15 (docs/realdata/16 R-58..R-61; cheatsheet "Alignment Visualization"):
same ObsCollector + camera-worker-subprocess capture used by
`scripts/capture_head.py` (one process, one camera, no extra ROS load), but
the node and camera worker are opened ONCE and stay up, so the base/table can
be nudged while watching alignment update instead of paying a fresh
node-startup/shutdown per frame. Overwrites `<out>/live_<camera>_latest.png`
in place (write to a temp file, then atomic rename) at roughly `--hz`; never
accumulates files, so there is nothing here to grow rig-side memory or disk
beyond what a single capture_head.py run already uses.

Run natively in the station's pixi shell with the DDS env sourced and
camelo's profile exported (16 §3.1 A):

    python -u scripts/stream_head.py [--out outputs/rig/t5] [--camera head] [--hz 1]

Must be a FILE, not stdin: the camera worker is spawned as a fresh process
that re-imports this module by path (same constraint as capture_head.py).
Nothing is commanded; the arms are not touched. Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/rig/t5")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--timeout-s", type=float, default=30.0)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    from camelo.contracts import topics_for
    from camelo.ros.obs_collector import ObsCollector
    from camelo.ros.session import ros_session
    from camelo.runner.episode_runner import wait_for_obs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"live_{args.camera}_latest.png"
    tmp = out / f".live_{args.camera}_latest.png.tmp"
    period = 1.0 / max(args.hz, 0.1)

    topics = topics_for("real")
    with ros_session("camelo_stream_head", world="real") as node:
        collector = ObsCollector(node, camera_keys=[args.camera], topics=topics)
        try:
            wait_for_obs(collector, timeout_s=args.timeout_s, require_images=True)
            print(f"streaming {args.camera} -> {dest} at {args.hz} Hz; Ctrl+C to stop", flush=True)
            n = 0
            heartbeat_every = max(int(args.hz * 30), 1)  # ~ every 30s
            while True:
                t0 = time.monotonic()
                img = collector.images[args.camera]
                if img is not None:
                    Image.fromarray(np.ascontiguousarray(img)).save(tmp, format="PNG")
                    tmp.replace(dest)  # atomic: the Mac-side scp never sees a partial file
                    n += 1
                    if n % heartbeat_every == 0:
                        print(f"wrote frame {n}", flush=True)
                else:
                    print("no frame yet", flush=True)
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            collector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
