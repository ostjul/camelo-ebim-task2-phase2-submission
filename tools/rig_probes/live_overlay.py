#!/usr/bin/env python3
"""Live head-camera alignment view (rig checklist C-15, docs/realdata/16
R-58..R-61, cheatsheet "Alignment Visualization").

Runs on the Mac. Polls the rig's `outputs/rig/t5/live_head_latest.png` over
SSH at ~1 Hz (written by `scripts/stream_head.py`, run on the station) and
redraws one matplotlib window with a live blend against a corpus frame-0
reference, so the base/table can be nudged while watching alignment update —
no repeated open/close of Preview like `overlay_latest.sh`.

    tools/rig_probes/live_overlay.py                                   # vs episode 163
    tools/rig_probes/live_overlay.py outputs/rig/t5/ep009_frame0_head.png
    RIG_HOST=ebim@192.168.0.5 RIG_REMOTE_DIR=camelo/camelo-ebim \
        tools/rig_probes/live_overlay.py --hz 1

`RIG_REMOTE_DIR` (or `--remote-dir`) is the checkout path on the rig relative
to the SSH user's home, e.g. `camelo/camelo-ebim` (the deploy target in the
cheatsheet's `M — rsync` step) — set it to match whatever checkout
`scripts/stream_head.py` is actually running from, which is NOT always that
default (a dev checkout can live elsewhere, e.g. `camelo/camelo-ebim-viz`).

A failed poll (wrong `--remote-dir`, rig not streaming yet, tunnel hiccup)
just leaves the last good frame on screen with the scp error in the title,
instead of crashing the view. Close the window or Ctrl+C to stop. Needs
python3 with numpy, PIL, matplotlib.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REMOTE_DIR_DEFAULT = "camelo/camelo-ebim"
REMOTE_SUBPATH = "outputs/rig/t5/live_head_latest.png"


def fetch(host: str, remote_path: str, local_path: Path, timeout_s: float) -> tuple[bool, str]:
    """scp the latest frame down. (True, "") on success; (False, reason) on
    any failure (wrong remote path, rig not streaming yet, tunnel hiccup) so
    the caller can show the last good frame AND say why it's stale, instead
    of either crashing or failing silently (a bad --remote-dir must not look
    identical to "no frame yet").

    Plain scp, no connection multiplexing (same as overlay_latest.sh): a
    fresh SSH handshake once a second is cheap on the rig's local network,
    and a shared ControlMaster socket that outlives a failed run is a
    stale-connection failure mode of its own ("scp: Connection closed" on
    every subsequent call, with no signal it's the *socket* that's dead
    rather than the rig or the path).
    """
    tmp = local_path.with_suffix(".tmp")
    cmd = [
        "scp", "-q",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={max(int(timeout_s), 1)}",
        f"{host}:{remote_path}", str(tmp),
    ]
    try:
        result = subprocess.run(cmd, timeout=timeout_s + 2, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False, f"scp timed out after {timeout_s + 2:.0f}s"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or f"scp exit {result.returncode}").strip()
    tmp.replace(local_path)
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", nargs="?", default="outputs/rig/t5/ep163_frame0_head.png")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--host", default=None)
    ap.add_argument("--remote-dir", default=None)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    host = args.host or os.environ.get("RIG_HOST", "ebim@192.168.0.5")
    remote_dir = args.remote_dir or os.environ.get("RIG_REMOTE_DIR", REMOTE_DIR_DEFAULT)
    remote_path = f"{remote_dir}/{REMOTE_SUBPATH}"
    repo_root = Path(__file__).resolve().parents[2]
    ref_path = Path(args.ref)
    if not ref_path.is_absolute():
        ref_path = repo_root / ref_path
    ref = Image.open(ref_path).convert("RGB")
    w, h = ref.size
    ref_arr = np.asarray(ref).astype(np.int16)

    out_dir = repo_root / "outputs/rig/live_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / "live_head_latest.png"

    period = 1.0 / max(args.hz, 0.1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
    ax.set_axis_off()
    title = ax.set_title("waiting for first frame...")
    fig.canvas.manager.set_window_title(f"live vs {ref_path.name}")
    plt.ion()
    plt.show()

    last_ok = None
    last_err = None
    print(
        f"polling {host}:{remote_path} at {args.hz} Hz, ref={ref_path.name} "
        "— close the window or Ctrl+C to stop"
    )
    try:
        while plt.fignum_exists(fig.number):
            t0 = time.monotonic()
            ok, err = fetch(host, remote_path, local_path, timeout_s=max(period, 2.0))
            if ok:
                try:
                    live = Image.open(local_path).convert("RGB")
                    if live.size != ref.size:
                        live = live.resize(ref.size, Image.BILINEAR)
                    live_arr = np.asarray(live).astype(np.int16)
                    blend = ((live_arr + ref_arr) // 2).astype(np.uint8)
                    diff = np.abs(live_arr - ref_arr).max(axis=2).mean()
                    im.set_data(blend)
                    title.set_text(
                        f"live+corpus blend  mean|diff|={diff:.1f}  {time.strftime('%H:%M:%S')}"
                    )
                    last_ok = time.monotonic()
                    last_err = None
                except OSError as exc:
                    ok, err = False, str(exc)
            if not ok:
                stale = (
                    "" if last_ok is None
                    else f" (last good frame {time.monotonic() - last_ok:.0f}s ago)"
                )
                title.set_text(f"NO FRAME{stale} — {err[:80]}")
                if err != last_err:  # don't spam the same scp error every tick
                    print(f"fetch failed: {err}", flush=True)
                    last_err = err
            fig.canvas.draw_idle()
            plt.pause(0.01)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
