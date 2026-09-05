#!/usr/bin/env python3
"""Red-pad slot check for the scored T6 block (protocol 15 §2.3).

Prints the red pad's centroid in a head-camera capture (1280x720) as pixels
and normalized x, and the verdict against a slot target from
``outputs/rig/munich_2026-09-01/t6_slot_sequence.csv`` (x048 = 614 px,
x051 = 653 px, x061 = 781 px; tolerance ±12 px ≈ ±0.01 of the frame).

    python3 tools/rig_probes/pad_centroid.py <capture.png> [--slot x051]
    tools/rig_probes/pad_centroid.sh --slot x051      # fetch newest capture from ebimHP first

The teal pads are printed too (their order and spacing must match the
corpus row: red first, three teals to its right at ≈ +80/+155/+225 px, or
red at the slot with the teals around it — the operator judges that part).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

SLOT_PX = {"x048": 614, "x051": 653, "x061": 781}
TOL_PX = 12


def blobs(arr: np.ndarray):
    a = arr.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 110) & (g < 90) & (b < 90) & (r - g > 60)
    teal = (g > 100) & (b > 100) & (r < 100) & (g - r > 40)
    out = {}
    for name, m in (("red", red), ("teal", teal)):
        ys, xs = np.nonzero(m)
        found = []
        if len(xs):
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            cl, cy = [[xs[0]]], [[ys[0]]]
            for x, y in zip(xs[1:], ys[1:], strict=False):
                if x - cl[-1][-1] > 25:
                    cl.append([x])
                    cy.append([y])
                else:
                    cl[-1].append(x)
                    cy[-1].append(y)
            found = [
                (int(np.mean(c)), int(np.mean(yy)), len(c))
                for c, yy in zip(cl, cy, strict=False)
                if len(c) > 120
            ]
        out[name] = found
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("--slot", choices=sorted(SLOT_PX), help="target slot from the sequence")
    args = ap.parse_args()
    img = Image.open(args.capture).convert("RGB")
    w, h = img.size
    arr = np.asarray(img)
    found = blobs(arr)
    print(f"{args.capture}: {w}x{h}")
    print(f"  teal pads: {[(x, y) for x, y, _ in found['teal']]}")
    if not found["red"]:
        print("  RED PAD NOT FOUND")
        return 2
    x, y, n = max(found["red"], key=lambda t: t[2])
    xn = x * 1280.0 / w
    print(f"  red pad: x={x} px ({x / w:.3f} of width, {xn:.0f} px at 1280), y={y}, {n} px")
    if args.slot:
        target = SLOT_PX[args.slot]
        d = xn - target
        ok = abs(d) <= TOL_PX
        verdict = "OK" if ok else "MOVE"
        print(f"  slot {args.slot}: target {target} px, diff {d:+.0f} px -> {verdict}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
