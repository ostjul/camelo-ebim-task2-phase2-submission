#!/usr/bin/env python3
"""overlay_head.py <live.png> <corpus_frame0.png> <out_prefix>

Rig checklist C-15 (docs/realdata/16 R-58..R-61): place the mobile base and the
table by overlaying a live head-camera frame (scripts/capture_head.py) on the
corpus episode's frame 0. Writes <out_prefix>_side.png (live | corpus, same
height), <out_prefix>_blend.png (50/50 blend at the corpus size) and
<out_prefix>_diff.png (abs difference); prints the mean absolute pixel
difference, which fell from 93 (base ~1 m off) to ~40 (placed) on 2026-09-02.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

live = Image.open(sys.argv[1]).convert("RGB")
ref = Image.open(sys.argv[2]).convert("RGB")
out = sys.argv[3]
if live.size != ref.size:
    live = live.resize(ref.size, Image.BILINEAR)
w, h = ref.size
side = Image.new("RGB", (2 * w + 10, h), (40, 40, 40))
side.paste(live, (0, 0))
side.paste(ref, (w + 10, 0))
d = ImageDraw.Draw(side)
d.text((10, 10), "LIVE (today)", fill=(255, 255, 0))
d.text((w + 20, 10), "CORPUS frame 0", fill=(255, 255, 0))
side.save(out + "_side.png")
blend = Image.blend(live, ref, 0.5)
ImageDraw.Draw(blend).text((10, 10), "50/50 blend: live + corpus", fill=(255, 255, 0))
blend.save(out + "_blend.png")
a = np.asarray(live).astype(np.int16)
b = np.asarray(ref).astype(np.int16)
diff = np.abs(a - b).max(axis=2).astype(np.uint8)
Image.fromarray(diff).save(out + "_diff.png")
print("wrote", out + "_side.png", out + "_blend.png", out + "_diff.png", "size", ref.size,
      f"mean|diff|={diff.mean():.1f}")
