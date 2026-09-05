#!/usr/bin/env bash
# Fetch the newest head-camera capture from ebimHP and overlay it on a corpus
# frame 0 (rig checklist C-15, docs/realdata/16 §3.2 T6-5). Run on the Mac:
#
#   tools/rig_probes/overlay_latest.sh                 # vs episode 163 frame 0
#   tools/rig_probes/overlay_latest.sh outputs/rig/t5/ep009_frame0_head.png
#
# Prints the mean pixel difference (Tuesday: 93 = base ~1 m off, ~30 = placed)
# and opens the side-by-side and the 50/50 blend. Needs python3 with numpy+PIL.
set -euo pipefail
cd "$(dirname "$0")/../.."
REF="${1:-outputs/rig/t5/ep163_frame0_head.png}"
HOST="${RIG_HOST:-ebim@192.168.0.5}"
OUT="outputs/rig/munich_2026-09-01/t6_overlays"
mkdir -p "$OUT"
REMOTE=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  'find ~/camelo/camelo-ebim/outputs/rig -type f -name "live_head_*.png" -printf "%T@ %p\n" | sort -rn | head -1 | cut -d" " -f2-')
NAME=$(basename "$REMOTE" .png)
scp -q "$HOST:$REMOTE" "$OUT/$NAME.png"
TAG=$(basename "$REF" _frame0_head.png)
python3 tools/rig_probes/overlay_head.py "$OUT/$NAME.png" "$REF" "$OUT/cmp_${TAG}_$NAME"
if command -v open >/dev/null; then
  open "$OUT/cmp_${TAG}_${NAME}_side.png" "$OUT/cmp_${TAG}_${NAME}_blend.png"
fi
