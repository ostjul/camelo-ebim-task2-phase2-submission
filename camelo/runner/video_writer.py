"""Write RGB frame lists to H.264 MP4 via system ffmpeg (no torchcodec)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def write_rgb_mp4(
    path: Path,
    frames: list[np.ndarray],
    fps: float,
) -> Path | None:
    """Encode HxWx3 uint8 RGB frames. Returns path on success, else None."""
    if not frames:
        log.warning("write_rgb_mp4: no frames for %s", path)
        return None
    if shutil.which("ffmpeg") is None:
        log.error("write_rgb_mp4: ffmpeg not on PATH — cannot write %s", path)
        return None

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frames, got shape {first.shape}")
    height, width = int(first.shape[0]), int(first.shape[1])
    fps = max(float(fps), 0.1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            arr = np.ascontiguousarray(frame, dtype=np.uint8)
            if arr.shape[0] != height or arr.shape[1] != width:
                raise ValueError(
                    f"frame shape {arr.shape[:2]} != first frame {(height, width)}"
                )
            proc.stdin.write(arr.tobytes())
    finally:
        proc.stdin.close()
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait(timeout=120)
    if rc != 0:
        log.error(
            "ffmpeg failed rc=%s for %s: %s",
            rc,
            path,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return None
    log.info("wrote %s (%d frames @ %.2f fps)", path, len(frames), fps)
    return path
