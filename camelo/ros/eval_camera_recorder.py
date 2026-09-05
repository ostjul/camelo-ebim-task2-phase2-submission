"""Record scene-camera RGB topics during a rollout into per-camera MP4s.

Eval Trigger only dumps stills (F-31); continuous video for a scored episode
has to be sampled from the live topics while the policy runs. Default set is
the top-down scoring camera plus the oblique demo_cam (benchmark
``cameras_*.yaml``, no LeRobot contract). ``demo_cam`` is optional: sims
without that publisher simply omit ``demo_cam.mp4``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from camelo.ros.camera_workers import CameraWorkers
from camelo.runner.video_writer import write_rgb_mp4

log = logging.getLogger(__name__)

EVAL_CAMERA_IMAGE_TOPIC = "/isaac/eval_camera/image_raw"
DEMO_CAMERA_IMAGE_TOPIC = "/isaac/demo_cam/image_raw"

# key → ROS image topic; MP4 basename is ``{key}.mp4``.
DEFAULT_CAMERAS: dict[str, str] = {
    "eval_camera": EVAL_CAMERA_IMAGE_TOPIC,
    "demo_cam": DEMO_CAMERA_IMAGE_TOPIC,
}

# Absent publisher → skip MP4, no warning (older sims / non-demo_cam branch).
OPTIONAL_CAMERAS: frozenset[str] = frozenset({"demo_cam"})


class EvalCameraRecorder:
    """One camera-worker process per topic; poll() from the control loop."""

    def __init__(
        self,
        cameras: dict[str, str] | None = None,
        optional: frozenset[str] | None = None,
    ):
        self._cameras = dict(cameras) if cameras is not None else dict(DEFAULT_CAMERAS)
        self._optional = OPTIONAL_CAMERAS if optional is None else frozenset(optional)
        self._workers = CameraWorkers(self._cameras)
        self.frames: dict[str, list] = {key: [] for key in self._cameras}
        self._last_id: dict[str, int | None] = {key: None for key in self._cameras}

    def poll(self) -> None:
        latest = self._workers.poll()
        for key in self._cameras:
            frame = latest.get(key)
            if frame is None:
                continue
            # Workers ship CameraFrame(array, t_stamp, t_wall); the MP4 wants
            # the array alone (a tuple here reached write_rgb_mp4 as an
            # inhomogeneous (3,) "frame" and killed the episode wrap-up).
            frame = getattr(frame, "array", frame)
            # Identity check: workers hand us a new ndarray per decode.
            frame_id = id(frame)
            if frame_id == self._last_id[key]:
                continue
            self._last_id[key] = frame_id
            self.frames[key].append(frame)

    def write(self, episode_dir: Path, duration_s: float) -> dict[str, Path]:
        """Write ``{key}.mp4`` for cameras that delivered frames.

        Required cameras with zero frames log a warning. Optional ones
        (``demo_cam``) are silent skips so older sims keep working.
        """
        written: dict[str, Path] = {}
        for key, frames in self.frames.items():
            path = episode_dir / f"{key}.mp4"
            if not frames:
                if key in self._optional:
                    log.info(
                        "scene camera recorder: skipping optional %s (no frames on %s)",
                        key,
                        self._cameras[key],
                    )
                else:
                    log.warning(
                        "scene camera recorder: no frames for %s (%s)",
                        key,
                        self._cameras[key],
                    )
                continue
            fps = len(frames) / max(float(duration_s), 1e-3)
            # Print the three numbers together: a caller that sizes duration_s
            # on the wrong span shows up here as an implausible fps rather than
            # as a silently time-scaled file.
            log.info(
                "scene camera recorder: %s — %d frames over %.1f sim-s -> %.2f fps",
                key,
                len(frames),
                float(duration_s),
                fps,
            )
            out = write_rgb_mp4(path, frames, fps)
            if out is not None:
                written[key] = out
        return written

    def close(self) -> None:
        self._workers.close()
        self.frames = {key: [] for key in self._cameras}
        self._last_id = {key: None for key in self._cameras}
