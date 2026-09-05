"""Fixed, hand-built observations for the intermediate "prove the wire works"
step (scripts/run_dummy_client.py) and for --dummy-obs on run_policy.py.

Numpy-only (no rclpy, no torch) so both a plain-Python smoke test and the
real ROS entry points can import it — camelo/__init__.py's layering rule.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from camelo import contracts as C
from camelo.policy.adapters import s27a15
from camelo.policy.base import Obs

log = logging.getLogger("camelo.policy.dummy_obs")

DEFAULT_TASK = {"canonical": C.TASK2_INSTRUCTION, "s27a15": s27a15.TASK_CAPTION}


def make_dummy_obs_canonical(t_sim: float) -> Obs:
    """One fixed canonical Obs: zero 37-dim state, mid-grey images at each
    SIM camera's resolution (camelo.contracts.CAMERAS) — for sim-trained
    (model16/canonical) checkpoints."""
    state = np.zeros(C.STATE_DIM, dtype=np.float32)
    images = {
        key: np.full(C.CAMERAS[key]["shape"], 128, dtype=np.uint8) for key in C.CAMERA_KEYS
    }
    image_t_sim = dict.fromkeys(images, t_sim)
    return Obs(t_sim=t_sim, state=state, images=images, image_t_sim=image_t_sim)


def make_dummy_obs_s27a15(t_sim: float) -> Obs:
    """One fixed s27a15 Obs: named rig groups (not a flat state — the real
    bridge's shape, see camelo.ros.obs_collector.rig_groups) plus mid-grey
    images at each REAL camera's resolution (camelo.contracts real topic
    map — the rig's wrists are 640x480, NOT the sim's 848x480)."""
    rig = {
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "right_gripper_rad": 0.0,  # open; within s27a15.KNUCKLE_RAD band
        "left_wrench": [0.0] * 6,
        "right_wrench": [0.0] * 6,
    }
    real_cameras = C.topics_for("real").cameras
    images = {
        key: np.full(real_cameras[key]["shape"], 128, dtype=np.uint8) for key in real_cameras
    }
    image_t_sim = dict.fromkeys(images, t_sim)
    return Obs(
        # wire.py validates state against STATE_DIMS=(37, 27) unconditionally
        # (rig or not) — 37 zeros, matching what ObsCollector.get_obs() always
        # builds alongside rig on a real run. The adapter ignores this and
        # reads `rig` instead whenever it is set (see LeRobotAdapter._rig_state).
        t_sim=t_sim,
        state=np.zeros(C.STATE_DIM, dtype=np.float32),
        images=images,
        image_t_sim=image_t_sim,
        rig=rig,
    )


OBS_BUILDERS = {"canonical": make_dummy_obs_canonical, "s27a15": make_dummy_obs_s27a15}

_LABEL_WIDTH = 10
_COL_WIDTH = 8


def _table(rows: list[tuple[str, list[float]]]) -> str:
    """rows = [(label, values), ...] -> a header of column names (j1, j2, ...)
    plus one right-aligned row per entry, fixed-width columns so left_arm and
    right_arm line up joint-for-joint underneath the same header."""
    n = len(rows[0][1])
    header = " " * _LABEL_WIDTH + "".join(f"j{i + 1}".rjust(_COL_WIDTH) for i in range(n))
    lines = ["  " + header]
    for label, values in rows:
        cells = "".join(f"{v:{_COL_WIDTH}.2f}" for v in values)
        lines.append(f"  {label.ljust(_LABEL_WIDTH)}{cells}")
    return "\n".join(lines)


def format_action_s27a15(step: np.ndarray) -> str:
    """One (15,) action step -> a left_arm/right_arm joint table plus the
    gripper — camelo.policy.adapters.s27a15."""
    table = _table(
        [
            ("left_arm", [float(v) for v in step[s27a15.A_LEFT_ARM]]),
            ("right_arm", [float(v) for v in step[s27a15.A_RIGHT_ARM]]),
        ]
    )
    grip = float(step[s27a15.A_RIGHT_GRIP])
    return f"{table}\n  {'gripper'.ljust(_LABEL_WIDTH)}right_open={grip:.2f}"


def format_action_canonical(step: np.ndarray) -> str:
    """One (20,) action step -> a left_arm/right_arm joint table plus
    gripper/base/spine — camelo.contracts A_* layout."""
    base = step[C.A_BASE]
    table = _table(
        [
            ("left_arm", [float(v) for v in step[C.A_LEFT_ARM]]),
            ("right_arm", [float(v) for v in step[C.A_RIGHT_ARM]]),
        ]
    )
    return (
        f"{table}\n"
        f"  {'gripper'.ljust(_LABEL_WIDTH)}left={float(step[C.A_LEFT_GRIP]):.2f} "
        f"right={float(step[C.A_RIGHT_GRIP]):.2f}\n"
        f"  {'base'.ljust(_LABEL_WIDTH)}vx={base[0]:.2f} vy={base[1]:.2f} wz={base[2]:.2f} "
        f"spine={float(step[C.A_SPINE]):.2f}"
    )


ACTION_FORMATTERS = {"canonical": format_action_canonical, "s27a15": format_action_s27a15}


def layout_for_args(action_layout: str | None, state_layout: str | None) -> str:
    """--action-layout/--state-layout -> which OBS_BUILDERS/ACTION_FORMATTERS
    key applies. Mirrors camelo.cli.action_space_from_args's own check."""
    return "s27a15" if "s27a15" in (action_layout, state_layout) else "canonical"


class FixedObsBackend:
    """Wraps a PolicyBackend so every infer() call is answered from a FIXED,
    hand-built observation (OBS_BUILDERS) instead of whatever the caller
    passed in — only ``obs.t_sim`` is kept, so a real control loop anchored
    on it (executor.needs_replan, chunk.t0 bookkeeping) still ticks
    correctly. This substitutes ONLY what the policy sees; the collector,
    the executor's max-delta clamp, the publisher, keepalive, and
    deactivate-on-exit all keep running on REAL measured state — so on a
    real-robot run the arms move on whatever the checkpoint predicts from
    this fake input, safely rate/delta-limited, but not sensibly.

    For scripts/run_policy.py --dummy-obs — verifying the command-publish
    path end to end before real sensor data is wired in.
    """

    def __init__(self, inner, layout: str):
        self.inner = inner
        self.layout = layout
        self._make_obs = OBS_BUILDERS[layout]
        self._format_action = ACTION_FORMATTERS[layout]

    @property
    def adapter(self):
        """Pass the wrapped adapter through so `action_space_from_args`'s
        adapter-vs-flags cross-check still sees it (a RemoteBackend has none)."""
        return getattr(self.inner, "adapter", None)

    def reset(self, task: str) -> None:
        self.inner.reset(task)

    def infer(self, obs: Obs):
        dummy = self._make_obs(obs.t_sim)
        chunk = self.inner.infer(dummy)
        log.info(
            "DUMMY-OBS chunk t0=%.2f next_action (step 0/%d):\n%s",
            chunk.t0,
            chunk.actions.shape[0],
            self._format_action(np.asarray(chunk.actions[0])),
        )
        return chunk

    def close(self) -> None:
        self.inner.close()

    def stats(self) -> dict:
        return self.inner.stats()


# ---------------------------------------------------------------------------
# U-35 probe: send a RECORDED corpus observation through the real client path
# (docs/realdata/16d_T6_LEARNINGS.md). Unlike OBS_BUILDERS above, this reads
# genuine rig data — real images, real joints, real wrenches — for one frame
# of a specific episode, so the returned chunk can be diffed against the
# demo's own recorded action_chunk. If they agree, the wire and server are
# right and the live scene/cameras/wrenches are the gap; if they disagree,
# the client -> server path itself is wrong. Numpy + stdlib + PIL only, so
# this stays importable with no ROS, no torch, no lerobot (layering rule).
# ---------------------------------------------------------------------------

#: dims 0..13 of an s27a15 action row (both arms), radians — see
#: `chunk_diff_verdict`.
_ARM_DIFF_TOL_RAD = 0.05
#: dim 14 (right gripper), open fraction (0 = closed, 1 = open).
_GRIPPER_DIFF_TOL = 0.1


def load_image_rgb(path: Path) -> np.ndarray:
    """PNG on disk -> HxWx3 uint8 RGB — exactly what ObsCollector produces
    from a live camera topic (camelo.policy.adapters.s27a15 docstring)."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_recorded_frame(obs_dir: Path, frame: int) -> tuple[dict, dict]:
    """`<obs_dir>/*_obs.json` -> (the whole manifest, the entry for `frame`).

    The manifest name is not hardcoded (today's is `ep163_obs.json`, a
    different episode would be `ep164_obs.json`, ...) — there must be
    exactly one `*_obs.json` in the directory. Raises with the available
    frame indices listed if `frame` is not one of them.
    """
    matches = sorted(Path(obs_dir).glob("*_obs.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one *_obs.json in {obs_dir}, found {[str(m) for m in matches]}"
        )
    manifest = json.loads(matches[0].read_text())
    frames = manifest.get("frames") or []
    for entry in frames:
        if entry.get("frame") == frame:
            return manifest, entry
    available = [entry.get("frame") for entry in frames]
    raise RuntimeError(f"frame {frame} not found in {matches[0]}; available frames: {available}")


def rig_groups_from_state27(state27) -> dict:
    """Recorded 27-dim `observation.state` (s27a15 order) -> the named rig
    groups `s27a15.RigObservation.from_mapping` expects.

    Uses the ALREADY-CONVERTED open fraction at dim 14 directly
    (`right_gripper_open`) rather than routing it back through
    `right_gripper_rad` — the corpus recorded the open fraction, and a
    rad -> open round trip would reapply the knuckle clip band
    (`s27a15.KNUCKLE_RAD` / `GRIPPER_CLOSED_RAD`) a second time for no
    reason. The wrenches are read straight from the recording, never
    zeroed (`RigObservation` requires them for exactly this reason).
    """
    vec = np.asarray(state27, dtype=np.float64).ravel()
    if vec.shape != (s27a15.STATE_DIM,):
        raise RuntimeError(
            f"recorded observation.state must be {s27a15.STATE_DIM}-dim, got {vec.shape}"
        )
    return {
        "left_arm": vec[s27a15.S_LEFT_ARM].tolist(),
        "right_arm": vec[s27a15.S_RIGHT_ARM].tolist(),
        "right_gripper_open": float(vec[s27a15.S_RIGHT_GRIP]),
        "left_wrench": vec[s27a15.S_LEFT_WRENCH].tolist(),
        "right_wrench": vec[s27a15.S_RIGHT_WRENCH].tolist(),
    }


def build_obs_from_recorded_frame(obs_dir: Path, frame_entry: dict) -> Obs:
    """A `load_recorded_frame` entry -> the exact `Obs` the live s27a15
    bridge would build for that instant: real images (RGB uint8 HxWx3, PIL
    `convert("RGB")` per `RigObservation`'s docstring), real joints, real
    wrenches, via `Obs.rig` — never the fixed mid-grey/zero-state dummy
    above. `Obs.state` stays 37 zeros, matching what a live run's
    `ObsCollector.get_obs()` always builds alongside `rig` (wire.py
    validates against `STATE_DIMS=(37, 27)` unconditionally); the adapter
    ignores it and reads `rig` instead whenever it is set.
    """
    obs_dir = Path(obs_dir)
    images: dict[str, np.ndarray] = {}
    for camera, info in frame_entry["images"].items():
        images[camera] = load_image_rgb(obs_dir / info["file"])
    t_sim = float(frame_entry.get("timestamp", 0.0))
    rig = rig_groups_from_state27(frame_entry["observation.state"])
    return Obs(
        t_sim=t_sim,
        state=np.zeros(C.STATE_DIM, dtype=np.float32),
        images=images,
        image_t_sim=dict.fromkeys(images, t_sim),
        rig=rig,
    )


def build_obs_from_mixed_frames(images_dir: Path, images_entry: dict, state_entry: dict) -> Obs:
    """Like `build_obs_from_recorded_frame`, but the images come from one
    obs-dir entry and the state (arms + gripper + wrenches) from a
    DIFFERENT one — `scripts/run_dummy_client.py --images-from/--state-from`
    (U-35 isolation: is a rollout mismatch coming from the live images or
    the live wrenches?). Either side may be a recorded corpus frame or a
    `--capture-live` capture — both share the `*_obs.json` schema
    (`load_recorded_frame` reads either one the same way), so this function
    does not care which is which.

    `t_sim`/`image_t_sim` are taken from the IMAGES entry (they stamp the
    images); the STATE entry's own timestamp is not used here.
    """
    images_dir = Path(images_dir)
    images: dict[str, np.ndarray] = {}
    for camera, info in images_entry["images"].items():
        images[camera] = load_image_rgb(images_dir / info["file"])
    t_sim = float(images_entry.get("timestamp", 0.0))
    rig = rig_groups_from_state27(state_entry["observation.state"])
    return Obs(
        t_sim=t_sim,
        state=np.zeros(C.STATE_DIM, dtype=np.float32),
        images=images,
        image_t_sim=dict.fromkeys(images, t_sim),
        rig=rig,
    )


#: Human-readable label per s27a15 `observation.state` dim, in order — arms
#: in radians, gripper as an open fraction, wrenches in N (force) / N*m
#: (torque). Used only for the `--capture-live --obs-dir` side-by-side
#: printout below; `s27a15.STATE_NAMES` is the machine-checked source of
#: truth for the actual dim order.
_STATE27_LABELS = (
    *(f"left_arm_j{i}" for i in range(1, 8)),
    *(f"right_arm_j{i}" for i in range(1, 8)),
    "right_gripper_open",
    *(f"left_wrench_{c}" for c in ("fx", "fy", "fz", "tx", "ty", "tz")),
    *(f"right_wrench_{c}" for c in ("fx", "fy", "fz", "tx", "ty", "tz")),
)


def format_state27_diff_table(live_state27, recorded_state27) -> str:
    """Side-by-side live-vs-recorded table, one row per named s27a15 state
    dim (arms in rad, gripper as an open fraction, wrenches in N/N*m) — the
    U-35 isolation printout for `scripts/run_dummy_client.py --capture-live
    --obs-dir`."""
    live = np.asarray(live_state27, dtype=np.float64).ravel()
    recorded = np.asarray(recorded_state27, dtype=np.float64).ravel()
    if live.shape != (s27a15.STATE_DIM,) or recorded.shape != (s27a15.STATE_DIM,):
        raise RuntimeError(
            f"state27 diff needs two {s27a15.STATE_DIM}-dim vectors, got "
            f"live={live.shape} recorded={recorded.shape}"
        )
    lines = [f"  {'dim':<20}{'live':>10}{'recorded':>10}{'diff':>10}"]
    for label, lv, rv in zip(_STATE27_LABELS, live, recorded, strict=True):
        lines.append(f"  {label:<20}{lv:10.4f}{rv:10.4f}{lv - rv:10.4f}")
    return "\n".join(lines)


def mean_abs_pixel_diff(live_image, recorded_image) -> float:
    """Mean |live - recorded| over every pixel/channel. Raises if the two
    images are not the same shape — a shape mismatch (e.g. the corpus's
    HD720 vs a station camera moved to VGA, docs/realdata/16 R-42) is
    itself the finding here and must be reported, not silently resized or
    cropped."""
    live_image = np.asarray(live_image)
    recorded_image = np.asarray(recorded_image)
    if live_image.shape != recorded_image.shape:
        raise RuntimeError(
            f"camera shape mismatch: live={live_image.shape} recorded={recorded_image.shape}"
        )
    return float(np.abs(live_image.astype(np.float64) - recorded_image.astype(np.float64)).mean())


def save_live_frame(outdir: Path, obs: Obs, state27, frame: int = 0) -> Path:
    """One live `Obs` (as captured by `camelo.ros.obs_collector.ObsCollector`
    + `camelo.runner.episode_runner.wait_for_obs`) plus its packed 27-dim
    state -> a `live_obs.json` manifest and `live_f{frame:03d}_{camera}.png`
    files, in EXACTLY the schema `load_recorded_frame` /
    `build_obs_from_recorded_frame` read back (see
    `tests/test_dummy_client_obs_dir.py::_write_frame_dir` for the
    recorded-corpus original this mirrors) — so a live capture can be
    pointed at by `--obs-dir`, `--images-from`, or `--state-from` exactly
    like a recorded episode directory.

    Deliberately carries NO `action`/`action_chunk` key: a live capture has
    no ground-truth action, unlike a recorded corpus frame.
    `load_recorded_frame`'s callers must treat a missing action_chunk as
    "skip the diff/verdict", not as an error
    (`scripts/run_dummy_client.py::_run_obs_dir`).

    Images are saved RGB uint8 exactly as the model receives them (PIL
    `Image.fromarray`, no re-encoding beyond the PNG container itself) —
    the same representation `load_image_rgb` reads back.

    Returns the path to the written manifest.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state27 = np.asarray(state27, dtype=np.float64).ravel()
    if state27.shape != (s27a15.STATE_DIM,):
        raise RuntimeError(f"state27 must be {s27a15.STATE_DIM}-dim, got {state27.shape}")

    from PIL import Image

    images_manifest: dict[str, dict] = {}
    camera_shapes: dict[str, list[int]] = {}
    for camera in s27a15.CAMERA_KEYS:
        if camera not in obs.images:
            continue
        array = np.asarray(obs.images[camera])
        fname = f"live_f{frame:03d}_{camera}.png"
        Image.fromarray(array).save(outdir / fname)
        images_manifest[camera] = {"file": fname, "shape": list(array.shape)}
        camera_shapes[f"observation.images.{camera}"] = list(array.shape)

    manifest = {
        "episode": None,
        "source": "live",
        "valid": 1,
        "state_feature_names": list(s27a15.STATE_NAMES),
        "action_feature_names": list(s27a15.ACTION_NAMES),
        "camera_shapes_hwc": camera_shapes,
        "frames": [
            {
                "frame": frame,
                "timestamp": float(obs.t_sim),
                "observation.state": state27.tolist(),
                "images": images_manifest,
            }
        ],
    }
    manifest_path = outdir / "live_obs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def chunk_diff_verdict(served, demo) -> tuple[np.ndarray, str]:
    """Served (T, 15) chunk vs the demo's own recorded `action_chunk` ->
    (per-row max|diff| over the 15 dims, one-line verdict).

    `WIRE OK` when every row agrees within `_ARM_DIFF_TOL_RAD` (0.05 rad) on
    the 14 arm dims and within `_GRIPPER_DIFF_TOL` (0.1 open fraction) on
    the gripper dim; `WIRE MISMATCH` otherwise. Agreement here means the
    client -> server wire and the server-side decode are right and a live
    rollout's disagreement comes from the scene/cameras/wrenches instead
    (U-35, docs/realdata/16d_T6_LEARNINGS.md).
    """
    served = np.asarray(served, dtype=np.float64)
    demo = np.asarray(demo, dtype=np.float64)
    if served.shape != demo.shape or served.ndim != 2 or served.shape[-1] != s27a15.ACTION_DIM:
        raise RuntimeError(
            f"chunk shape mismatch: served={served.shape} demo={demo.shape}, expected "
            f"matching (T, {s27a15.ACTION_DIM})"
        )
    diff = np.abs(served - demo)
    per_row_max = diff.max(axis=1)
    arm_ok = bool(
        np.all(diff[:, s27a15.A_LEFT_ARM] <= _ARM_DIFF_TOL_RAD)
        and np.all(diff[:, s27a15.A_RIGHT_ARM] <= _ARM_DIFF_TOL_RAD)
    )
    gripper_ok = bool(np.all(diff[:, s27a15.A_RIGHT_GRIP] <= _GRIPPER_DIFF_TOL))
    verdict = "WIRE OK" if (arm_ok and gripper_ok) else "WIRE MISMATCH"
    return per_row_max, verdict
