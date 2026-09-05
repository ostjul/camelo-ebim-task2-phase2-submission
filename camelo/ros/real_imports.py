"""Preflight for the ROS packages the real-robot (`--world real`) path needs.

MEASURED on the rig 2026-09-02 (run t1r_120415, docs/realdata/16 R-38):
`scripts/sine_probe_real.py --activate-arms` — and therefore
`scripts/run_policy.py --world real --activate-arms` / `--wait-for-activation`,
which drive the same `ArmControllers.__init__` — exited at start with
``ModuleNotFoundError: No module named 'controller_manager_msgs'`` in BOTH
the station's RoboStack Humble pixi env (`ros-humble-desktop` does not carry
the ros2_control message packages) and the `camelo-ebim/ros:latest`
container (`docker/Dockerfile.ros` never installed it either). The failure
was safe — it happened before any publish — but it meant the activation
choreography had never been runnable anywhere.

This module lets an entry point check every real-path import up front and
report ALL that are missing, with a fix hint, instead of dying on whichever
import a given code path happens to hit first. Kept out of
`camelo/ros/arm_activation.py` so it can be imported (and unit-tested) with
no ROS installed and without pulling in that module's activation logic.

Layering: only `camelo/ros` may import rclpy/message packages, and only
lazily, inside functions — this module is no exception. `importlib` is the
only import at module scope.
"""

from __future__ import annotations

import importlib

#: One entry per package actually imported somewhere under camelo/ros for a
#: `--world real` run (grepped 2026-09-02: session.py, qos.py,
#: camera_workers.py, obs_collector.py, command_publisher.py,
#: arm_activation.py). Checked at module granularity — e.g. `sensor_msgs.msg`
#: covers both `JointState` and `Image`, whichever submodule imports them.
REAL_WORLD_MODULES: tuple[str, ...] = (
    "rclpy",
    "controller_manager_msgs.srv",
    "std_msgs.msg",
    "sensor_msgs.msg",
    "geometry_msgs.msg",
    "nav_msgs.msg",
    "rosgraph_msgs.msg",
)

#: `controller_manager_msgs` is the one package `ros-${ROS_DISTRO}-desktop`
#: does not carry (it ships with ros2_control, not ros-base/desktop) — the
#: only module here likely to actually be missing. The others are common
#: interfaces bundled with any ROS desktop/base install.
FIX_HINT = (
    "pixi add ros-humble-controller-manager-msgs in ~/teleoperation/station "
    "(site env change, operator decision), or rebuild the camelo-ros image "
    "after the docker/Dockerfile.ros change (docs/realdata/16 R-38)"
)


def check_real_imports(modules: tuple[str, ...] = REAL_WORLD_MODULES) -> dict[str, str | None]:
    """``{module: None}`` if importable, ``{module: str(error)}`` if not.

    Never raises: a missing package is data for the caller to report, not a
    reason for this function itself to blow up.
    """
    results: dict[str, str | None] = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            results[module_name] = str(exc)
        except Exception as exc:  # pragma: no cover - defensive: report, don't hide
            results[module_name] = f"{type(exc).__name__}: {exc}"
        else:
            results[module_name] = None
    return results


def missing_real_imports(results: dict[str, str | None] | None = None) -> list[str]:
    """Module names from `results` (or a fresh `check_real_imports()`) that
    failed to import, in the order they were checked."""
    if results is None:
        results = check_real_imports()
    return [name for name, error in results.items() if error is not None]


def report_real_imports(results: dict[str, str | None] | None = None) -> str:
    """Human-readable present/missing table plus the fix hint, for an entry
    point to print (and refuse early on) before touching the ROS graph."""
    if results is None:
        results = check_real_imports()
    lines = ["real-world ROS imports:"]
    for name, error in results.items():
        lines.append(f"  ok      {name}" if error is None else f"  MISSING {name} ({error})")
    if missing_real_imports(results):
        lines.append(f"fix: {FIX_HINT}")
    return "\n".join(lines)
