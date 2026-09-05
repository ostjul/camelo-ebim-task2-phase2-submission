"""Shared helpers for GT-trajectory debugging notebooks.

Numpy / pandas / matplotlib only — no ROS, no Isaac, no lerobot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from camelo.control.gt_traj import GT_TRAJ_KEYS

_START_MARKERS = ("o", "s", "D", "^")


def wrap_pi(yaw: np.ndarray) -> np.ndarray:
    """Wrap radians to ``[-pi, pi)``."""
    return (np.asarray(yaw, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def fmt_xyz_mm(xyz: np.ndarray) -> str:
    mm = np.asarray(xyz, dtype=float).reshape(3) * 1000.0
    return f"x={mm[0]:+.4f} y={mm[1]:+.4f} z={mm[2]:+.4f} mm"


def every_nth_idx(n_samples: int, n: int) -> np.ndarray:
    n = max(1, int(n))
    idx = list(range(0, n_samples, n))
    last = n_samples - 1
    if not idx or idx[-1] != last:
        idx.append(last)
    return np.asarray(idx, dtype=int)


def draw_heading(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    *,
    color: str,
    tick_m: float,
    lw: float = 1.2,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    dx = tick_m * np.cos(yaw)
    dy = tick_m * np.sin(yaw)
    ax.plot(
        np.vstack([x, x + dx]),
        np.vstack([y, y + dy]),
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=4,
    )


def resample_xy_yaw(
    t_src: np.ndarray,
    xy_yaw: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Linear interp of ``(x, y, yaw)`` onto ``t_dst``.

    Clamps to the first/last source sample outside the source time span.
    Yaw is unwrapped before interp, then wrapped to ``[-pi, pi)``.
    """
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    xy_yaw = np.asarray(xy_yaw, dtype=np.float64)
    if t_src.size < 1:
        raise ValueError("t_src must be non-empty")
    if xy_yaw.shape != (t_src.shape[0], 3):
        raise ValueError(
            f"xy_yaw must be ({t_src.shape[0]}, 3), got {xy_yaw.shape}"
        )
    order = np.argsort(t_src, kind="mergesort")
    t_src = t_src[order]
    xy_yaw = xy_yaw[order]
    x = np.interp(t_dst, t_src, xy_yaw[:, 0])
    y = np.interp(t_dst, t_src, xy_yaw[:, 1])
    yaw = np.interp(t_dst, t_src, np.unwrap(xy_yaw[:, 2]))
    return np.column_stack([x, y, wrap_pi(yaw)])


def resample_xyz(
    t_src: np.ndarray,
    xyz: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Linear interp of ``(x, y, z)`` onto ``t_dst`` (clamp outside span)."""
    t_src = np.asarray(t_src, dtype=np.float64).reshape(-1)
    t_dst = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float64)
    if t_src.size < 1:
        raise ValueError("t_src must be non-empty")
    if xyz.shape != (t_src.shape[0], 3):
        raise ValueError(f"xyz must be ({t_src.shape[0]}, 3), got {xyz.shape}")
    order = np.argsort(t_src, kind="mergesort")
    t_src = t_src[order]
    xyz = xyz[order]
    return np.column_stack(
        [np.interp(t_dst, t_src, xyz[:, i]) for i in range(3)]
    )


def ensure_tabular(repo_id: str, dataset_dir: Path) -> Path:
    """Reuse cached meta+data, or snapshot_download them (no videos)."""
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    parquet = (
        list((dataset_dir / "data").rglob("*.parquet"))
        if dataset_dir.is_dir()
        else []
    )
    if info_path.is_file() and parquet:
        print(f"reusing {dataset_dir} ({len(parquet)} parquet files)")
        return dataset_dir
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id} (meta+data only) -> {dataset_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=["meta/**", "data/**"],
        )
    except GatedRepoError:
        raise SystemExit(
            f"gated: accept terms at https://huggingface.co/datasets/{repo_id} "
            "and export HF_TOKEN"
        ) from None
    return dataset_dir


def load_episode(dataset_dir: Path, episode: int) -> tuple[Any, float]:
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_dir}/data")
    cols = ["episode_index", "frame_index", "observation.state", "action"]
    parts = []
    for path in files:
        df = pd.read_parquet(path, columns=cols)
        part = df[df.episode_index == episode]
        if len(part):
            parts.append(part)
    if not parts:
        raise SystemExit(f"episode {episode} not found in {dataset_dir}")
    ep = pd.concat(parts, ignore_index=True).sort_values("frame_index")
    return ep.reset_index(drop=True), fps


def load_gt_traj(path: Path) -> dict[str, np.ndarray]:
    """Load ``epXXX_gt_traj.npz`` from the extract notebook."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"GT trajectory not found at {path} — run "
            "scripts/debugging/extract_GT_trajectory.ipynb for that episode"
        )
    with np.load(path) as data:
        missing = [k for k in GT_TRAJ_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"{path} missing keys {missing}")
        return {k: np.asarray(data[k]) for k in data.files}


def plot_mobile_base_overlay(
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    *,
    title: str,
    plot_every_n: int = 20,
    heading_tick_m: float = 0.15,
    figsize: tuple[float, float] = (16, 3.6),
):
    """4-panel base plot. Each series is ``(t, base_xy_yaw, color, label)``.

    Native time per series (lengths may differ). Square path box, independent
    x/y scales. Start green, end gold; marker shape differs per series.
    """
    import matplotlib.pyplot as plt

    if not series:
        raise ValueError("series must be non-empty")
    fig, (ax_path, ax_x, ax_y, ax_yaw) = plt.subplots(
        1, 4, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1, 1, 1]}
    )
    n = len(series)
    for i, (t, base, color, label) in enumerate(series):
        t = np.asarray(t, dtype=float).reshape(-1)
        base = np.asarray(base, dtype=float)
        if base.shape != (t.shape[0], 3):
            raise ValueError(
                f"{label}: base_xy_yaw must be ({t.shape[0]}, 3), got {base.shape}"
            )
        bx, by, byaw = base[:, 0], base[:, 1], base[:, 2]
        idx = every_nth_idx(len(t), plot_every_n)
        ax_path.plot(bx, by, color=color, lw=1.0, zorder=2, label=label)
        draw_heading(
            ax_path,
            bx[idx],
            by[idx],
            byaw[idx],
            color=color,
            tick_m=heading_tick_m,
        )
        marker = _START_MARKERS[i % len(_START_MARKERS)]
        start_lab = "start" if n == 1 else f"{label} start"
        end_lab = "end" if n == 1 else f"{label} end"
        ax_path.scatter(
            bx[0], by[0], c="green", s=40, zorder=5, marker=marker, label=start_lab
        )
        ax_path.scatter(
            bx[-1], by[-1], c="gold", s=40, zorder=5, marker=marker, label=end_lab
        )
        ax_x.plot(t, bx, color=color, lw=1.0, label=label)
        ax_y.plot(t, by, color=color, lw=1.0, label=label)
        ax_yaw.plot(t, np.degrees(byaw), color=color, lw=1.0, label=label)

    ax_path.set_box_aspect(1)
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("base path (world)")
    ax_path.legend(loc="best", fontsize=8)
    ax_path.grid(True, alpha=0.3)
    for ax, ylabel in zip(
        (ax_x, ax_y, ax_yaw),
        ("x (m)", "y (m)", "yaw (deg)"),
        strict=True,
    ):
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if n > 1:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_path, ax_x, ax_y, ax_yaw)


def plot_mobile_base_pose(
    t: np.ndarray,
    base_xy_yaw: np.ndarray,
    *,
    title: str,
    color: str = "tab:blue",
    label: str = "base",
    plot_every_n: int = 20,
    heading_tick_m: float = 0.15,
):
    """Single-series wrapper around ``plot_mobile_base_overlay``."""
    return plot_mobile_base_overlay(
        [(t, base_xy_yaw, color, label)],
        title=title,
        plot_every_n=plot_every_n,
        heading_tick_m=heading_tick_m,
    )
