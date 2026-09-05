"""Locate and import from the ebim-benchmark checkout.

The benchmark is not pip-installable (its own convention: scripts reach
each other via sys.path.insert). We locate the checkout and path-load the
few import-safe modules we rely on. Resolution order:

1. ``EBIM_BENCHMARK_ROOT`` environment variable
2. ``ebim-benchmark`` sibling directory of this repository

Everything here is stdlib-only so it imports anywhere (GB10, H100, CI).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_SENTINEL = Path("task2_isaacsim") / "config" / "topics.yaml"


class BenchmarkNotFound(RuntimeError):
    pass


def find_benchmark_root(required: bool = True) -> Path | None:
    """Resolve the ebim-benchmark checkout, or None (only when not required)."""
    candidates = []
    env = os.environ.get("EBIM_BENCHMARK_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(REPO_ROOT.parent / "ebim-benchmark")

    for root in candidates:
        if (root / _SENTINEL).is_file():
            return root.resolve()

    if not required:
        return None
    tried = "\n  ".join(str(c) for c in candidates)
    raise BenchmarkNotFound(
        "ebim-benchmark checkout not found (looked for "
        f"{_SENTINEL} under):\n  {tried}\n"
        "Clone it as a sibling of this repository or set EBIM_BENCHMARK_ROOT."
    )


def load_module(relpath: str, module_name: str):
    """Import a benchmark file by path (e.g. 'task2_isaacsim/scripts/topics.py')."""
    root = find_benchmark_root()
    path = root / relpath
    if not path.is_file():
        raise BenchmarkNotFound(f"benchmark module missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_topics() -> dict:
    """The benchmark's live task2 topic contract (config/topics.yaml)."""
    topics_mod = load_module("task2_isaacsim/scripts/topics.py", "ebim_topics")
    return topics_mod.load_topics()
