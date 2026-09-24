"""Benchmark data location. Stdlib only.

    SPEAKER_BENCH_DIR   datasets, manifests, caches and runs
                        default ${XDG_DATA_HOME:-~/.local/share}/meeting-documenter/bench-data
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "meeting-documenter"


def bench_dir() -> Path:
    raw = os.environ.get("SPEAKER_BENCH_DIR")
    if raw:
        return Path(raw).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / APP / "bench-data"


def dataset_dir(name: str) -> Path:
    return bench_dir() / name


def manifest_path(name: str) -> Path:
    return bench_dir() / f"manifest-{name}.json"
