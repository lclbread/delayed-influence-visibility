"""Shared path helpers for the released analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA_FILES = {"data_by_artist.csv", "influence_data.csv", "full_music_data.csv"}


def _has_required_data(path):
    path = Path(path)
    return path.exists() and all((path / name).exists() for name in REQUIRED_DATA_FILES)


def _first_data_dir(candidates):
    for candidate in candidates:
        if candidate and _has_required_data(candidate):
            return Path(candidate).resolve()
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    return RELEASE_ROOT / "data"


DATA_DIR = _first_data_dir(
    [
        os.environ.get("MUSIC_DATA_DIR"),
        RELEASE_ROOT / "data",
        RELEASE_ROOT.parent / "data",
        Path.cwd() / "data",
    ]
)

FIGURE_DIR = Path(os.environ.get("MUSIC_OUTPUT_DIR", RELEASE_ROOT / "outputs" / "figures")).resolve()
