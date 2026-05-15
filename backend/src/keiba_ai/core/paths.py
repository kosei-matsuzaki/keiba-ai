"""Canonical path resolution for all data directories.

All paths are resolved relative to the repository root so that the package
works regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this file to find the repository root (contains .git)."""
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback: assume cwd is inside the repo
    return Path.cwd()


def data_dir() -> Path:
    """Return the data directory, creating it if necessary."""
    env_val = os.getenv("KEIBA_DATA_DIR", "")
    if env_val:
        d = Path(env_val)
    else:
        d = _repo_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    return data_dir() / "raw"


def raw_path(yyyy: str, mm: str, filename: str) -> Path:
    """Return (and create) the cache path for a dated HTML file."""
    p = raw_dir() / yyyy / mm
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def db_path() -> Path:
    return data_dir() / "keiba.db"
