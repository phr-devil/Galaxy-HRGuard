"""Filesystem layout for the reproduction package.

Every location is derived from the repository root or from an environment
variable, so no absolute path from the machine that produced the frozen
results survives in the code.

Environment variables (all optional)
------------------------------------
HRGUARD_DATA     raw cohort folders, one directory per participant (P01..P24)
HRGUARD_SEALED   held-out cohort folders, only for analysts with access
HRGUARD_OUTPUT   directory for regenerated artefacts (default: outputs/)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Directory holding the licensed raw cohort folders (P01..P24)."""
    env = os.environ.get("HRGUARD_DATA")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data" / "raw"


def sealed_root() -> Path:
    """Directory holding the held-out cohort folders.

    Empty by default: the held-out files are not redistributed here and are
    only present for analysts who were granted access to them.
    """
    env = os.environ.get("HRGUARD_SEALED")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data" / "sealed"


def output_dir(sub: str | None = None) -> Path:
    """Directory for regenerated artefacts; created on demand."""
    env = os.environ.get("HRGUARD_OUTPUT")
    base = Path(env).expanduser().resolve() if env else REPO_ROOT / "outputs"
    path = base / sub if sub else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def frozen_dir() -> Path:
    """Committed aggregate artefacts produced by the frozen analysis."""
    return REPO_ROOT / "evidence"
