"""Path adapter for the release layout.

Set environment variable BADMINTONSKILL23_ROOT to the dataset root, e.g.:
  export BADMINTONSKILL23_ROOT=/path/to/BadmintonSkill-23_data
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("BADMINTONSKILL23_ROOT", ".")).resolve()

PATHS = {
    "metadata": ROOT / "Metadata",
    "annotations": ROOT / "Annotations",
    "splits": ROOT / "Splits",
    "rgb": ROOT / "RGB",
    "imu": ROOT / "IMU",
    "skeleton": ROOT / "Skeleton",
    "features_tad": ROOT / "Features_TAD",
}


def p(key: str) -> Path:
    return PATHS[key]
