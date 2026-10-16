"""Example path placeholders for BadmintonSkill-23 benchmark code.

Copy or symlink your local dataset tree to match these paths, or override
via environment variables before training/evaluation.

No absolute host paths appear in benchmark configs — only these examples.
"""

from __future__ import annotations

import os
from pathlib import Path

# Root of the released dataset (RGB + skeleton + IMU + annotations).
DATASET_ROOT = Path(os.environ.get("BADMINTONSKILL23_ROOT", "example/BadmintonSkill23"))

# --- Classification (trimmed clips) ---
RGB_FRAMES_ROOT = DATASET_ROOT / "rgb" / "frames"
RGB_ANNOTATIONS = DATASET_ROOT / "annotations" / "rgb"
SKELETON_ANNOTATIONS = DATASET_ROOT / "annotations" / "skeleton"
IMU_ROOT = DATASET_ROOT / "imu"
CLASSIFICATION_MANIFEST = DATASET_ROOT / "metadata" / "instances.csv"

# Example protocol files (XSUB / XACQ).
PROTOCOLS_ROOT = DATASET_ROOT / "protocols"

# Pretrained checkpoints (download separately).
CHECKPOINTS_ROOT = DATASET_ROOT / "checkpoints"

# --- TAD (untrimmed continuous sequences) ---
TAD_ROOT = DATASET_ROOT / "tad"
TAD_ANNOTATIONS = TAD_ROOT / "annotations"
TAD_FEATURES = TAD_ROOT / "features"
TAD_EXPERIMENTS = Path(os.environ.get("BADMINTONSKILL23_EXP_ROOT", "example/experiments/tad"))

# --- Work directories (training outputs) ---
WORK_ROOT = Path(os.environ.get("BADMINTONSKILL23_WORK_ROOT", "example/work"))


def rgb_ann(protocol: str, split: str) -> Path:
    """e.g. example/BadmintonSkill23/annotations/rgb/xsub/train_frames.txt"""
    return RGB_ANNOTATIONS / protocol / f"{split}_frames.txt"


def skeleton_ann(protocol: str) -> Path:
    """e.g. example/BadmintonSkill23/annotations/skeleton/xsub/badmintonskill23.pkl"""
    return SKELETON_ANNOTATIONS / protocol / "badmintonskill23.pkl"


def tad_annotation(protocol: str) -> Path:
    return TAD_ANNOTATIONS / protocol / "anno.json"


def tad_imu_annotation(protocol: str) -> Path:
    return TAD_ANNOTATIONS / "imu" / protocol / "anno.json"


def tad_feature(*parts: str) -> Path:
    return TAD_FEATURES.joinpath(*parts)


def rgb_work(protocol: str, model: str) -> Path:
    return WORK_ROOT / "rgb" / protocol / model


def skeleton_work(protocol: str, model: str) -> Path:
    return WORK_ROOT / "skeleton" / protocol / model


def tad_work(*parts: str) -> Path:
    return TAD_EXPERIMENTS.joinpath(*parts)
