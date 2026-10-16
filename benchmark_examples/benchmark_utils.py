"""Shared helpers for BadmintonSkill-23 additional benchmark code."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Validation holdouts (shared with full benchmark codebase).
XSUB_VAL_SUBJECTS = {"S003", "S010"}
XACQ_VAL_DATES = {"1029"}

IMU_FEATURE_COLUMNS = [
    "waist_acc_x", "waist_acc_y", "waist_acc_z",
    "waist_gyr_x", "waist_gyr_y", "waist_gyr_z",
    "palm_acc_x", "palm_acc_y", "palm_acc_z",
    "palm_gyr_x", "palm_gyr_y", "palm_gyr_z",
    "upper_arm_acc_x", "upper_arm_acc_y", "upper_arm_acc_z",
    "upper_arm_gyr_x", "upper_arm_gyr_y", "upper_arm_gyr_z",
    "forearm_acc_x", "forearm_acc_y", "forearm_acc_z",
    "forearm_gyr_x", "forearm_gyr_y", "forearm_gyr_z",
    "back_acc_x", "back_acc_y", "back_acc_z",
    "back_gyr_x", "back_gyr_y", "back_gyr_z",
]
NUM_CLASSES = 11


def get_dataset_root() -> Path:
    """Return BadmintonSkill23_Release root (parent of benchmark/)."""
    return Path(__file__).resolve().parent.parent


def load_action_labels(root: Optional[Path] = None) -> Tuple[Dict[str, int], List[str]]:
    root = root or get_dataset_root()
    path = root / "metadata" / "action_labels.csv"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = row["action_code"]
            if code and code != "AXX":
                rows.append((int(row["action_id"]), row["action_class"], code))
    rows.sort(key=lambda x: x[0])
    code_to_id = {code: aid for aid, _, code in rows}
    class_names = [name for _, name, _ in rows]
    return code_to_id, class_names


def load_split(protocol: str, dataset_root: Optional[Path] = None) -> dict:
    """Load XSUB/XACQ split JSON. Uses full-dataset metadata when available."""
    root = dataset_root or get_dataset_root().parent.parent
    split_path = root / "metadata" / "splits" / f"{protocol}_train_test.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Point dataset_root to the full BadmintonSkill-23 repository."
        )
    return json.loads(split_path.read_text(encoding="utf-8"))


def assign_subset(row: dict, protocol: str, split: dict) -> Optional[str]:
    """Map a metadata row to train / val / test (same logic as full benchmark)."""
    if protocol == "xsub":
        sid = row["subject_id"]
        if sid in split["test_subjects"]:
            return "test"
        if sid in XSUB_VAL_SUBJECTS:
            return "val"
        if sid in split["train_subjects"]:
            return "train"
        return None
    if protocol.startswith("xacq"):
        date = row["date"] if "date" in row else row.get("acquisition_id", "")
        if date in split["test_dates"]:
            return "test"
        if date in XACQ_VAL_DATES:
            return "val"
        if date in split["train_dates"]:
            return "train"
        return None
    raise ValueError(f"unknown protocol: {protocol}")


def load_protocol_recording_ids(
    protocol: str,
    subset: str,
    root: Optional[Path] = None,
) -> List[str]:
    """Load recording IDs from additional protocol txt files."""
    root = root or get_dataset_root()
    path = root / "protocols" / f"{protocol}_{subset}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_protocol_sequence_ids(
    protocol: str,
    subset: str,
    root: Optional[Path] = None,
) -> List[str]:
    """Backward-compatible alias for TAD recording ID lists."""
    return load_protocol_recording_ids(protocol, subset, root=root)


def mean_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = NUM_CLASSES) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for gt, pred in zip(y_true, y_pred):
        if 0 <= gt < n_classes and 0 <= pred < n_classes:
            cm[gt, pred] += 1
    per_class = []
    for c in range(n_classes):
        n = cm[c].sum()
        if n:
            per_class.append(cm[c, c] / n)
    return float(np.mean(per_class)) if per_class else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    mca = mean_class_accuracy(y_true, y_pred)
    return {"top1_accuracy": acc, "mean_class_accuracy": mca, "n_samples": int(len(y_true))}
