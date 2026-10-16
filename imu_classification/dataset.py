""" metadata/instances.csv  IMU ， train/val/test。

 5  →  shape (N, T, 30)。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    ACTION_LABELS_CSV,
    CACHE_DIR,
    CACHE_TAG,
    DATASET_ROOT,
    DEFAULT_WIN_SIZE,
    DEFAULT_WINDOW_MODE,
    IMU_CHANNELS,
    INSTANCES_CSV,
    MAX_WIN_SIZE,
    N_CHANNELS,
    SENSOR_FILES,
    SPLIT_FILES,
    XACQ_VAL_DATES,
    XSUB_VAL_SUBJECTS,
)


@dataclass
class SplitArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    class_names: List[str]
    meta: dict


def load_action_labels() -> Tuple[Dict[str, int], List[str]]:
    rows = []
    with ACTION_LABELS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["action_code"]
            if code and code != "AXX":
                rows.append((int(row["action_id"]), row["cn_name"], code))
    rows.sort(key=lambda x: x[0])
    code_to_id = {code: aid for aid, _, code in rows}
    class_names = [name for _, name, _ in rows]
    return code_to_id, class_names


def load_split(protocol: str) -> dict:
    path = SPLIT_FILES[protocol]
    return json.loads(path.read_text(encoding="utf-8"))


def read_imu_segment(
    csv_path: Path,
    start: int,
    end: int,
    file_cache: Optional[dict] = None,
) -> np.ndarray:
    """Read one sensor CSV → (T, 6)."""
    import pandas as pd

    key = str(csv_path)
    if file_cache is not None:
        df = file_cache.get(key)
        if df is None:
            df = pd.read_csv(csv_path)
            file_cache[key] = df
    else:
        df = pd.read_csv(csv_path)
    seg = df.iloc[start:end][IMU_CHANNELS].to_numpy(dtype=np.float32)
    return seg


def read_multinode_segment(
    sensor06_path: Path,
    start: int,
    end: int,
    file_cache: Optional[dict] = None,
) -> np.ndarray:
    """Load all 5 sensor CSVs at the same [start:end) and concat → (T, 30).

    Missing sensors are zero-filled. Length is taken from the longest available
    node in-range (typically Sensor06); shorter nodes are end-padded with zeros.
    """
    parent = sensor06_path.parent
    length = max(0, end - start)
    parts: List[np.ndarray] = []
    for fname in SENSOR_FILES:
        path = parent / fname
        if path.exists():
            seg = read_imu_segment(path, start, end, file_cache)
        else:
            seg = np.zeros((0, len(IMU_CHANNELS)), dtype=np.float32)
        if seg.shape[0] < length:
            pad = np.zeros((length - seg.shape[0], len(IMU_CHANNELS)), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0) if seg.shape[0] else pad
        elif seg.shape[0] > length:
            seg = seg[:length]
        parts.append(seg.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1)  # (T, 30)


def fix_length(x: np.ndarray, win_size: int) -> np.ndarray:
    """Legacy center crop / center pad (128 )。"""
    return resize_segment(x, win_size, mode="center_crop")


def resize_segment(x: np.ndarray, win_size: int, mode: str = "full") -> np.ndarray:
    """Resize IMU segment to fixed length.

    full:  [start,end)，，（）
    center_crop: ，/
    """
    n, c = x.shape
    if mode == "center_crop":
        if n == win_size:
            return x
        if n > win_size:
            start = (n - win_size) // 2
            return x[start : start + win_size]
        out = np.zeros((win_size, c), dtype=np.float32)
        pad_left = (win_size - n) // 2
        out[pad_left : pad_left + n] = x
        return out
    if mode != "full":
        raise ValueError(f"unknown window mode: {mode}")
    if n == win_size:
        return x
    if n > win_size:
        return x[:win_size]
    out = np.zeros((win_size, c), dtype=np.float32)
    out[:n] = x
    return out


def resolve_win_size(
    lengths: List[int],
    win_size: int,
    window_mode: str = DEFAULT_WINDOW_MODE,
) -> int:
    if win_size > 0:
        return win_size
    if not lengths:
        return DEFAULT_WIN_SIZE
    if window_mode == "full":
        target = int(np.ceil(np.percentile(lengths, 99)))
        target = max(target, DEFAULT_WIN_SIZE)
        return min(target, MAX_WIN_SIZE)
    return DEFAULT_WIN_SIZE


def load_instances(
    require_valid_imu: bool = True,
    only_valid_class: bool = True,
) -> List[dict]:
    rows = list(csv.DictReader(INSTANCES_CSV.open(encoding="utf-8")))
    out = []
    for row in rows:
        if only_valid_class and row["is_valid_class"] != "True":
            continue
        if require_valid_imu and row["has_valid_imu"] != "True":
            continue
        out.append(row)
    return out


def assign_subset(row: dict, protocol: str, split: dict) -> Optional[str]:
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
        date = row["date"]
        if date in split["test_dates"]:
            return "test"
        if date in XACQ_VAL_DATES:
            return "val"
        if date in split["train_dates"]:
            return "train"
        return None
    raise ValueError(f"unknown protocol: {protocol}")


def _cache_path(protocol: str, window_mode: str, win_size: int) -> Path:
    wtag = win_size if win_size > 0 else "auto"
    return CACHE_DIR / f"{protocol}_{window_mode}_w{wtag}_{CACHE_TAG}.npz"


def build_arrays(
    protocol: str,
    win_size: int = DEFAULT_WIN_SIZE,
    window_mode: str = DEFAULT_WINDOW_MODE,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
) -> SplitArrays:
    cache_path = _cache_path(protocol, window_mode, win_size)
    if use_cache and max_samples is None and cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(npz["meta"]))
        return SplitArrays(
            X_train=npz["X_train"],
            y_train=npz["y_train"],
            X_val=npz["X_val"],
            y_val=npz["y_val"],
            X_test=npz["X_test"],
            y_test=npz["y_test"],
            n_classes=int(npz["n_classes"]),
            class_names=list(npz["class_names"]),
            meta=meta,
        )

    split = load_split(protocol)
    code_to_id, class_names = load_action_labels()

    buckets = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    skipped = {"missing_file": 0, "too_short": 0, "unassigned": 0, "truncated_tail": 0}
    file_cache: dict = {}
    pending: List[Tuple[str, int, np.ndarray]] = []
    raw_lengths: List[int] = []

    for row in load_instances():
        subset = assign_subset(row, protocol, split)
        if subset is None:
            skipped["unassigned"] += 1
            continue

        csv_path = DATASET_ROOT / row["sensor_file"]
        if not csv_path.exists():
            skipped["missing_file"] += 1
            continue

        start = int(float(row["imu_start_sample"]))
        end = int(float(row["imu_end_sample"]))
        if end - start < 4:
            skipped["too_short"] += 1
            continue

        action_id = code_to_id.get(row["action_code"])
        if action_id is None:
            continue

        raw = read_multinode_segment(csv_path, start, end, file_cache)
        raw_lengths.append(len(raw))
        pending.append((subset, action_id, raw))

        if max_samples is not None and len(pending) >= max_samples:
            break

    resolved_win = resolve_win_size(raw_lengths, win_size, window_mode=window_mode)
    if win_size <= 0:
        cache_path = _cache_path(protocol, window_mode, resolved_win)
        if use_cache and max_samples is None and cache_path.exists():
            npz = np.load(cache_path, allow_pickle=True)
            meta = json.loads(str(npz["meta"]))
            return SplitArrays(
                X_train=npz["X_train"],
                y_train=npz["y_train"],
                X_val=npz["X_val"],
                y_val=npz["y_val"],
                X_test=npz["X_test"],
                y_test=npz["y_test"],
                n_classes=int(npz["n_classes"]),
                class_names=list(npz["class_names"]),
                meta=meta,
            )

    for subset, action_id, raw in pending:
        if len(raw) > resolved_win:
            skipped["truncated_tail"] += 1
        seg = resize_segment(raw, resolved_win, mode=window_mode)
        buckets[subset][0].append(seg)
        buckets[subset][1].append(action_id)

    def stack(name: str) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = buckets[name]
        if not xs:
            return (
                np.zeros((0, resolved_win, N_CHANNELS), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.stack(xs), np.array(ys, dtype=np.int64)

    X_train, y_train = stack("train")
    X_val, y_val = stack("val")
    X_test, y_test = stack("test")
    n_classes = len(class_names)

    meta = {
        "protocol": protocol,
        "win_size": resolved_win,
        "window_mode": window_mode,
        "n_sensors": len(SENSOR_FILES),
        "n_channels": N_CHANNELS,
        "sensor_files": list(SENSOR_FILES),
        "segment_length": {
            "min": int(min(raw_lengths)) if raw_lengths else 0,
            "median": float(np.median(raw_lengths)) if raw_lengths else 0,
            "p99": float(np.percentile(raw_lengths, 99)) if raw_lengths else 0,
            "max": int(max(raw_lengths)) if raw_lengths else 0,
        },
        "skipped": skipped,
        "shapes": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape),
        },
    }
    result = SplitArrays(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_classes=n_classes,
        class_names=class_names,
        meta=meta,
    )

    if use_cache and max_samples is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            n_classes=n_classes,
            class_names=np.array(class_names, dtype=object),
            meta=json.dumps(meta, ensure_ascii=False),
        )
    return result


def normalize_train_stats(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def apply(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        shape = (1, 1, mean.shape[-1])
        return (x - mean.reshape(shape)) / std.reshape(shape)

    return apply(X_train), apply(X_val), apply(X_test), mean.squeeze(), std.squeeze()


def to_categorical(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out
