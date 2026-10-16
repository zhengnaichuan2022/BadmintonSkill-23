import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..builder import PIPELINES

_DATASET_ROOT = Path(__file__).resolve().parents[3].parent

# PN3 Pro five-node filenames (waist, back, upper_arm, forearm, palm)
DEFAULT_SENSOR_FILES = (
    "Sensor01_ID1.csv",  # Waist
    "Sensor02_ID2.csv",  # Palm
    "Sensor03_ID3.csv",  # Upper arm
    "Sensor04_ID4.csv",  # Forearm
    "Sensor06_ID6.csv",  # Back
)
DEFAULT_IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]


def _read_csv_segment(csv_path: Path, columns: list, start: int, end: int, scale: float) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=columns, dtype=np.float32)
    data = df.to_numpy(dtype=np.float32)
    if scale != 1.0:
        data = data * scale
    end = min(end, data.shape[0])
    start = max(0, min(start, end))
    return data[start:end]


def _read_multinode_segment(
    sensor06_path: Path,
    columns: list,
    start: int,
    end: int,
    scale: float,
    sensor_files: tuple,
) -> np.ndarray:
    """Load sibling sensor CSVs and concat → (T, 6*n_sensors). Missing nodes zero-filled."""
    parent = sensor06_path.parent
    length = max(0, end - start)
    parts = []
    for fname in sensor_files:
        path = parent / fname
        if path.is_file():
            seg = _read_csv_segment(path, columns, start, end, scale)
        else:
            seg = np.zeros((0, len(columns)), dtype=np.float32)
        if seg.shape[0] < length:
            pad = np.zeros((length - seg.shape[0], len(columns)), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=0) if seg.shape[0] else pad
        elif seg.shape[0] > length:
            seg = seg[:length]
        parts.append(seg.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1)


@PIPELINES.register_module()
class LoadIMURaw:
    """Load a window of raw IMU samples from session CSV.

    Args:
        columns: IMU column names (acc/gyro xyz).
        normalize: ``"window"`` for per-window z-score, or ``None``.
        dataset_root: dataset root (default: parent of OpenTAD-main).
        multinode: if True, load all sensors in ``sensor_files`` next to the
            annotated Sensor06 CSV and concatenate channels (default 30-ch).
        sensor_files: filenames for multinode mode.
    """

    def __init__(
        self,
        columns=None,
        normalize: str = "window",
        dataset_root: str | None = None,
        multinode: bool = False,
        sensor_files=None,
    ):
        self.columns = list(columns or DEFAULT_IMU_COLUMNS)
        self.normalize = normalize
        self.dataset_root = Path(dataset_root) if dataset_root else _DATASET_ROOT
        self.multinode = bool(multinode)
        self.sensor_files = tuple(sensor_files or DEFAULT_SENSOR_FILES)

    def __call__(self, results):
        imu_csv = results.get("imu_csv")
        if not imu_csv:
            raise KeyError("imu_csv missing in annotation entry")

        csv_path = self.dataset_root / imu_csv
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        scale = float(results.get("sensor_scale_factor", 1.0))
        snippet_stride = int(results["snippet_stride"])
        start = int(results["window_start_frame"])
        end = int(results["feature_end_idx"] * snippet_stride + snippet_stride)

        if self.multinode:
            # Temporal extent from Sensor06 (annotation key); siblings zero-padded if shorter.
            df0 = pd.read_csv(csv_path, usecols=self.columns, dtype=np.float32)
            end = min(end, len(df0))
            start = max(0, min(start, end))
            window = _read_multinode_segment(
                csv_path, self.columns, start, end, scale, self.sensor_files
            )
        else:
            df = pd.read_csv(csv_path, usecols=self.columns, dtype=np.float32)
            data = df.to_numpy(dtype=np.float32)
            if scale != 1.0:
                data = data * scale
            end = min(end, data.shape[0])
            start = max(0, min(start, end))
            window = data[start:end]

        if self.normalize == "window" and window.shape[0] > 1:
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            window = (window - mean) / std

        results["feats"] = torch.from_numpy(window.astype(np.float32, copy=False))
        results["raw_start"] = start
        results["raw_end"] = end
        return results


@PIPELINES.register_module()
class IMUSlidingWindowTrunc:
    """Pad/truncate raw IMU window to fixed raw length for batching."""

    def __init__(self, with_mask: bool = True):
        self.with_mask = with_mask

    def __call__(self, results):
        assert "window_size" in results
        snippet_stride = int(results["snippet_stride"])
        window_size = int(results["window_size"])
        target_len = window_size * snippet_stride

        feats = results["feats"]
        valid_len = feats.shape[0]

        if valid_len < target_len:
            pad = torch.zeros(target_len - valid_len, feats.shape[1], dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=0)
        elif valid_len > target_len:
            feats = feats[:target_len]
            valid_len = target_len

        if self.with_mask:
            if valid_len < target_len:
                masks = torch.cat([torch.ones(valid_len), torch.zeros(target_len - valid_len)])
            else:
                masks = torch.ones(target_len)
            results["masks"] = masks.bool()

        results["feats"] = feats.float()
        return results
