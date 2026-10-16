"""Load five-node IMU CSV from a additional sample directory."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from benchmark_utils import IMU_FEATURE_COLUMNS, get_dataset_root


def load_imu_sample(sample_dir: Path | str) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """Load synchronized 5-node IMU from imu.csv.

    Returns
    -------
    imu_array : np.ndarray, shape (T, 30) — 5 nodes × 6 channels (acc+gyr)
    df        : full pandas DataFrame (includes timestamp)
    meta      : dict with sample_rate_hz estimate, columns, frame_count
    """
    sample_dir = Path(sample_dir)
    imu_path = sample_dir / "imu.csv"
    if not imu_path.is_file():
        raise FileNotFoundError(f"Missing imu.csv in {sample_dir}")

    df = pd.read_csv(imu_path)
    missing = [c for c in IMU_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"imu.csv missing columns: {missing[:5]}...")

    imu_array = df[IMU_FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    if "timestamp" in df.columns and len(df) > 1:
        dt = float(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
        sample_rate = round(1.0 / dt) if dt > 0 else 120.0
    else:
        sample_rate = 120.0

    meta = {
        "path": str(imu_path),
        "columns": IMU_FEATURE_COLUMNS,
        "n_nodes": 5,
        "n_channels_per_node": 6,
        "frame_count": int(len(df)),
        "sample_rate_hz": sample_rate,
    }
    return imu_array, df, meta


if __name__ == "__main__":
    root = get_dataset_root()
    sample = root / "samples/classification_previews/A01_forehand_net"
    imu, df, meta = load_imu_sample(sample)
    print(f"Loaded IMU {imu.shape} from {sample.name}, rate={meta['sample_rate_hz']} Hz")
