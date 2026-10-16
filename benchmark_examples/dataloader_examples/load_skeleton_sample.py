"""Load skeleton array from a additional sample directory."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from benchmark_utils import get_dataset_root

SKELETON_JOINTS = 17
SKELETON_DIMS = 3  # x, y, confidence


def load_skeleton_sample(sample_dir: Path | str) -> Tuple[np.ndarray, dict]:
    """Load COCO-17 skeleton from skeleton.npy.

    Returns
    -------
    skeleton : np.ndarray, shape (T, 17, 3)
    meta     : dict with joints, dims, frame_count
    """
    sample_dir = Path(sample_dir)
    skel_path = sample_dir / "skeleton.npy"
    if not skel_path.is_file():
        raise FileNotFoundError(f"Missing skeleton.npy in {sample_dir}")

    skeleton = np.load(skel_path)
    if skeleton.ndim != 3 or skeleton.shape[1] != SKELETON_JOINTS or skeleton.shape[2] != SKELETON_DIMS:
        raise ValueError(
            f"Expected skeleton shape (T, {SKELETON_JOINTS}, {SKELETON_DIMS}), got {skeleton.shape}"
        )

    meta = {
        "path": str(skel_path),
        "joints": SKELETON_JOINTS,
        "dims": SKELETON_DIMS,
        "frame_count": int(skeleton.shape[0]),
        "dtype": str(skeleton.dtype),
    }
    return skeleton, meta


if __name__ == "__main__":
    root = get_dataset_root()
    sample = root / "samples/classification_previews/A01_forehand_net"
    skeleton, meta = load_skeleton_sample(sample)
    print(f"Loaded skeleton {skeleton.shape} from {sample.name}, meta={meta}")
