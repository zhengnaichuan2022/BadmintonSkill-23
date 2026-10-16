"""Load RGB video from a additional sample directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required: pip install opencv-python") from exc

from benchmark_utils import get_dataset_root


def load_rgb_sample(
    sample_dir: Path | str,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Load RGB frames from rgb.mp4.

    Returns
    -------
    frames : np.ndarray, shape (T, H, W, 3), dtype uint8, RGB order
    meta   : dict with fps, frame_count, width, height
    """
    sample_dir = Path(sample_dir)
    video_path = sample_dir / "rgb.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing rgb.mp4 in {sample_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    meta = {
        "path": str(video_path),
        "fps": float(fps),
        "frame_count": len(frames),
        "width": width,
        "height": height,
    }
    return np.stack(frames, axis=0), meta


if __name__ == "__main__":
    root = get_dataset_root()
    sample = root / "samples/classification_previews/A01_forehand_net"
    frames, meta = load_rgb_sample(sample, max_frames=10)
    print(f"Loaded {frames.shape} from {sample.name}, meta={meta}")
