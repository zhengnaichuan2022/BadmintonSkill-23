#!/usr/bin/env python3
"""Check MMAction2 training environment before run_train.sh."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MMACTION2_ROOT = Path(os.environ.get("MMACTION2_ROOT", "/home/ubuntu/users/dyc/code/mmaction2"))
VIDEO_DIR = Path(__file__).resolve().parent

DEFAULT_PYTHON = Path("/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python")


def main() -> int:
    py = Path(sys.executable)
    print(f"python: {py}")
    print(f"PYTHONPATH should include:\n  {MMACTION2_ROOT}\n  {VIDEO_DIR}")

    missing = []
    for mod in ("mmengine", "mmaction", "mmcv", "torch", "cv2", "decord"):
        try:
            __import__(mod)
            print(f"  ok  {mod}")
        except ImportError:
            print(f"  MISSING  {mod}")
            missing.append(mod)

    if missing:
        print("\nEnvironment incomplete.")
        if DEFAULT_PYTHON.exists():
            print(f"Try:\n  export MMACTION_PYTHON={DEFAULT_PYTHON}")
        return 1

    os.environ["PYTHONPATH"] = f"{MMACTION2_ROOT}:{VIDEO_DIR}:{os.environ.get('PYTHONPATH', '')}"
    sys.path.insert(0, str(VIDEO_DIR))
    from mmaction_ext.beiyu_dataset import BeiyuVideoDataset  # noqa: F401
    from mmaction_ext.transforms import BeiyuApplyFrameRange  # noqa: F401

    print("\nEnvironment OK for Beiyu video training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
