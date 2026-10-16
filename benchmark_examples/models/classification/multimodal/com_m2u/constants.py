"""CoM-M2U experiment constants (env-configurable, no hardcoded host paths)."""

from __future__ import annotations

import glob
import os
from pathlib import Path

# Full project root: video/, training/, skeleton/ trees (not in anonymous zip).
ROOT = Path(os.environ.get("BADMINTONSKILL23_PROJECT_ROOT", "example/project"))

# Benchmark package roots (vendored in additional).
_BENCH = Path(__file__).resolve().parents[3]
MMACTION2_ROOT = Path(os.environ.get("MMACTION2_ROOT", str(_BENCH / "models/classification/rgb/mmaction2")))
PYSKL_ROOT = Path(os.environ.get("PYSKL_ROOT", str(_BENCH / "models/classification/skeleton/pyskl")))
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

OUT_ROOT = Path(os.environ.get("BADMINTONSKILL23_COM_M2U_OUT", "example/work/com_m2u"))

NUM_CLASSES = 11
PROTOCOLS = ("xsub", "xacq")

VIDEO_TAG = "r2plus1d_badminton_r34_kinetics"
IMU_MODEL = "ispl_inception"
IMU_TAG = "ispl_inception_full_w256"
SKEL_MODEL = "protogcn"
SKEL_STREAMS = ("j", "b", "jm", "bm")

BASELINE_ACC = {
    "xsub": {"rgb": 0.5596, "imu": 0.5827, "skeleton": 0.6784},
    "xacq": {"rgb": 0.5963, "imu": 0.6153, "skeleton": 0.7766},
}

DEFAULT_T = 4.0
DEFAULT_LAMBDA = 0.5
DEFAULT_GAMMA = 1.0

MMACTION_PYTHON = os.environ.get("MMACTION_PYTHON", "python")
PYSKL_PYTHON = os.environ.get("PYSKL_PYTHON", "python")


def video_workdir(protocol: str) -> Path:
    return ROOT / "video" / "outputs" / protocol / VIDEO_TAG


def video_ckpt(protocol: str) -> Path:
    work = video_workdir(protocol)
    cands = sorted(work.glob("best_acc*.pth"))
    if not cands:
        raise FileNotFoundError(work)
    return cands[-1]


def video_manifest(protocol: str) -> Path:
    return ROOT / "video" / "annotations" / protocol / "manifest.json"


def video_ann_frames(protocol: str, split: str) -> Path:
    return ROOT / "video" / "annotations" / protocol / f"{split}_frames.txt"


def imu_ckpt_dir(protocol: str) -> Path:
    pat = str(ROOT / "training" / "outputs" / f"{protocol}_{IMU_MODEL}_full_w256_*")
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(pat)
    return Path(dirs[-1])


def imu_cache(protocol: str) -> Path:
    return ROOT / "training" / "cache" / f"{protocol}_full_w256.npz"


def skel_workdir(protocol: str, stream: str) -> Path:
    name = SKEL_MODEL if stream == "j" else f"{SKEL_MODEL}_{stream}"
    return ROOT / "skeleton" / "outputs" / protocol / name


def skel_cfg(protocol: str, stream: str) -> Path:
    name = SKEL_MODEL if stream == "j" else f"{SKEL_MODEL}_{stream}"
    return ROOT / "skeleton" / "configs" / "pyskl" / protocol / f"{name}.py"


def skel_ckpt(protocol: str, stream: str) -> Path:
    work = skel_workdir(protocol, stream)
    cands = sorted(work.glob("best_top1_acc*.pth"))
    if not cands:
        raise FileNotFoundError(work)
    return cands[-1]


def logits_dir(protocol: str) -> Path:
    return OUT_ROOT / protocol / "logits"
