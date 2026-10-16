"""CoM-M2U experiment constants: best unimodal backbones + baselines."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("${BADMINTONSKILL23_ROOT}")
OUT_ROOT = ROOT / "multimodal" / "outputs" / "com_m2u"
NUM_CLASSES = 11
PROTOCOLS = ("xsub", "xacq")

# Best backbones
VIDEO_TAG = "r2plus1d_badminton_r34_kinetics"
IMU_MODEL = "ispl_inception"
IMU_TAG = "ispl_inception_full_w256"
SKEL_MODEL = "protogcn"
SKEL_STREAMS = ("j", "b", "jm", "bm")

# Recorded CE baselines (test Acc)
BASELINE_ACC = {
    "xsub": {
        "rgb": 0.5596,
        "imu": 0.5827,
        "skeleton": 0.6784,  # protogcn 4-stream
    },
    "xacq": {
        "rgb": 0.5963,
        "imu": 0.6153,
        "skeleton": 0.7766,  # protogcn 4-stream
    },
}

# Default KD hyperparams (.md)
DEFAULT_T = 4.0
DEFAULT_LAMBDA = 0.5
DEFAULT_GAMMA = 1.0

MMACTION_PYTHON = "/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python"
MMACTION2_ROOT = "/home/ubuntu/users/dyc/code/mmaction2"
PYSKL_PYTHON = "/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python"
PYSKL_ROOT = ROOT / "skeleton" / "pyskl-main" / "pyskl-main"


def video_workdir(protocol: str) -> Path:
    return ROOT / "video" / "outputs" / protocol / VIDEO_TAG


def video_ckpt(protocol: str) -> Path:
    work = video_workdir(protocol)
    cands = sorted(work.glob("best_acc*.pth"))
    if not cands:
        raise FileNotFoundError(work)
    return cands[-1]


def imu_ckpt_dir(protocol: str) -> Path:
    import glob
    import os

    pat = str(ROOT / "training" / "outputs" / f"{protocol}_{IMU_MODEL}_full_w256_*")
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(pat)
    return Path(dirs[-1])


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
