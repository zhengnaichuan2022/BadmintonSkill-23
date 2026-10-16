#!/usr/bin/env python3
""" IMU / Video test  test_scores.pkl（）。

Skeleton  result.pkl，。

Video （ video/run_benchmark_*.sh ）:
  export MMACTION_PYTHON=/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python
  export MMACTION2_ROOT=/home/ubuntu/users/dyc/code/mmaction2
  export PYTHONPATH=$MMACTION2_ROOT:${BADMINTONSKILL23_ROOT}/video:$PYTHONPATH
  cd $MMACTION2_ROOT && $MMACTION_PYTHON tools/test.py <config> <ckpt> --dump ...

Usage:
  python3 export_test_scores.py --protocol xsub --modality imu
  CUDA_VISIBLE_DEVICES=0 python3 export_test_scores.py --protocol xsub --modality video
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "video"
sys.path.insert(0, str(ROOT))

from fusion_utils import eval_scores, save_scores, scores_path

IMU_MODEL = "ispl_inception"
VIDEO_MODEL = "r2plus1d_badminton_r34"
VIDEO_SUFFIX = "_kinetics"
MMACTION_ROOT = Path(os.environ.get("MMACTION2_ROOT", "/home/ubuntu/users/dyc/code/mmaction2"))
MMACTION_PYTHON = Path(
    os.environ.get("MMACTION_PYTHON", "/home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python")
)


def latest_imu_dir(protocol: str, model: str) -> Path:
    pat = str(ROOT / "training/outputs" / f"{protocol}_{model}_full_w256_*")
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    if not dirs:
        raise FileNotFoundError(f"no IMU output: {pat}")
    return Path(dirs[-1])


def export_imu(protocol: str) -> Path:
    import tensorflow as tf

    out_dir = latest_imu_dir(protocol, IMU_MODEL)
    cache = np.load(ROOT / "training/cache" / f"{protocol}_full_w256.npz", allow_pickle=True)
    X_test = cache["X_test"].astype(np.float32)
    y_test = cache["y_test"].astype(np.int64)

    mean = np.load(out_dir / "norm_mean.npy")
    std = np.load(out_dir / "norm_std.npy")
    Xn = (X_test - mean) / std

    model_path = out_dir / "best.keras"
    if not model_path.is_file():
        model_path = out_dir / "final.keras"
    model = tf.keras.models.load_model(str(model_path), compile=False)
    probs = model.predict(Xn, batch_size=64, verbose=0)
    acc, mca = eval_scores(probs, y_test)

    path = scores_path("imu", protocol, f"{IMU_MODEL}_full_w256")
    save_scores(
        path,
        probs.astype(np.float32),
        y_test,
        {
            "modality": "imu",
            "protocol": protocol,
            "model": IMU_MODEL,
            "source_dir": str(out_dir),
            "accuracy": acc,
            "macro_f1": mca,
        },
    )
    print(f"IMU {protocol}: acc={acc:.4f} mca={mca:.4f} -> {path}")
    return path


def _video_checkpoint(work: Path) -> Path:
    tr = work / "test_results.json"
    if tr.is_file():
        meta = json.loads(tr.read_text(encoding="utf-8"))
        ckpt = Path(meta["checkpoint"])
        if ckpt.is_file():
            return ckpt
    cands = sorted(work.glob("best_acc*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {work}")
    return cands[-1]


def _video_config(work: Path, protocol: str) -> Path:
    """Pick config that matches the successful benchmark (frames_256 for kinetics full)."""
    # Training snapshot under work_dir usually has the correct data_root.
    snaps = sorted(work.glob("*/vis_data/config.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    for cfg in snaps:
        text = cfg.read_text(encoding="utf-8")
        if "frames_256" in text and "frames_roi" not in text:
            return cfg

    top = work / f"{VIDEO_MODEL}.py"
    if top.is_file() and "frames_256" in top.read_text(encoding="utf-8"):
        return top

    fallback = VIDEO_DIR / "configs/mmaction" / protocol / f"{VIDEO_MODEL}.py"
    if not fallback.is_file():
        raise FileNotFoundError(f"no video config for {protocol}/{VIDEO_MODEL}")
    return fallback


def _mmaction_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{MMACTION_ROOT}:{VIDEO_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _parse_dump_results(dump_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse MMAction DumpResults pkl (needs torch) via RTMLib python."""
    out_npz = dump_path.with_suffix(".npz")
    code = f"""
import pickle
import numpy as np
raw = pickle.load(open({str(dump_path)!r}, "rb"))
scores, labels = [], []
for item in raw:
    s = item.pred_score if hasattr(item, "pred_score") else item["pred_score"]
    y = item.gt_label if hasattr(item, "gt_label") else item["gt_label"]
    scores.append(np.asarray(s).reshape(-1))
    labels.append(int(np.asarray(y).reshape(-1)[0]))
np.savez({str(out_npz)!r}, scores=np.stack(scores), labels=np.array(labels, dtype=np.int64))
"""
    subprocess.run(
        [str(MMACTION_PYTHON), "-c", code],
        cwd=str(MMACTION_ROOT),
        env=_mmaction_env(),
        check=True,
    )
    data = np.load(out_npz)
    scores_arr = data["scores"].astype(np.float32)
    labels_arr = data["labels"].astype(np.int64)
    out_npz.unlink(missing_ok=True)
    return scores_arr, labels_arr


def export_video(protocol: str, *, reuse_dump: bool = False) -> Path:
    if not MMACTION_PYTHON.is_file():
        raise FileNotFoundError(f"MMACTION_PYTHON not found: {MMACTION_PYTHON}")
    if not (MMACTION_ROOT / "tools/test.py").is_file():
        raise FileNotFoundError(f"MMACTION2_ROOT invalid: {MMACTION_ROOT}")

    work = ROOT / "video/outputs" / protocol / f"{VIDEO_MODEL}{VIDEO_SUFFIX}"
    ckpt = _video_checkpoint(work)
    cfg_path = _video_config(work, protocol)
    dump_path = work / "_export_test_dump.pkl"

    if not (reuse_dump and dump_path.is_file()):
        cmd = [
            str(MMACTION_PYTHON),
            str(MMACTION_ROOT / "tools/test.py"),
            str(cfg_path),
            str(ckpt),
            "--work-dir",
            str(work),
            "--dump",
            str(dump_path),
        ]
        print(f"Video {protocol}:")
        print(f"  python : {MMACTION_PYTHON}")
        print(f"  mmaction: {MMACTION_ROOT}")
        print(f"  config : {cfg_path}")
        print(f"  ckpt   : {ckpt}")
        subprocess.run(cmd, cwd=str(MMACTION_ROOT), env=_mmaction_env(), check=True)
    else:
        print(f"Video {protocol}: reuse dump {dump_path}")

    scores_arr, labels_arr = _parse_dump_results(dump_path)
    acc, mca = eval_scores(scores_arr, labels_arr)

    ref = None
    tr = work / "test_results.json"
    if tr.is_file():
        ref = json.loads(tr.read_text(encoding="utf-8")).get("accuracy")
    if ref is not None and abs(acc - ref) > 0.02:
        print(f"  WARNING: exported acc={acc:.4f} differs from test_results.json={ref:.4f}")

    path = scores_path("video", protocol, f"{VIDEO_MODEL}{VIDEO_SUFFIX}")
    save_scores(
        path,
        scores_arr,
        labels_arr,
        {
            "modality": "video",
            "protocol": protocol,
            "model": VIDEO_MODEL,
            "benchmark": "kinetics",
            "checkpoint": str(ckpt),
            "config": str(cfg_path),
            "mmaction_python": str(MMACTION_PYTHON),
            "mmaction_root": str(MMACTION_ROOT),
            "accuracy": acc,
            "macro_f1": mca,
        },
    )
    dump_path.unlink(missing_ok=True)
    print(f"  -> acc={acc:.4f} mca={mca:.4f} saved {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export IMU/Video test_scores.pkl")
    parser.add_argument("--protocol", choices=["xsub", "xacq"], required=True)
    parser.add_argument("--modality", choices=["imu", "video", "both"], default="both")
    parser.add_argument(
        "--reuse-dump",
        action="store_true",
        help="video only: skip tools/test.py if _export_test_dump.pkl exists",
    )
    args = parser.parse_args()

    if args.modality in ("imu", "both"):
        export_imu(args.protocol)
    if args.modality in ("video", "both"):
        export_video(args.protocol, reuse_dump=args.reuse_dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
