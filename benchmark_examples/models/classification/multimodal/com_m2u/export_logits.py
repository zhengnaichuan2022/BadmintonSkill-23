#!/usr/bin/env python3
"""Export offline train/val/test logits for RGB / Skeleton / IMU teachers."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))

from com_m2u.constants import (  # noqa: E402
    IMU_MODEL,
    MMACTION2_ROOT,
    MMACTION_PYTHON,
    OUT_ROOT,
    PYSKL_PYTHON,
    PYSKL_ROOT,
    ROOT,
    SKEL_STREAMS,
    VIDEO_TAG,
    imu_cache,
    imu_ckpt_dir,
    logits_dir,
    skel_cfg,
    skel_ckpt,
    skel_workdir,
    video_ann_frames,
    video_ckpt,
    video_manifest,
    video_workdir,
)


def _save_split(protocol: str, modality: str, split: str, logits: np.ndarray, labels: np.ndarray, ids: list[str], meta: dict):
    out = logits_dir(protocol) / modality
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"
    np.savez_compressed(
        path,
        logits=logits.astype(np.float32),
        labels=labels.astype(np.int64),
        ids=np.array(ids),
        meta=json.dumps(meta),
    )
    pred = logits.argmax(1)
    acc = float((pred == labels).mean())
    print(f"[export] {protocol}/{modality}/{split}: n={len(labels)} acc={acc:.4f} -> {path}")
    return path


def _manifest_ids_labels(protocol: str, split: str):
    man = json.loads(video_manifest(protocol).read_text(encoding="utf-8"))
    items = man["subsets"][split]
    ids = [x["instance_id"] for x in items]
    labels = np.array([int(x["label"]) for x in items], dtype=np.int64)
    return ids, labels


def export_imu(protocol: str, splits=("train", "val", "test")):
    import tensorflow as tf

    cache = np.load(imu_cache(protocol), allow_pickle=True)
    out_dir = imu_ckpt_dir(protocol)
    mean = np.load(out_dir / "norm_mean.npy")
    std = np.load(out_dir / "norm_std.npy")
    model_path = out_dir / "best.keras"
    if not model_path.is_file():
        model_path = out_dir / "final.keras"
    model = tf.keras.models.load_model(str(model_path), compile=False)

    for split in splits:
        ids, labels = _manifest_ids_labels(protocol, split)
        X = cache[f"X_{split}"].astype(np.float32)
        y = cache[f"y_{split}"].astype(np.int64)
        assert len(ids) == len(y) and (y == labels).all()
        Xn = (X - mean) / std
        probs = model.predict(Xn, batch_size=64, verbose=0).astype(np.float32)
        # Keras end softmax → pseudo-logits
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        _save_split(
            protocol,
            "imu",
            split,
            logits,
            labels,
            ids,
            {"model": IMU_MODEL, "source": str(out_dir), "ckpt": str(model_path)},
        )


def _mmaction_env():
    env = os.environ.copy()
    video_dir = str(ROOT / "video")
    env["PYTHONPATH"] = f"{MMACTION2_ROOT}:{video_dir}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _parse_mmaction_dump(dump_path: Path) -> tuple[np.ndarray, np.ndarray]:
    out_npz = dump_path.with_suffix(".npz")
    code = f"""
import pickle, numpy as np
raw = pickle.load(open({str(dump_path)!r}, "rb"))
scores, labels = [], []
for item in raw:
    s = item.pred_score if hasattr(item, "pred_score") else item["pred_score"]
    y = item.gt_label if hasattr(item, "gt_label") else item["gt_label"]
    scores.append(np.asarray(s).reshape(-1))
    labels.append(int(np.asarray(y).reshape(-1)[0]))
np.savez({str(out_npz)!r}, scores=np.stack(scores).astype(np.float32), labels=np.array(labels, dtype=np.int64))
"""
    subprocess.run([MMACTION_PYTHON, "-c", code], cwd=MMACTION2_ROOT, env=_mmaction_env(), check=True)
    data = np.load(out_npz)
    return data["scores"], data["labels"]


def export_video(protocol: str, splits=("train", "val", "test"), gpu: str = "0"):
    work = video_workdir(protocol)
    cfg = work / "config.py"
    ckpt = video_ckpt(protocol)
    tmp = OUT_ROOT / protocol / "tmp_video"
    tmp.mkdir(parents=True, exist_ok=True)

    for split in splits:
        ids, labels = _manifest_ids_labels(protocol, split)
        # Reuse existing test_scores.pkl when available
        if split == "test":
            tsp = work / "test_scores.pkl"
            if tsp.is_file():
                with open(tsp, "rb") as f:
                    d = pickle.load(f)
                scores = d["scores"].astype(np.float32)
                y = d["labels"].astype(np.int64)
                assert len(scores) == len(labels) and (y == labels).all()
                logits = np.log(np.clip(scores, 1e-8, 1.0))
                _save_split(
                    protocol,
                    "rgb",
                    split,
                    logits,
                    labels,
                    ids,
                    {"model": VIDEO_TAG, "ckpt": str(ckpt), "source": str(tsp)},
                )
                continue

        ann = video_ann_frames(protocol, split)
        dump = tmp / f"{split}_dump.pkl"
        text = cfg.read_text(encoding="utf-8")
        text = text.replace("test_frames.txt", f"{split}_frames.txt")
        text = text.replace("_export_test_dump.pkl", f"{split}_dump.pkl")
        text = text.replace(str(work / f"{split}_dump.pkl"), str(dump))
        import re

        text = re.sub(
            r"out_file_path=\s*'[^']*'",
            f"out_file_path={str(dump)!r}",
            text,
        )
        text = re.sub(
            r'out_file_path=\s*"[^"]*"',
            f"out_file_path={str(dump)!r}",
            text,
        )
        text = re.sub(
            r"out_file_path=\s*\n\s*'[^']*'",
            f"out_file_path=\n        {str(dump)!r}",
            text,
        )
        ov = tmp / f"{split}_cfg.py"
        ov.write_text(text, encoding="utf-8")
        cmd = [
            MMACTION_PYTHON,
            f"{MMACTION2_ROOT}/tools/test.py",
            str(ov),
            str(ckpt),
            "--work-dir",
            str(tmp / f"wd_{split}"),
            "--dump",
            str(dump),
        ]
        env = _mmaction_env()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print("[video]", " ".join(cmd))
        subprocess.run(cmd, cwd=MMACTION2_ROOT, env=env, check=True)

        scores, y = _parse_mmaction_dump(dump)
        assert len(scores) == len(labels), (len(scores), len(labels))
        assert (y == labels).all(), "video label mismatch"
        logits = np.log(np.clip(scores, 1e-8, 1.0))
        _save_split(
            protocol,
            "rgb",
            split,
            logits,
            labels,
            ids,
            {"model": VIDEO_TAG, "ckpt": str(ckpt)},
        )


def export_skeleton_stream(protocol: str, stream: str, splits=("train", "val", "test"), gpu: str = "0"):
    """Delegate to single-GPU infer_skeleton (ProtoGCN custom modules)."""
    cmd = [
        PYSKL_PYTHON,
        "-m",
        "com_m2u.infer_skeleton",
        "--protocol",
        protocol,
        "--stream",
        stream,
        "--splits",
        ",".join(splits),
        "--device",
        "cuda:0",
        "--reuse_test_pkl",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{_PKG}:{PYSKL_ROOT}:{env.get('PYTHONPATH', '')}"
    print("[skel]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(_PKG), env=env, check=True)


def fuse_skeleton_4stream(protocol: str, splits=("train", "val", "test")):
    for split in splits:
        ids, labels = _manifest_ids_labels(protocol, split)
        arrs = []
        for stream in SKEL_STREAMS:
            p = logits_dir(protocol) / f"skel_{stream}" / f"{split}.npz"
            d = np.load(p, allow_pickle=True)
            # convert logits→prob then average then back to logits
            probs = np.exp(d["logits"])
            probs = probs / probs.sum(axis=1, keepdims=True)
            arrs.append(probs)
        fused = np.mean(arrs, axis=0).astype(np.float32)
        logits = np.log(np.clip(fused, 1e-8, 1.0))
        _save_split(
            protocol,
            "skeleton",
            split,
            logits,
            labels,
            ids,
            {"model": "protogcn", "streams": list(SKEL_STREAMS), "fusion": "equal_prob_avg"},
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument(
        "--modality",
        default="all",
        choices=["all", "imu", "rgb", "skeleton", "skel_streams", "fuse_skel"],
    )
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--stream", default=None, help="single skeleton stream")
    args = ap.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    if args.modality in ("all", "imu"):
        export_imu(args.protocol, splits)
    if args.modality in ("all", "rgb"):
        export_video(args.protocol, splits, gpu=args.gpu)
    if args.modality in ("all", "skel_streams", "skeleton"):
        streams = [args.stream] if args.stream else list(SKEL_STREAMS)
        for st in streams:
            export_skeleton_stream(args.protocol, st, splits, gpu=args.gpu)
        if args.modality in ("all", "skeleton") and args.stream is None:
            fuse_skeleton_4stream(args.protocol, splits)
    if args.modality == "fuse_skel":
        fuse_skeleton_4stream(args.protocol, splits)


if __name__ == "__main__":
    main()
