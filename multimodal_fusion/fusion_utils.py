""" test （）。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
NUM_CLASSES = 11


def load_labels(protocol: str) -> Tuple[List[str], np.ndarray]:
    manifest = json.loads(
        (ROOT / "video/annotations" / protocol / "manifest.json").read_text(encoding="utf-8")
    )
    test = manifest["subsets"]["test"]
    ids = [x["instance_id"] for x in test]
    labels = np.array([int(x["label"]) for x in test], dtype=np.int64)
    return ids, labels


def load_skeleton_scores(protocol: str, model: str, stream: str) -> np.ndarray:
    if model == "posec3d_joint" and stream == "b":
        cfg = "posec3d_limb"
    elif stream == "j":
        cfg = model
    else:
        cfg = f"{model}_{stream}"
    pkl = ROOT / "skeleton/outputs" / protocol / cfg / "result.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return np.stack(data).astype(np.float32)


def scores_path(modality: str, protocol: str, tag: str) -> Path:
    if modality == "imu":
        return ROOT / "training/outputs" / f"{protocol}_{tag}" / "test_scores.pkl"
    if modality == "video":
        return ROOT / "video/outputs" / protocol / tag / "test_scores.pkl"
    raise ValueError(modality)


def save_scores(path: Path, scores: np.ndarray, labels: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"scores": scores, "labels": labels, "meta": meta}, f)


def load_scores(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["scores"], d["labels"], d.get("meta", {})


def mean_class_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    pred = scores.argmax(axis=1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for p, g in zip(pred, labels):
        cm[g, p] += 1
    cls_acc = []
    for c in range(NUM_CLASSES):
        n = cm[c].sum()
        if n:
            cls_acc.append(cm[c, c] / n)
    return float(np.mean(cls_acc)) if cls_acc else 0.0


def eval_scores(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    pred = scores.argmax(axis=1)
    acc = float((pred == labels).mean())
    mca = mean_class_accuracy(scores, labels)
    return acc, mca


def fuse_average(arrays: List[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("no score arrays")
    if len(set(a.shape for a in arrays)) != 1:
        raise ValueError(f"shape mismatch: {[a.shape for a in arrays]}")
    return np.mean(arrays, axis=0).astype(np.float32)
