#!/usr/bin/env python3
"""Export IMU session-level classification scores for GTAD StandardClassifier.

Uses the best action-recognition IMU model: ``ispl_inception`` · ``full_w256``.

Output schema (same as RGB/skeleton classifier JSON)::

  {"class": [...11 names...], "results": {video_id: [11 floats]}}

For each IMU TAD session (``data/beiyu/annotations/imu/{protocol}/anno.json``):
  1. Load Sensor06 CSV (acc.xyz + gyro.xyz)
  2. For every annotated stroke, take a 256-sample window (center-crop / pad)
  3. Z-score with the training ``norm_mean.npy`` / ``norm_std.npy``
  4. Softmax average over windows → one 11-d vector per session

Writes:
  data/beiyu/classifiers/ispl_inception_{protocol}.json
  data/beiyu/classifiers/imu_{protocol}.json   # alias for GTAD configs

Usage:
  cd OpenTAD-main
  # system python3 has TensorFlow (not vmamba)
  python3 tools/extract_features/export_imu_cls.py --protocol xsub
  python3 tools/extract_features/export_imu_cls.py --protocol xacq
  python3 tools/extract_features/export_imu_cls.py              # both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = OPENTAD_ROOT.parent
sys.path.insert(0, str(OPENTAD_ROOT))

from config import PROTOCOLS  # noqa: E402

CLS_DIR = OPENTAD_ROOT / "data" / "beiyu" / "classifiers"
CATEGORY_FILE = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "category_idx.txt"
IMU_ANNO_DIR = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "imu"
TRAINING_OUT = DATASET_ROOT / "training" / "outputs"

IMU_MODEL = "ispl_inception"
WIN_SIZE = 256
IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]


def load_class_names() -> list[str]:
    return [ln.strip() for ln in CATEGORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve_keras(out_dir: Path) -> Path | None:
    for name in ("best.keras", "final.keras"):
        p = out_dir / name
        if p.is_file():
            return p
    return None


def best_imu_dir(protocol: str) -> tuple[Path, float, float]:
    """Pick the highest test-Acc full_w256 run that has a keras checkpoint.

    Prefer ``ispl_inception`` (known best), but fall back to any model if needed.
    """
    candidates: list[tuple[float, float, Path, str]] = []
    for d in TRAINING_OUT.glob(f"{protocol}_*_full_w256*"):
        if not d.is_dir():
            continue
        keras = resolve_keras(d)
        rj = d / "results.json"
        if keras is None or not rj.is_file():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        test = r.get("test") or {}
        acc = test.get("accuracy")
        mf1 = test.get("macro_f1") or 0.0
        if acc is None:
            continue
        model = r.get("model") or d.name
        candidates.append((float(acc), float(mf1), d, model))

    if not candidates:
        raise FileNotFoundError(f"no usable IMU checkpoints under {TRAINING_OUT} for {protocol}")

    # Prefer ispl_inception when Acc ties / near-tie; otherwise max Acc
    preferred = [c for c in candidates if c[3] == IMU_MODEL]
    pool = preferred if preferred else candidates
    pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
    acc, mf1, path, model = pool[0]
    print(f"[info] best {protocol}: model={model} Acc={acc:.4f} MF1={mf1:.4f}")
    print(f"[info] dir={path}")
    return path, acc, mf1

def segment_window(data: np.ndarray, start: int, end: int, win: int = WIN_SIZE) -> np.ndarray:
    """Center-crop or zero-pad a stroke segment to fixed length ``win``."""
    start = max(0, min(start, data.shape[0]))
    end = max(start, min(end, data.shape[0]))
    seg = data[start:end]
    if seg.shape[0] == 0:
        return np.zeros((win, data.shape[1]), dtype=np.float32)
    if seg.shape[0] >= win:
        mid = seg.shape[0] // 2
        half = win // 2
        lo = mid - half
        return seg[lo : lo + win].astype(np.float32, copy=False)
    out = np.zeros((win, data.shape[1]), dtype=np.float32)
    offset = (win - seg.shape[0]) // 2
    out[offset : offset + seg.shape[0]] = seg
    return out


def load_imu_csv(rel_path: str, scale: float = 1.0) -> np.ndarray:
    csv_path = DATASET_ROOT / rel_path
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, usecols=IMU_COLUMNS, dtype=np.float32)
    data = df.to_numpy(dtype=np.float32)
    if scale != 1.0:
        data = data * float(scale)
    return data


def export_protocol(
    protocol: str,
    *,
    batch_size: int = 64,
    limit: int = 0,
    overwrite: bool = False,
) -> Path:
    import tensorflow as tf

    out_path = CLS_DIR / f"{IMU_MODEL}_{protocol}.json"
    alias_path = CLS_DIR / f"imu_{protocol}.json"
    if out_path.is_file() and not overwrite:
        print(f"[skip] {out_path} exists (pass --overwrite)")
        if not alias_path.is_file():
            alias_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[ok] alias -> {alias_path}")
        return out_path

    anno_path = IMU_ANNO_DIR / protocol / "anno.json"
    if not anno_path.is_file():
        raise FileNotFoundError(f"missing {anno_path}; run bash run_prepare_imu.sh")

    class_names = load_class_names()
    ckpt_dir, best_acc, best_mf1 = best_imu_dir(protocol)
    keras_path = resolve_keras(ckpt_dir)
    if keras_path is None:
        raise FileNotFoundError(f"no .keras in {ckpt_dir}")
    mean = np.load(ckpt_dir / "norm_mean.npy").astype(np.float32)
    std = np.load(ckpt_dir / "norm_std.npy").astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    print(f"[info] protocol={protocol} Acc={best_acc:.4f} MF1={best_mf1:.4f}")
    print(f"[info] checkpoint={keras_path}")
    print(f"[info] norm mean={mean.tolist()}")

    model = tf.keras.models.load_model(str(keras_path), compile=False)

    database = json.loads(anno_path.read_text(encoding="utf-8"))["database"]
    items = sorted(database.items())
    if limit > 0:
        items = items[:limit]

    results: dict[str, list[float]] = {}
    ok = fail = 0

    for vid, info in items:
        try:
            data = load_imu_csv(info["imu_csv"], float(info.get("sensor_scale_factor", 1.0)))
            fps = float(info["fps"])
            windows = []
            for anno in info.get("annotations", []):
                t0, t1 = float(anno["segment"][0]), float(anno["segment"][1])
                s0 = int(round(t0 * fps))
                s1 = int(round(t1 * fps))
                windows.append(segment_window(data, s0, s1))
            if not windows:
                # fallback: evenly spaced windows over whole recording
                step = WIN_SIZE // 2
                for s0 in range(0, max(1, data.shape[0] - WIN_SIZE + 1), step):
                    windows.append(data[s0 : s0 + WIN_SIZE].astype(np.float32))
                    if len(windows) >= 32:
                        break
            if not windows:
                raise RuntimeError("no windows")

            X = np.stack(windows, axis=0)  # [N, 256, 6]
            Xn = (X - mean) / std
            probs = model.predict(Xn, batch_size=batch_size, verbose=0)
            vec = probs.mean(axis=0).astype(np.float32)
            results[vid] = vec.tolist()
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {vid}: {exc}")
            fail += 1

    CLS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"class": class_names, "results": results}, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    alias_path.write_text(payload, encoding="utf-8")
    print(f"[ok] {protocol} videos={ok} fail={fail} -> {out_path}")
    print(f"[ok] alias -> {alias_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IMU GTAD classifier JSON (ispl_inception)")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="debug: only first N sessions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or list(PROTOCOLS)
    for protocol in protocols:
        export_protocol(
            protocol,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
