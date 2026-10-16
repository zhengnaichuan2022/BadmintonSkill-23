#!/usr/bin/env python3
"""Export IMU **5-node** session-level classification scores for GTAD.

Uses best 5-node recognition checkpoint under ``training/outputs/5node/``.

Writes:
  data/beiyu/classifiers/{model}_5node_{protocol}.json
  data/beiyu/classifiers/imu5_{protocol}.json   # alias for GTAD imu5 configs

Usage:
  cd OpenTAD-main
  # system python3 with TensorFlow
  python3 tools/extract_features/export_imu5_cls.py --protocol xsub
  python3 tools/extract_features/export_imu5_cls.py --protocol xacq
  python3 tools/extract_features/export_imu5_cls.py
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

from config import PROTOCOLS  # noqa: E402  OpenTAD config

# training/ for models.registry (after OpenTAD config is imported)
sys.path.insert(0, str(DATASET_ROOT / "training"))

CLS_DIR = OPENTAD_ROOT / "data" / "beiyu" / "classifiers"
CATEGORY_FILE = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "category_idx.txt"
IMU_ANNO_DIR = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "imu"
TRAINING_5NODE = DATASET_ROOT / "training" / "outputs" / "5node"

WIN_SIZE = 256
IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
SENSOR_FILES = [
    "Sensor01_ID1.csv",
    "Sensor02_ID2.csv",
    "Sensor03_ID3.csv",
    "Sensor04_ID4.csv",
    "Sensor06_ID6.csv",
]
N_CH = len(SENSOR_FILES) * len(IMU_COLUMNS)  # 30

# Prefer these models when Acc is close (match recognition benchmark winners)
PREFERRED = {
    "xsub": ("resnet_se", "ispl_inception"),
    "xacq": ("residual_bilstm", "stacked_lstm", "ispl_inception"),
}


def load_class_names() -> list[str]:
    return [ln.strip() for ln in CATEGORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve_keras(out_dir: Path) -> Path | None:
    for name in ("best.keras", "final.keras"):
        p = out_dir / name
        if p.is_file():
            return p
    return None


def best_5node_dir(protocol: str) -> tuple[Path, str, float, float]:
    candidates: list[tuple[float, float, Path, str]] = []
    for d in TRAINING_5NODE.glob(f"{protocol}_*_full_w256_*"):
        if not d.is_dir() or "_broken" in str(d):
            continue
        keras = resolve_keras(d)
        rj = d / "results.json"
        if keras is None or not rj.is_file():
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        test = r.get("test") or {}
        acc = test.get("accuracy")
        if acc is None:
            continue
        model = str(r.get("model") or d.name)
        if model == "r_tsf_net":
            continue
        mf1 = float(test.get("macro_f1") or 0.0)
        candidates.append((float(acc), mf1, d, model))

    if not candidates:
        raise FileNotFoundError(f"no usable 5-node checkpoints under {TRAINING_5NODE} for {protocol}")

    preferred_names = PREFERRED.get(protocol, ())
    preferred = [c for c in candidates if c[3] in preferred_names]
    pool = preferred if preferred else candidates
    # Among preferred, pick by Acc; else global max Acc
    if preferred:
        order = {n: i for i, n in enumerate(preferred_names)}
        pool.sort(key=lambda x: (x[0], -order.get(x[3], 99), x[1]), reverse=True)
    else:
        pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
    acc, mf1, path, model = pool[0]
    print(f"[info] best {protocol} 5node: model={model} Acc={acc:.4f} MF1={mf1:.4f}")
    print(f"[info] dir={path}")
    return path, model, acc, mf1


def segment_window(data: np.ndarray, start: int, end: int, win: int = WIN_SIZE) -> np.ndarray:
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


def load_imu5(rel_sensor06: str, scale: float = 1.0) -> np.ndarray:
    """Load 5-node session → (T, 30). Length = Sensor06 length."""
    s06 = DATASET_ROOT / rel_sensor06
    if not s06.is_file():
        raise FileNotFoundError(s06)
    df0 = pd.read_csv(s06, usecols=IMU_COLUMNS, dtype=np.float32)
    length = len(df0)
    parent = s06.parent
    parts = []
    for fname in SENSOR_FILES:
        path = parent / fname
        if path.is_file():
            df = pd.read_csv(path, usecols=IMU_COLUMNS, dtype=np.float32)
            seg = df.to_numpy(dtype=np.float32)
            if scale != 1.0:
                seg = seg * float(scale)
            if seg.shape[0] < length:
                pad = np.zeros((length - seg.shape[0], len(IMU_COLUMNS)), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=0)
            elif seg.shape[0] > length:
                seg = seg[:length]
        else:
            seg = np.zeros((length, len(IMU_COLUMNS)), dtype=np.float32)
        parts.append(seg)
    return np.concatenate(parts, axis=1)


def _keras_custom_objects(model_name: str) -> dict:
    from models.registry import load_model_module  # type: ignore

    mod = load_model_module(model_name)
    objs = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name[:1].isupper():
            objs[name] = obj
    return objs


def export_protocol(
    protocol: str,
    *,
    batch_size: int = 64,
    limit: int = 0,
    overwrite: bool = False,
) -> Path:
    import tensorflow as tf

    anno_path = IMU_ANNO_DIR / protocol / "anno.json"
    if not anno_path.is_file():
        raise FileNotFoundError(f"missing {anno_path}; run bash run_prepare_imu.sh")

    ckpt_dir, model_name, best_acc, best_mf1 = best_5node_dir(protocol)
    out_path = CLS_DIR / f"{model_name}_5node_{protocol}.json"
    alias_path = CLS_DIR / f"imu5_{protocol}.json"
    if alias_path.is_file() and not overwrite:
        print(f"[skip] {alias_path} exists (pass --overwrite)")
        return alias_path

    class_names = load_class_names()
    keras_path = resolve_keras(ckpt_dir)
    if keras_path is None:
        raise FileNotFoundError(f"no .keras in {ckpt_dir}")
    mean = np.load(ckpt_dir / "norm_mean.npy").astype(np.float32)
    std = np.load(ckpt_dir / "norm_std.npy").astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    if mean.shape[-1] != N_CH:
        raise ValueError(f"norm mean channels {mean.shape} != {N_CH}")

    print(f"[info] protocol={protocol} Acc={best_acc:.4f} MF1={best_mf1:.4f}")
    print(f"[info] checkpoint={keras_path}")

    custom = _keras_custom_objects(model_name)
    model = tf.keras.models.load_model(str(keras_path), compile=False, custom_objects=custom)

    database = json.loads(anno_path.read_text(encoding="utf-8"))["database"]
    items = sorted(database.items())
    if limit > 0:
        items = items[:limit]

    results: dict[str, list[float]] = {}
    ok = fail = 0

    for vid, info in items:
        try:
            data = load_imu5(info["imu_csv"], float(info.get("sensor_scale_factor", 1.0)))
            fps = float(info["fps"])
            windows = []
            for anno in info.get("annotations", []):
                t0, t1 = float(anno["segment"][0]), float(anno["segment"][1])
                s0 = int(round(t0 * fps))
                s1 = int(round(t1 * fps))
                windows.append(segment_window(data, s0, s1))
            if not windows:
                step = WIN_SIZE // 2
                for s0 in range(0, max(1, data.shape[0] - WIN_SIZE + 1), step):
                    windows.append(data[s0 : s0 + WIN_SIZE].astype(np.float32))
                    if len(windows) >= 32:
                        break
            if not windows:
                raise RuntimeError("no windows")

            X = np.stack(windows, axis=0)  # [N, 256, 30]
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
    return alias_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IMU5 GTAD classifier JSON")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for protocol in args.protocol or list(PROTOCOLS):
        export_protocol(
            protocol,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
