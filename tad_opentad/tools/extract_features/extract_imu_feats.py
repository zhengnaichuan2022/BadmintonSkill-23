#!/usr/bin/env python3
"""Extract IMU TAD features aligned to the **video** feature timeline (stride=4).

For each session in ``annotations/{protocol}/anno.json``:
  1. Snippet centers on video frames: ``0, 4, 8, …``  (same as RGB/skeleton)
  2. Map video time → IMU sample via per-session ``alignment_offset_seconds``
  3. Crop a 1024-sample raw window (256×stride4, matches IMU e2e window)
  4. Run ``IMUInceptionBackbone`` from a trained IMU e2e ActionFormer ckpt
  5. Mean-pool over the backbone time axis → one 256-d vector per snippet

Outputs (float32 ``[T_video, 256]``)::

  data/beiyu/features/imu/inception_{protocol}_stride4/{video_id}.npy      # 1-node (6ch)
  data/beiyu/features/imu/inception5_{protocol}_stride4/{video_id}.npy     # 5-node (30ch)

Usage::

  cd OpenTAD-main
  # 5-node (recommended for RGB+skel+IMU fusion)
  python tools/extract_features/extract_imu_feats.py --protocol xsub --nodes 5 --device cuda:0
  python tools/extract_features/extract_imu_feats.py --protocol xacq --nodes 5 --device cuda:0

  # 1-node
  python tools/extract_features/extract_imu_feats.py --protocol both --nodes 1 --device cuda:0

  # smoke
  python tools/extract_features/extract_imu_feats.py --protocol xsub --nodes 5 --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPENTAD_ROOT))

from config import DATASET_ROOT, FEATURE_STRIDE, IMU_SAMPLE_RATE, PROTOCOLS  # noqa: E402
from opentad.models.imu_inception import IMUInceptionBackbone  # noqa: E402
from tools.extract_features.beiyu_common import load_anno, snippet_centers  # noqa: E402

INSTANCES_CSV = DATASET_ROOT / "metadata" / "instances.csv"
IMU_ANNO_DIR = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "imu"
FEATURES_IMU = OPENTAD_ROOT / "data" / "beiyu" / "features" / "imu"

SENSOR_FILES = (
    "Sensor01_ID1.csv",
    "Sensor02_ID2.csv",
    "Sensor03_ID3.csv",
    "Sensor04_ID4.csv",
    "Sensor06_ID6.csv",
)
IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]

# Match IMU e2e sliding window: window_size=256 × feature_stride=4
WIN_RAW = 256 * FEATURE_STRIDE  # 1024
FEAT_DIM = 256

DEFAULT_CKPT = {
    (1, "xsub"): "exps/beiyu/imu_e2e/xsub/actionformer_inception_fix/gpu1_id0/checkpoint/best.pth",
    (1, "xacq"): "exps/beiyu/imu_e2e/xacq/actionformer_inception_fix/gpu1_id0/checkpoint/best.pth",
    (5, "xsub"): "exps/beiyu/imu5_e2e/xsub/actionformer_inception5_fix/gpu1_id0/checkpoint/best.pth",
    (5, "xacq"): "exps/beiyu/imu5_e2e/xacq/actionformer_inception5_fix/gpu1_id0/checkpoint/best.pth",
}


def video_id_from_ann(ann: str) -> str:
    rel = Path(ann).as_posix()
    rel = rel.replace("2：-/", "F/")
    rel = rel.replace("2：-/", "S/")
    rel = rel.replace("/", "__")
    rel = re.sub(r"__+", "__", rel)
    return rel.replace("_annotations.json", "")


def load_offsets() -> dict[str, float]:
    df = pd.read_csv(INSTANCES_CSV, low_memory=False)
    df = df[df["video_annotation_json"].notna() & (df["video_annotation_json"] != "")]
    df["vid"] = df["video_annotation_json"].map(video_id_from_ann)
    off = df.groupby("vid")["alignment_offset_seconds"].median().to_dict()
    return {str(k): float(v) for k, v in off.items()}


def load_imu_anno(protocol: str) -> dict:
    path = IMU_ANNO_DIR / protocol / "anno.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run bash run_prepare_imu.sh")
    return json.loads(path.read_text(encoding="utf-8"))["database"]


def resolve_ckpt(nodes: int, protocol: str, ckpt: str | None) -> Path:
    if ckpt:
        p = Path(ckpt)
        if not p.is_absolute():
            p = OPENTAD_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    rel = DEFAULT_CKPT.get((nodes, protocol))
    if not rel:
        raise KeyError(f"no default ckpt for nodes={nodes} protocol={protocol}")
    p = OPENTAD_ROOT / rel
    if not p.is_file():
        # try alternate gpu*_id* layout
        parent = p.parent.parent  # .../checkpoint → work_dir
        cands = sorted(parent.glob("**/checkpoint/best.pth"))
        if not cands:
            raise FileNotFoundError(f"missing {p}")
        p = cands[0]
    return p


def load_backbone(nodes: int, ckpt_path: Path, device: torch.device) -> IMUInceptionBackbone:
    in_ch = 6 * nodes if nodes in (1, 5) else nodes  # nodes=1→6, nodes=5→30
    if nodes == 5:
        in_ch = 30
    elif nodes == 1:
        in_ch = 6
    model = IMUInceptionBackbone(
        in_channels=in_ch,
        out_channels=FEAT_DIM,
        network_depth=5,
        filters_number=64,
        max_kernel_size=68,
        use_residual=True,
        use_bottleneck=True,
        stride=FEATURE_STRIDE,
        freeze_backbone=True,
    )
    blob = torch.load(str(ckpt_path), map_location="cpu")
    state = blob.get("state_dict_ema") or blob.get("state_dict") or blob
    bb = {}
    for k, v in state.items():
        if "backbone." not in k:
            continue
        # strip prefixes: module.backbone. / backbone.
        nk = k.split("backbone.", 1)[1]
        bb[nk] = v
    missing, unexpected = model.load_state_dict(bb, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}…")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}…")
    model.eval().to(device)
    return model


def read_sensor_csv(path: Path, scale: float) -> np.ndarray:
    df = pd.read_csv(path, usecols=IMU_COLUMNS, dtype=np.float32)
    data = df.to_numpy(dtype=np.float32)
    if scale != 1.0:
        data = data * float(scale)
    return data


def load_imu_array(imu_csv_rel: str, nodes: int, scale: float) -> np.ndarray:
    """Return (T, C) with C=6 (1-node) or 30 (5-node). Length = Sensor06 length."""
    s06 = DATASET_ROOT / imu_csv_rel
    if not s06.is_file():
        raise FileNotFoundError(s06)
    base = read_sensor_csv(s06, scale)
    if nodes == 1:
        return base
    length = base.shape[0]
    parent = s06.parent
    parts = []
    for fname in SENSOR_FILES:
        path = parent / fname
        if path.is_file():
            seg = read_sensor_csv(path, scale)
            if seg.shape[0] < length:
                pad = np.zeros((length - seg.shape[0], 6), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=0)
            elif seg.shape[0] > length:
                seg = seg[:length]
        else:
            seg = np.zeros((length, 6), dtype=np.float32)
        parts.append(seg)
    return np.concatenate(parts, axis=1)


def crop_window(data: np.ndarray, center: int, win: int = WIN_RAW) -> np.ndarray:
    """Centered crop with edge pad → (win, C)."""
    n, c = data.shape
    half = win // 2
    start = center - half
    out = np.zeros((win, c), dtype=np.float32)
    src0 = max(0, start)
    src1 = min(n, start + win)
    dst0 = src0 - start
    dst1 = dst0 + (src1 - src0)
    if src1 > src0:
        out[dst0:dst1] = data[src0:src1]
    return out


@torch.no_grad()
def encode_windows(
    model: IMUInceptionBackbone,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """windows: [N, W, C] → feats [N, 256]."""
    n = windows.shape[0]
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    for i in range(0, n, batch_size):
        batch = windows[i : i + batch_size]
        # [B, C, T]
        x = torch.from_numpy(batch).permute(0, 2, 1).contiguous().to(device)
        # per-window z-score (match LoadIMURaw normalize=window)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        y = model(x)  # [B, 256, T//4]
        vec = y.mean(dim=-1)  # [B, 256]
        feats[i : i + batch_size] = vec.float().cpu().numpy()
    return feats


def out_dir(nodes: int, protocol: str) -> Path:
    tag = "inception5" if nodes == 5 else "inception"
    return FEATURES_IMU / f"{tag}_{protocol}_stride4"


def extract_protocol(
    protocol: str,
    *,
    nodes: int,
    device: torch.device,
    batch_size: int,
    limit: int,
    overwrite: bool,
    ckpt: str | None,
) -> None:
    video_db = load_anno(protocol)
    imu_db = load_imu_anno(protocol)
    offsets = load_offsets()
    ckpt_path = resolve_ckpt(nodes, protocol, ckpt)
    print(f"[info] protocol={protocol} nodes={nodes} ckpt={ckpt_path}")
    model = load_backbone(nodes, ckpt_path, device)

    dest = out_dir(nodes, protocol)
    dest.mkdir(parents=True, exist_ok=True)

    keys = sorted(k for k in video_db if k not in {".", ""} and k in imu_db)
    if limit > 0:
        keys = keys[:limit]

    ok = skip = fail = 0
    for vid in tqdm(keys, desc=f"imu{nodes}-{protocol}"):
        out_path = dest / f"{vid}.npy"
        if out_path.is_file() and not overwrite:
            skip += 1
            continue
        try:
            vinfo = video_db[vid]
            iinfo = imu_db[vid]
            n_frames = int(vinfo["frame"])
            v_fps = float(vinfo["fps"])
            centers = snippet_centers(n_frames, FEATURE_STRIDE)
            # Prefer RGB feature length if present (skel/rgb may differ by 1–2)
            rgb_feat = (
                OPENTAD_ROOT
                / "data"
                / "beiyu"
                / "features"
                / "rgb"
                / f"r2plus1d_{protocol}_stride4"
                / f"{vid}.npy"
            )
            if rgb_feat.is_file():
                t_rgb = int(np.load(rgb_feat, mmap_mode="r").shape[0])
                if t_rgb != len(centers):
                    # re-sample centers to match RGB T exactly
                    if t_rgb <= 0:
                        raise RuntimeError("empty rgb feat")
                    centers = np.linspace(0, max(n_frames - 1, 0), t_rgb).astype(np.int64)

            if vid not in offsets:
                raise KeyError(f"no alignment_offset_seconds for {vid}")
            t_off = offsets[vid]
            scale = float(iinfo.get("sensor_scale_factor", 1.0))
            imu = load_imu_array(iinfo["imu_csv"], nodes=nodes, scale=scale)

            # video frame → IMU sample
            imu_centers = np.round((centers.astype(np.float64) / v_fps + t_off) * IMU_SAMPLE_RATE).astype(
                np.int64
            )
            windows = np.stack([crop_window(imu, int(c)) for c in imu_centers], axis=0)
            feats = encode_windows(model, windows, device, batch_size)
            np.save(out_path, feats.astype(np.float32))
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {vid}: {exc}")

    print(f"[done] {protocol} nodes={nodes} ok={ok} skip={skip} fail={fail} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract IMU features on video timeline")
    parser.add_argument("--protocol", choices=["xsub", "xacq", "both"], default="both")
    parser.add_argument("--nodes", type=int, choices=[1, 5], default=5, help="1=6ch Sensor06, 5=30ch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ckpt", default=None, help="override backbone checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    protos = PROTOCOLS if args.protocol == "both" else [args.protocol]
    for proto in protos:
        extract_protocol(
            proto,
            nodes=args.nodes,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
            ckpt=args.ckpt,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
