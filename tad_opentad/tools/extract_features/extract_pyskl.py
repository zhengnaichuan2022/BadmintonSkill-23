#!/usr/bin/env python3
"""Extract skeleton 4-stream features with fine-tuned ProtoGCN (pyskl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from tqdm import tqdm

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
SKELETON_DIR = OPENTAD_ROOT.parent / "skeleton"
sys.path.insert(0, str(OPENTAD_ROOT))
sys.path.insert(0, str(SKELETON_DIR))

from config import FEATURES_DIR, FEATURE_STRIDE, PROTOCOLS  # noqa: E402
from tools.extract_features.beiyu_common import (  # noqa: E402
    PYSKL_ROOT,
    SKELETON_DIM,
    SKELETON_STREAMS,
    collect_videos,
    find_best_checkpoint,
    load_video_pose,
    resolve_skeleton_json,
    skeleton_config_path,
    skeleton_output_dir,
    snippet_centers,
    centered_indices,
)

CLIP_LEN = 100


def _setup_pyskl():
    if str(PYSKL_ROOT) not in sys.path:
        sys.path.insert(0, str(PYSKL_ROOT))
    import pyskl.utils.mmcv_compat  # noqa: F401
    import pyskl.models.gcns.protogcn_ext  # noqa: F401
    import pyskl.models.heads.protogcn_head  # noqa: F401
    import pyskl.models.recognizers.recognizer_protogcn  # noqa: F401
    from pyskl.datasets.pipelines import Compose
    from pyskl.models import build_model

    return Compose, build_model


def _build_stream_pipeline(cfg: Config, Compose):
    pipeline_cfg = [
        dict(type="PreNormalize2D"),
        cfg.data.test.pipeline[1],
        dict(type="PoseDecode"),
        dict(type="FormatGCNInput", num_person=2),
        dict(type="Collect", keys=["keypoint"], meta_keys=[]),
        dict(type="ToTensor", keys=["keypoint"]),
    ]
    return Compose(pipeline_cfg)


def _pool_protogcn_feat(feat) -> np.ndarray:
    x = feat[0] if isinstance(feat, (tuple, list)) else feat
    vec = x.mean(dim=(0, 1, 3, 4))
    return vec.cpu().numpy().astype(np.float32)


def _load_stream_model(protocol: str, stream: str, device: str):
    Compose, build_model = _setup_pyskl()
    cfg = Config.fromfile(str(skeleton_config_path(protocol, stream)))
    model = build_model(cfg.model)
    ckpt = find_best_checkpoint(skeleton_output_dir(protocol, stream))
    load_checkpoint(model, str(ckpt), map_location="cpu")
    model.to(device)
    model.eval()
    pipeline = _build_stream_pipeline(cfg, Compose)
    return model, pipeline, ckpt


def _prepare_window(keypoints: np.ndarray, scores: np.ndarray, img_shape, center: int):
    idx = centered_indices(center, keypoints.shape[0], CLIP_LEN)
    return {
        "keypoint": keypoints[idx][None, ...],
        "keypoint_score": scores[idx][None, ...],
        "total_frames": CLIP_LEN,
        "frame_inds": np.arange(CLIP_LEN),
        "img_shape": img_shape,
        "original_shape": img_shape,
        "label": -1,
    }


@torch.inference_mode()
def _extract_video(model, pipeline, keypoints, scores, img_shape, device: str, batch_size: int) -> np.ndarray:
    centers = snippet_centers(keypoints.shape[0], FEATURE_STRIDE)
    feats = []
    for start in range(0, len(centers), batch_size):
        batch = []
        for center in centers[start : start + batch_size]:
            batch.append(_prepare_window(keypoints, scores, img_shape, int(center)))
        batch = [pipeline(sample) for sample in batch]
        batch = collate(batch, samples_per_gpu=len(batch))
        kp = batch["keypoint"]
        if kp.ndim == 6:
            bs, nc = kp.shape[:2]
            kp = kp.reshape((bs * nc,) + kp.shape[2:])
        kp = kp.to(device)
        raw = model.extract_feat(kp)
        for i in range(kp.shape[0]):
            if isinstance(raw, (tuple, list)):
                item = tuple(r[i : i + 1] for r in raw)
            else:
                item = raw[i : i + 1]
            feats.append(_pool_protogcn_feat(item))
    return np.stack(feats, axis=0)


def extract_stream(
    protocol: str,
    stream: str,
    *,
    device: str = "cuda:0",
    batch_size: int = 16,
    limit: int = 0,
    overwrite: bool = False,
) -> dict:
    out_dir = FEATURES_DIR / "skeleton" / f"protogcn_{stream}_{protocol}_stride{FEATURE_STRIDE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, pipeline, ckpt = _load_stream_model(protocol, stream, device)
    items = collect_videos([protocol])
    if limit > 0:
        items = items[:limit]

    ok, skip, fail, missing_skel = 0, 0, 0, 0
    for vid, info in tqdm(items, desc=f"skel/protogcn_{stream}/{protocol}"):
        out_path = out_dir / f"{vid}.npy"
        if out_path.is_file() and not overwrite:
            skip += 1
            continue
        video_path = OPENTAD_ROOT.parent / info["video_path"]
        skel_json = resolve_skeleton_json(video_path)
        if skel_json is None:
            missing_skel += 1
            fail += 1
            continue
        try:
            keypoints, scores, img_shape = load_video_pose(skel_json)
            feats = _extract_video(model, pipeline, keypoints, scores, img_shape, device, batch_size)
            np.save(out_path, feats)
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {vid}/{stream}: {exc}")
            fail += 1

    meta = {
        "backend": "protogcn",
        "stream": stream,
        "protocol": protocol,
        "config": str(skeleton_config_path(protocol, stream)),
        "checkpoint": str(ckpt),
        "stride": FEATURE_STRIDE,
        "dim": SKELETON_DIM,
        "count": len(items),
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "missing_skeleton": missing_skel,
        "out_dir": str(out_dir.relative_to(OPENTAD_ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skeleton ProtoGCN 4-stream TAD features")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--stream", choices=SKELETON_STREAMS, action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"

    protocols = args.protocol or PROTOCOLS
    streams = args.stream or list(SKELETON_STREAMS)
    for protocol in protocols:
        for stream in streams:
            meta = extract_stream(
                protocol,
                stream,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
                overwrite=args.overwrite,
            )
            print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
