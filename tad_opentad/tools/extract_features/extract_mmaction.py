#!/usr/bin/env python3
"""Extract RGB features with fine-tuned R(2+1)D classifier (MMAction2).

Uses ``video/outputs/{protocol}/r2plus1d_badminton_r34_kinetics/best_acc_top1*.pth``
and the same val/test preprocessing (Resize short=256, CenterCrop 224, clip 32x2).
Features are backbone outputs pooled by the trained cls_head (512-d), before FC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPENTAD_ROOT))

from config import FEATURES_DIR, FEATURE_STRIDE, PROTOCOLS, RGB_FEATURE_DIMS  # noqa: E402
from tools.extract_features.beiyu_common import (  # noqa: E402
    collect_videos,
    resolve_rgb_checkpoint,
    rgb_config_path,
    rgb_output_dir,
    snippet_centers,
)

MMACTION_ROOT = Path(os.environ.get("MMACTION2_ROOT", "/home/ubuntu/users/dyc/code/mmaction2"))
VIDEO_DIR = OPENTAD_ROOT.parent / "video"

CLIP_LEN = 32
FRAME_INTERVAL = 2
SHORT_SIDE = 256
CROP_SIZE = 224


def _import_mmaction():
    if str(MMACTION_ROOT) not in sys.path:
        sys.path.insert(0, str(MMACTION_ROOT))
    if str(VIDEO_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEO_DIR))
    from mmengine.config import Config
    from mmaction.apis import init_recognizer

    return Config, init_recognizer


def _load_model(config_path: Path, checkpoint_path: Path, device: str):
    Config, init_recognizer = _import_mmaction()
    cfg = Config.fromfile(str(config_path))
    model = init_recognizer(cfg, str(checkpoint_path), device=device)
    model.eval()
    return model, cfg


def _frame_indices_at_center(center: int, total_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(CLIP_LEN, dtype=np.int64)
    offset = int(center) - ((CLIP_LEN - 1) * FRAME_INTERVAL + 1) // 2
    idx = offset + np.arange(CLIP_LEN, dtype=np.int64) * FRAME_INTERVAL
    return np.mod(idx, total_frames)


def _resize_short_side(frame: np.ndarray, short_side: int = SHORT_SIDE) -> np.ndarray:
    h, w = frame.shape[:2]
    if min(h, w) == short_side:
        return frame
    if h < w:
        nh, nw = short_side, int(w * short_side / h)
    else:
        nw, nh = short_side, int(h * short_side / w)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _center_crop(frame: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    h, w = frame.shape[:2]
    y0 = max(0, (h - crop_size) // 2)
    x0 = max(0, (w - crop_size) // 2)
    return frame[y0 : y0 + crop_size, x0 : x0 + crop_size]


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Match MMAction test pipeline: Resize(-1,256) + CenterCrop(224), RGB uint8."""
    frame = _resize_short_side(frame)
    frame = _center_crop(frame)
    if frame.shape[0] != CROP_SIZE or frame.shape[1] != CROP_SIZE:
        frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    return frame


def _frames_to_clip_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    """[1, C, T, H, W] float32 in 0-255, ready for ActionDataPreprocessor."""
    arr = np.stack(frames, axis=0).astype(np.float32)  # T,H,W,C
    clip = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return clip


@torch.inference_mode()
def _pool_classifier_feat(model, clip_batch: torch.Tensor) -> np.ndarray:
    """Backbone + cls_head spatial pool (512-d), without classification FC."""
    inputs_list = [clip_batch[i] for i in range(clip_batch.size(0))]
    batch_inputs, _ = model.data_preprocessor.preprocess(inputs_list, [], training=False)
    feats, _ = model.extract_feat(batch_inputs, stage="backbone", test_mode=True)
    if model.with_cls_head and getattr(model.cls_head, "avg_pool", None) is not None:
        pooled = model.cls_head.avg_pool(feats)
        return pooled.flatten(1).cpu().numpy().astype(np.float32)
    if feats.ndim == 5:
        return feats.mean(dim=(2, 3, 4)).cpu().numpy().astype(np.float32)
    return feats.flatten(1).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def _extract_video(
    model,
    video_path: Path,
    num_frames: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    if num_frames <= 0:
        num_frames = len(vr)
    num_frames = min(num_frames, len(vr))
    centers = snippet_centers(num_frames, FEATURE_STRIDE)
    all_idx = [_frame_indices_at_center(int(center), num_frames) for center in centers]

    frame_cache: dict[int, np.ndarray] = {}
    feats = []
    for start in range(0, len(all_idx), batch_size):
        batch_idx = all_idx[start : start + batch_size]
        missing = sorted({int(fid) for idx in batch_idx for fid in idx if int(fid) not in frame_cache})
        if missing:
            batch = vr.get_batch(np.asarray(missing, dtype=np.int64))
            frames_arr = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)
            for i, fid in enumerate(missing):
                frame_cache[fid] = _preprocess_frame(frames_arr[i])

        clips = []
        for idx in batch_idx:
            frames = [frame_cache[int(fid)] for fid in idx]
            clips.append(_frames_to_clip_tensor(frames))

        clip_batch = torch.stack(clips).to(device)
        feats.append(_pool_classifier_feat(model, clip_batch))

    return np.concatenate(feats, axis=0)


def extract_protocol(
    protocol: str,
    *,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 8,
    limit: int = 0,
    overwrite: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict:
    config_path = Path(config_path or rgb_config_path(protocol))
    checkpoint_path = Path(checkpoint_path or resolve_rgb_checkpoint(protocol))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    out_dir = FEATURES_DIR / "rgb" / f"r2plus1d_{protocol}_stride{FEATURE_STRIDE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(config_path, checkpoint_path, device)

    items = collect_videos([protocol])
    if num_shards > 1:
        items = items[shard_id::num_shards]
    if limit > 0:
        items = items[:limit]

    ok, skip, fail = 0, 0, 0
    desc = f"rgb/r2plus1d/{protocol}"
    if num_shards > 1:
        desc += f" shard{shard_id}/{num_shards}"
    for vid, info in tqdm(items, desc=desc):
        out_path = out_dir / f"{vid}.npy"
        if out_path.is_file() and not overwrite:
            skip += 1
            continue
        video_path = OPENTAD_ROOT.parent / info["video_path"]
        if not video_path.is_file():
            print(f"[FAIL] missing video: {video_path}")
            fail += 1
            continue
        try:
            feats = _extract_video(
                model,
                video_path,
                int(info.get("frame", 0)),
                device,
                batch_size,
            )
            np.save(out_path, feats)
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {vid}: {exc}")
            fail += 1

    meta = {
        "backend": "r2plus1d",
        "protocol": protocol,
        "model": rgb_output_dir(protocol).name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "extractor": "finetuned_classifier_backbone_pool",
        "preprocess": "Resize(-1,256)+CenterCrop(224)+clip32x2",
        "stride": FEATURE_STRIDE,
        "dim": RGB_FEATURE_DIMS["r2plus1d"],
        "count": len(items),
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "out_dir": str(out_dir.relative_to(OPENTAD_ROOT)),
    }
    meta_name = "meta.json" if num_shards <= 1 else f"meta_shard{shard_id}.json"
    (out_dir / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB R(2+1)D TAD features")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"

    protocols = args.protocol or PROTOCOLS
    for protocol in protocols:
        meta = extract_protocol(
            protocol,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            overwrite=args.overwrite,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
