#!/usr/bin/env python3
"""Export full-video classification scores for GTAD StandardClassifier.

Outputs JSON schema:
  {"class": [...11 action names...], "results": {video_id: [11 floats]}}

Backends (protocol-aligned, same checkpoints as TAD feature extraction):
  rgb      → r2plus1d_badminton_r34_kinetics
  protogcn → j / b / jm / bm streams

Composite keys (mean of snippet-level softmax, then mean across backends):
  skel_j, skel_b, skel_jb, skel_4stream,
  rgb, rgb_skel_j, rgb_skel_b, rgb_skel_jb, rgb_skel_4stream

Usage:
  cd OpenTAD-main
  # RGB (MMAction env)
  export MMACTION2_ROOT=/home/ubuntu/users/dyc/code/mmaction2
  export PYTHONPATH=$MMACTION2_ROOT:../video:$PYTHONPATH
  CUDA_VISIBLE_DEVICES=0 python tools/extract_features/export_video_cls.py --modality rgb --protocol xsub

  # Skeleton (vmamba / pyskl env)
  CUDA_VISIBLE_DEVICES=1 python tools/extract_features/export_video_cls.py --modality skel --protocol xsub --stream j

  # Merge composites + all base keys for one protocol
  python tools/extract_features/export_video_cls.py --modality merge --protocol xsub

  # Everything
  bash run_export_video_cls.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPENTAD_ROOT))

from config import PROTOCOLS  # noqa: E402
from tools.extract_features.beiyu_common import (  # noqa: E402
    SKELETON_STREAMS,
    collect_videos,
)

CLS_DIR = OPENTAD_ROOT / "data" / "beiyu" / "classifiers"
CATEGORY_FILE = OPENTAD_ROOT / "data" / "beiyu" / "annotations" / "category_idx.txt"

# GTAD benchmark feature key → list of base classifier stems (without protocol suffix)
COMPOSITE_SOURCES = {
    "rgb": ["r2plus1d"],
    "skel_j": ["protogcn_j"],
    "skel_b": ["protogcn_b"],
    "skel_jb": ["protogcn_j", "protogcn_b"],
    "skel_4stream": ["protogcn_j", "protogcn_b", "protogcn_jm", "protogcn_bm"],
    "rgb_skel_j": ["r2plus1d", "protogcn_j"],
    "rgb_skel_b": ["r2plus1d", "protogcn_b"],
    "rgb_skel_jb": ["r2plus1d", "protogcn_j", "protogcn_b"],
    "rgb_skel_4stream": [
        "r2plus1d",
        "protogcn_j",
        "protogcn_b",
        "protogcn_jm",
        "protogcn_bm",
    ],
}


def load_class_names() -> list[str]:
    return [ln.strip() for ln in CATEGORY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def cls_path(stem: str, protocol: str) -> Path:
    return CLS_DIR / f"{stem}_{protocol}.json"


def write_cls_json(path: Path, results: dict[str, list[float]], class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"class": class_names, "results": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_cls_json(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float32) for k, v in data["results"].items()}


def export_rgb_protocol(
    protocol: str,
    *,
    device: str = "cuda:0",
    batch_size: int = 8,
    limit: int = 0,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    import torch
    from decord import VideoReader, cpu
    from tqdm import tqdm

    from config import FEATURE_STRIDE  # noqa: E402
    from tools.extract_features.extract_mmaction import (
        CLIP_LEN,
        _frames_to_clip_tensor,
        _frame_indices_at_center,
        _load_model,
        _preprocess_frame,
    )
    from tools.extract_features.beiyu_common import (
        resolve_rgb_checkpoint,
        rgb_config_path,
        snippet_centers,
    )

    out_path = cls_path("r2plus1d", protocol)
    if out_path.is_file() and not overwrite and not resume:
        print(f"[skip] {out_path}")
        return out_path

    results: dict[str, list[float]] = {}
    if resume and out_path.is_file() and not overwrite:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        results = {k: list(v) for k, v in data.get("results", {}).items()}
        print(f"[resume] {out_path} existing={len(results)}")

    config_path = rgb_config_path(protocol)
    checkpoint_path = resolve_rgb_checkpoint(protocol)
    model, _ = _load_model(config_path, checkpoint_path, device)

    @torch.inference_mode()
    def _classify_video(model, video_path: Path, num_frames: int) -> np.ndarray:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        if num_frames <= 0:
            num_frames = len(vr)
        num_frames = min(num_frames, len(vr))
        centers = snippet_centers(num_frames, FEATURE_STRIDE)
        all_idx = [_frame_indices_at_center(int(c), num_frames) for c in centers]

        frame_cache: dict[int, np.ndarray] = {}
        probs = []
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

            inputs_list = [clip_batch[i] for i in range(clip_batch.size(0))]
            batch_inputs, _ = model.data_preprocessor.preprocess(inputs_list, [], training=False)
            feats, _ = model.extract_feat(batch_inputs, stage="backbone", test_mode=True)
            logits = model.cls_head(feats)
            if logits.ndim > 2:
                logits = logits.view(logits.size(0), -1)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

        arr = np.concatenate(probs, axis=0)
        return arr.mean(axis=0).astype(np.float32)

    class_names = load_class_names()
    items = collect_videos([protocol])
    if limit > 0:
        items = items[:limit]

    ok, fail, skipped = 0, 0, 0
    for vid, info in tqdm(items, desc=f"cls/r2plus1d/{protocol}"):
        if vid in results:
            skipped += 1
            continue
        video_path = OPENTAD_ROOT.parent / info["video_path"]
        if not video_path.is_file():
            fail += 1
            continue
        try:
            vec = _classify_video(model, video_path, int(info.get("frame", 0)))
            results[vid] = vec.tolist()
            ok += 1
            write_cls_json(out_path, results, class_names)
        except Exception as exc:
            print(f"[FAIL] rgb {vid}: {exc}")
            fail += 1

    write_cls_json(out_path, results, class_names)
    print(f"[ok] rgb/{protocol} new={ok} skip={skipped} fail={fail} total={len(results)} -> {out_path}")
    return out_path


def export_skel_stream(
    protocol: str,
    stream: str,
    *,
    device: str = "cuda:0",
    batch_size: int = 16,
    limit: int = 0,
    overwrite: bool = False,
) -> Path:
    import torch
    from mmcv.parallel import collate
    from tqdm import tqdm

    from tools.extract_features.extract_pyskl import (
        _load_stream_model,
        _prepare_window,
    )
    from tools.extract_features.beiyu_common import load_video_pose, resolve_skeleton_json, snippet_centers

    out_path = cls_path(f"protogcn_{stream}", protocol)
    if out_path.is_file() and not overwrite:
        print(f"[skip] {out_path}")
        return out_path

    model, pipeline, _ = _load_stream_model(protocol, stream, device)
    class_names = load_class_names()
    items = collect_videos([protocol])
    if limit > 0:
        items = items[:limit]

    results: dict[str, list[float]] = {}
    ok, fail, missing = 0, 0, 0

    @torch.inference_mode()
    def _classify_pose(keypoints, scores, img_shape) -> np.ndarray:
        centers = snippet_centers(keypoints.shape[0])
        probs = []
        for start in range(0, len(centers), batch_size):
            batch = []
            for center in centers[start : start + batch_size]:
                batch.append(_prepare_window(keypoints, scores, img_shape, int(center)))
            batch = [pipeline(sample) for sample in batch]
            batch = collate(batch, samples_per_gpu=len(batch))
            kp = batch["keypoint"].to(device)
            out = model.forward_test(kp)
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 1:
                out = out[None, :]
            probs.append(out)
        arr = np.concatenate(probs, axis=0)
        return arr.mean(axis=0).astype(np.float32)

    for vid, info in tqdm(items, desc=f"cls/protogcn_{stream}/{protocol}"):
        video_path = OPENTAD_ROOT.parent / info["video_path"]
        skel_json = resolve_skeleton_json(video_path)
        if skel_json is None:
            missing += 1
            fail += 1
            continue
        try:
            keypoints, sc, img_shape = load_video_pose(skel_json)
            vec = _classify_pose(keypoints, sc, img_shape)
            results[vid] = vec.tolist()
            ok += 1
        except Exception as exc:
            print(f"[FAIL] skel/{stream} {vid}: {exc}")
            fail += 1

    write_cls_json(out_path, results, class_names)
    print(f"[ok] skel/{stream}/{protocol} videos={ok} fail={fail} missing_skel={missing} -> {out_path}")
    return out_path


def merge_protocol(protocol: str, *, overwrite: bool = True) -> list[Path]:
    class_names = load_class_names()
    written: list[Path] = []
    for feat_key, stems in COMPOSITE_SOURCES.items():
        out_path = cls_path(feat_key, protocol)
        if out_path.is_file() and not overwrite:
            print(f"[skip] {out_path}")
            written.append(out_path)
            continue

        arrays: list[dict[str, np.ndarray]] = []
        for stem in stems:
            src = cls_path(stem, protocol)
            if not src.is_file():
                raise FileNotFoundError(f"missing base classifier json: {src}")
            arrays.append(read_cls_json(src))

        common = set(arrays[0]).intersection(*arrays[1:]) if len(arrays) > 1 else set(arrays[0])
        results: dict[str, list[float]] = {}
        for vid in sorted(common):
            vec = np.mean([arr[vid] for arr in arrays], axis=0)
            results[vid] = vec.astype(np.float32).tolist()

        write_cls_json(out_path, results, class_names)
        print(f"[ok] merge {feat_key}/{protocol} videos={len(results)} -> {out_path}")
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GTAD video-level classifier JSON")
    parser.add_argument("--modality", choices=["rgb", "skel", "merge", "all"], default="all")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--stream", choices=SKELETON_STREAMS, action="append")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="load partial JSON and continue; save after each video")
    args = parser.parse_args()

    protocols = args.protocol or list(PROTOCOLS)
    streams = args.stream or list(SKELETON_STREAMS)

    if args.modality in {"rgb", "all"}:
        for protocol in protocols:
            export_rgb_protocol(
                protocol,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
                overwrite=args.overwrite,
                resume=args.resume,
            )

    if args.modality in {"skel", "all"}:
        for protocol in protocols:
            for stream in streams:
                export_skel_stream(
                    protocol,
                    stream,
                    device=args.device,
                    batch_size=max(args.batch_size, 8),
                    limit=args.limit,
                    overwrite=args.overwrite,
                )

    if args.modality in {"merge", "all"}:
        for protocol in protocols:
            merge_protocol(protocol, overwrite=True)


if __name__ == "__main__":
    main()
