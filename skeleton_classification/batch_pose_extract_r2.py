#!/usr/bin/env python3
"""Batch RTMPose-L skeleton extraction for Research 2 videos."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MMPOSE_ROOT = Path("/home/ubuntu/users/syq/mmpose")
RTMPOSE_CONFIG = (
    MMPOSE_ROOT
    / "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py"
)

DATASET_ROOT = Path(__file__).resolve().parent.parent
SKELETON_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    SKELETON_ROOT
    / "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth"
)
DEFAULT_DET_CONFIG = (
    MMPOSE_ROOT / "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
)
DEFAULT_DET_WEIGHTS = (
    SKELETON_ROOT / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
DEFAULT_OUTPUT = SKELETON_ROOT
MANIFEST_PATH = SKELETON_ROOT / "pose_extract_manifest.jsonl"

R2_DIRS = [
    DATASET_ROOT / "2：-",
    DATASET_ROOT / "2：-",
]
VIDEO_EXTS = {".mp4", ".mov", ".MP4", ".MOV"}
DUPLICATE_COPY_MARKER = "(1)"


@dataclass
class WorkerConfig:
    weights: str
    output_dir: str
    det_config: str
    det_weights: str
    bbox_thr: float
    nms_thr: float
    whole_image: bool
    force: bool
    manifest_path: str


def setup_mmpose() -> None:
    os.chdir(MMPOSE_ROOT)
    os.environ["PYTHONPATH"] = str(MMPOSE_ROOT)
    if str(MMPOSE_ROOT) not in sys.path:
        sys.path.insert(0, str(MMPOSE_ROOT))


def is_duplicate_copy(path: Path) -> bool:
    """Skip macOS/Windows duplicate files like ``forehand_drive(1).mov``."""
    return DUPLICATE_COPY_MARKER in path.stem


def collect_videos(limit: int | None = None) -> list[Path]:
    videos: list[Path] = []
    for root in R2_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in VIDEO_EXTS and not is_duplicate_copy(path):
                videos.append(path)
    videos.sort(key=lambda p: str(p.relative_to(DATASET_ROOT)))
    if limit is not None:
        videos = videos[:limit]
    return videos


def video_dir_to_skeleton_dir(name: str) -> str:
    if "-" not in name:
        raise ValueError(f"Video folder name has no '-' suffix: {name}")
    return name.replace("-", "-", 1)


def output_json_path(video_path: Path, output_root: Path) -> Path:
    rel = video_path.relative_to(DATASET_ROOT)
    parts = list(rel.parts)
    for i, part in enumerate(parts):
        if "-" in part:
            parts[i] = video_dir_to_skeleton_dir(part)
            break
    return output_root / Path(*parts).with_suffix(".json")


def append_manifest(path: Path, record: dict, lock: Any | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock is not None:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def summarize_instances(payload: list[dict]) -> dict:
    from collections import Counter

    per_frame = [len(fr.get("instances", [])) for fr in payload]
    counts = Counter(per_frame)
    return {
        "max_instances_per_frame": max(per_frame) if per_frame else 0,
        "frames_with_multiple_persons": sum(1 for n in per_frame if n > 1),
        "frames_with_zero_persons": counts.get(0, 0),
        "instance_count_distribution": {str(k): v for k, v in sorted(counts.items())},
    }


def parse_gpu_list(value: str) -> list[int]:
    ids = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    if not ids:
        raise ValueError("GPU list is empty")
    return ids


def shard_videos(videos: list[Path], shard_id: int, num_shards: int) -> list[Path]:
    return [v for i, v in enumerate(videos) if i % num_shards == shard_id]


def split_round_robin(videos: list[Path], n: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(n)]
    for i, video in enumerate(videos):
        buckets[i % n].append(video)
    return buckets


def log(msg: str, gpu_id: int | None = None) -> None:
    prefix = f"[GPU{gpu_id}] " if gpu_id is not None else ""
    print(f"{prefix}{msg}", flush=True)


def process_videos(
    videos: list[Path],
    cfg: WorkerConfig,
    device: str,
    gpu_id: int | None = None,
    manifest_lock: Any | None = None,
) -> tuple[int, int, int]:
    setup_mmpose()
    from mmpose.apis.inferencers import MMPoseInferencer

    output_root = Path(cfg.output_dir)
    manifest_path = Path(cfg.manifest_path)

    inferencer_kwargs: dict[str, Any] = dict(
        pose2d=str(RTMPOSE_CONFIG),
        pose2d_weights=cfg.weights,
        device=device,
        scope="mmpose",
    )
    if cfg.whole_image:
        inferencer_kwargs["det_model"] = "whole_image"
    else:
        inferencer_kwargs.update(
            det_model=cfg.det_config,
            det_weights=cfg.det_weights,
            det_cat_ids=0,
        )

    inferencer = MMPoseInferencer(**inferencer_kwargs)
    infer_args = {}
    if not cfg.whole_image:
        infer_args = {"bbox_thr": cfg.bbox_thr, "nms_thr": cfg.nms_thr}

    done = skipped = failed = 0
    total = len(videos)

    for idx, video_path in enumerate(videos, start=1):
        rel = str(video_path.relative_to(DATASET_ROOT))
        out_json = output_json_path(video_path, output_root)

        if out_json.exists() and not cfg.force:
            skipped += 1
            log(f"[{idx}/{total}] skip existing: {rel}", gpu_id)
            continue

        out_json.parent.mkdir(parents=True, exist_ok=True)
        log(f"[{idx}/{total}] processing: {rel}", gpu_id)
        started = time.time()
        status = "ok"
        err = ""
        frames = 0
        instance_stats: dict = {}

        try:
            for _ in inferencer(
                inputs=str(video_path),
                pred_out_dir=str(out_json.parent),
                **infer_args,
            ):
                pass
            if not out_json.exists():
                raise FileNotFoundError(f"Expected output not created: {out_json}")
            with out_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            frames = len(payload)
            instance_stats = summarize_instances(payload)
            done += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = str(exc)
            failed += 1
            log(f"  ERROR: {exc}", gpu_id)

        elapsed = round(time.time() - started, 2)
        record = {
            "video": rel,
            "output": str(out_json.relative_to(output_root)),
            "status": status,
            "frames": frames,
            "seconds": elapsed,
            "time": datetime.now().isoformat(timespec="seconds"),
            "gpu": gpu_id,
            "device": device,
            "det_mode": "whole_image" if cfg.whole_image else "rtmdet_multi",
            "instance_stats": instance_stats,
            "error": err,
        }
        append_manifest(manifest_path, record, manifest_lock)
        if status == "ok":
            max_inst = instance_stats.get("max_instances_per_frame", 0)
            multi = instance_stats.get("frames_with_multiple_persons", 0)
            log(
                f"  -> {frames} frames, max_persons={max_inst}, "
                f"multi_person_frames={multi}, {elapsed}s",
                gpu_id,
            )

    return done, skipped, failed


def worker_entry(
    gpu_id: int,
    video_strs: list[str],
    cfg: WorkerConfig,
    manifest_lock: Any | None,
) -> tuple[int, int, int]:
    # ， CUDA_VISIBLE_DEVICES  invalid device ordinal
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"
    videos = [Path(v) for v in video_strs]
    log(f"worker start, {len(videos)} videos on physical GPU {gpu_id} ({device})", gpu_id)
    return process_videos(videos, cfg, device, gpu_id=gpu_id, manifest_lock=manifest_lock)


def build_worker_config(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        weights=str(args.weights),
        output_dir=str(args.output_dir),
        det_config=str(args.det_config),
        det_weights=str(args.det_weights),
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        whole_image=args.whole_image,
        force=args.force,
        manifest_path=str(MANIFEST_PATH),
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not RTMPOSE_CONFIG.exists():
        raise SystemExit(f"MMPose config not found: {RTMPOSE_CONFIG}")
    if not args.whole_image:
        if not args.det_config.exists():
            raise SystemExit(f"Detector config not found: {args.det_config}")
        if not args.det_weights.exists():
            raise SystemExit(
                f"Detector weights not found: {args.det_weights}\n"
                "Download to skeleton/:\n"
                "  curl -L -o skeleton/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth "
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
            )
    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None:
            raise SystemExit("--shard-id and --num-shards must be used together")
        if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise SystemExit("Invalid --shard-id / --num-shards")
    if args.gpus and (args.shard_id is not None or args.device != "cuda:0"):
        raise SystemExit("Use either --gpus or --shard-id/--device, not both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RTMPose-L skeletons for Research 2")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0", help="Single-GPU mode device")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Multi-GPU mode, comma-separated ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Manual sharding: worker index (use with --num-shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Manual sharding: total workers (use with --shard-id)",
    )
    parser.add_argument("--det-config", type=Path, default=DEFAULT_DET_CONFIG)
    parser.add_argument("--det-weights", type=Path, default=DEFAULT_DET_WEIGHTS)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--whole-image", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = collect_videos(limit=args.limit)
    if not videos:
        raise SystemExit("No Research 2 videos found.")

    if args.shard_id is not None:
        videos = shard_videos(videos, args.shard_id, args.num_shards)

    print(f"Found {len(videos)} videos for this worker.")
    if args.dry_run:
        for video in videos:
            print(output_json_path(video, args.output_dir))
        return

    validate_args(args)
    cfg = build_worker_config(args)

    if not args.whole_image:
        print(f"Mode: RTMDet multi-person (bbox_thr={args.bbox_thr}, nms_thr={args.nms_thr})")
    else:
        print("Mode: whole_image (single person per frame)")

    t0 = time.time()

    if args.gpus:
        gpu_ids = parse_gpu_list(args.gpus)
        buckets = split_round_robin(videos, len(gpu_ids))
        print(f"Multi-GPU launch: {gpu_ids}")
        for gpu_id, bucket in zip(gpu_ids, buckets):
            print(f"  GPU {gpu_id}: {len(bucket)} videos")

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        manifest_lock = manager.Lock()
        processes = []
        results: list[Any] = []

        for gpu_id, bucket in zip(gpu_ids, buckets):
            video_strs = [str(v) for v in bucket]
            proc = ctx.Process(
                target=worker_entry,
                args=(gpu_id, video_strs, cfg, manifest_lock),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise SystemExit(
                    f"GPU worker failed with exit code {proc.exitcode}. "
                    "Check log for 'invalid device ordinal' or OOM."
                )

        print(f"All {len(gpu_ids)} GPU workers finished.")
    else:
        done, skipped, failed = process_videos(
            videos, cfg, args.device, gpu_id=None, manifest_lock=None
        )
        print(f"Finished: done={done}, skipped={skipped}, failed={failed}")

    total = round(time.time() - t0, 1)
    print(f"Total elapsed: {total}s")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
