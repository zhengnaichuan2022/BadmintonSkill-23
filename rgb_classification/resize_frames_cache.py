#!/usr/bin/env python3
"""Downscale existing JPEG frame cache (e.g. 1080p -> short_edge=256).

Reads cache/frames/{instance_id}/img_*.jpg and writes cache/frames_256/...
Frame indices and annotation files stay unchanged.

Usage:
  cd /video
  python3 resize_frames_cache.py --workers 8
  python3 resize_frames_cache.py --limit 10 --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from config import (
    CACHE_DIR,
    FRAME_FILENAME_TPL,
    FRAME_JPEG_QUALITY,
    FRAME_SHORT_EDGE_TRAIN,
    FRAMES_256_DIR,
    FRAMES_DIR,
)
from extract_frames import resize_short_edge


def resize_instance(task: dict) -> dict:
    instance_id = task["instance_id"]
    src_dir = Path(task["src_dir"])
    dst_dir = Path(task["dst_dir"])
    short_edge = task["short_edge"]
    jpeg_quality = task["jpeg_quality"]
    force = task["force"]

    meta_src = src_dir / "meta.json"
    if not meta_src.exists():
        return {"instance_id": instance_id, "status": "fail", "error": "missing meta.json"}

    src_meta = json.loads(meta_src.read_text(encoding="utf-8"))
    num_expected = int(src_meta.get("num_frames", 0))
    jpgs = sorted(src_dir.glob("img_*.jpg"))
    if num_expected <= 0:
        num_expected = len(jpgs)
    if len(jpgs) != num_expected:
        return {
            "instance_id": instance_id,
            "status": "fail",
            "error": f"frame count mismatch: jpg={len(jpgs)} meta={num_expected}",
        }

    done_marker = dst_dir / ".done"
    if done_marker.exists() and not force:
        meta_dst = dst_dir / "meta.json"
        if meta_dst.exists():
            dst_meta = json.loads(meta_dst.read_text(encoding="utf-8"))
            if (
                dst_meta.get("short_edge") == short_edge
                and dst_meta.get("num_frames") == num_expected
                and dst_meta.get("source_dir") == str(src_dir)
            ):
                return {
                    "instance_id": instance_id,
                    "status": "skip",
                    "num_frames": num_expected,
                }

    if force and dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            return {"instance_id": instance_id, "status": "fail", "error": f"read failed: {jpg.name}"}
        if short_edge > 0:
            img = resize_short_edge(img, short_edge)
        out_path = dst_dir / jpg.name
        if not cv2.imwrite(str(out_path), img, encode_params):
            return {"instance_id": instance_id, "status": "fail", "error": f"write failed: {out_path.name}"}

    dst_meta = dict(src_meta)
    dst_meta["short_edge"] = short_edge
    dst_meta["source_dir"] = str(src_dir)
    dst_meta["resized_from_short_edge"] = src_meta.get("short_edge", 0)
    (dst_dir / "meta.json").write_text(
        json.dumps(dst_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    done_marker.write_text("", encoding="utf-8")
    return {"instance_id": instance_id, "status": "ok", "num_frames": num_expected}


def collect_tasks(
    src_root: Path,
    dst_root: Path,
    short_edge: int,
    jpeg_quality: int,
    force: bool,
    limit: int | None,
) -> list[dict]:
    tasks: list[dict] = []
    if not src_root.is_dir():
        return tasks
    for child in sorted(src_root.iterdir()):
        if not child.is_dir():
            continue
        meta = child / "meta.json"
        done = child / ".done"
        if not meta.exists() and not done.exists():
            continue
        tasks.append(
            {
                "instance_id": child.name,
                "src_dir": str(child),
                "dst_dir": str(dst_root / child.name),
                "short_edge": short_edge,
                "jpeg_quality": jpeg_quality,
                "force": force,
            }
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize existing frame cache to short_edge=256")
    parser.add_argument("--src", type=Path, default=FRAMES_DIR, help="Source cache (default: cache/frames)")
    parser.add_argument("--dst", type=Path, default=FRAMES_256_DIR, help="Output cache (default: cache/frames_256)")
    parser.add_argument("--short-edge", type=int, default=FRAME_SHORT_EDGE_TRAIN)
    parser.add_argument("--jpeg-quality", type=int, default=FRAME_JPEG_QUALITY)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-process even if .done exists")
    args = parser.parse_args()

    dst_root = args.dst
    dst_root.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(
        args.src,
        dst_root,
        args.short_edge,
        args.jpeg_quality,
        args.force,
        args.limit or None,
    )
    if not tasks:
        raise SystemExit(f"No instances found under {args.src}")

    manifest_path = dst_root / "resize_manifest.jsonl"
    print(f"instances: {len(tasks)}")
    print(f"src: {args.src}")
    print(f"dst: {dst_root}")
    print(f"short_edge={args.short_edge}, workers={args.workers}")

    ok = skip = fail = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        if args.workers <= 1:
            results = [resize_instance(t) for t in tasks]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(resize_instance, t): t for t in tasks}
                for fut in as_completed(futures):
                    results.append(fut.result())

        for rec in results:
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st = rec["status"]
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
                print(f"FAIL {rec['instance_id']}: {rec.get('error')}")

    index = {
        "source_frames_dir": str(args.src),
        "frames_dir": str(dst_root),
        "filename_tmpl": FRAME_FILENAME_TPL,
        "short_edge": args.short_edge,
        "total_instances": len(tasks),
        "ok": ok,
        "skip": skip,
        "fail": fail,
    }
    (dst_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done: ok={ok} skip={skip} fail={fail}")
    print(f"Use for training: FRAMES_CACHE=frames_256 bash run_prepare.sh")


if __name__ == "__main__":
    main()
