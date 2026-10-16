#!/usr/bin/env python3
"""Extract action-segment frames once; training reads JPEGs via RawframeDataset.

Each instance -> cache/frames/{instance_id}/img_00001.jpg ...

Usage:
  cd /video
  python3 extract_frames.py
  python3 extract_frames.py --workers 8 --limit 100
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
    DATASET_ROOT,
    FRAME_FILENAME_TPL,
    FRAME_JPEG_QUALITY,
    FRAME_SHORT_EDGE,
    FRAME_START_INDEX,
    FRAMES_DIR,
    INSTANCES_CSV,
)
from dataset_utils import (
    build_all_session_indexes,
    load_instances,
    parse_frame_range,
    resolve_video_path,
)


def resize_short_edge(img: np.ndarray, short_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) <= short_edge:
        return img
    if h < w:
        new_h = short_edge
        new_w = int(w * short_edge / h)
    else:
        new_w = short_edge
        new_h = int(h * short_edge / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clamp_frame_range(start: int, end: int, total: int) -> tuple[int, int]:
    if total <= 0:
        return 0, -1
    start = max(0, min(start, total - 1))
    end = max(start, min(end, total - 1))
    return start, end


def opencv_last_readable_frame(cap: cv2.VideoCapture, total: int) -> int:
    """OpenCV CAP_PROP_FRAME_COUNT is often 1 higher than the last readable index."""
    for idx in range(max(total - 1, 0), -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, _ = cap.read()
        if ok:
            return idx
    return -1


def extract_one(task: dict) -> dict:
    instance_id = task["instance_id"]
    video_path = Path(task["video_path"])
    req_start = task["start_frame"]
    req_end = task["end_frame"]
    out_dir = Path(task["out_dir"])
    short_edge = task["short_edge"]
    jpeg_quality = task["jpeg_quality"]

    expected = req_end - req_start + 1
    done_marker = out_dir / ".done"
    stale = False
    if done_marker.exists():
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("video_path") != str(video_path):
                stale = True
            elif task.get("video_annotation_json") and meta.get("video_annotation_json") != task.get("video_annotation_json"):
                stale = True
            elif meta.get("short_edge") != short_edge:
                stale = True
            elif (
                meta.get("num_frames", 0) > 0
            ):
                return {
                    "instance_id": instance_id,
                    "status": "skip",
                    "num_frames": meta.get("num_frames", expected),
                }

    if stale and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    start = end = 0
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total = len(vr)
        start, end = clamp_frame_range(req_start, req_end, total)
        if end < start:
            raise RuntimeError("empty range")
        idxs = list(range(start, end + 1))
        batch = vr.get_batch(idxs).asnumpy()
        frames = [batch[i] for i in range(batch.shape[0])]
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"instance_id": instance_id, "status": "fail", "error": "open failed"}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame = opencv_last_readable_frame(cap, total)
        if last_frame < 0:
            cap.release()
            return {"instance_id": instance_id, "status": "fail", "error": "no frames"}
        start, end = clamp_frame_range(req_start, req_end, last_frame + 1)
        frames = []
        for frame_idx in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    if not frames:
        return {"instance_id": instance_id, "status": "fail", "error": "no frames"}

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    for i, frame in enumerate(frames, start=FRAME_START_INDEX):
        if short_edge > 0:
            frame = resize_short_edge(frame, short_edge)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path = out_dir / FRAME_FILENAME_TPL.format(i)
        cv2.imwrite(str(out_path), frame_bgr, encode_params)

    num_frames = len(frames)
    meta = {
        "instance_id": instance_id,
        "video_annotation_json": task.get("video_annotation_json", ""),
        "video_path": str(video_path),
        "requested_start_frame": req_start,
        "requested_end_frame": req_end,
        "start_frame": start,
        "end_frame": end,
        "num_frames": num_frames,
        "short_edge": short_edge,
        "clamped_to_video_end": req_end > end or req_start > start,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    done_marker.write_text("", encoding="utf-8")
    return {"instance_id": instance_id, "status": "ok", "num_frames": num_frames}


def collect_tasks(limit: int | None = None) -> list[dict]:
    rows = load_instances(require_valid_imu=True)
    session_index = build_all_session_indexes(rows)
    seen: set[str] = set()
    tasks: list[dict] = []

    for row in rows:
        iid = row["instance_id"]
        if iid in seen:
            continue
        video_path = resolve_video_path(row, session_index)
        frame_range = parse_frame_range(row)
        if video_path is None or frame_range is None:
            continue
        seen.add(iid)
        start, end = frame_range
        tasks.append(
            {
                "instance_id": iid,
                "video_path": str(video_path),
                "video_annotation_json": row.get("video_annotation_json", ""),
                "start_frame": start,
                "end_frame": end,
                "out_dir": str(FRAMES_DIR / iid),
                "short_edge": FRAME_SHORT_EDGE,
                "jpeg_quality": FRAME_JPEG_QUALITY,
            }
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def find_wrong_video_instances() -> list[dict]:
    """Instances whose cache was extracted from a different video file."""
    rows = load_instances(require_valid_imu=True)
    session_index = build_all_session_indexes(rows)
    task_map = {t["instance_id"]: t for t in collect_tasks()}
    wrong: list[dict] = []

    for row in rows:
        iid = row["instance_id"]
        if iid not in task_map:
            continue
        task = task_map[iid]
        meta_path = FRAMES_DIR / iid / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("video_path") != task["video_path"]:
            wrong.append(
                {
                    "instance_id": iid,
                    "cached_video": meta.get("video_path"),
                    "expected_video": task["video_path"],
                    "cached_frames": meta.get("num_frames"),
                    "clamped": meta.get("clamped_to_video_end", False),
                }
            )
    return wrong


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Beiyu action clips to JPEG frames")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--limit", type=int, default=0, help="Debug: only process N instances")
    parser.add_argument("--short-edge", type=int, default=FRAME_SHORT_EDGE)
    parser.add_argument("--jpeg-quality", type=int, default=FRAME_JPEG_QUALITY)
    parser.add_argument(
        "--repair-wrong-video",
        action="store_true",
        help="Only re-extract instances whose cache points to the wrong video file",
    )
    parser.add_argument(
        "--list-wrong-video",
        action="store_true",
        help="Print wrong-video instances and exit",
    )
    args = parser.parse_args()

    if args.list_wrong_video:
        wrong = find_wrong_video_instances()
        report_path = FRAMES_DIR / "wrong_video_report.json"
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(wrong, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrong-video instances: {len(wrong)}")
        print(f"report: {report_path}")
        for item in wrong[:10]:
            print(f"  {item['instance_id']}")
            print(f"    cached:   {item['cached_video']}")
            print(f"    expected: {item['expected_video']}")
        if len(wrong) > 10:
            print(f"  ... and {len(wrong) - 10} more")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    if args.repair_wrong_video:
        wrong_ids = {item["instance_id"] for item in find_wrong_video_instances()}
        all_tasks = collect_tasks()
        tasks = [t for t in all_tasks if t["instance_id"] in wrong_ids]
        manifest_path = FRAMES_DIR / "repair_manifest.jsonl"
        print(f"repair wrong-video instances: {len(tasks)}")
    else:
        tasks = collect_tasks(limit=args.limit or None)
        manifest_path = FRAMES_DIR / "extract_manifest.jsonl"
    for task in tasks:
        task["short_edge"] = args.short_edge
        task["jpeg_quality"] = args.jpeg_quality

    print(f"instances to extract: {len(tasks)}")
    print(f"output: {FRAMES_DIR}")
    print(f"short_edge={args.short_edge} ({'original resolution' if args.short_edge == 0 else 'px'}), workers={args.workers}")

    ok = skip = fail = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        if args.workers <= 1:
            results = [extract_one(task) for task in tasks]
        else:
            results = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(extract_one, task): task for task in tasks}
                for fut in as_completed(futures):
                    results.append(fut.result())

        for rec in results:
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["status"] == "ok":
                ok += 1
            elif rec["status"] == "skip":
                skip += 1
            else:
                fail += 1
                print(f"FAIL {rec['instance_id']}: {rec.get('error')}")

    index = {
        "dataset_root": str(DATASET_ROOT),
        "frames_dir": str(FRAMES_DIR),
        "filename_tmpl": FRAME_FILENAME_TPL,
        "start_index": FRAME_START_INDEX,
        "short_edge": args.short_edge,
        "total_instances": len(tasks),
        "ok": ok,
        "skip": skip,
        "fail": fail,
    }
    (FRAMES_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"done: ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
