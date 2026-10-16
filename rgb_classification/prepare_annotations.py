#!/usr/bin/env python3
"""Generate MMAction2 annotation files for Beiyu video benchmark.

Uses the same instance set and split rules as training/dataset.py (IMU benchmark):
  - valid 11-class instances with has_valid_imu=True
  - xsub / xacq / xacq_conservative protocols
  - val: xsub -> S003,S010 ; xacq -> date 1029

Annotation line format (4 fields, space-separated):
  relative_video_path start_frame end_frame label_id

Example:
  2：-/10--/.../backhand_drive.mp4 598 635 9

Usage:
  cd /video
  python3 prepare_annotations.py
  python3 prepare_annotations.py --protocol xsub
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import ANNOTATIONS_DIR, DATASET_ROOT, FRAME_SHORT_EDGE, FRAMES_DIR, PROTOCOLS
from dataset_utils import (
    assign_subset,
    build_all_session_indexes,
    load_action_labels,
    load_instances,
    load_split,
    parse_frame_range,
    resolve_video_path,
)


def cached_num_frames(instance_id: str) -> int | None:
    """Use extracted JPEG count when frame cache exists."""
    meta_path = FRAMES_DIR / instance_id / "meta.json"
    done_path = FRAMES_DIR / instance_id / ".done"
    if not (meta_path.exists() and done_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("short_edge") != FRAME_SHORT_EDGE:
        return None
    num_frames = int(meta.get("num_frames") or 0)
    return num_frames if num_frames > 0 else None


def prepare_protocol(protocol: str, dry_run: bool = False) -> dict:
    split = load_split(protocol)
    code_to_id, class_names = load_action_labels()
    rows = load_instances(require_valid_imu=True)
    session_index = build_all_session_indexes(rows)

    buckets = {"train": [], "val": [], "test": []}
    skipped = Counter()

    for row in rows:
        subset = assign_subset(row, protocol, split)
        if subset is None:
            skipped["unassigned"] += 1
            continue

        video_path = resolve_video_path(row, session_index)
        if video_path is None:
            skipped["missing_video"] += 1
            continue

        frame_range = parse_frame_range(row)
        if frame_range is None:
            skipped["bad_frames"] += 1
            continue

        action_id = code_to_id.get(row["action_code"])
        if action_id is None:
            skipped["bad_label"] += 1
            continue

        rel_path = video_path.relative_to(DATASET_ROOT).as_posix()
        start_frame, end_frame = frame_range
        num_frames = end_frame - start_frame + 1
        cached_frames = cached_num_frames(row["instance_id"])
        if cached_frames is not None:
            num_frames = cached_frames
        buckets[subset].append(
            {
                "line": f"{rel_path} {start_frame} {end_frame} {action_id}",
                "frame_line": f"{row['instance_id']} {num_frames} {action_id}",
                "instance_id": row["instance_id"],
                "video_path": str(video_path),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "label": action_id,
                "cn_name": row["cn_name"],
            }
        )

    out_dir = ANNOTATIONS_DIR / protocol
    stats = {
        "protocol": protocol,
        "class_names": class_names,
        "counts": {k: len(v) for k, v in buckets.items()},
        "skipped": dict(skipped),
    }

    if dry_run:
        return stats

    out_dir.mkdir(parents=True, exist_ok=True)
    for subset, items in buckets.items():
        ann_path = out_dir / f"{subset}.txt"
        with ann_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(item["line"] + "\n")
        frame_ann_path = out_dir / f"{subset}_frames.txt"
        with frame_ann_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(item["frame_line"] + "\n")

    manifest = {
        "protocol": protocol,
        "data_root": str(DATASET_ROOT),
        "class_names": class_names,
        "counts": stats["counts"],
        "skipped": stats["skipped"],
        "subsets": {
            subset: [
                {k: v for k, v in item.items() if k != "line"}
                for item in items
            ]
            for subset, items in buckets.items()
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stats["annotation_dir"] = str(out_dir)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Beiyu video MMAction2 annotations")
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        action="append",
        help="Protocol to generate (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or PROTOCOLS
    for protocol in protocols:
        stats = prepare_protocol(protocol, dry_run=args.dry_run)
        print(f"[{protocol}] train={stats['counts']['train']} val={stats['counts']['val']} "
              f"test={stats['counts']['test']} skipped={stats['skipped']}")
        if not args.dry_run:
            print(f"  -> {stats.get('annotation_dir')}")


if __name__ == "__main__":
    main()
