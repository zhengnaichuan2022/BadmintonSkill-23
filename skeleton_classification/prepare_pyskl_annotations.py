#!/usr/bin/env python3
"""Convert cleaned RTMPose JSON into PYSKL pickle annotations for Beiyu benchmark.

Split assignment is identical to video/prepare_annotations.py and training/dataset.py:
  - metadata/splits/*.json + video.dataset_utils.assign_subset
  - is_valid_class + has_valid_imu filters (same instance pool as IMU/video)
  - all views: single + front + rear

Usage:
  cd /skeleton
  python3 prepare_pyskl_annotations.py
  python3 prepare_pyskl_annotations.py --protocol xsub
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config import ANNOTATIONS_DIR, DEFAULT_IMG_SHAPE, PROTOCOLS, VIDEO_ANNOTATIONS_DIR
from pose_clean import extract_clip_from_video_json, infer_img_shape
from skeleton_utils import (
    assign_subset,
    build_all_session_indexes,
    load_action_labels,
    load_instances,
    load_split,
    parse_frame_range,
    resolve_skeleton_json,
)


def load_frames_by_id(json_path: Path, cache: dict) -> dict:
    key = str(json_path)
    if key not in cache:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        cache[key] = {int(fr["frame_id"]): fr for fr in payload}
    return cache[key]


def video_manifest_subset_map(protocol: str) -> Dict[str, str]:
    """instance_id -> train/val/test from video/annotations (full IMU/video pool)."""
    manifest_path = VIDEO_ANNOTATIONS_DIR / protocol / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Video manifest missing: {manifest_path}\n"
            f"Run: cd video && python3 prepare_annotations.py --protocol {protocol}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for subset, items in manifest["subsets"].items():
        for item in items:
            mapping[item["instance_id"]] = subset
    return mapping


def verify_against_video(protocol: str, split_ids: Dict[str, List[str]]) -> None:
    """Each skeleton instance must land in the same subset as video/IMU."""
    expected = video_manifest_subset_map(protocol)
    mismatches: List[str] = []
    for subset, ids in split_ids.items():
        for iid in ids:
            exp_subset = expected.get(iid)
            if exp_subset is None:
                mismatches.append(f"{iid}: not in video single-view manifest")
            elif exp_subset != subset:
                mismatches.append(f"{iid}: skeleton={subset} video={exp_subset}")
    if mismatches:
        raise RuntimeError(
            f"[{protocol}] split alignment failed ({len(mismatches)} errors), "
            f"first: {mismatches[0]}"
        )


def prepare_protocol(protocol: str, dry_run: bool = False) -> dict:
    split = load_split(protocol)
    code_to_id, class_names = load_action_labels()
    rows = load_instances(require_valid_imu=True)
    session_index = build_all_session_indexes(rows)

    split_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    annotations: List[dict] = []
    skipped = Counter()
    json_cache: dict = {}
    stats_counter = Counter()

    bar = tqdm(
        rows,
        desc=f"[{protocol}] processing",
        unit="inst",
        total=len(rows),
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for row in bar:
        subset = assign_subset(row, protocol, split)
        if subset is None:
            skipped["unassigned"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        skel_path = resolve_skeleton_json(row, session_index)
        if skel_path is None:
            skipped["missing_skeleton"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        frame_range = parse_frame_range(row)
        if frame_range is None:
            skipped["bad_frames"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        action_id = code_to_id.get(row["action_code"])
        if action_id is None:
            skipped["bad_label"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        start_frame, end_frame = frame_range
        try:
            frames_by_id = load_frames_by_id(skel_path, json_cache)
            keypoints, scores, clip_stats = extract_clip_from_video_json(
                frames_by_id, start_frame, end_frame
            )
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            skipped["bad_skeleton"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        if clip_stats["total_frames"] < 2:
            skipped["too_short"] += 1
            bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))
            continue

        img_shape = infer_img_shape(keypoints, scores, default=DEFAULT_IMG_SHAPE)
        instance_id = row["instance_id"]

        annotations.append({
            "frame_dir": instance_id,
            "total_frames": clip_stats["total_frames"],
            "img_shape": img_shape,
            "original_shape": img_shape,
            "label": action_id,
            "keypoint": keypoints[np.newaxis, ...],
            "keypoint_score": scores[np.newaxis, ...],
        })
        split_ids[subset].append(instance_id)
        stats_counter["instances"] += 1
        stats_counter["multi_person_frames"] += clip_stats["multi_person_frames"]
        stats_counter["cleaned_frames"] += clip_stats["cleaned_frames"]
        bar.set_postfix(ok=stats_counter["instances"], skip=sum(skipped.values()))

    bar.close()

    tqdm.write(f"[{protocol}] verifying split alignment with video/IMU...")
    verify_against_video(protocol, split_ids)

    out_dir = ANNOTATIONS_DIR / protocol
    result = {
        "protocol": protocol,
        "class_names": class_names,
        "counts": {k: len(v) for k, v in split_ids.items()},
        "skipped": dict(skipped),
        "clip_stats": dict(stats_counter),
    }

    if dry_run:
        return result

    tqdm.write(f"[{protocol}] writing pickle ({len(annotations)} annotations)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "beiyu.pkl"
    payload = {"split": split_ids, "annotations": annotations}
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    video_manifest = VIDEO_ANNOTATIONS_DIR / protocol / "manifest.json"
    manifest = {
        "protocol": protocol,
        "pickle": str(pkl_path),
        "class_names": class_names,
        "counts": result["counts"],
        "skipped": result["skipped"],
        "clip_stats": result["clip_stats"],
        "keypoint_layout": "coco-17",
        "source": "skeleton/cleaned",
        "views": ["single", "front", "rear"],
        "split_assignment": "video/dataset_utils.assign_subset (same as IMU/video)",
        "aligned_with": str(video_manifest),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["pickle"] = str(pkl_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PYSKL pickle annotations for Beiyu skeleton benchmark")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or PROTOCOLS
    for i, protocol in enumerate(protocols, 1):
        if len(protocols) > 1:
            tqdm.write(f"=== protocol {i}/{len(protocols)}: {protocol} ===")
        stats = prepare_protocol(protocol, dry_run=args.dry_run)
        print(
            f"[{protocol}] train={stats['counts']['train']} val={stats['counts']['val']} "
            f"test={stats['counts']['test']} skipped={stats['skipped']} "
            f"(split aligned with video/IMU, all views)"
        )
        if stats.get("clip_stats"):
            print(f"  clip_stats={stats['clip_stats']}")
        if not args.dry_run:
            print(f"  -> {stats.get('pickle')}")


if __name__ == "__main__":
    main()
