#!/usr/bin/env python3
"""Run all additional dataloader examples on bundled samples."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "dataloader_examples"))

from benchmark_utils import get_dataset_root, load_protocol_recording_ids
from load_classification_annotation import load_classification_annotation
from load_imu_sample import load_imu_sample
from load_rgb_sample import load_rgb_sample
from load_skeleton_sample import load_skeleton_sample
from load_tad_annotation import load_tad_annotation


def main() -> None:
    root = get_dataset_root()
    manifest = root / "samples" / "samples_manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}")

    print("=== BadmintonSkill-23 dataloader smoke test ===\n")

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    cls_rows = [r for r in rows if r["type"] == "classification_preview"]
    tad_rows = [r for r in rows if r["type"] == "tad"]

    print(f"Classification preview samples ({len(cls_rows)}):")
    for row in cls_rows:
        sample_dir = root / "samples" / row["output_dir"]
        ann, ann_meta = load_classification_annotation(sample_dir)
        rgb, rgb_meta = load_rgb_sample(sample_dir, max_frames=5)
        skel, skel_meta = load_skeleton_sample(sample_dir)
        imu, _, imu_meta = load_imu_sample(sample_dir)

        assert ann_meta["preview_id"] == row["preview_id"]
        assert ann_meta["recording_id"] == row["recording_id"]
        assert ann_meta["segment_count"] == int(float(row["segment_count"]))
        assert rgb_meta["frame_count"] > 0
        assert skel_meta["frame_count"] > 0
        assert imu_meta["frame_count"] > 0

        print(
            f"  OK {row['preview_id']}: recording={row['recording_id']} "
            f"segments={ann_meta['segment_count']} rgb={rgb.shape} skel={skel.shape} "
            f"imu={imu.shape} class={ann_meta['class_name']}"
        )

    print(f"\nTAD samples ({len(tad_rows)}):")
    for row in tad_rows:
        sample_dir = root / "samples" / row["output_dir"]
        ann, segments, meta = load_tad_annotation(sample_dir)
        rgb, rgb_meta = load_rgb_sample(sample_dir, max_frames=5)
        imu, _, imu_meta = load_imu_sample(sample_dir)

        assert meta["video_id"] == row["recording_id"]
        assert len(segments) == int(float(row["segment_count"]))
        assert rgb_meta["frame_count"] > 0
        assert imu_meta["frame_count"] > 0

        print(
            f"  OK {meta['video_id']}: segments={meta['segment_count']} "
            f"rgb={rgb.shape} imu={imu.shape}"
        )

    print("\nProtocol files (xsub train, first 3 recording IDs):")
    train_ids = load_protocol_recording_ids("xsub", "train")[:3]
    print(f"  {train_ids}")

    print("\nAll dataloader checks passed.")


if __name__ == "__main__":
    main()
