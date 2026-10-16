#!/usr/bin/env python3
"""Convert Beiyu IMU stroke annotations to OpenTAD TAD format (IMU time axis).

One entry per IMU recording session (grouped by ``video_annotation_json``).
Segments use ``imu_start_time`` / ``imu_end_time``; ``frame`` = total IMU samples.

Usage:
  cd OpenTAD-main
  python3 tools/prepare_data/beiyu/convert_imu_annotations.py
  python3 tools/prepare_data/beiyu/convert_imu_annotations.py --protocol xsub
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OPENTAD_ROOT = Path(__file__).resolve().parents[3]


def _load_opentad_config():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "opentad_beiyu_config", OPENTAD_ROOT / "config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load_opentad_config()
ACTION_LABELS_CSV = cfg.ACTION_LABELS_CSV
ANNOTATIONS_DIR = cfg.ANNOTATIONS_DIR
DATASET_ROOT = cfg.DATASET_ROOT
INSTANCES_CSV = cfg.INSTANCES_CSV
PROTOCOLS = cfg.PROTOCOLS
SPLIT_FILES = cfg.SPLIT_FILES
XACQ_VAL_DATES = cfg.XACQ_VAL_DATES
XSUB_VAL_SUBJECTS = cfg.XSUB_VAL_SUBJECTS

SUBSET_MAP = {"train": "training", "val": "validation", "test": "testing"}


def load_action_names() -> list[str]:
    rows = []
    with ACTION_LABELS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["action_code"] and row["action_code"] != "AXX":
                rows.append((int(row["action_id"]), row["cn_name"]))
    rows.sort(key=lambda x: x[0])
    return [name for _, name in rows]


def load_instances() -> list[dict]:
    out = []
    with INSTANCES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_valid_class"] != "True":
                continue
            if row["has_valid_imu"] != "True":
                continue
            if row["is_ambiguous"] == "True":
                continue
            out.append(row)
    return out


def assign_subset(row: dict, protocol: str, split: dict) -> str | None:
    if protocol == "xsub":
        sid = row["subject_id"]
        if sid in split["test_subjects"]:
            return "test"
        if sid in XSUB_VAL_SUBJECTS:
            return "val"
        if sid in split["train_subjects"]:
            return "train"
        return None
    if protocol == "xacq":
        date = row["date"]
        if date in split["test_dates"]:
            return "test"
        if date in XACQ_VAL_DATES:
            return "val"
        if date in split["train_dates"]:
            return "train"
        return None
    raise ValueError(protocol)


def session_key(row: dict) -> str:
    ann = (row.get("video_annotation_json") or "").strip()
    if ann:
        return ann
    sensor = (row.get("sensor_file") or "").strip()
    if sensor:
        return sensor
    return f"{row['subject_id']}_{row['session_id']}_{row['cn_name']}"


def video_id_from_row(row: dict) -> str:
    ann = (row.get("video_annotation_json") or "").strip()
    if ann:
        rel = Path(ann).as_posix()
        rel = rel.replace("2：-/", "F/")
        rel = rel.replace("2：-/", "S/")
        rel = rel.replace("/", "__")
        rel = re.sub(r"__+", "__", rel)
        return rel.replace("_annotations.json", "")
    sensor = (row.get("sensor_file") or "").strip()
    if sensor:
        rel = Path(sensor).as_posix()
        rel = rel.replace("2：-/", "F/")
        rel = rel.replace("2：-/", "S/")
        rel = rel.replace("/", "__")
        rel = re.sub(r"__+", "__", rel)
        return Path(rel).parent.as_posix().replace("/", "__")
    raise ValueError(f"cannot derive video_id for subject={row.get('subject_id')}")


def convert_protocol(protocol: str, dry_run: bool = False) -> dict:
    split = json.loads(SPLIT_FILES[protocol].read_text(encoding="utf-8"))
    rows = load_instances()
    class_names = load_action_names()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[session_key(row)].append(row)

    database = {}
    stats = Counter()
    skipped = Counter()

    for ann_json, items in sorted(groups.items()):
        row0 = items[0]
        subset = assign_subset(row0, protocol, split)
        if subset is None:
            skipped["unassigned"] += 1
            continue

        imu_csv = (row0.get("sensor_file") or "").strip()
        imu_path = DATASET_ROOT / imu_csv
        if not imu_csv or not imu_path.is_file():
            skipped["missing_imu"] += 1
            continue

        try:
            fps = float(row0["imu_sample_rate"])
            total_samples = int(float(row0["imu_samples_total"]))
        except (TypeError, ValueError):
            skipped["bad_meta"] += 1
            continue

        if fps <= 0 or total_samples <= 0:
            skipped["bad_meta"] += 1
            continue

        duration = total_samples / fps
        vid = video_id_from_row(row0)
        annotations = []
        max_end_time = 0.0
        max_end_sample = 0

        for item in items:
            label = item["cn_name"]
            if label not in class_names:
                skipped["bad_label"] += 1
                continue
            try:
                t0 = float(item["imu_start_time"])
                t1 = float(item["imu_end_time"])
            except (TypeError, ValueError):
                skipped["bad_time"] += 1
                continue
            if t1 <= t0:
                continue
            annotations.append({"label": label, "segment": [t0, t1]})
            max_end_time = max(max_end_time, t1)
            try:
                max_end_sample = max(max_end_sample, int(item["imu_end_sample"]))
            except (TypeError, ValueError):
                pass

        if not annotations:
            skipped["empty"] += 1
            continue

        if max_end_sample > total_samples:
            total_samples = max_end_sample
            duration = total_samples / fps

        try:
            scale = float(row0.get("sensor_scale_factor") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0

        database[vid] = {
            "subset": SUBSET_MAP[subset],
            "duration": round(duration, 4),
            "frame": int(total_samples),
            "fps": round(fps, 3),
            "imu_csv": imu_csv,
            "sensor_scale_factor": scale,
            "video_annotation_json": ann_json,
            "annotations": annotations,
        }
        stats[subset] += 1

    out_dir = ANNOTATIONS_DIR / "imu" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    anno_path = out_dir / "anno.json"
    class_map_path = ANNOTATIONS_DIR / "category_idx.txt"
    payload = {
        "version": "beiyu_imu_v1",
        "protocol": protocol,
        "modality": "imu",
        "database": database,
    }

    if not dry_run:
        anno_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not class_map_path.is_file():
            class_map_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
        (out_dir / "video_list.txt").write_text("\n".join(sorted(database.keys())) + "\n", encoding="utf-8")

    return {
        "protocol": protocol,
        "sessions": len(database),
        "stats": dict(stats),
        "skipped": dict(skipped),
        "anno_path": str(anno_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Beiyu IMU → OpenTAD annotations")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or PROTOCOLS
    for protocol in protocols:
        info = convert_protocol(protocol, dry_run=args.dry_run)
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
