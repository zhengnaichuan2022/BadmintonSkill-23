"""Skeleton helpers. Split assignment is delegated to video/dataset_utils (same as IMU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

from config import CLEANED_ROOT, DATASET_ROOT, SKELETON_VIEW

VIDEO_UTILS = DATASET_ROOT / "video" / "dataset_utils.py"
_spec = importlib.util.spec_from_file_location("beiyu_video_utils", VIDEO_UTILS)
_video_utils = importlib.util.module_from_spec(_spec)
sys.modules["beiyu_video_utils"] = _video_utils
assert _spec.loader is not None
_spec.loader.exec_module(_video_utils)

# Re-export split helpers — must stay identical to video/IMU benchmark
assign_subset = _video_utils.assign_subset
load_split = _video_utils.load_split
load_action_labels = _video_utils.load_action_labels
parse_frame_range = _video_utils.parse_frame_range
resolve_video_path = _video_utils.resolve_video_path
build_all_session_indexes = _video_utils.build_all_session_indexes
stem_key = _video_utils.stem_key


def load_instances(require_valid_imu: bool = True, view: str | None = SKELETON_VIEW) -> List[dict]:
    """Same instance pool as video/IMU; optional view filter (default: all views)."""
    rows = _video_utils.load_instances(require_valid_imu=require_valid_imu)
    if view:
        rows = [row for row in rows if row.get("view") == view]
    return rows


def video_dir_to_skeleton_dir(name: str) -> str:
    if "-" not in name:
        raise ValueError(f"video folder has no '-' suffix: {name}")
    return name.replace("-", "-", 1)


def skeleton_json_from_video(video_path: Path) -> Path:
    rel = video_path.relative_to(DATASET_ROOT)
    parts = list(rel.parts)
    for i, part in enumerate(parts):
        if "-" in part:
            parts[i] = video_dir_to_skeleton_dir(part)
            break
    return CLEANED_ROOT / Path(*parts).with_suffix(".json")


def _candidate_json_names(row: dict, video_path: Path) -> List[str]:
    names = [video_path.with_suffix(".json").name, f"{video_path.stem}.json"]
    subject = row.get("subject_name", "")
    date = row.get("date", "")
    cn = row.get("cn_name", "")
    if subject and date and cn:
        names.extend([
            f"{subject}-{date}-{cn}.json",
            f"{subject}-{date}-{cn}-1.json",
            f"{subject}-{date}-{cn}-2.json",
        ])
    return names


def resolve_skeleton_json(row: dict, session_index: dict) -> Optional[Path]:
    video_path = resolve_video_path(row, session_index)
    if video_path is None:
        return None

    direct = skeleton_json_from_video(video_path)
    if direct.is_file():
        return direct

    skel_dir = direct.parent
    if not skel_dir.is_dir():
        return None

    target_key = stem_key(video_path.name)
    for name in _candidate_json_names(row, video_path):
        path = skel_dir / name
        if path.is_file():
            return path

    matches = [
        path for path in sorted(skel_dir.glob("*.json"))
        if stem_key(path.name) == target_key
    ]
    if matches:
        matches.sort(key=lambda p: ("(1)" in p.name, p.name))
        return matches[0]

    cn = row.get("cn_name", "")
    if cn:
        for path in sorted(skel_dir.glob("*.json")):
            if cn in path.stem:
                return path
    return None
