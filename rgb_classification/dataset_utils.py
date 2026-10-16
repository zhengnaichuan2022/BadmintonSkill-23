"""Shared helpers for the Beiyu video benchmark (same splits as IMU training)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    ACTION_LABELS_CSV,
    DATASET_ROOT,
    INSTANCES_CSV,
    SPLIT_FILES,
    XACQ_VAL_DATES,
    XSUB_VAL_SUBJECTS,
)

VIDEO_EXTS = {".mp4", ".mov", ".MP4", ".MOV"}
VIEW_DIR_HINTS = {
    "front": ("-", "--"),
    "rear": ("-", "--"),
    "single": ("-",),
}
# Known filename / label aliases from metadata typos or naming drift.
FILENAME_ALIASES = {
    "": "forehand_drive",
}
CN_MATCH_ALIASES = {
    "forehand_clear": "forehand_clear",
    "backhand_clear": "backhand_clear",
}


def load_action_labels() -> Tuple[Dict[str, int], List[str]]:
    rows = []
    with ACTION_LABELS_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = row["action_code"]
            if code and code != "AXX":
                rows.append((int(row["action_id"]), row["cn_name"], code))
    rows.sort(key=lambda x: x[0])
    code_to_id = {code: aid for aid, _, code in rows}
    class_names = [name for _, name, _ in rows]
    return code_to_id, class_names


def load_split(protocol: str) -> dict:
    return json.loads(SPLIT_FILES[protocol].read_text(encoding="utf-8"))


def load_instances(require_valid_imu: bool = True) -> List[dict]:
    rows = list(csv.DictReader(INSTANCES_CSV.open(encoding="utf-8")))
    out = []
    for row in rows:
        if row["is_valid_class"] != "True":
            continue
        if require_valid_imu and row["has_valid_imu"] != "True":
            continue
        out.append(row)
    return out


def assign_subset(row: dict, protocol: str, split: dict) -> Optional[str]:
    if protocol == "xsub":
        sid = row["subject_id"]
        if sid in split["test_subjects"]:
            return "test"
        if sid in XSUB_VAL_SUBJECTS:
            return "val"
        if sid in split["train_subjects"]:
            return "train"
        return None
    if protocol.startswith("xacq"):
        date = row["date"]
        if date in split["test_dates"]:
            return "test"
        if date in XACQ_VAL_DATES:
            return "val"
        if date in split["train_dates"]:
            return "train"
        return None
    raise ValueError(f"unknown protocol: {protocol}")


def normalize_video_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\.{2,}", ".", name)
    name = name.replace(".-mp4", "-")
    name = name.replace(".-mp4", "-")
    name = name.replace("..mov", ".mov")
    name = name.replace("..mp4", ".mp4")
    for src, dst in FILENAME_ALIASES.items():
        name = name.replace(src, dst)
    return name


def cn_match_variants(cn: str) -> List[str]:
    variants = [cn]
    if cn in CN_MATCH_ALIASES:
        variants.append(CN_MATCH_ALIASES[cn])
    if cn.endswith(""):
        variants.append(cn.replace("", ""))
    return variants


def stem_key(name: str) -> str:
    stem = Path(normalize_video_name(name)).stem.lower()
    stem = re.sub(r"\(\d+\)$", "", stem)
    stem = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", stem)
    return stem


def iter_session_video_dirs(session_dir: Path, view: str) -> List[Path]:
    dirs = [d for d in session_dir.glob("**") if d.is_dir()]
    if view in VIEW_DIR_HINTS:
        hinted = []
        for d in dirs:
            for hint in VIEW_DIR_HINTS[view]:
                if hint in d.name:
                    hinted.append(d)
                    break
        if hinted:
            return sorted(hinted)
    plain = [d for d in dirs if "" not in d.name]
    return sorted(plain or dirs)


def build_session_video_index(session_dir: Path) -> Dict[str, List[Path]]:
    """Map normalized stem -> candidate video paths within a session."""
    index: Dict[str, List[Path]] = {}
    for video_dir in session_dir.glob("**"):
        if not video_dir.is_dir() or "" in video_dir.name:
            continue
        for path in video_dir.iterdir():
            if path.is_file() and path.suffix in VIDEO_EXTS:
                key = stem_key(path.name)
                index.setdefault(key, []).append(path)
                index.setdefault(path.name, [path])
    return index


def candidate_names(row: dict) -> List[str]:
    names = []
    if row.get("video_filename"):
        names.append(row["video_filename"])
        names.append(normalize_video_name(row["video_filename"]))
    ann = row.get("video_annotation_json", "")
    if ann:
        stem = Path(ann).stem
        for suffix in ("_annotations", "_annotations_aligned", "-annotations"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        for ext in (".mp4", ".mov", ".MP4", ".MOV"):
            names.append(stem + ext)
    cn = row.get("cn_name", "")
    if cn:
        view = row.get("view", "")
        date = row.get("date", "")
        name = row.get("subject_name", "")
        for action_cn in cn_match_variants(cn):
            if view == "front":
                names.append(f"{name}-{date}-{action_cn}-.mp4")
            elif view == "rear":
                names.append(f"{name}-{date}-{action_cn}-.mov")
            else:
                names.append(f"{action_cn}.mov")
                names.append(f"{action_cn}.mp4")
                names.append(f"{action_cn}.MOV")
    out = []
    seen = set()
    for name in names:
        name = normalize_video_name(name)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def session_dir_from_row(row: dict) -> Optional[Path]:
    """Resolve session root even when video_annotation_json is missing."""
    for field in ("video_annotation_json", "imu_annotation_json"):
        rel = row.get(field, "").strip()
        if not rel:
            continue
        path = DATASET_ROOT / rel
        if path.exists():
            return path.parent.parent

    sensor = row.get("sensor_file", "").strip()
    if sensor:
        path = DATASET_ROOT / sensor
        if path.exists():
            return path.parent.parent.parent
    return None


def paired_video_dirs(session_dir: Path, ann_dir: Path, view: str) -> List[Path]:
    """Map  folder to the matching  folder(s) in the same session."""
    dirs: List[Path] = []
    if ann_dir.name.endswith(""):
        paired = session_dir / ann_dir.name.replace("", "", 1)
        if paired.is_dir():
            dirs.append(paired)
    dirs.extend(iter_session_video_dirs(session_dir, view))
    out: List[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if path.is_dir() and "" not in path.name and key not in seen:
            seen.add(key)
            out.append(path)
    return out


def candidate_video_names(row: dict, ann_path: Path) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = normalize_video_name(name)
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    try:
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        add(str(payload.get("video_filename", "")))
    except (OSError, json.JSONDecodeError):
        pass

    add(row.get("video_filename", ""))

    stem = ann_path.stem
    for suffix in ("_annotations", "_annotations_aligned", "-annotations"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for ext in (".mp4", ".mov", ".MP4", ".MOV"):
        add(stem + ext)
        add(Path(stem).name + ext)

    view = row.get("view", "")
    if view in {"front", "rear"}:
        hint = "-" if view == "front" else "-"
        expanded: List[str] = []
        for name in list(names):
            if hint not in name:
                base = Path(name)
                expanded.append(base.stem + hint + base.suffix)
        for name in expanded:
            add(name)

    alias_expanded: List[str] = []
    for name in list(names):
        if "" in name:
            alias_expanded.append(name.replace("", ""))
        for src, dst in FILENAME_ALIASES.items():
            if src in name:
                alias_expanded.append(name.replace(src, dst))
    for name in alias_expanded:
        add(name)

    return names


def resolve_video_from_annotation(row: dict) -> Optional[Path]:
    """Resolve video strictly from the instance's video_annotation_json session."""
    ann_rel = row.get("video_annotation_json", "").strip()
    if not ann_rel:
        return None

    ann_path = DATASET_ROOT / ann_rel
    if not ann_path.exists():
        return None

    session_dir = ann_path.parent.parent
    ann_dir = ann_path.parent
    search_dirs = paired_video_dirs(session_dir, ann_dir, row.get("view", "single"))
    if not search_dirs:
        return None

    names = candidate_video_names(row, ann_path)
    for video_dir in search_dirs:
        for name in names:
            direct = video_dir / name
            if direct.is_file() and direct.suffix in VIDEO_EXTS:
                return direct

        for name in names:
            target_key = stem_key(name)
            matches = []
            for path in sorted(video_dir.iterdir()):
                if path.suffix not in VIDEO_EXTS:
                    continue
                if stem_key(path.name) == target_key:
                    matches.append(path)
            if matches:
                matches.sort(key=lambda p: ("(1)" in p.name, p.name))
                return matches[0]
    return None


def resolve_video_path(row: dict, session_index: Dict[Tuple[str, str], Dict[str, List[Path]]]) -> Optional[Path]:
    ann_rel = row.get("video_annotation_json", "").strip()
    if ann_rel:
        vp = resolve_video_from_annotation(row)
        if vp is not None:
            return vp
        session_dir = session_dir_from_row(row)
        if session_dir is None:
            return None
        search_dirs = iter_session_video_dirs(session_dir, row.get("view", "single"))
        cn = row.get("cn_name", "")
        if cn and search_dirs:
            for video_dir in search_dirs:
                for path in sorted(video_dir.iterdir()):
                    if path.suffix not in VIDEO_EXTS:
                        continue
                    for variant in cn_match_variants(cn):
                        if variant in path.name:
                            return path
        return None

    session_dir = session_dir_from_row(row)
    if session_dir is None:
        return None

    files = build_session_video_index(session_dir)
    search_dirs = iter_session_video_dirs(session_dir, row.get("view", "single"))

    for name in candidate_names(row):
        if name in files:
            paths = files[name]
            if search_dirs:
                for path in paths:
                    if any(str(path).startswith(str(d)) for d in search_dirs):
                        return path
            return paths[0]

        norm = normalize_video_name(name)
        skey = stem_key(norm)
        paths = files.get(skey)
        if paths:
            if search_dirs:
                for path in paths:
                    if any(str(path).startswith(str(d)) for d in search_dirs):
                        return path
            return paths[0]

    cn = row.get("cn_name", "")
    if cn and search_dirs:
        for video_dir in search_dirs:
            for path in video_dir.iterdir():
                if path.suffix not in VIDEO_EXTS:
                    continue
                for variant in cn_match_variants(cn):
                    if variant in path.name:
                        return path
    return None


def build_all_session_indexes(rows: List[dict]) -> Dict[Tuple[str, str], Dict[str, List[Path]]]:
    sessions: Dict[str, Path] = {}
    for row in rows:
        session_id = row.get("session_id", "")
        if not session_id or session_id in sessions:
            continue
        session_dir = session_dir_from_row(row)
        if session_dir and session_dir.exists():
            sessions[session_id] = session_dir

    index: Dict[Tuple[str, str], Dict[str, List[Path]]] = {}
    for session_id, session_dir in sessions.items():
        index[(session_id, "")] = build_session_video_index(session_dir)
    return index


def parse_frame_range(row: dict) -> Optional[Tuple[int, int]]:
    try:
        start = int(float(row["start_frame"]))
        end = int(float(row["end_frame"]))
    except (TypeError, ValueError):
        start = end = -1
    if end <= start:
        try:
            fps = 30.0
            start = int(float(row["video_start_time"]) * fps)
            end = int(float(row["video_end_time"]) * fps)
        except (TypeError, ValueError):
            return None
    if end <= start:
        return None
    return start, end
