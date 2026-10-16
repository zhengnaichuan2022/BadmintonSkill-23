"""Shared helpers for Beiyu TAD feature extraction."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from config import (
    ANNOTATIONS_DIR,
    DATASET_ROOT,
    FEATURE_STRIDE,
    PROTOCOLS,
    SKELETON_ROOT,
    VIDEO_ROOT,
)

SKELETON_DIR = SKELETON_ROOT
if str(SKELETON_DIR) not in sys.path:
    sys.path.insert(0, str(SKELETON_DIR))

from pose_clean import (  # noqa: E402
    _instance_arrays,
    _pick_instance,
    clean_clip_pose,
    infer_img_shape,
)

CLEANED_ROOT = SKELETON_ROOT / "cleaned"
PYSKL_ROOT = SKELETON_ROOT / "pyskl-main" / "pyskl-main"

SKELETON_STREAMS = ("j", "b", "jm", "bm")
STREAM_SUFFIX = {"j": "", "b": "_b", "jm": "_jm", "bm": "_bm"}

RGB_MODEL = "r2plus1d_badminton_r34_kinetics"
SKELETON_MODEL = "protogcn"
SKELETON_DIM = 384


def find_best_checkpoint(out_dir: Path) -> Path:
    cands = sorted(out_dir.glob("best_top1_acc_epoch_*.pth"))
    if not cands:
        cands = sorted(out_dir.glob("best_acc_top1_epoch_*.pth"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint in {out_dir}")
    return cands[-1]


def rgb_output_dir(protocol: str) -> Path:
    return VIDEO_ROOT / "outputs" / protocol / RGB_MODEL


def rgb_config_path(protocol: str) -> Path:
    return rgb_output_dir(protocol) / "config.py"


def resolve_rgb_checkpoint(protocol: str) -> Path:
    return find_best_checkpoint(rgb_output_dir(protocol))


def rgb_defaults(protocol: str) -> dict:
    return {
        "config": rgb_config_path(protocol),
        "checkpoint": resolve_rgb_checkpoint(protocol),
    }


RGB_DEFAULTS = {
    "xsub": {"config": rgb_config_path("xsub"), "checkpoint": None},
    "xacq": {"config": rgb_config_path("xacq"), "checkpoint": None},
}


def load_anno(protocol: str) -> dict:
    path = ANNOTATIONS_DIR / protocol / "anno.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run tools/prepare_data/beiyu/convert_annotations.py")
    return json.loads(path.read_text(encoding="utf-8"))["database"]


def collect_videos(protocols: Iterable[str] | None = None) -> List[Tuple[str, dict]]:
    protocols = list(protocols or PROTOCOLS)
    seen = set()
    items: List[Tuple[str, dict]] = []
    for protocol in protocols:
        db = load_anno(protocol)
        for vid, info in db.items():
            if vid in seen or vid in {".", ""}:
                continue
            seen.add(vid)
            items.append((vid, info))
    return items


def snippet_centers(num_frames: int, stride: int = FEATURE_STRIDE) -> np.ndarray:
    if num_frames <= 0:
        return np.array([0], dtype=np.int64)
    return np.arange(0, num_frames, stride, dtype=np.int64)


def centered_indices(center: int, total_frames: int, clip_len: int) -> np.ndarray:
    half = clip_len // 2
    start = center - half
    idx = start + np.arange(clip_len, dtype=np.int64)
    idx = np.clip(idx, 0, max(total_frames - 1, 0))
    return idx


def skeleton_json_from_video(video_path: Path) -> Path:
    rel = video_path.relative_to(DATASET_ROOT)
    parts = list(rel.parts)
    for i, part in enumerate(parts):
        if "-" in part:
            parts[i] = part.replace("-", "-", 1)
            break
    return CLEANED_ROOT / Path(*parts).with_suffix(".json")


def resolve_skeleton_json(video_path: Path) -> Path | None:
    direct = skeleton_json_from_video(video_path)
    if direct.is_file():
        return direct
    skel_dir = direct.parent
    if not skel_dir.is_dir():
        return None
    stem = video_path.stem.lower().replace(" ", "")
    for path in sorted(skel_dir.glob("*.json")):
        if path.stem.lower().replace(" ", "") == stem:
            return path
    return None


def load_video_pose(json_path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames_by_id = {int(fr["frame_id"]): fr for fr in payload}
    if not frames_by_id:
        raise ValueError(f"empty skeleton json: {json_path}")
    max_id = max(frames_by_id)
    num_frames = max_id + 1

    keypoints = np.zeros((num_frames, 17, 2), dtype=np.float32)
    scores = np.zeros((num_frames, 17), dtype=np.float32)
    prev_kp = None
    prev_sc = None

    multi_flags = np.zeros(num_frames, dtype=bool)
    missing_flags = np.zeros(num_frames, dtype=bool)

    for frame_id in range(num_frames):
        frame = frames_by_id.get(frame_id)
        if frame is None:
            missing_flags[frame_id] = True
            continue
        instances = frame.get("instances") or []
        if not instances:
            missing_flags[frame_id] = True
            continue
        if len(instances) > 1:
            multi_flags[frame_id] = True
        chosen = _pick_instance(instances, prev_kp, prev_sc)
        if chosen is None:
            missing_flags[frame_id] = True
            continue
        kp, sc = _instance_arrays(chosen)
        keypoints[frame_id] = kp
        scores[frame_id] = sc
        prev_kp, prev_sc = kp, sc

    keypoints, scores = clean_clip_pose(keypoints, scores, multi_flags, missing_flags)
    img_shape = infer_img_shape(keypoints, scores)
    return keypoints, scores, img_shape


def skeleton_config_path(protocol: str, stream: str) -> Path:
    suffix = STREAM_SUFFIX[stream]
    if stream == "b" and SKELETON_MODEL == "posec3d_joint":
        name = "posec3d_limb"
    else:
        name = f"{SKELETON_MODEL}{suffix}"
    return SKELETON_ROOT / "configs" / "pyskl" / protocol / f"{name}.py"


def skeleton_output_dir(protocol: str, stream: str) -> Path:
    suffix = STREAM_SUFFIX[stream]
    if stream == "b" and SKELETON_MODEL == "posec3d_joint":
        name = "posec3d_limb"
    else:
        name = f"{SKELETON_MODEL}{suffix}"
    return SKELETON_ROOT / "outputs" / protocol / name



def safe_video_id(name: str) -> str:
    name = name.strip()
    if name in {".", ""}:
        raise ValueError("invalid video id")
    return re.sub(r"[^\w\-.]+", "_", name)
