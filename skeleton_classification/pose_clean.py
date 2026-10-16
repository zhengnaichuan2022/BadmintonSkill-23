"""Temporal cleaning for single-view badminton skeleton clips.

COCO-17 keypoint order (matches PYSKL ``layout='coco'`` and RTMPose-L):
  0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
  5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
  9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
  13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

NUM_KEYPOINTS = 17


def _instance_arrays(instance: dict) -> Tuple[np.ndarray, np.ndarray]:
    kp = np.asarray(instance["keypoints"], dtype=np.float32)
    sc = np.asarray(instance.get("keypoint_scores", instance.get("keypoint_score", [])), dtype=np.float32)
    if kp.shape != (NUM_KEYPOINTS, 2):
        raise ValueError(f"expected keypoints shape ({NUM_KEYPOINTS}, 2), got {kp.shape}")
    if sc.shape != (NUM_KEYPOINTS,):
        sc = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    return kp, sc


def _mean_distance(a: np.ndarray, b: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, thr: float = 0.3) -> float:
    mask = (scores_a > thr) & (scores_b > thr)
    if not np.any(mask):
        return float(np.linalg.norm(a - b))
    return float(np.mean(np.linalg.norm(a[mask] - b[mask], axis=1)))


def _pick_instance(instances: List[dict], prev_kp: Optional[np.ndarray], prev_sc: Optional[np.ndarray]) -> Optional[dict]:
    if not instances:
        return None
    if len(instances) == 1:
        return instances[0]
    if prev_kp is None:
        return max(instances, key=lambda x: float(x.get("bbox_score", 0.0)))
    best = None
    best_dist = float("inf")
    for inst in instances:
        kp, sc = _instance_arrays(inst)
        dist = _mean_distance(kp, prev_kp, sc, prev_sc if prev_sc is not None else sc)
        if dist < best_dist:
            best_dist = dist
            best = inst
    return best


def _interpolate_between(
    left_idx: int,
    left_kp: np.ndarray,
    left_sc: np.ndarray,
    right_idx: int,
    right_kp: np.ndarray,
    right_sc: np.ndarray,
    t: int,
) -> Tuple[np.ndarray, np.ndarray]:
    span = right_idx - left_idx
    alpha = (t - left_idx) / span
    kp = (1.0 - alpha) * left_kp + alpha * right_kp
    sc = (1.0 - alpha) * left_sc + alpha * right_sc
    return kp.astype(np.float32), sc.astype(np.float32)


def _smooth_neighbors(
    keypoints: np.ndarray,
    scores: np.ndarray,
    flags: np.ndarray,
    t: int,
    radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    weights: List[float] = []
    kp_sum = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    sc_sum = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
    for dt in range(-radius, radius + 1):
        if dt == 0:
            continue
        j = t + dt
        if j < 0 or j >= len(flags) or flags[j]:
            continue
        w = 1.0 / (abs(dt) + 1e-6)
        weights.append(w)
        kp_sum += w * keypoints[j]
        sc_sum += w * scores[j]
    if not weights:
        return keypoints[t].copy(), scores[t].copy()
    wtot = sum(weights)
    return (kp_sum / wtot).astype(np.float32), (sc_sum / wtot).astype(np.float32)


def clean_clip_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    multi_person_flags: np.ndarray,
    missing_flags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill missing / multi-person frames using temporal neighbors."""
    T = keypoints.shape[0]
    out_kp = keypoints.copy()
    out_sc = scores.copy()
    needs_fix = multi_person_flags | missing_flags

    for t in range(T):
        if not needs_fix[t]:
            continue
        prev_idx = next((i for i in range(t - 1, -1, -1) if not needs_fix[i]), None)
        next_idx = next((i for i in range(t + 1, T) if not needs_fix[i]), None)
        if prev_idx is not None and next_idx is not None:
            out_kp[t], out_sc[t] = _interpolate_between(
                prev_idx, out_kp[prev_idx], out_sc[prev_idx],
                next_idx, out_kp[next_idx], out_sc[next_idx],
                t,
            )
        elif prev_idx is not None or next_idx is not None:
            anchor = prev_idx if prev_idx is not None else next_idx
            out_kp[t] = out_kp[anchor]
            out_sc[t] = out_sc[anchor]
        else:
            out_kp[t], out_sc[t] = _smooth_neighbors(out_kp, out_sc, needs_fix, t)

    for t in range(T):
        if multi_person_flags[t]:
            sm_kp, sm_sc = _smooth_neighbors(out_kp, out_sc, multi_person_flags, t)
            out_kp[t] = 0.5 * out_kp[t] + 0.5 * sm_kp
            out_sc[t] = 0.5 * out_sc[t] + 0.5 * sm_sc

    return out_kp, out_sc


def extract_clip_from_video_json(
    frames_by_id: dict,
    start_frame: int,
    end_frame: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build [T,17,2] clip with primary-person tracking and temporal cleaning."""
    T = end_frame - start_frame + 1
    keypoints = np.zeros((T, NUM_KEYPOINTS, 2), dtype=np.float32)
    scores = np.zeros((T, NUM_KEYPOINTS), dtype=np.float32)
    multi_flags = np.zeros(T, dtype=bool)
    missing_flags = np.zeros(T, dtype=bool)

    prev_kp: Optional[np.ndarray] = None
    prev_sc: Optional[np.ndarray] = None

    for offset, frame_id in enumerate(range(start_frame, end_frame + 1)):
        frame = frames_by_id.get(frame_id)
        if frame is None:
            missing_flags[offset] = True
            continue
        instances = frame.get("instances") or []
        if len(instances) == 0:
            missing_flags[offset] = True
            continue
        if len(instances) > 1:
            multi_flags[offset] = True
        chosen = _pick_instance(instances, prev_kp, prev_sc)
        if chosen is None:
            missing_flags[offset] = True
            continue
        kp, sc = _instance_arrays(chosen)
        keypoints[offset] = kp
        scores[offset] = sc
        prev_kp, prev_sc = kp, sc

    keypoints, scores = clean_clip_pose(keypoints, scores, multi_flags, missing_flags)
    stats = {
        "total_frames": T,
        "missing_frames": int(missing_flags.sum()),
        "multi_person_frames": int(multi_flags.sum()),
        "cleaned_frames": int((multi_flags | missing_flags).sum()),
    }
    return keypoints, scores, stats


def infer_img_shape(keypoints: np.ndarray, scores: np.ndarray, default=(1080, 1920)) -> Tuple[int, int]:
    mask = scores > 0.3
    if not np.any(mask):
        return default
    ys = keypoints[mask, 1]
    xs = keypoints[mask, 0]
    h = int(max(default[0], np.ceil(ys.max()) + 1))
    w = int(max(default[1], np.ceil(xs.max()) + 1))
    return h, w
