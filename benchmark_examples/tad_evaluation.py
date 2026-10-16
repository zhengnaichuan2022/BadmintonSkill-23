#!/usr/bin/env python3
"""Temporal action detection (TAD) evaluation — mAP @ tIoU for additional samples.

Provides a lightweight mAP implementation compatible with OpenTAD prediction format.
Full-scale TAD benchmarks (RGB / skeleton / IMU / multimodal) run via OpenTAD-main;
see benchmark_readme.md and config_examples/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "dataloader_examples"))

from benchmark_utils import get_dataset_root
from load_tad_annotation import load_tad_annotation

DEFAULT_TIOU_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def segment_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    inter = max(0.0, end - start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth_from_samples(root: Path) -> Dict[str, List[dict]]:
    """Build GT dict: video_id -> [{label, segment}, ...]."""
    manifest = root / "samples" / "samples_manifest.csv"
    import csv

    gt: Dict[str, List[dict]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["type"] != "tad":
                continue
            sample_dir = root / "samples" / row["output_dir"]
            ann, segments, _ = load_tad_annotation(sample_dir)
            vid = ann["video_id"]
            gt[vid] = [
                {"label": s["label_id"], "segment": s["segment"]}
                for s in segments
            ]
    return gt


def load_predictions(path: Path) -> Dict[str, List[dict]]:
    """OpenTAD-style JSON: {results: {video_id: [{label, score, segment}, ...]}}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        return data["results"]
    return data


def make_perfect_predictions(root: Path, score: float = 1.0) -> Dict[str, List[dict]]:
    gt = load_ground_truth_from_samples(root)
    preds = {}
    for vid, segments in gt.items():
        preds[vid] = [
            {"label": s["label"], "score": score, "segment": s["segment"]}
            for s in segments
        ]
    return preds


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum((recall[indices] - recall[indices - 1]) * precision[indices]))


def evaluate_video(
    gt_segments: List[dict],
    pred_segments: List[dict],
    tiou_threshold: float,
) -> Tuple[float, int]:
    """Average precision for one video at one tIoU."""
    gt_used = np.zeros(len(gt_segments), dtype=bool)
    pred_sorted = sorted(pred_segments, key=lambda x: -x.get("score", 0.0))

    tp = np.zeros(len(pred_sorted))
    fp = np.zeros(len(pred_sorted))

    for i, pred in enumerate(pred_sorted):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_segments):
            if gt_used[j] or pred["label"] != gt["label"]:
                continue
            iou = segment_iou(tuple(pred["segment"]), tuple(gt["segment"]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= tiou_threshold and best_j >= 0:
            tp[i] = 1
            gt_used[best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt = max(len(gt_segments), 1)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    return compute_ap(recall, precision), len(gt_segments)


def evaluate_map(
    ground_truth: Dict[str, List[dict]],
    predictions: Dict[str, List[dict]],
    tiou_thresholds: List[float],
    activity_ids: List[int] | None = None,
) -> dict:
    if activity_ids is None:
        labels = set()
        for segs in ground_truth.values():
            for s in segs:
                labels.add(int(s["label"]))
        activity_ids = sorted(labels)

    results = {}
    for tiou in tiou_thresholds:
        class_aps = []
        for label in activity_ids:
            video_aps = []
            for vid, gt_segs in ground_truth.items():
                gt_label = [s for s in gt_segs if int(s["label"]) == label]
                if not gt_label:
                    continue
                pred_label = [
                    s for s in predictions.get(vid, [])
                    if int(s["label"]) == label
                ]
                ap, _ = evaluate_video(gt_label, pred_label, tiou)
                video_aps.append(ap)
            class_aps.append(float(np.mean(video_aps)) if video_aps else 0.0)
        results[f"mAP@{tiou:.1f}"] = float(np.mean(class_aps)) if class_aps else 0.0

    avg = float(np.mean(list(results.values()))) if results else 0.0
    results["average_mAP"] = avg
    results["n_videos"] = len(ground_truth)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="BadmintonSkill-23 TAD mAP evaluation")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tiou",
        type=float,
        nargs="+",
        default=DEFAULT_TIOU_THRESHOLDS,
        help="tIoU thresholds (default: 0.3–0.7)",
    )
    args = parser.parse_args()

    root = args.root or get_dataset_root()
    gt = load_ground_truth_from_samples(root)

    if args.predictions:
        preds = load_predictions(args.predictions)
    else:
        preds = make_perfect_predictions(root)

    metrics = evaluate_map(gt, preds, args.tiou)
    print(json.dumps(metrics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
