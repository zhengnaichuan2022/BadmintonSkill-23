#!/usr/bin/env python3
"""Classification evaluation on additional samples (top-1 / mean class accuracy).

For full-dataset benchmarks, see the repository modules documented in
benchmark_readme.md (training/, video/, skeleton/).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "dataloader_examples"))

from benchmark_utils import classification_metrics, get_dataset_root, load_action_labels
from load_classification_annotation import load_classification_annotation


def load_samples_manifest(root: Path) -> List[dict]:
    manifest = root / "samples" / "samples_manifest.csv"
    with manifest.open(encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if r["type"] == "classification"]


def load_predictions(path: Path) -> Dict[str, int]:
    """Load predictions: JSON {instance_id: class_id} or CSV instance_id,pred_class_id."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in data.items()}
    preds = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("instance_id") or row.get("sample_id")
            preds[key] = int(row["pred_class_id"])
    return preds


def make_perfect_predictions(root: Path) -> Dict[str, int]:
    """Ground-truth predictions for demo (one sample per class)."""
    preds = {}
    for row in load_samples_manifest(root):
        sample_dir = root / "samples" / row["output_dir"]
        ann, _ = load_classification_annotation(sample_dir)
        key = ann.get("instance_id") or ann.get("sample_id")
        preds[key] = int(ann["class_id"])
    return preds


def make_random_predictions(root: Path, seed: int = 0) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    _, class_names = load_action_labels(root)
    preds = {}
    for row in load_samples_manifest(root):
        preds[row.get("instance_id") or row.get("sample_id")] = int(rng.integers(0, len(class_names)))
    return preds


def evaluate(preds: Dict[str, int], root: Path) -> dict:
    y_true, y_pred, ids = [], [], []
    for row in load_samples_manifest(root):
        sample_dir = root / "samples" / row["output_dir"]
        ann, _ = load_classification_annotation(sample_dir)
        sid = ann.get("instance_id") or ann.get("sample_id")
        if sid not in preds:
            continue
        y_true.append(int(ann["class_id"]))
        y_pred.append(int(preds[sid]))
        ids.append(sid)

    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    metrics["instance_ids"] = ids
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="BadmintonSkill-23 classification evaluation")
    parser.add_argument("--root", type=Path, default=None, help="additional root")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Predictions file (.json or .csv). Default: perfect GT demo.",
    )
    parser.add_argument("--demo-random", action="store_true", help="Use random predictions for demo")
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON here")
    args = parser.parse_args()

    root = args.root or get_dataset_root()

    if args.demo_random:
        preds = make_random_predictions(root)
    elif args.predictions:
        preds = load_predictions(args.predictions)
    else:
        preds = make_perfect_predictions(root)

    metrics = evaluate(preds, root)
    print(json.dumps({k: v for k, v in metrics.items() if k != "instance_ids"}, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
