#!/usr/bin/env python3
"""Smoke test IMU end-to-end TAD pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmengine.config import Config
from torch.utils.data import DataLoader
from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.models import build_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/beiyu/actionformer_imu_e2e_xsub.py",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = Config.fromfile(str(ROOT / args.config))
    dataset = build_dataset(cfg.dataset.train)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    batch = next(iter(loader))
    print("inputs:", batch["inputs"].shape, "masks:", batch["masks"].shape)
    if "gt_segments" in batch:
        print("gt_segments:", len(batch["gt_segments"]), "example:", batch["gt_segments"][0][:2])

    model = build_detector(cfg.model).to(args.device)
    model.eval()
    batch["inputs"] = batch["inputs"].to(args.device)
    batch["masks"] = batch["masks"].to(args.device)
    if "gt_segments" in batch:
        batch["gt_segments"] = [x.to(args.device) for x in batch["gt_segments"]]
        batch["gt_labels"] = [x.to(args.device) for x in batch["gt_labels"]]

    with torch.no_grad():
        losses = model(
            inputs=batch["inputs"],
            masks=batch["masks"],
            metas=batch["metas"],
            gt_segments=batch.get("gt_segments"),
            gt_labels=batch.get("gt_labels"),
            return_loss=True,
        )
    print("loss keys:", {k: float(v) for k, v in losses.items() if hasattr(v, "item")})
    print("OK")


if __name__ == "__main__":
    main()
