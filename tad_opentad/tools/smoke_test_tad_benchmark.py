#!/usr/bin/env python3
"""Full smoke test for all Beiyu TAD benchmark configs (72).

Checks per config:
  1) Config loads
  2) Train dataset + one batch
  3) Model builds
  4) One training forward pass

Usage:
  cd OpenTAD-main
  python tools/smoke_test_tad_benchmark.py
  CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_tad_benchmark.py
  python tools/smoke_test_tad_benchmark.py --task 1
  python tools/smoke_test_tad_benchmark.py --config configs/beiyu/benchmark/skel_j_xsub_gtad.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from mmengine.config import Config

OPENTAD_ROOT = Path(__file__).resolve().parents[1]
if str(OPENTAD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENTAD_ROOT))

from opentad.datasets import build_dataset
from opentad.models import build_detector

sys.path.insert(0, str(OPENTAD_ROOT / "tools"))
from filter_tad_benchmark import filter_manifest as _filter_manifest  # noqa: E402

MANIFEST = OPENTAD_ROOT / "configs" / "beiyu" / "benchmark_manifest.json"


def draw_bar(done: int, total: int, width: int = 30) -> str:
    filled = done * width // max(total, 1)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def filter_manifest(task: str, feature: str, protocol: str, model: str) -> list[dict]:
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return _filter_manifest(items, task, feature, protocol, model)


def smoke_one(cfg_path: Path, device: torch.device) -> tuple[bool, str]:
    name = cfg_path.stem
    try:
        cfg = Config.fromfile(str(cfg_path))
        sched = dict(cfg.scheduler)
        sched["max_epoch"] = 1
        cfg.merge_from_dict(
            {
                "scheduler": sched,
                "workflow": {"val_start_epoch": 99, "end_epoch": 1},
                "solver": {
                    "train": {"batch_size": 1, "num_workers": 0},
                    "val": {"batch_size": 1, "num_workers": 0},
                    "test": {"batch_size": 1, "num_workers": 0},
                },
            }
        )

        train_dataset = build_dataset(cfg.dataset.train)
        sample = train_dataset[0]

        model = build_detector(cfg.model)
        model = model.to(device)
        model.train()

        data_dict = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                data_dict[key] = val.unsqueeze(0).to(device)
            else:
                data_dict[key] = val

        with torch.cuda.amp.autocast(enabled=False):
            losses = model(**data_dict, return_loss=True)

        if isinstance(losses, dict):
            loss_val = sum(v.item() if torch.is_tensor(v) else float(v) for v in losses.values())
        else:
            loss_val = float(losses)
        return True, f"loss={loss_val:.4f} windows={len(train_dataset)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all TAD benchmark configs")
    parser.add_argument("--config", default="", help="single config path")
    parser.add_argument("--task", default="")
    parser.add_argument("--feature", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.config:
        entries = [{"config": args.config, "feature": "", "protocol": "", "model": ""}]
    else:
        if not MANIFEST.is_file():
            print(f"Missing manifest: {MANIFEST}")
            print("Run: python tools/generate_beiyu_tad_benchmark.py")
            sys.exit(1)
        entries = filter_manifest(args.task, args.feature, args.protocol, args.model)

    device = torch.device(args.device)
    total = len(entries)
    ok_count = 0
    fail_count = 0
    results: list[tuple[str, bool, str, float]] = []

    print(f"TAD benchmark smoke test — {total} configs on {device}")
    print("=" * 60)

    for idx, entry in enumerate(entries, 1):
        cfg_rel = entry["config"]
        cfg_path = OPENTAD_ROOT / cfg_rel
        name = cfg_path.stem
        t0 = time.time()
        ok, msg = smoke_one(cfg_path, device)
        elapsed = time.time() - t0
        results.append((name, ok, msg, elapsed))
        if ok:
            ok_count += 1
            mark = "OK"
        else:
            fail_count += 1
            mark = "FAIL"
        bar = draw_bar(idx, total)
        print(f"{bar} {idx}/{total}  {mark:<4}  {name}  ({elapsed:.1f}s)  {msg}")

    print("=" * 60)
    print(f"Done: ok={ok_count} fail={fail_count} / {total}")

    if fail_count:
        print("\nFailed configs:")
        for name, ok, msg, _ in results:
            if not ok:
                print(f"  - {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
