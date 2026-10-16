#!/usr/bin/env python3
"""ROI256 video benchmark — list completed / pending train jobs.

Completion: training reached max_epochs (default 50) in the latest ROI256 log,
or epoch_50.pth exists under outputs/{protocol}/{model}_roi256_scratch/.

Usage:
  python3 benchmark_roi256_jobs.py              # table
  python3 benchmark_roi256_jobs.py --pending    # protocol model (for shell)
  python3 benchmark_roi256_jobs.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
WORK_TAG = "roi256_scratch"
MAX_EPOCHS = 50

PROTOCOLS = ("xsub", "xacq")
MODELS = (
    "tsn_badminton_r50",
    "tsm_badminton_r50",
    "slowfast_badminton_r50",
    "c3d_badminton",
    "i3d_badminton_r50",
    "r2plus1d_badminton_r34",
    "x3d_badminton_m",
    "tin_badminton_r50",
    "trn_badminton_r50",
    "tanet_badminton_r50",
)


def _log_is_roi256(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:12000]
    except OSError:
        return False
    return "roi256_scratch" in head or "frames_roi_256" in head


def work_dir(protocol: str, model: str) -> Path:
    return VIDEO_DIR / "outputs" / protocol / f"{model}_{WORK_TAG}"


def collect_logs(protocol: str, model: str) -> list[Path]:
    cands: list[Path] = []
    log_dir = VIDEO_DIR / "logs"
    cands.extend(log_dir.glob(f"{protocol}_{model}_*.log"))
    for batch_dir in log_dir.glob("benchmark_roi256_*"):
        p = batch_dir / f"{protocol}_{model}.log"
        if p.is_file():
            cands.append(p)
    wd = work_dir(protocol, model)
    if wd.is_dir():
        cands.extend(wd.rglob("*.log"))
    return [p for p in cands if _log_is_roi256(p)]


def latest_log(protocol: str, model: str) -> Path | None:
    logs = collect_logs(protocol, model)
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def has_final_checkpoint(protocol: str, model: str) -> bool:
    wd = work_dir(protocol, model)
    if not wd.is_dir():
        return False
    if (wd / "epoch_50.pth").is_file():
        return True
    return any(wd.rglob("epoch_50.pth"))


def parse_progress(log_path: Path | None) -> tuple[int, int, str]:
    """Return (current_epoch, max_epochs, state). state: done|running|failed|pending."""
    if log_path is None:
        return 0, MAX_EPOCHS, "pending"

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    max_ep = MAX_EPOCHS
    m = re.search(r"max_epochs=(\d+)", text)
    if m:
        max_ep = int(m.group(1))

    train_eps = [int(x) for x in re.findall(r"Epoch\(train\)\s+\[(\d+)\]", text)]
    val_eps = [int(x) for x in re.findall(r"Epoch\(val\)\s+\[(\d+)\]", text)]
    cur = max(train_eps[-1] if train_eps else 0, val_eps[-1] if val_eps else 0)

    if cur >= max_ep or re.search(rf"Saving checkpoint at {max_ep} epochs", text):
        return cur, max_ep, "done"
    if "FileNotFoundError" in text or "Traceback (most recent call last)" in text:
        if cur == 0:
            return cur, max_ep, "failed"
    if cur > 0:
        return cur, max_ep, "running"
    return cur, max_ep, "pending"


def job_info(protocol: str, model: str) -> dict:
    log_path = latest_log(protocol, model)
    cur, max_ep, state = parse_progress(log_path)
    if state != "done" and has_final_checkpoint(protocol, model):
        state = "done"
        cur = max(cur, max_ep)
    return {
        "protocol": protocol,
        "model": model,
        "state": state,
        "current_epoch": cur,
        "max_epochs": max_ep,
        "done": state == "done",
        "log": str(log_path.relative_to(VIDEO_DIR.parent)) if log_path else "",
    }


def all_jobs() -> list[dict]:
    return [job_info(p, m) for p in PROTOCOLS for m in MODELS]


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI256 benchmark job status")
    parser.add_argument("--pending", action="store_true", help="only incomplete jobs")
    parser.add_argument("--done", action="store_true", help="only completed jobs")
    parser.add_argument(
        "--format",
        choices=("table", "tsv", "protocol:model", "json"),
        default="table",
    )
    args = parser.parse_args()

    jobs = all_jobs()
    if args.pending:
        jobs = [j for j in jobs if not j["done"]]
    elif args.done:
        jobs = [j for j in jobs if j["done"]]

    if args.format == "json":
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return 0

    if args.format == "protocol:model":
        for j in jobs:
            print(f"{j['protocol']} {j['model']}")
        return 0

    if args.format == "tsv":
        print("protocol\tmodel\tstate\tepoch\tmax\tlog")
        for j in jobs:
            print(
                f"{j['protocol']}\t{j['model']}\t{j['state']}\t"
                f"{j['current_epoch']}\t{j['max_epochs']}\t{j['log']}"
            )
        return 0

    done_n = sum(1 for j in all_jobs() if j["done"])
    print(f"ROI256 benchmark: {done_n}/{len(PROTOCOLS)*len(MODELS)} done\n")
    print(f"{'protocol':6} {'model':28} {'state':8} {'epoch':12} log")
    print("-" * 90)
    for j in jobs:
        ep = f"{j['current_epoch']}/{j['max_epochs']}"
        log = j["log"].split("/")[-1] if j["log"] else "—"
        print(f"{j['protocol']:6} {j['model']:28} {j['state']:8} {ep:12} {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
