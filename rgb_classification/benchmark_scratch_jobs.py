#!/usr/bin/env python3
"""Video scratch benchmark — list completed / pending jobs (xsub or xacq).

Completion: training reached max_epochs (50) in latest scratch log,
or epoch_50.pth / best_acc under outputs/{protocol}/{model}_scratch/.

Usage:
  python3 benchmark_scratch_jobs.py --protocol xacq
  python3 benchmark_scratch_jobs.py --protocol xacq --pending
  python3 benchmark_scratch_jobs.py --protocol xacq --format protocol:model
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
WORK_SUFFIX = "_scratch"
MAX_EPOCHS = 50

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


def _log_is_scratch(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:80000]
    except OSError:
        return False
    if "roi256" in head or "frames_roi_256" in head:
        return False
    return f"{WORK_SUFFIX}" in head or "Epoch(train)" in head


def _log_has_training(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:120000]
    except OSError:
        return False
    return "Epoch(train)" in text


def work_dir(protocol: str, model: str) -> Path:
    return VIDEO_DIR / "outputs" / protocol / f"{model}{WORK_SUFFIX}"


def collect_logs(protocol: str, model: str) -> list[Path]:
    cands: list[Path] = []
    log_dir = VIDEO_DIR / "logs"
    cands.extend(log_dir.glob(f"{protocol}_{model}_*.log"))
    wd = work_dir(protocol, model)
    if wd.is_dir():
        cands.extend(wd.rglob("*.log"))
    out = [p for p in cands if "vis_data" not in str(p) and _log_is_scratch(p)]
    train = [p for p in out if _log_has_training(p)]
    return train if train else out


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
    if any(wd.glob("best_acc*.pth")):
        for p in wd.glob("best_acc*.pth"):
            m = re.search(r"epoch_(\d+)", p.name)
            if m and int(m.group(1)) >= MAX_EPOCHS:
                return True
    return any(wd.rglob("epoch_50.pth"))


def has_test(protocol: str, model: str) -> bool:
    return (work_dir(protocol, model) / "test_results.json").is_file()


def parse_progress(log_path: Path | None) -> tuple[int, int, str]:
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
    if "CUDA out of memory" in text or (
        "Traceback" in text and cur == 0
    ):
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
        "has_test": has_test(protocol, model),
        "log": str(log_path.relative_to(VIDEO_DIR.parent)) if log_path else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Video scratch benchmark job status")
    parser.add_argument("--protocol", choices=("xsub", "xacq"), required=True)
    parser.add_argument("--pending", action="store_true", help="only incomplete train jobs")
    parser.add_argument("--need-test", action="store_true", help="train done but no test_results.json")
    parser.add_argument(
        "--format",
        choices=("table", "tsv", "protocol:model", "json"),
        default="table",
    )
    args = parser.parse_args()

    jobs = [job_info(args.protocol, m) for m in MODELS]
    if args.pending:
        jobs = [j for j in jobs if not j["done"]]
    elif args.need_test:
        jobs = [j for j in jobs if j["done"] and not j["has_test"]]

    if args.format == "json":
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return 0

    if args.format == "protocol:model":
        for j in jobs:
            print(f"{j['protocol']} {j['model']}")
        return 0

    if args.format == "tsv":
        print("protocol\tmodel\tstate\tepoch\ttest\tlog")
        for j in jobs:
            print(
                f"{j['protocol']}\t{j['model']}\t{j['state']}\t"
                f"{j['current_epoch']}/{j['max_epochs']}\t{j['has_test']}\t{j['log']}"
            )
        return 0

    done_n = sum(1 for j in jobs if j["done"])
    print(f"scratch {args.protocol}: {done_n}/{len(MODELS)} train done\n")
    print(f"{'model':28} {'state':8} {'epoch':12} {'test':5} log")
    print("-" * 90)
    for j in jobs:
        ep = f"{j['current_epoch']}/{j['max_epochs']}"
        log = j["log"].split("/")[-1] if j["log"] else "—"
        print(
            f"{j['model']:28} {j['state']:8} {ep:12} "
            f"{str(j['has_test']):5} {log}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
