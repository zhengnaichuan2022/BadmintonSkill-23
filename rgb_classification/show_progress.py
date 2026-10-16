#!/usr/bin/env python3
""" MMAction2 （ + epoch/loss/eta）。"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent
LOG_DIR = VIDEO_DIR / "logs"

TRAIN_RE = re.compile(
    r"Epoch\(train\)\s+\[(\d+)/(\d+)\]\[\s*(\d+)/(\d+)\].*?"
    r"eta:\s*([\d:]+).*?loss:\s*([\d.]+)"
)
VAL_RE = re.compile(
    r"Epoch\(val\)\s+\[(\d+)/(\d+)\].*?acc/top1:\s*([\d.]+)"
)
MASTER_START_RE = re.compile(r"========== START (\S+)")


def draw_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} ({ratio * 100:.1f}%)"


def find_latest_train_log() -> Path | None:
    candidates = sorted(
        LOG_DIR.glob("*_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        name = path.name
        if name.startswith("benchmark"):
            continue
        if re.match(r"(xsub|xacq)_\w+_\d{8}_\d{6}\.log$", name):
            return path
    return None


def read_master_progress() -> str:
    masters = sorted(
        LOG_DIR.glob("benchmark_*/master.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not masters:
        return ""

    text = masters[0].read_text(encoding="utf-8", errors="ignore")
    starts = MASTER_START_RE.findall(text)
    done = text.count("========== DONE ")
    fail = text.count("========== FAIL ")
    total = 20
    current = starts[-1] if starts else "-"
    finished = done + fail
    return (
        f" {draw_bar(finished, total, 24)}  "
        f": {current}  :{done}  :{fail}"
    )


def tail_log(log_path: Path) -> None:
    print(f": {log_path}")
    print(" Ctrl+C （）\n")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(0, 2)
        last_train = ""
        last_val = ""
        while True:
            master = read_master_progress()
            if master:
                print(f"\r{master}", end="")

            line = handle.readline()
            if not line:
                time.sleep(0.5)
                continue

            val_match = VAL_RE.search(line)
            if val_match:
                epoch, max_epoch, acc = val_match.groups()
                last_val = f" epoch {epoch}/{max_epoch}  acc={acc}"
                print(f"\n{last_val}")

            train_match = TRAIN_RE.search(line)
            if train_match:
                epoch, max_epoch, step, max_step, eta, loss = train_match.groups()
                epoch_i, max_epoch_i = int(epoch), int(max_epoch)
                step_i, max_step_i = int(step), int(max_step)
                epoch_bar = draw_bar(step_i, max_step_i, 36)
                overall_done = (epoch_i - 1) * max_step_i + step_i
                overall_total = max_epoch_i * max_step_i
                overall_bar = draw_bar(overall_done, overall_total, 36)
                last_train = (
                    f"epoch {epoch_i}/{max_epoch_i}  {epoch_bar}  "
                    f"loss={loss}  eta={eta}"
                )
                print(f"\r{last_train}\n   {overall_bar}", end="", flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_train_log()
    if log_path is None or not log_path.exists():
        print("。，：")
        print("  python3 show_progress.py logs/xsub_tsn_badminton_r50_20260611_103937.log")
        sys.exit(1)
    try:
        tail_log(log_path)
    except KeyboardInterrupt:
        print("\n")


if __name__ == "__main__":
    main()
