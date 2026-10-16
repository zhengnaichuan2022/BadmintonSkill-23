#!/usr/bin/env python3
"""Summarize Beiyu video benchmark logs (best val acc + test acc)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BEST_RE = re.compile(r"best checkpoint with ([0-9.]+) acc/top1 at (\d+) epoch")
TEST_RE = re.compile(r"acc/top1:\s*([0-9.]+)")


def parse_log(path: Path) -> dict:
    best_acc = best_epoch = None
    test_accs = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = BEST_RE.search(line)
        if m:
            best_acc = float(m.group(1))
            best_epoch = int(m.group(2))
        if "Epoch(test)" in line or "acc/top1" in line:
            m2 = TEST_RE.search(line)
            if m2:
                test_accs.append(float(m2.group(1)))
    return {
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "test_acc": test_accs[-1] if test_accs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.log_dir.glob("*.log")):
        if path.name == "master.log":
            continue
        name = path.stem
        parts = name.split("_")
        if len(parts) < 3:
            continue
        protocol = parts[0]
        model = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
        stats = parse_log(path)
        rows.append((protocol, model, stats))

    lines = ["| protocol | model | val_acc | test_acc | best_epoch |", "|---|---|---:|---:|---:|"]
    for protocol, model, stats in sorted(rows):
        val_s = f"{stats['best_val_acc']*100:.2f}%" if stats["best_val_acc"] is not None else "-"
        test_s = f"{stats['test_acc']*100:.2f}%" if stats["test_acc"] is not None else "-"
        ep = stats["best_epoch"] if stats["best_epoch"] is not None else "-"
        lines.append(f"| {protocol} | {model} | {val_s} | {test_s} | {ep} |")

    text = "\n".join(lines) + "\n"
    print(text)
    out = args.output or args.log_dir / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
