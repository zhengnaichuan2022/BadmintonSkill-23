#!/usr/bin/env python3
"""Monitor frame cache extraction: progress bar, counts, disk usage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from config import ANNOTATIONS_DIR, FRAME_SHORT_EDGE, FRAMES_DIR


def draw_bar(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0%"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} ({ratio * 100:.1f}%)"


def fmt_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], stderr=subprocess.DEVNULL, text=True)
        return int(out.split()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
        return total


@dataclass
class CacheStats:
    total: int
    done: int
    stale: int
    partial: int
    pending: int
    total_frames: int
    used_bytes: int
    short_edge: int

    @property
    def remaining(self) -> int:
        return max(self.total - self.done, 0)

    @property
    def avg_bytes_per_done(self) -> float:
        if self.done <= 0:
            return 0.0
        return self.used_bytes / self.done

    @property
    def est_total_bytes(self) -> int:
        if self.done <= 0 or self.total <= 0:
            return self.used_bytes
        return int(self.avg_bytes_per_done * self.total)

    @property
    def est_remaining_bytes(self) -> int:
        return max(self.est_total_bytes - self.used_bytes, 0)


def total_instances() -> int:
    marker = FRAMES_DIR / "extract_total.json"
    if marker.exists():
        try:
            return int(json.loads(marker.read_text(encoding="utf-8"))["total"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    seen: set[str] = set()
    for ann_path in ANNOTATIONS_DIR.rglob("*_frames.txt"):
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(line.split()[0])
    total = len(seen)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"total": total}, ensure_ascii=False) + "\n", encoding="utf-8")
    return total


def expected_ids() -> set[str]:
    ids: set[str] = set()
    for ann_path in ANNOTATIONS_DIR.rglob("*_frames.txt"):
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(line.split()[0])
    return ids


def scan_cache(expected: set[str], total: int, short_edge: int, cache_root: Path) -> CacheStats:
    frames_dir = cache_root / "frames" if (cache_root / "frames").is_dir() else cache_root

    done = stale = partial = 0
    total_frames = 0

    if frames_dir.is_dir():
        for child in frames_dir.iterdir():
            if not child.is_dir() or child.name not in expected:
                continue

            done_marker = child / ".done"
            meta_path = child / "meta.json"

            if done_marker.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    partial += 1
                    continue
                if meta.get("short_edge") == short_edge:
                    done += 1
                    total_frames += int(meta.get("num_frames") or 0)
                else:
                    stale += 1
                continue

            if any(child.glob("img_*.jpg")):
                partial += 1

    pending = max(total - done - stale - partial, 0)
    used_bytes = dir_size_bytes(cache_root)
    return CacheStats(
        total=total,
        done=done,
        stale=stale,
        partial=partial,
        pending=pending,
        total_frames=total_frames,
        used_bytes=used_bytes,
        short_edge=short_edge,
    )


def render(stats: CacheStats, cache_root: Path, eta: str = "", updated: str = "") -> str:
    lines = [
        "",
        f": {cache_root}",
        f": {' (short_edge=0)' if stats.short_edge == 0 else f' {stats.short_edge}px'}",
    ]
    if updated:
        lines.append(f": {updated}")
    lines.extend(
        [
            "",
            draw_bar(stats.done, stats.total),
            f": {stats.done}  : {stats.remaining}  : {stats.total}",
        ]
    )
    if stats.stale or stats.partial:
        lines.append(
            f"(): {stats.stale}  ( .done): {stats.partial}  : {stats.pending}"
        )
    lines.extend(
        [
            "",
            f" JPEG : {stats.total_frames:,}",
            f": {fmt_bytes(stats.used_bytes)}",
        ]
    )
    if stats.done > 0:
        lines.extend(
            [
                f": {fmt_bytes(int(stats.avg_bytes_per_done))}  "
                f"({stats.total_frames / stats.done:.1f} /)",
                f": {fmt_bytes(stats.est_total_bytes)}  "
                f": {fmt_bytes(stats.est_remaining_bytes)}",
            ]
        )
    if eta:
        lines.append(f": {eta}")
    return "\n".join(lines)


def format_eta(seconds: float) -> str:
    if seconds <= 0 or not (seconds < float("inf")):
        return "..."
    if seconds < 60:
        return f"{seconds:.0f} "
    if seconds < 3600:
        return f"{seconds / 60:.1f} "
    if seconds < 86400:
        return f"{seconds / 3600:.1f} "
    return f"{seconds / 86400:.1f} "


def main() -> None:
    parser = argparse.ArgumentParser(description="Show frame cache extraction progress")
    parser.add_argument(
        "--cache",
        type=Path,
        default=FRAMES_DIR.parent,
        help="Cache root (default: video/cache)",
    )
    parser.add_argument(
        "--short-edge",
        type=int,
        default=FRAME_SHORT_EDGE,
        help="Expected short_edge in meta.json (default from config.py)",
    )
    parser.add_argument(
        "-w",
        "--watch",
        type=float,
        default=0,
        metavar="SEC",
        help="Refresh every SEC seconds (0 = print once)",
    )
    args = parser.parse_args()

    cache_root = args.cache.resolve()

    total = total_instances()
    expected = expected_ids()
    prev_done = -1
    prev_time = time.time()

    while True:
        stats = scan_cache(expected, total, args.short_edge, cache_root)
        eta = ""
        now = time.time()
        if args.watch > 0 and prev_done >= 0:
            delta_done = stats.done - prev_done
            delta_time = now - prev_time
            if delta_done > 0 and delta_time > 0:
                rate = delta_done / delta_time
                eta = format_eta(stats.remaining / rate)
        prev_done = stats.done
        prev_time = now

        text = render(
            stats,
            cache_root,
            eta=eta,
            updated=time.strftime("%Y-%m-%d %H:%M:%S") if args.watch > 0 else "",
        )
        if args.watch > 0:
            sys.stdout.write("\033[2J\033[H")
        print(text)
        sys.stdout.flush()

        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
