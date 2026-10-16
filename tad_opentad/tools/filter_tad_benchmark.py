#!/usr/bin/env python3
"""Filter Beiyu TAD benchmark manifest by task / feature / protocol / model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OPENTAD_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = OPENTAD_ROOT / "configs" / "beiyu" / "benchmark_manifest.json"


def filter_manifest(
    manifest: list[dict],
    task: str = "",
    feature: str = "",
    protocol: str = "",
    model: str = "",
) -> list[dict]:
    out: list[dict] = []
    for item in manifest:
        t = item["task"]
        if task == "1" and not t.startswith("1_"):
            continue
        if task == "2" and t != "2_jb":
            continue
        if task == "3" and t != "3_4stream":
            continue
        if task == "4" and t != "4_rgb":
            continue
        if task == "5" and not t.startswith("5_"):
            continue
        if feature and item["feature"] != feature:
            continue
        if protocol and item["protocol"] != protocol:
            continue
        if model:
            allowed = {m.strip() for m in model.split(",") if m.strip()}
            if allowed and item["model"] not in allowed:
                continue
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter TAD benchmark manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--task", default="")
    parser.add_argument("--feature", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--field", default="config", help="output field: config | json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = filter_manifest(manifest, args.task, args.feature, args.protocol, args.model)
    if args.limit > 0:
        items = items[: args.limit]

    if args.field == "json":
        print(json.dumps(items, indent=2))
        return

    for item in items:
        print(item["config"])


if __name__ == "__main__":
    main()
