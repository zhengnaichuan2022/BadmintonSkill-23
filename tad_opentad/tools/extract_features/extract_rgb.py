#!/usr/bin/env python3
"""Extract per-video RGB features for Beiyu TAD (offline, frozen backbone).

， **** /。
 RGB → ``features/rgb/r2plus1d_{protocol}_stride4/{video_id}.npy``。

Supported backends:
  - dummy: random features for pipeline smoke test
  - r2plus1d: fine-tuned R(2+1)D from video/outputs (best RGB classifier)

Usage:
  python3 tools/extract_features/extract_rgb.py --backend dummy
  CUDA_VISIBLE_DEVICES=0 python3 tools/extract_features/extract_rgb.py --backend r2plus1d --protocol xsub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

OPENTAD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPENTAD_ROOT))

from config import (  # noqa: E402
    ANNOTATIONS_DIR,
    FEATURE_STRIDE,
    FEATURES_DIR,
    PROTOCOLS,
    RGB_FEATURE_DIMS,
)
from tools.extract_features.beiyu_common import collect_videos  # noqa: E402


def load_anno(protocol: str) -> dict:
    path = ANNOTATIONS_DIR / protocol / "anno.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run tools/prepare_data/beiyu/convert_annotations.py")
    return json.loads(path.read_text(encoding="utf-8"))["database"]


def extract_video_dummy(video_info: dict, dim: int, stride: int, seed: int) -> np.ndarray:
    n = max(1, int(np.ceil(video_info["frame"] / stride)))
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RGB TAD features")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--backend", default="r2plus1d", choices=list(RGB_FEATURE_DIMS.keys()))
    parser.add_argument("--stride", type=int, default=FEATURE_STRIDE)
    parser.add_argument("--limit", type=int, default=0, help="debug: max videos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if args.backend == "r2plus1d":
        from tools.extract_features.extract_mmaction import extract_protocol

        protocols = args.protocol or PROTOCOLS
        for protocol in protocols:
            extract_protocol(
                protocol,
                device=args.device,
                batch_size=args.batch_size,
                limit=args.limit,
                overwrite=args.overwrite,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
            )
        return

    dim = RGB_FEATURE_DIMS[args.backend]
    out_dir = FEATURES_DIR / "rgb" / f"{args.backend}_stride{args.stride}"
    out_dir.mkdir(parents=True, exist_ok=True)

    protocols = args.protocol or PROTOCOLS
    items = collect_videos(protocols)
    if args.limit > 0:
        items = items[: args.limit]

    for i, (vid, info) in enumerate(tqdm(items, desc=f"rgb/{args.backend}")):
        out_path = out_dir / f"{vid}.npy"
        if out_path.is_file() and not args.overwrite:
            continue
        feats = extract_video_dummy(info, dim, args.stride, seed=i + 17)
        np.save(out_path, feats)

    meta = {
        "backend": args.backend,
        "stride": args.stride,
        "dim": dim,
        "count": len(items),
        "out_dir": str(out_dir.relative_to(OPENTAD_ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
