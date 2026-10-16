#!/usr/bin/env python3
"""Extract per-video skeleton features for Beiyu TAD.

 j/b/jm/bm **** ，：
``features/skeleton/protogcn_{stream}_{protocol}_stride4/{video_id}.npy``

Supported backends:
  - dummy: random features for smoke test
  - protogcn: fine-tuned ProtoGCN 4 streams (best skeleton classifier)

Usage:
  python3 tools/extract_features/extract_skeleton.py --backend protogcn --protocol xsub --stream j
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

from config import FEATURE_STRIDE, FEATURES_DIR, PROTOCOLS, SKELETON_FEATURE_DIMS  # noqa: E402
from tools.extract_features.beiyu_common import SKELETON_STREAMS, collect_videos  # noqa: E402


def extract_video_dummy(video_info: dict, dim: int, stride: int, seed: int) -> np.ndarray:
    n = max(1, int(np.ceil(video_info["frame"] / stride)))
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skeleton TAD features")
    parser.add_argument("--protocol", choices=PROTOCOLS, action="append")
    parser.add_argument("--stream", choices=SKELETON_STREAMS, action="append")
    parser.add_argument("--backend", default="protogcn", choices=list(SKELETON_FEATURE_DIMS.keys()))
    parser.add_argument("--stride", type=int, default=FEATURE_STRIDE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.backend == "protogcn":
        from tools.extract_features.extract_pyskl import extract_stream

        protocols = args.protocol or PROTOCOLS
        streams = args.stream or list(SKELETON_STREAMS)
        for protocol in protocols:
            for stream in streams:
                extract_stream(
                    protocol,
                    stream,
                    device=args.device,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    overwrite=args.overwrite,
                )
        return

    dim = SKELETON_FEATURE_DIMS[args.backend]
    out_dir = FEATURES_DIR / "skeleton" / f"{args.backend}_stride{args.stride}"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_videos(args.protocol or PROTOCOLS)
    if args.limit > 0:
        items = items[: args.limit]

    for i, (vid, info) in enumerate(tqdm(items, desc=f"skel/{args.backend}")):
        out_path = out_dir / f"{vid}.npy"
        if out_path.is_file() and not args.overwrite:
            continue
        feats = extract_video_dummy(info, dim, args.stride, seed=i + 101)
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
