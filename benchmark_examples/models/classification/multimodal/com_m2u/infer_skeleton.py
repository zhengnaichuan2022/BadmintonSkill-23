#!/usr/bin/env python3
"""Single-GPU ProtoGCN inference for CoM-M2U logits export (avoids pyskl tools/test.py dist issues)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, os.environ.get("PYSKL_ROOT", str(_PKG.parents[1] / "skeleton" / "pyskl")))

import pyskl.models.gcns.protogcn_ext  # noqa: F401
import pyskl.models.heads.protogcn_head  # noqa: F401
import pyskl.models.recognizers.recognizer_protogcn  # noqa: F401

from pyskl.datasets import build_dataset
from pyskl.models import build_model

from com_m2u.constants import SKEL_STREAMS, logits_dir, skel_cfg, skel_ckpt, skel_workdir, video_manifest


def _manifest(protocol, split):
    man = json.loads(video_manifest(protocol).read_text(encoding="utf-8"))
    items = man["subsets"][split]
    ids = [x["instance_id"] for x in items]
    labels = np.array([int(x["label"]) for x in items], dtype=np.int64)
    return ids, labels


def _save(protocol, stream, split, logits, labels, ids, meta):
    out = logits_dir(protocol) / f"skel_{stream}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"
    np.savez_compressed(
        path,
        logits=logits.astype(np.float32),
        labels=labels.astype(np.int64),
        ids=np.array(ids),
        meta=json.dumps(meta),
    )
    acc = float((logits.argmax(1) == labels).mean())
    print(f"[skel-infer] {protocol}/{stream}/{split}: n={len(labels)} acc={acc:.4f} -> {path}")
    return path


@torch.no_grad()
def infer_split(protocol: str, stream: str, split: str, batch_size: int = 16, device: str = "cuda:0"):
    ids, labels = _manifest(protocol, split)
    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    # Use val pipeline (1 clip) for speed on train/val; for test use test pipeline if present
    ds_cfg = cfg.data.test.copy()
    ds_cfg.split = split
    # Prefer 1-clip for export speed unless split==test and config already uses multi-clip
    if "pipeline" in ds_cfg:
        for step in ds_cfg.pipeline:
            if step.get("type") == "UniformSample" and split != "test":
                step["num_clips"] = 1

    dataset = build_dataset(ds_cfg, dict(test_mode=True))
    assert len(dataset) == len(labels), (len(dataset), len(labels))

    model = build_model(cfg.model)
    load_checkpoint(model, str(skel_ckpt(protocol, stream)), map_location="cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate(b, samples_per_gpu=len(b)),
        pin_memory=True,
    )

    all_scores = []
    for data in loader:
        # scatter to device
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        result = model(return_loss=False, **data)
        # result: list of score arrays or ndarray
        if isinstance(result, list):
            for r in result:
                all_scores.append(np.asarray(r).reshape(-1))
        else:
            arr = np.asarray(result)
            if arr.ndim == 1:
                all_scores.append(arr.reshape(-1))
            else:
                for r in arr:
                    all_scores.append(np.asarray(r).reshape(-1))

    scores = np.stack(all_scores).astype(np.float32)
    assert len(scores) == len(labels), (len(scores), len(labels))
    logits = np.log(np.clip(scores, 1e-8, 1.0))
    _save(
        protocol,
        stream,
        split,
        logits,
        labels,
        ids,
        {"model": "protogcn", "stream": stream, "ckpt": str(skel_ckpt(protocol, stream))},
    )


def import_existing_test_pkl(protocol: str, stream: str):
    """Reuse already-exported pyskl result.pkl for test split."""
    from com_m2u.constants import skel_workdir

    pkl = skel_workdir(protocol, stream) / "result.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(pkl)
    ids, labels = _manifest(protocol, "test")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    scores = np.stack(data).astype(np.float32)
    assert len(scores) == len(labels)
    logits = np.log(np.clip(scores, 1e-8, 1.0))
    _save(
        protocol,
        stream,
        "test",
        logits,
        labels,
        ids,
        {"model": "protogcn", "stream": stream, "source": str(pkl)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument("--stream", default="all")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reuse_test_pkl", action="store_true", default=True)
    args = ap.parse_args()
    streams = list(SKEL_STREAMS) if args.stream == "all" else [args.stream]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for st in streams:
        for sp in splits:
            if sp == "test" and args.reuse_test_pkl:
                try:
                    import_existing_test_pkl(args.protocol, st)
                    continue
                except Exception as e:
                    print(f"[warn] reuse test pkl failed ({e}), inferring...")
            infer_split(args.protocol, st, sp, args.batch_size, args.device)


if __name__ == "__main__":
    main()
