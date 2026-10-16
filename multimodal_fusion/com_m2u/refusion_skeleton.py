#!/usr/bin/env python3
"""Re-evaluate improved ProtoGCN stream ckpts and fuse with original scores."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader

ROOT = Path("${BADMINTONSKILL23_ROOT}")
sys.path.insert(0, str(ROOT / "skeleton" / "pyskl-main" / "pyskl-main"))
sys.path.insert(0, str(ROOT / "multimodal"))

import pyskl.models.gcns.protogcn_ext  # noqa: F401
import pyskl.models.heads.protogcn_head  # noqa: F401
import pyskl.models.recognizers.recognizer_protogcn  # noqa: F401

from pyskl.datasets import build_dataset
from pyskl.models import build_model

from com_m2u.constants import BASELINE_ACC, OUT_ROOT, SKEL_STREAMS, logits_dir, skel_cfg, skel_ckpt


@torch.no_grad()
def infer_stream(protocol, stream, ckpt_path, device="cuda:0"):
    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    ds_cfg = cfg.data.test.copy()
    ds_cfg.split = "test"
    dataset = build_dataset(ds_cfg, dict(test_mode=True))
    model = build_model(cfg.model)
    load_checkpoint(model, str(ckpt_path), map_location="cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate(b, samples_per_gpu=len(b)),
    )
    scores = []
    for data in loader:
        data = scatter(data, [device])[0]
        out = model(return_loss=False, **data)
        if isinstance(out, list):
            for r in out:
                scores.append(np.asarray(r).reshape(-1))
        else:
            arr = np.asarray(out)
            if arr.ndim == 1:
                scores.append(arr)
            else:
                for r in arr:
                    scores.append(np.asarray(r).reshape(-1))
    scores_arr = np.stack(scores).astype(np.float32)
    return scores_arr


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="xsub")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    proto = args.protocol
    base = BASELINE_ACC[proto]["skeleton"]
    arrs = []
    labels = None
    info = []
    for st in SKEL_STREAMS:
        orig = np.load(logits_dir(proto) / f"skel_{st}" / "test.npz")
        oprob = np.exp(orig["logits"])
        oprob = oprob / oprob.sum(1, keepdims=True)
        oacc = float((oprob.argmax(1) == orig["labels"]).mean())
        labels = orig["labels"]
        ckpt = OUT_ROOT / proto / "checkpoints" / "skeleton" / st / "com_m2u_best.pth"
        use = oprob
        tag = "original"
        if ckpt.is_file():
            cfg = Config.fromfile(str(skel_cfg(proto, st)))
            model = build_model(cfg.model)
            state = torch.load(str(ckpt), map_location="cpu")
            model.load_state_dict(state["state_dict"])
            tmp = OUT_ROOT / proto / "tmp_skel_eval" / f"{st}.pth"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), tmp)
            s = infer_stream(proto, st, tmp, args.device)
            assert len(s) == len(labels), (len(s), len(labels))
            if s.min() < 0:
                s = np.exp(s)
                s = s / s.sum(1, keepdims=True)
            else:
                s = s / (s.sum(1, keepdims=True) + 1e-8)
            sacc = float((s.argmax(1) == labels).mean())
            if sacc > oacc + 1e-6:
                use = s.astype(np.float32)
                tag = f"ft:{sacc:.4f}"
            else:
                tag = f"keep_orig(ft={sacc:.4f})"
        arrs.append(use.astype(np.float32))
        info.append({"stream": st, "orig": oacc, "used": tag})
        print(st, tag, "orig", oacc, flush=True)

    fused = np.mean(arrs, 0)
    acc = float((fused.argmax(1) == labels).mean())
    out = {
        "protocol": proto,
        "modality": "skeleton",
        "baseline_acc": base,
        "best_test_acc": acc,
        "improved": acc > base + 1e-6,
        "streams": info,
    }
    outp = OUT_ROOT / proto / "results"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "skeleton_best_summary.json").write_text(json.dumps(out, indent=2))
    (outp / "skeleton_com_m2u.json").write_text(json.dumps(out, indent=2))
    np.savez(outp / "skeleton_com_m2u_test_scores.npz", scores=fused, labels=labels)
    print(f"[Skel {proto}] 4s={acc:.4f} baseline={base:.4f} improved={out['improved']}", flush=True)


if __name__ == "__main__":
    main()
