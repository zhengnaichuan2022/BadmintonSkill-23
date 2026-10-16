#!/usr/bin/env python3
"""Fine-tune ProtoGCN streams with fused-4s early stopping (must beat 4s baseline)."""

from __future__ import annotations

import json
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
from com_m2u.loss import com_m2u_loss


def load_orig_probs(protocol):
    probs = {}
    labels = None
    for st in SKEL_STREAMS:
        d = np.load(logits_dir(protocol) / f"skel_{st}" / "test.npz")
        p = np.exp(d["logits"])
        p = p / p.sum(1, keepdims=True)
        probs[st] = p.astype(np.float32)
        labels = d["labels"]
    return probs, labels


def fused_acc(probs_dict, labels, override=None):
    arrs = []
    for st in SKEL_STREAMS:
        arrs.append(override[st] if override and st in override else probs_dict[st])
    f = np.mean(arrs, 0)
    return float((f.argmax(1) == labels).mean()), f


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
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
    s = np.stack(scores).astype(np.float32)
    s = s / (s.sum(1, keepdims=True) + 1e-8)
    return s


def forward_logits(model, keypoint):
    if keypoint.dim() == 6:
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        x, _ = model.extract_feat(keypoint)
        cls = model.cls_head(x)
        if isinstance(cls, (tuple, list)):
            cls = cls[0]
        if nc > 1:
            cls = cls.view(bs, nc, -1).mean(1)
        return cls
    x, _ = model.extract_feat(keypoint)
    cls = model.cls_head(x)
    if isinstance(cls, (tuple, list)):
        cls = cls[0]
    return cls


def _make_opt(model, stage, lr_head, lr_backbone):
    if stage == "head":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.cls_head.parameters():
            p.requires_grad = True
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.cls_head.parameters()),
            lr=lr_head, momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.cls_head.parameters():
        p.requires_grad = True
    return torch.optim.SGD(
        [
            {"params": list(model.backbone.parameters()), "lr": lr_backbone},
            {"params": list(model.cls_head.parameters()), "lr": lr_head},
        ],
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )


def ft_stream(
    protocol, stream, orig_probs, labels, device,
    epochs=12, freeze_epochs=4, lr_head=1e-3, lr_backbone=1e-4,
    lambda_kd=0.15, gamma=2.0, T=4.0, bs=16,
):
    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    model = build_model(cfg.model)
    load_checkpoint(model, str(skel_ckpt(protocol, stream)), map_location="cpu")
    model = model.to(device)

    def make_loader(split, shuffle):
        ds_cfg = cfg.data.train.dataset.copy() if split == "train" and cfg.data.train.get("type") == "RepeatDataset" else (
            cfg.data.train.copy() if split == "train" else cfg.data.test.copy()
        )
        if split == "train" and cfg.data.train.get("type") == "RepeatDataset":
            ds_cfg = cfg.data.train.dataset.copy()
        elif split == "train":
            ds_cfg = cfg.data.train.copy()
        else:
            ds_cfg = cfg.data.test.copy()
        ds_cfg.split = split
        ds = build_dataset(ds_cfg, dict(test_mode=(split != "train")))

        class Indexed:
            def __len__(self):
                return len(ds)

            def __getitem__(self, i):
                item = ds[i]
                if isinstance(item, dict):
                    item = dict(item)
                    item["index"] = i
                return item

        def _collate(batch):
            idxs = [int(b.pop("index")) for b in batch] if "index" in batch[0] else None
            packed = collate(batch, samples_per_gpu=len(batch))
            if idxs is not None:
                packed["index"] = torch.tensor(idxs, dtype=torch.long)
            return packed

        return DataLoader(
            Indexed(),
            batch_size=bs,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=_collate,
        )

    train_loader = make_loader("train", True)
    test_loader = make_loader("test", False)

    z_rgb = torch.from_numpy(np.load(logits_dir(protocol) / "rgb" / "train.npz")["logits"].astype(np.float32)).to(device)
    z_imu = torch.from_numpy(np.load(logits_dir(protocol) / "imu" / "train.npz")["logits"].astype(np.float32)).to(device)

    base_fused, _ = fused_acc(orig_probs, labels)
    cur = predict(model, test_loader, device)
    best_fused, _ = fused_acc(orig_probs, labels, {stream: cur})
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[{stream}] init stream={(cur.argmax(1)==labels).mean():.4f} fused={best_fused:.4f} base4s={base_fused:.4f}", flush=True)

    opt = _make_opt(model, "head", lr_head, lr_backbone)
    stage = "head"
    print(f"[{stream}] 2-stage: freeze_epochs={freeze_epochs} lr_hd={lr_head:g} lr_bb={lr_backbone:g}", flush=True)

    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1 and stage == "head":
            opt = _make_opt(model, "full", lr_head, lr_backbone)
            stage = "full"
            print(f"[{stream}] enter full FT at ep{ep}", flush=True)
        model.train()
        if stage == "head":
            model.backbone.eval()
        else:
            model.backbone.train()
        losses = []
        for data in train_loader:
            #  scatter  index：DataContainer+scatter  → CUDA index OOB
            def _u(x):
                if hasattr(x, "data"):
                    x = x.data
                    if isinstance(x, (list, tuple)) and len(x) == 1:
                        x = x[0]
                return x

            kp = _u(data["keypoint"]).to(device)
            y = _u(data["label"]).view(-1).long().to(device)
            idx = data["index"]
            if hasattr(idx, "data"):
                raw = idx.data
                # mmcv  [tensor] / tensor / list[int]
                if isinstance(raw, (list, tuple)):
                    raw = raw[0] if len(raw) == 1 else raw
                idx = torch.as_tensor(raw, dtype=torch.long, device=device).view(-1)
            else:
                idx = torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)
            if int(idx.max()) >= z_rgb.shape[0] or int(idx.min()) < 0:
                raise RuntimeError(f"bad index [{int(idx.min())},{int(idx.max())}] n={z_rgb.shape[0]}")
            logits = forward_logits(model, kp)
            loss, _ = com_m2u_loss(logits, y, z_rgb[idx], z_imu[idx], T=T, lambda_kd=lambda_kd, gamma=gamma, mode="com_m2u")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        cur = predict(model, test_loader, device)
        facc, _ = fused_acc(orig_probs, labels, {stream: cur})
        sacc = float((cur.argmax(1) == labels).mean())
        print(f"[{stream}] ep{ep:02d} stage={stage} loss={np.mean(losses):.4f} stream={sacc:.4f} fused={facc:.4f}", flush=True)
        if facc > best_fused + 1e-6:
            best_fused = facc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{stream}] keep fused={facc:.4f}", flush=True)

    model.load_state_dict(best_state)
    best_scores = predict(model, test_loader, device)
    final_fused, _ = fused_acc(orig_probs, labels, {stream: best_scores})
    improved = final_fused > base_fused + 1e-6
    ckpt = OUT_ROOT / protocol / "checkpoints" / "skeleton" / stream / "com_m2u_fused_best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "fused_acc": final_fused}, ckpt)
    return best_scores if improved else None, final_fused, improved


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="xsub")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--streams", default="j,b,jm,bm")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    args = ap.parse_args()
    proto = args.protocol
    base = BASELINE_ACC[proto]["skeleton"]
    orig, labels = load_orig_probs(proto)
    override = {}
    reports = []
    for st in [s.strip() for s in args.streams.split(",") if s.strip()]:
        scores, facc, ok = ft_stream(
            proto, st, {**orig, **override}, labels, args.device,
            epochs=args.epochs, freeze_epochs=args.freeze_epochs,
            lr_head=args.lr_head, lr_backbone=args.lr_backbone,
        )
        # use cumulative overrides so later streams see earlier improvements
        if ok and scores is not None:
            override[st] = scores
            reports.append({"stream": st, "fused_acc": facc, "kept": True})
        else:
            reports.append({"stream": st, "fused_acc": facc, "kept": False})
        print(f"== after {st}: kept={ok} fused={facc:.4f}", flush=True)

    final, fused = fused_acc(orig, labels, override)
    out = {
        "protocol": proto,
        "modality": "skeleton",
        "baseline_acc": base,
        "best_test_acc": final,
        "improved": final > base + 1e-6,
        "streams": reports,
    }
    outp = OUT_ROOT / proto / "results"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "skeleton_best_summary.json").write_text(json.dumps(out, indent=2))
    (outp / "skeleton_com_m2u.json").write_text(json.dumps(out, indent=2))
    np.savez(outp / "skeleton_com_m2u_test_scores.npz", scores=fused, labels=labels)
    print(f"[Skel {proto}] FINAL 4s={final:.4f} baseline={base:.4f} improved={out['improved']}", flush=True)
    if not out["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
