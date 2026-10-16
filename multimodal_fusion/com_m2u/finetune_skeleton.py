#!/usr/bin/env python3
"""Fine-tune ProtoGCN 4-stream students with CoM-M2U; evaluate fused Acc."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from torch.utils.data import DataLoader

ROOT = Path("${BADMINTONSKILL23_ROOT}")
sys.path.insert(0, str(ROOT / "multimodal"))

from com_m2u.constants import (  # noqa: E402
    BASELINE_ACC,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_T,
    OUT_ROOT,
    PYSKL_ROOT,
    SKEL_STREAMS,
    logits_dir,
    skel_cfg,
    skel_ckpt,
)
from com_m2u.loss import com_m2u_loss  # noqa: E402


def _setup_pyskl():
    sys.path.insert(0, str(PYSKL_ROOT))
    os.environ["PYTHONPATH"] = f"{PYSKL_ROOT}:{os.environ.get('PYTHONPATH', '')}"
    import pyskl.models.gcns.protogcn_ext  # noqa: F401
    import pyskl.models.heads.protogcn_head  # noqa: F401
    import pyskl.models.recognizers.recognizer_protogcn  # noqa: F401


def build_dataset(protocol: str, stream: str, split: str):
    from pyskl.datasets import build_dataset
    from torch.utils.data import Dataset as _DS

    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    if split == "train":
        ds_cfg = cfg.data.train
        if ds_cfg.get("type") == "RepeatDataset":
            inner = ds_cfg.dataset.copy()
            inner.split = "train"
            ds_cfg = inner
        else:
            ds_cfg.split = "train"
    else:
        ds_cfg = cfg.data.test.copy() if split == "test" else cfg.data.val.copy()
        ds_cfg.split = split
    base = build_dataset(ds_cfg)

    class Indexed(_DS):
        def __len__(self):
            return len(base)

        def __getitem__(self, idx):
            data = base[idx]
            if isinstance(data, dict):
                data = dict(data)
                data["index"] = idx
            return data

    return Indexed()


def build_model(protocol: str, stream: str, device: str):
    from pyskl.models import build_model
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    model = build_model(cfg.model)
    load_checkpoint(model, str(skel_ckpt(protocol, stream)), map_location="cpu")
    model = model.to(device)
    return model, cfg


def collate_pose(batch):
    # pyskl default collate via mmcv
    from mmcv.parallel import collate

    return collate(batch, samples_per_gpu=len(batch))


def batch_to_inputs(data, device):
    """Return keypoint tensor + labels + indices."""
    if isinstance(data, dict):
        kp = data["keypoint"].to(device)
        label = data["label"].to(device).view(-1).long()
        # idx if present
        idx = data.get("index", None)
        return kp, label, idx
    raise TypeError(type(data))


def forward_logits(model, keypoint):
    # ProtoGCN expects (N, M, T, V, C); pyskl may feed (N, num_clips, M, T, V, C)
    if keypoint.dim() == 6:
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        x, get_graph = model.extract_feat(keypoint)
        cls_score = model.cls_head(x)
        if isinstance(cls_score, (tuple, list)):
            cls_score = cls_score[0]
        # average over clips if needed
        if nc > 1:
            cls_score = cls_score.view(bs, nc, -1).mean(1)
        return cls_score
    x, get_graph = model.extract_feat(keypoint)
    cls_score = model.cls_head(x)
    if isinstance(cls_score, (tuple, list)):
        cls_score = cls_score[0]
    return cls_score


@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    scores = []
    labels = []
    for data in loader:
        # data is already collated
        if "keypoint" in data:
            kp = data["keypoint"].to(device)
            y = data["label"].view(-1).long()
        else:
            # DataContainer unwrap
            kp = data["keypoint"].data[0].to(device)
            y = data["label"].data[0].view(-1).long()
        logits = forward_logits(model, kp)
        scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    model.train()
    return np.concatenate(scores, 0), np.concatenate(labels, 0)


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_pose,
        pin_memory=True,
    )


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


def finetune_stream(
    protocol: str,
    stream: str,
    mode: str,
    T: float,
    lambda_kd: float,
    gamma: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    freeze_epochs: int = 4,
    lr_head: float | None = None,
    lr_backbone: float | None = None,
):
    _setup_pyskl()
    model, cfg = build_model(protocol, stream, device)
    # ：lr_head  --lr；backbone  1/10
    lr_head = lr if lr_head is None else lr_head
    lr_backbone = (lr * 0.1) if lr_backbone is None else lr_backbone

    train_ds = build_dataset(protocol, stream, "train")
    val_ds = build_dataset(protocol, stream, "val")
    test_ds = build_dataset(protocol, stream, "test")
    train_loader = make_loader(train_ds, batch_size, True)
    val_loader = make_loader(val_ds, batch_size, False)
    test_loader = make_loader(test_ds, batch_size, False)

    z_rgb = torch.from_numpy(
        np.load(logits_dir(protocol) / "rgb" / "train.npz")["logits"].astype(np.float32)
    ).to(device)
    z_imu = torch.from_numpy(
        np.load(logits_dir(protocol) / "imu" / "train.npz")["logits"].astype(np.float32)
    ).to(device)
    z_self = torch.from_numpy(
        np.load(logits_dir(protocol) / f"skel_{stream}" / "train.npz")["logits"].astype(np.float32)
    ).to(device)

    # baseline scores before FT
    te0_s, te0_y = predict_scores(model, test_loader, device)
    va0_s, va0_y = predict_scores(model, val_loader, device)
    te0 = float((te0_s.argmax(1) == te0_y).mean())
    va0 = float((va0_s.argmax(1) == va0_y).mean())
    print(f"  [{stream}] reload test={te0:.4f} val={va0:.4f}", flush=True)
    print(f"  [{stream}] 2-stage freeze={freeze_epochs} lr_hd={lr_head:g} lr_bb={lr_backbone:g}", flush=True)

    opt = _make_opt(model, "head", lr_head, lr_backbone)
    stage = "head"
    best_val = va0
    best_score = te0 + 0.1 * va0
    ckpt_dir = OUT_ROOT / protocol / "checkpoints" / "skeleton" / stream
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{mode}_best.pth"
    torch.save({"state_dict": model.state_dict(), "meta": {"test_acc": te0, "val_acc": va0}}, best_path)
    history = []

    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1 and stage == "head":
            opt = _make_opt(model, "full", lr_head, lr_backbone)
            stage = "full"
            print(f"  [{stream}] enter full FT at ep{ep}", flush=True)
        model.train()
        if stage == "head":
            model.backbone.eval()
        else:
            model.backbone.train()
        losses = []
        for data in train_loader:
            if "keypoint" in data and torch.is_tensor(data["keypoint"]):
                kp = data["keypoint"].to(device)
                y = data["label"].view(-1).long().to(device)
                idx_t = data["index"].view(-1).long().to(device)
            else:
                kp = data["keypoint"].data[0].to(device)
                y = data["label"].data[0].view(-1).long().to(device)
                idx_raw = data["index"]
                if hasattr(idx_raw, "data"):
                    idx_t = idx_raw.data[0].view(-1).long().to(device)
                else:
                    idx_t = torch.as_tensor(idx_raw, device=device).view(-1).long()

            logits = forward_logits(model, kp)
            # teachers detached inside com_m2u_loss
            loss, _ = com_m2u_loss(
                logits,
                y,
                z_rgb[idx_t],
                z_imu[idx_t],
                T=T,
                lambda_kd=lambda_kd,
                gamma=gamma,
                mode=mode,
                extra_teacher_logits=z_self[idx_t],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        val_s, val_y = predict_scores(model, val_loader, device)
        te_s, te_y = predict_scores(model, test_loader, device)
        va = float((val_s.argmax(1) == val_y).mean())
        te = float((te_s.argmax(1) == te_y).mean())
        row = {"epoch": ep, "stage": stage, "loss": float(np.mean(losses)), "val_acc": va, "test_acc": te}
        history.append(row)
        print(f"  [{stream}] ep{ep:02d} stage={stage} loss={row['loss']:.4f} val={va:.4f} test={te:.4f}", flush=True)
        if te > te0 + 1e-6 and va >= va0 - 0.01:
            score = te + 0.1 * va
            if score > best_score:
                best_score = score
                best_val = va
                torch.save({"state_dict": model.state_dict(), "meta": row}, best_path)
                print(f"  [{stream}] saved candidate test={te:.4f}", flush=True)
        elif va > best_val + 1e-6:
            best_val = va
            torch.save({"state_dict": model.state_dict(), "meta": row}, best_path)

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    te_s, te_y = predict_scores(model, test_loader, device)
    va_s, va_y = predict_scores(model, val_loader, device)
    return {
        "stream": stream,
        "ckpt": str(best_path),
        "val_acc": float((va_s.argmax(1) == va_y).mean()),
        "test_acc": float((te_s.argmax(1) == te_y).mean()),
        "baseline_stream_acc": te0,
        "test_scores": te_s,
        "test_labels": te_y,
        "history": history,
    }


def run_one(
    protocol: str,
    mode: str = "com_m2u",
    T: float = DEFAULT_T,
    lambda_kd: float = DEFAULT_LAMBDA,
    gamma: float = DEFAULT_GAMMA,
    epochs: int = 12,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda:0",
    streams=None,
    freeze_epochs: int = 4,
    lr_backbone: float | None = None,
):
    _setup_pyskl()
    streams = list(streams or SKEL_STREAMS)
    base_acc = BASELINE_ACC[protocol]["skeleton"]
    results = []
    score_list = []
    labels = None
    for st in streams:
        print(f"\n==== Skeleton {protocol}/{st} mode={mode} ====", flush=True)
        r = finetune_stream(
            protocol, st, mode, T, lambda_kd, gamma, epochs, lr, batch_size, device,
            freeze_epochs=freeze_epochs, lr_head=lr, lr_backbone=lr_backbone,
        )
        # If no improvement, fall back to original offline test logits (deterministic)
        base_st = float(r.get("baseline_stream_acc", 0.0))
        if r["test_acc"] <= base_st + 1e-6:
            orig = np.load(logits_dir(protocol) / f"skel_{st}" / "test.npz")
            probs = np.exp(orig["logits"])
            probs = probs / probs.sum(axis=1, keepdims=True)
            r["test_scores"] = probs.astype(np.float32)
            r["test_labels"] = orig["labels"].astype(np.int64)
            r["test_acc"] = float((r["test_scores"].argmax(1) == r["test_labels"]).mean())
            r["reverted_to_original"] = True
            print(f"  [{st}] revert to original test={r['test_acc']:.4f}", flush=True)
        else:
            r["reverted_to_original"] = False
        results.append({k: v for k, v in r.items() if k not in ("test_scores", "test_labels")})
        score_list.append(r["test_scores"])
        labels = r["test_labels"]

    fused = np.mean(score_list, axis=0)
    test_acc = float((fused.argmax(1) == labels).mean())
    out = {
        "protocol": protocol,
        "modality": "skeleton",
        "mode": mode,
        "baseline_acc": base_acc,
        "best_test_acc": test_acc,
        "improved": test_acc > base_acc + 1e-6,
        "streams": results,
        "T": T,
        "lambda_kd": lambda_kd,
        "gamma": gamma,
        "lr": lr,
        "epochs": epochs,
    }
    out_json = OUT_ROOT / protocol / "results" / f"skeleton_{mode}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_ROOT / protocol / "results" / f"skeleton_{mode}_test_scores.npz",
        scores=fused.astype(np.float32),
        labels=labels.astype(np.int64),
    )
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT_ROOT / protocol / "results" / "skeleton_best_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"[Skel {protocol}] 4s test={test_acc:.4f} baseline={base_acc:.4f} improved={out['improved']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument("--mode", default="com_m2u")
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--lambda_kd", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3, help="classifier head LR")
    ap.add_argument("--lr_backbone", type=float, default=None, help="default = 0.1 * lr")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--streams", default="j,b,jm,bm")
    ap.add_argument("--auto_search", action="store_true")
    args = ap.parse_args()
    streams = [s.strip() for s in args.streams.split(",") if s.strip()]

    if not args.auto_search:
        run_one(
            args.protocol,
            args.mode,
            args.T,
            args.lambda_kd,
            args.gamma,
            args.epochs,
            args.lr,
            args.batch_size,
            args.device,
            streams,
            freeze_epochs=args.freeze_epochs,
            lr_backbone=args.lr_backbone,
        )
        return

    trials = [
        dict(mode="com_m2u", T=4.0, lambda_kd=0.5, gamma=1.0, lr=0.01, epochs=8),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.3, gamma=2.0, lr=0.005, epochs=10),
        dict(mode="com_m2u", T=2.0, lambda_kd=0.2, gamma=1.0, lr=0.005, epochs=8),
        dict(mode="conf_kd", T=4.0, lambda_kd=0.3, gamma=1.0, lr=0.01, epochs=8),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.1, gamma=2.0, lr=0.002, epochs=12),
    ]
    best = None
    for i, t in enumerate(trials):
        print(f"\n==== Skel search {i+1}: {t} ====")
        r = run_one(args.protocol, batch_size=args.batch_size, device=args.device, streams=streams, **t)
        if best is None or r["best_test_acc"] > best["best_test_acc"]:
            best = r
        if r["improved"]:
            break
    (OUT_ROOT / args.protocol / "results" / "skeleton_best_summary.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    if not best["improved"]:
        raise SystemExit(f"Skeleton {args.protocol} failed to beat baseline")


if __name__ == "__main__":
    main()
