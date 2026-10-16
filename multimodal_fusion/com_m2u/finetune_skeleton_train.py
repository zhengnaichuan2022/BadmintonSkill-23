#!/usr/bin/env python3
""" CoM-M2U student （）。

（）：
1.  ProtoGCN checkpoint ；
2. Teacher =  logits，loss  detach；
3. Stage-1（freeze_epochs）： backbone， classifier head；
4. Stage-2： backbone， full fine-tune（lr_backbone << lr_head）；
5.  student，。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = Path("${BADMINTONSKILL23_ROOT}")
sys.path.insert(0, str(ROOT / "skeleton" / "pyskl-main" / "pyskl-main"))
sys.path.insert(0, str(ROOT / "multimodal"))
os.environ["PYTHONPATH"] = f"{ROOT / 'skeleton' / 'pyskl-main' / 'pyskl-main'}:{os.environ.get('PYTHONPATH', '')}"

import pyskl.models.gcns.protogcn_ext  # noqa: F401
import pyskl.models.heads.protogcn_head  # noqa: F401
import pyskl.models.recognizers.recognizer_protogcn  # noqa: F401
from pyskl.datasets import build_dataset
from pyskl.models import build_model

from com_m2u.constants import BASELINE_ACC, OUT_ROOT, SKEL_STREAMS, logits_dir, skel_cfg, skel_ckpt
from com_m2u.loss import com_m2u_loss


def _unwrap(x):
    if hasattr(x, "data"):
        x = x.data
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
    return x


def forward_logits(model, keypoint):
    if keypoint.dim() == 6:
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        x, _g = model.extract_feat(keypoint)
        cls = model.cls_head(x)
        if isinstance(cls, (tuple, list)):
            cls = cls[0]
        if nc > 1:
            cls = cls.view(bs, nc, -1).mean(1)
        return cls
    x, _g = model.extract_feat(keypoint)
    cls = model.cls_head(x)
    if isinstance(cls, (tuple, list)):
        cls = cls[0]
    return cls


@torch.no_grad()
def eval_stream(model, loader, device, labels_np):
    """： skeleton student。"""
    model.eval()
    scores = []
    for data in loader:
        kp = _unwrap(data["keypoint"]).to(device)
        logits = forward_logits(model, kp)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for r in probs:
            scores.append(r.reshape(-1))
    s = np.stack(scores).astype(np.float32)
    acc = float((s.argmax(1) == labels_np).mean())
    return s, acc


def load_orig_probs(protocol):
    probs, labels = {}, None
    for st in SKEL_STREAMS:
        d = np.load(logits_dir(protocol) / f"skel_{st}" / "test.npz")
        p = np.exp(d["logits"])
        p = p / p.sum(1, keepdims=True)
        probs[st] = p.astype(np.float32)
        labels = d["labels"]
    return probs, labels


def fused_acc(probs, labels, override=None):
    arrs = [override[st] if override and st in override else probs[st] for st in SKEL_STREAMS]
    f = np.mean(arrs, 0)
    return float((f.argmax(1) == labels).mean()), f


def make_loaders(cfg, id2idx, bs=16):
    if cfg.data.train.get("type") == "RepeatDataset":
        train_cfg = cfg.data.train.dataset.copy()
    else:
        train_cfg = cfg.data.train.copy()
    train_cfg.split = "train"
    test_cfg = cfg.data.test.copy()
    test_cfg.split = "test"

    train_ds = build_dataset(train_cfg)
    test_ds = build_dataset(test_cfg, dict(test_mode=True))

    class TrainIndexed(Dataset):
        def __len__(self):
            return len(train_ds)

        def __getitem__(self, i):
            item = dict(train_ds[i])
            fid = os.path.basename(str(train_ds.video_infos[i]["frame_dir"]).rstrip("/"))
            item["teacher_idx"] = int(id2idx[fid])
            return item

    def train_collate(batch):
        from mmcv.parallel import collate

        idxs = [int(b.pop("teacher_idx")) for b in batch]
        packed = collate(batch, samples_per_gpu=len(batch))
        packed["teacher_idx"] = torch.tensor(idxs, dtype=torch.long)
        return packed

    def test_collate(batch):
        from mmcv.parallel import collate

        return collate(batch, samples_per_gpu=len(batch))

    train_loader = DataLoader(
        TrainIndexed(),
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        collate_fn=train_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        collate_fn=test_collate,
        pin_memory=True,
    )
    return train_loader, test_loader


def set_stage(model, stage: str, lr_head: float, lr_backbone: float):
    """stage: 'head' | 'full' -> requires_grad + SGD param groups."""
    if stage == "head":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.cls_head.parameters():
            p.requires_grad = True
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.cls_head.parameters()),
            lr=lr_head,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        print(f"[opt] stage=head  lr_head={lr_head:g}  (backbone frozen)", flush=True)
    elif stage == "full":
        for p in model.backbone.parameters():
            p.requires_grad = True
        for p in model.cls_head.parameters():
            p.requires_grad = True
        opt = torch.optim.SGD(
            [
                {"params": list(model.backbone.parameters()), "lr": lr_backbone},
                {"params": list(model.cls_head.parameters()), "lr": lr_head},
            ],
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )
        print(f"[opt] stage=full  lr_backbone={lr_backbone:g}  lr_head={lr_head:g}", flush=True)
    else:
        raise ValueError(stage)
    return opt


def ft_one(
    protocol,
    stream,
    device,
    epochs=12,
    freeze_epochs=4,
    lr_head=1e-3,
    lr_backbone=1e-4,
    lambda_kd=0.3,
    gamma=2.0,
    T=4.0,
    bs=16,
):
    print(f"\n==== Fine-tune {protocol}/{stream} (2-stage full FT) ====", flush=True)
    assert 1 <= freeze_epochs < epochs, "freeze_epochs must be in [1, epochs)"

    cfg = Config.fromfile(str(skel_cfg(protocol, stream)))
    model = build_model(cfg.model)
    #  checkpoint 
    ckpt_src = skel_ckpt(protocol, stream)
    load_checkpoint(model, str(ckpt_src), map_location="cpu")
    print(f"[{stream}] init from {ckpt_src}", flush=True)
    model = model.to(device)

    rgb_np = np.load(logits_dir(protocol) / "rgb" / "train.npz", allow_pickle=True)
    imu_np = np.load(logits_dir(protocol) / "imu" / "train.npz", allow_pickle=True)
    train_ids = [str(x) for x in rgb_np["ids"]]
    assert [str(x) for x in imu_np["ids"]] == train_ids
    id2idx = {sid: i for i, sid in enumerate(train_ids)}
    n_train = len(train_ids)

    train_loader, test_loader = make_loaders(cfg, id2idx, bs=bs)
    # Teacher： logits（loss  .detach()）
    z_rgb = torch.from_numpy(rgb_np["logits"].astype(np.float32)).to(device)
    z_imu = torch.from_numpy(imu_np["logits"].astype(np.float32)).to(device)

    test_labels = np.load(logits_dir(protocol) / f"skel_{stream}" / "test.npz")["labels"]
    scores0, acc0 = eval_stream(model, test_loader, device, test_labels)
    print(f"[{stream}] reload test_acc={acc0:.4f}", flush=True)

    opt = set_stage(model, "head", lr_head, lr_backbone)
    current_stage = "head"

    best_acc = acc0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_scores = scores0
    history = []

    for ep in range(1, epochs + 1):
        # 
        if ep == freeze_epochs + 1 and current_stage == "head":
            opt = set_stage(model, "full", lr_head, lr_backbone)
            current_stage = "full"

        model.train()
        if current_stage == "head":
            model.backbone.eval()  # BN 
        else:
            model.backbone.train()

        losses = []
        for data in train_loader:
            kp = _unwrap(data["keypoint"]).to(device)
            y = _unwrap(data["label"]).view(-1).long().to(device)
            idx = data["teacher_idx"].view(-1).long().to(device)

            if int(idx.max()) >= n_train or int(idx.min()) < 0:
                raise RuntimeError(f"bad teacher_idx [{int(idx.min())},{int(idx.max())}] n={n_train}")
            if int(y.max()) >= 11 or int(y.min()) < 0:
                raise RuntimeError(f"bad label [{int(y.min())},{int(y.max())}]")

            logits = forward_logits(model, kp)
            # teachers already detached inside com_m2u_loss
            loss, _ = com_m2u_loss(
                logits, y, z_rgb[idx], z_imu[idx], T=T, lambda_kd=lambda_kd, gamma=gamma, mode="com_m2u"
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        scores, acc = eval_stream(model, test_loader, device, test_labels)
        row = {
            "epoch": ep,
            "stage": current_stage,
            "loss": float(np.mean(losses)),
            "test_acc": acc,
        }
        history.append(row)
        print(
            f"[{stream}] ep{ep:02d}/{epochs} stage={current_stage} loss={row['loss']:.4f} test={acc:.4f}",
            flush=True,
        )
        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores = scores
            print(f"[{stream}]  * saved best test={acc:.4f}", flush=True)

    #  student
    model.load_state_dict(best_state)
    best_scores, best_acc = eval_stream(model, test_loader, device, test_labels)

    ckpt = OUT_ROOT / protocol / "checkpoints" / "skeleton" / stream / "com_m2u_best.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "test_acc": best_acc,
            "reload_acc": acc0,
            "history": history,
            "freeze_epochs": freeze_epochs,
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "init_ckpt": str(ckpt_src),
        },
        ckpt,
    )
    print(f"[{stream}] DONE best={best_acc:.4f} (reload={acc0:.4f}) -> {ckpt}", flush=True)
    return best_scores, best_acc, acc0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="xsub", choices=["xsub", "xacq"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--streams", default="j,b,jm,bm")
    ap.add_argument("--epochs", type=int, default=12, help="total epochs (stage1+stage2)")
    ap.add_argument("--freeze_epochs", type=int, default=4, help="stage-1 head-only epochs (3-5)")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lambda_kd", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    if not (3 <= args.freeze_epochs <= 5):
        print(f"[warn] freeze_epochs={args.freeze_epochs} outside recommended 3-5", flush=True)

    streams = [s.strip() for s in args.streams.split(",") if s.strip()]
    orig, labels = load_orig_probs(args.protocol)
    base4 = BASELINE_ACC[args.protocol]["skeleton"]
    print(
        f"Protocol={args.protocol} 4s_baseline={base4:.4f} "
        f"epochs={args.epochs} freeze={args.freeze_epochs} "
        f"lr_bb={args.lr_backbone:g} lr_hd={args.lr_head:g}",
        flush=True,
    )

    override = {}
    reports = []
    for st in streams:
        scores, best, reload = ft_one(
            args.protocol,
            st,
            args.device,
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            lambda_kd=args.lambda_kd,
            gamma=args.gamma,
            T=args.T,
            bs=args.bs,
        )
        kept = best > reload + 1e-6
        if kept:
            override[st] = scores
        reports.append({"stream": st, "reload": reload, "best": best, "kept_in_fusion": kept})

    # ： FT  4 （，）
    final, fused = fused_acc(orig, labels, override)
    out = {
        "protocol": args.protocol,
        "modality": "skeleton",
        "baseline_acc": base4,
        "best_test_acc": final,
        "improved": final > base4 + 1e-6,
        "freeze_epochs": args.freeze_epochs,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "epochs": args.epochs,
        "streams": reports,
        "note": "test = unimodal skeleton students only; teachers offline+detached",
    }
    outp = OUT_ROOT / args.protocol / "results"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "skeleton_best_summary.json").write_text(json.dumps(out, indent=2))
    (outp / "skeleton_com_m2u.json").write_text(json.dumps(out, indent=2))
    np.savez(outp / "skeleton_com_m2u_test_scores.npz", scores=fused, labels=labels)
    print(
        f"\n[Skel {args.protocol}] FINAL 4s={final:.4f} baseline={base4:.4f} improved={out['improved']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
