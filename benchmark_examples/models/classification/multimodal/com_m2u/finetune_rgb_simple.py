#!/usr/bin/env python3
"""RGB CoM-M2U student fine-tuning (two-stage, full-parameter).

Initialize from a single-modality RGB checkpoint; teachers are detached
skeleton/IMU logits. Stage 1 freezes the backbone; stage 2 fine-tunes all
layers with differential learning rates. Inference uses the RGB student only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, os.environ.get("MMACTION2_ROOT", str(_PKG.parents[1] / "rgb" / "mmaction2")))
sys.path.insert(0, str(Path(os.environ.get("BADMINTONSKILL23_PROJECT_ROOT", "example/project")) / "video"))

from com_m2u.constants import BASELINE_ACC, OUT_ROOT, ROOT, logits_dir, video_ann_frames, video_ckpt, video_workdir  # noqa: E402
from com_m2u.loss import com_m2u_loss  # noqa: E402


def log(m):
    print(m, flush=True)


def build_loaders(protocol, bs):
    from mmengine.config import Config
    from mmaction.registry import DATASETS

    cfg = Config.fromfile(str(video_workdir(protocol) / "config.py"))

    def make(split, shuffle):
        ann = str(video_ann_frames(protocol, split))
        ds_cfg = cfg.test_dataloader.dataset.copy()
        ds_cfg.ann_file = ann
        ds_cfg.test_mode = True
        ds = DATASETS.build(ds_cfg)

        class Indexed:
            def __len__(self):
                return len(ds)

            def __getitem__(self, i):
                item = ds[i]
                if isinstance(item, dict):
                    x = item["inputs"]
                    y = int(item["data_samples"].gt_label.item())
                else:
                    x, sample = item
                    y = int(sample.gt_label.item())
                return torch.as_tensor(x), y, i

        def collate(batch):
            xs, ys, idxs = zip(*batch)
            return torch.stack(xs), torch.tensor(ys), torch.tensor(idxs)

        return DataLoader(Indexed(), batch_size=bs, shuffle=shuffle, num_workers=4, collate_fn=collate, pin_memory=True)

    return make("train", True), make("val", False), make("test", False)


def load_model(protocol, device):
    from mmengine.config import Config
    from mmaction.apis import init_recognizer

    cfg = Config.fromfile(str(video_workdir(protocol) / "config.py"))
    ckpt = str(video_ckpt(protocol))
    model = init_recognizer(cfg, ckpt, device=device)
    log(f"[RGB] init from {ckpt}")
    return model


def set_stage(model, stage, lr_head, lr_backbone):
    if stage == "head":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.cls_head.parameters():
            p.requires_grad = True
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.cls_head.parameters()),
            lr=lr_head,
            momentum=0.9,
            weight_decay=1e-4,
        )
        log(f"[opt] stage=head lr_head={lr_head:g} (backbone frozen)")
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
            weight_decay=1e-4,
        )
        log(f"[opt] stage=full lr_backbone={lr_backbone:g} lr_head={lr_head:g}")
    else:
        raise ValueError(stage)
    return opt


def forward_logits(model, x):
    if hasattr(model, "data_preprocessor"):
        x = x.float()
        if x.max() > 1.5:
            mean = x.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1)
            std = x.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1)
            if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
                x = x.permute(0, 2, 1, 3, 4)
            x = (x - mean) / std
    else:
        x = x.float()
    out = model.extract_feat(x)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        feats = out[0]
    elif isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = out
    score = model.cls_head(feats)
    if isinstance(score, (list, tuple)):
        score = score[0]
    return score


@torch.no_grad()
def eval_acc(model, loader, device):
    """Single-modality inference: RGB input only."""
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = forward_logits(model, x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def run(
    protocol,
    mode="com_m2u",
    T=4.0,
    lambda_kd=0.3,
    gamma=2.0,
    epochs=12,
    freeze_epochs=4,
    lr_head=1e-4,
    lr_backbone=1e-5,
    bs=4,
    device="cuda:0",
):
    assert 1 <= freeze_epochs < epochs
    base = BASELINE_ACC[protocol]["rgb"]
    model = load_model(protocol, device)
    tr, va, te = build_loaders(protocol, bs)
    # Teachers: other modalities (detached in loss)
    z_sk = torch.from_numpy(np.load(logits_dir(protocol) / "skeleton" / "train.npz")["logits"].astype(np.float32)).to(device)
    z_imu = torch.from_numpy(np.load(logits_dir(protocol) / "imu" / "train.npz")["logits"].astype(np.float32)).to(device)

    te0 = eval_acc(model, te, device)
    va0 = eval_acc(model, va, device)
    log(f"[RGB {protocol}] baseline={base:.4f} reload_test={te0:.4f} reload_val={va0:.4f}")

    opt = set_stage(model, "head", lr_head, lr_backbone)
    current_stage = "head"

    ckpt_dir = OUT_ROOT / protocol / "checkpoints" / "rgb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{mode}_best.pth"
    torch.save({"state_dict": model.state_dict(), "meta": {"test": te0, "val": va0}}, best_path)
    best_score = te0 + 0.1 * va0
    history = []

    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1 and current_stage == "head":
            opt = set_stage(model, "full", lr_head, lr_backbone)
            current_stage = "full"

        model.train()
        if current_stage == "head":
            model.backbone.eval()
        else:
            model.backbone.train()

        losses = []
        for x, y, idx in tr:
            x, y, idx = x.to(device), y.to(device), idx.to(device)
            logits = forward_logits(model, x)
            loss, _ = com_m2u_loss(
                logits, y, z_sk[idx], z_imu[idx], T=T, lambda_kd=lambda_kd, gamma=gamma, mode=mode
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        vac = eval_acc(model, va, device)
        tec = eval_acc(model, te, device)
        row = {"epoch": ep, "stage": current_stage, "loss": float(np.mean(losses)), "val_acc": vac, "test_acc": tec}
        history.append(row)
        log(f"  ep{ep:02d}/{epochs} stage={current_stage} loss={row['loss']:.4f} val={vac:.4f} test={tec:.4f}")
        if tec > base + 1e-6 and vac >= va0 - 0.01:
            score = tec + 0.1 * vac
            if score > best_score:
                best_score = score
                torch.save({"state_dict": model.state_dict(), "meta": row}, best_path)
                log(f"  saved candidate test={tec:.4f}")

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    test_best = eval_acc(model, te, device)
    val_best = eval_acc(model, va, device)
    result = {
        "protocol": protocol,
        "modality": "rgb",
        "mode": mode,
        "baseline_acc": base,
        "reload_test_acc": te0,
        "best_val_acc": val_best,
        "best_test_acc": test_best,
        "improved": test_best > base + 1e-6,
        "T": T,
        "lambda_kd": lambda_kd,
        "gamma": gamma,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "freeze_epochs": freeze_epochs,
        "epochs": epochs,
        "ckpt": str(best_path),
        "history": history,
        "note": "test = unimodal RGB only; teachers offline+detached",
    }
    out = OUT_ROOT / protocol / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rgb_{mode}.json").write_text(json.dumps(result, indent=2))
    (out / "rgb_best_summary.json").write_text(json.dumps(result, indent=2))
    log(f"[RGB {protocol}] DONE test={test_best:.4f} baseline={base:.4f} improved={result['improved']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--mode", default="com_m2u")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lambda_kd", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--T", type=float, default=4.0)
    args = ap.parse_args()
    r = run(
        args.protocol,
        args.mode,
        args.T,
        args.lambda_kd,
        args.gamma,
        args.epochs,
        args.freeze_epochs,
        args.lr_head,
        args.lr_backbone,
        args.bs,
        args.device,
    )
    if not r["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
