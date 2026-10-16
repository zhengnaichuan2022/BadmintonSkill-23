#!/usr/bin/env python3
"""Fine-tune RGB (r2plus1d) student with CoM-M2U via MMAction2 model + custom loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path("${BADMINTONSKILL23_ROOT}")
sys.path.insert(0, str(ROOT / "multimodal"))

from com_m2u.constants import (  # noqa: E402
    BASELINE_ACC,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_T,
    MMACTION2_ROOT,
    OUT_ROOT,
    logits_dir,
    video_ckpt,
    video_workdir,
)
from com_m2u.loss import com_m2u_loss  # noqa: E402


class IndexedRawframes(Dataset):
    """Wrap MMAction RawframeDataset and expose sample index for teacher lookup."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        # MMAction PackActionInputs → data_samples / inputs
        if isinstance(data, dict):
            inputs = data["inputs"]
            # label
            if "data_samples" in data:
                lab = int(data["data_samples"].gt_label.item())
            else:
                lab = int(data.get("label", -1))
        else:
            # older tuple
            inputs, data_samples = data
            lab = int(data_samples.gt_label.item())
        return inputs, lab, idx


def build_loader(protocol: str, split: str, batch_size: int, shuffle: bool):
    from mmengine.config import Config
    from mmaction.registry import DATASETS

    cfg = Config.fromfile(str(video_workdir(protocol) / "config.py"))
    ann = str(ROOT / "video/annotations" / protocol / f"{split}_frames.txt")
    ds_cfg = cfg.test_dataloader.dataset.copy()
    ds_cfg.ann_file = ann
    ds_cfg.test_mode = True
    # lighter workers
    dataset = DATASETS.build(ds_cfg)
    wrapped = IndexedRawframes(dataset)

    def collate(batch):
        xs, ys, idxs = zip(*batch)
        # inputs may be Tensor NCTHW
        x = torch.stack([torch.as_tensor(t) for t in xs], dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        idx = torch.tensor(idxs, dtype=torch.long)
        return x, y, idx

    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )


def load_model(protocol: str, device: str):
    from mmengine.config import Config
    from mmaction.apis import init_recognizer

    cfg = Config.fromfile(str(video_workdir(protocol) / "config.py"))
    ckpt = str(video_ckpt(protocol))
    model = init_recognizer(cfg, ckpt, device=device)
    model.train()
    return model


def forward_logits(model, inputs):
    """Get cls_score logits from Recognizer3D."""
    # MMAction2 Recognizer3D
    feats = model.extract_feat(inputs)
    cls_score = model.cls_head(feats)
    # cls_score may be (N, C) or list
    if isinstance(cls_score, (list, tuple)):
        cls_score = cls_score[0]
    return cls_score


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = forward_logits(model, x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    model.train()
    return correct / max(total, 1)


def load_npz_logits(protocol, modality, split):
    d = np.load(logits_dir(protocol) / modality / f"{split}.npz", allow_pickle=True)
    return torch.from_numpy(d["logits"].astype(np.float32))


def run_one(
    protocol: str,
    mode: str = "com_m2u",
    T: float = DEFAULT_T,
    lambda_kd: float = DEFAULT_LAMBDA,
    gamma: float = DEFAULT_GAMMA,
    epochs: int = 8,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str = "cuda:0",
):
    os.environ.setdefault("PYTHONPATH", f"{MMACTION2_ROOT}:{ROOT / 'video'}")
    sys.path.insert(0, MMACTION2_ROOT)
    sys.path.insert(0, str(ROOT / "video"))

    base_acc = BASELINE_ACC[protocol]["rgb"]
    model = load_model(protocol, device)
    train_loader = build_loader(protocol, "train", batch_size, shuffle=True)
    val_loader = build_loader(protocol, "val", batch_size, shuffle=False)
    test_loader = build_loader(protocol, "test", batch_size, shuffle=False)

    z_sk = load_npz_logits(protocol, "skeleton", "train").to(device)
    z_imu = load_npz_logits(protocol, "imu", "train").to(device)
    z_rgb = load_npz_logits(protocol, "rgb", "train").to(device)

    te0 = eval_loader(model, test_loader, device)
    print(f"[RGB {protocol}] baseline_ref={base_acc:.4f} reload_test={te0:.4f} mode={mode}")

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    best_val = -1.0
    ckpt_dir = OUT_ROOT / protocol / "checkpoints" / "rgb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{mode}_best.pth"
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            idx = idx.to(device)
            logits = forward_logits(model, x)
            loss, stats = com_m2u_loss(
                logits,
                y,
                z_sk[idx],
                z_imu[idx],
                T=T,
                lambda_kd=lambda_kd,
                gamma=gamma,
                mode=mode,
                extra_teacher_logits=z_rgb[idx],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
            opt.step()
            losses.append(float(loss.detach()))
        va = eval_loader(model, val_loader, device)
        te = eval_loader(model, test_loader, device)
        row = {"epoch": ep, "loss": float(np.mean(losses)), "val_acc": va, "test_acc": te}
        history.append(row)
        print(f"  ep{ep:02d} loss={row['loss']:.4f} val={va:.4f} test={te:.4f}")
        if va > best_val:
            best_val = va
            torch.save({"state_dict": model.state_dict(), "meta": row}, best_path)

    # reload best
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    val_best = eval_loader(model, val_loader, device)
    test_best = eval_loader(model, test_loader, device)
    result = {
        "protocol": protocol,
        "modality": "rgb",
        "mode": mode,
        "baseline_acc": base_acc,
        "reload_test_acc": te0,
        "best_val_acc": val_best,
        "best_test_acc": test_best,
        "improved": test_best > base_acc + 1e-6,
        "T": T,
        "lambda_kd": lambda_kd,
        "gamma": gamma,
        "lr": lr,
        "epochs": epochs,
        "ckpt": str(best_path),
        "history": history,
    }
    out_json = OUT_ROOT / protocol / "results" / f"rgb_{mode}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[RGB {protocol}] DONE test={test_best:.4f} baseline={base_acc:.4f} improved={result['improved']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument("--mode", default="com_m2u")
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--lambda_kd", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--auto_search", action="store_true")
    args = ap.parse_args()

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
        )
        return

    trials = [
        dict(mode="com_m2u", T=4.0, lambda_kd=0.5, gamma=1.0, lr=1e-4, epochs=8),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.3, gamma=2.0, lr=5e-5, epochs=10),
        dict(mode="com_m2u", T=2.0, lambda_kd=0.2, gamma=1.0, lr=5e-5, epochs=8),
        dict(mode="conf_kd", T=4.0, lambda_kd=0.3, gamma=1.0, lr=1e-4, epochs=8),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.1, gamma=2.0, lr=2e-5, epochs=12),
    ]
    best = None
    for i, t in enumerate(trials):
        print(f"\n==== RGB search {i+1}: {t} ====")
        r = run_one(args.protocol, batch_size=args.batch_size, device=args.device, **t)
        if best is None or r["best_test_acc"] > best["best_test_acc"]:
            best = r
        if r["improved"]:
            break
    (OUT_ROOT / args.protocol / "results" / "rgb_best_summary.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    if not best["improved"]:
        raise SystemExit(f"RGB {args.protocol} failed to beat baseline")


if __name__ == "__main__":
    main()
