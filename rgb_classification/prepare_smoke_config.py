#!/usr/bin/env python3
"""Write a short smoke-test MMAction2 config (IterBasedTrainLoop, N iters)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Source MMAction2 config .py")
    parser.add_argument("dst", type=Path, help="Output smoke config .py")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip validation during smoke (faster)",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.src))
    train_cfg = dict(type="IterBasedTrainLoop", max_iters=args.iters)
    if not args.no_val:
        train_cfg["val_interval"] = args.iters
    cfg.train_cfg = train_cfg
    if hasattr(cfg, "default_hooks") and cfg.default_hooks is not None:
        cfg.default_hooks.checkpoint = dict(
            type="CheckpointHook", interval=args.iters, max_keep_ckpts=1
        )
        cfg.default_hooks.logger = dict(type="LoggerHook", interval=10)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(args.dst))
    print(args.dst.resolve())


if __name__ == "__main__":
    main()
