#!/usr/bin/env python3
"""Smoke-test all PYSKL skeleton configs: build model, load batch, forward+backward."""

from __future__ import annotations

import os
import sys
import traceback

import pyskl.utils.mmcv_compat  # noqa: F401
import torch
from functools import partial
from mmcv import Config
from mmcv.parallel import collate, scatter
from torch.utils.data import DataLoader

PYSKL_ROOT = os.path.join(os.path.dirname(__file__), 'pyskl-main', 'pyskl-main')
sys.path.insert(0, PYSKL_ROOT)

from pyskl.datasets import build_dataset  # noqa: E402
from pyskl.models import build_model  # noqa: E402

PROTOCOLS = ('xsub', 'xacq')
MODELS = (
    'stgcnpp',
    'stgcn',
    'ctrgcn',
    'msg3d',
    'aagcn',
    'posec3d_joint',
)
SKELETON_DIR = os.path.dirname(os.path.abspath(__file__))


def unwrap_batch(batch, device):
    batch = scatter(batch, [device])[0]
    return batch


def validate_one(protocol: str, model_name: str, device: torch.device) -> tuple[bool, str, float | None]:
    cfg_path = os.path.join(SKELETON_DIR, 'configs', 'pyskl', protocol, f'{model_name}.py')
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.train)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate, samples_per_gpu=1),
    )
    batch = next(iter(loader))
    batch = unwrap_batch(batch, device)

    model = build_model(cfg.model).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    outputs = model.train_step(batch, optimizer)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()

    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return True, f'loss={float(loss):.4f}', mem_mb


def main() -> int:
    if not torch.cuda.is_available():
        print('CUDA not available')
        return 1

    device = torch.device('cuda:0')
    _ = torch.zeros(1, device=device)

    print(f'Validating {len(PROTOCOLS) * len(MODELS)} configs on {device}')
    print('-' * 72)

    ok_count = 0
    fail_count = 0
    rows = []

    for protocol in PROTOCOLS:
        for model_name in MODELS:
            tag = f'{protocol}/{model_name}'
            torch.cuda.empty_cache()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            try:
                ok, msg, mem = validate_one(protocol, model_name, device)
                if ok:
                    ok_count += 1
                    rows.append((tag, 'OK', msg, f'{mem:.0f} MB'))
                else:
                    fail_count += 1
                    rows.append((tag, 'FAIL', msg, '-'))
            except Exception as exc:
                fail_count += 1
                err = f'{type(exc).__name__}: {exc}'
                rows.append((tag, 'FAIL', err, '-'))
                traceback.print_exc()
                print()

    for tag, status, msg, mem in rows:
        print(f'[{status:4}] {tag:25} {msg:40} peak={mem}')

    print('-' * 72)
    print(f'Result: {ok_count} OK, {fail_count} FAIL / {ok_count + fail_count}')
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
