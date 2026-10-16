#!/usr/bin/env python3
"""Generate Beiyu TAD benchmark configs.

Experiment matrix (72 configs):
  1) skel_j, skel_b  ×  ActionFormer / TriDet / GTAD / TemporalMaxer  ×  xsub / xacq
  2) skel_jb (j+b concat at train time)  ×  same models  ×  xsub / xacq
  3) skel_4stream (j+b+jm+bm)  ×  same models  ×  xsub / xacq
  4) rgb (R2+1D)  ×  same models  ×  xsub / xacq
  5) RGB + skeleton multimodal (LoadFeats channel concat):
       rgb_skel_j      (512+384=896)
       rgb_skel_b      (512+384=896)
       rgb_skel_jb     (512+768=1280)
       rgb_skel_4stream (512+1536=2048)
     ×  same models  ×  xsub / xacq

Training schedule scales with input feature dimension (epochs, val_start, batch, lr).
Multimodal concat dim = sum of per-modality channels; projection.in_channels must match.
Neck/head hidden dim (512) stays fixed — only TriDet sgp_mlp_dim scales with input width.

Usage:
  cd OpenTAD-main
  python3 tools/generate_beiyu_tad_benchmark.py
  python3 tools/generate_beiyu_tad_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OPENTAD_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = OPENTAD_ROOT / "configs" / "beiyu" / "benchmark"
MANIFEST_PATH = OPENTAD_ROOT / "configs" / "beiyu" / "benchmark_manifest.json"

PROTOCOLS = ("xsub", "xacq")
NUM_CLASSES = 11

FEATURE_SETS = {
    "skel_j": {
        "title": "Skeleton joint (j)",
        "streams": ("j",),
        "dim": 384,
        "task": "1_j",
        "feat_type": "skeleton",
    },
    "skel_b": {
        "title": "Skeleton bone (b)",
        "streams": ("b",),
        "dim": 384,
        "task": "1_b",
        "feat_type": "skeleton",
    },
    "skel_jb": {
        "title": "Skeleton j+b concat",
        "streams": ("j", "b"),
        "dim": 768,
        "task": "2_jb",
        "feat_type": "skeleton",
    },
    "skel_4stream": {
        "title": "Skeleton 4-stream j+b+jm+bm",
        "streams": ("j", "b", "jm", "bm"),
        "dim": 1536,
        "task": "3_4stream",
        "feat_type": "skeleton",
    },
    "rgb": {
        "title": "RGB R2+1D",
        "streams": ("rgb",),
        "dim": 512,
        "task": "4_rgb",
        "feat_type": "rgb",
    },
    "rgb_skel_j": {
        "title": "RGB R2+1D + Skeleton joint (j)",
        "streams": ("rgb", "j"),
        "dim": 896,  # 512 + 384
        "task": "5_j",
        "feat_type": "multimodal",
    },
    "rgb_skel_b": {
        "title": "RGB R2+1D + Skeleton bone (b)",
        "streams": ("rgb", "b"),
        "dim": 896,
        "task": "5_b",
        "feat_type": "multimodal",
    },
    "rgb_skel_jb": {
        "title": "RGB R2+1D + Skeleton j+b",
        "streams": ("rgb", "j", "b"),
        "dim": 1280,  # 512 + 768
        "task": "5_jb",
        "feat_type": "multimodal",
    },
    "rgb_skel_4stream": {
        "title": "RGB R2+1D + Skeleton 4-stream (ProtoGCN j/b/jm/bm)",
        "streams": ("rgb", "j", "b", "jm", "bm"),
        "dim": 2048,  # 512 + 384*4
        "task": "5_4stream",
        "feat_type": "multimodal",
    },
}


def train_hparams(dim: int) -> dict:
    """Scale epochs / lr / batch size with feature channel count."""
    if dim <= 384:
        return {
            "max_epoch": 35,
            "warmup_epoch": 5,
            "val_start_epoch": 25,
            "batch_size": 2,
            "lr_adamw": 1e-4,
            "weight_decay": 0.05,
            "lr_adam": 1e-3,
            "gtad_milestones": [20, 30],
            "gtad_val_start": 20,
        }
    if dim <= 512:
        return {
            "max_epoch": 40,
            "warmup_epoch": 5,
            "val_start_epoch": 30,
            "batch_size": 2,
            "lr_adamw": 1e-4,
            "weight_decay": 0.05,
            "lr_adam": 1e-3,
            "gtad_milestones": [24, 34],
            "gtad_val_start": 24,
        }
    if dim <= 768:
        return {
            "max_epoch": 45,
            "warmup_epoch": 6,
            "val_start_epoch": 35,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "weight_decay": 0.05,
            "lr_adam": 8e-4,
            "gtad_milestones": [28, 38],
            "gtad_val_start": 26,
        }
    if dim <= 896:
        return {
            "max_epoch": 50,
            "warmup_epoch": 7,
            "val_start_epoch": 38,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "weight_decay": 0.05,
            "lr_adam": 8e-4,
            "gtad_milestones": [32, 44],
            "gtad_val_start": 28,
        }
    if dim <= 1280:
        return {
            "max_epoch": 52,
            "warmup_epoch": 8,
            "val_start_epoch": 40,
            "batch_size": 1,
            "lr_adamw": 8e-5,
            "weight_decay": 0.05,
            "lr_adam": 8e-4,
            "gtad_milestones": [34, 46],
            "gtad_val_start": 30,
        }
    if dim <= 1536:
        return {
            "max_epoch": 55,
            "warmup_epoch": 8,
            "val_start_epoch": 42,
            "batch_size": 1,
            "lr_adamw": 8e-5,
            "weight_decay": 0.05,
            "lr_adam": 8e-4,
            "gtad_milestones": [35, 48],
            "gtad_val_start": 32,
        }
    if dim <= 2048:
        return {
            "max_epoch": 60,
            "warmup_epoch": 8,
            "val_start_epoch": 45,
            "batch_size": 1,
            "lr_adamw": 5e-5,
            "weight_decay": 0.05,
            "lr_adam": 5e-4,
            "gtad_milestones": [38, 52],
            "gtad_val_start": 35,
        }
    return {
        "max_epoch": 60,
        "warmup_epoch": 8,
        "val_start_epoch": 45,
        "batch_size": 1,
        "lr_adamw": 5e-5,
        "weight_decay": 0.05,
        "lr_adam": 5e-4,
        "gtad_milestones": [38, 52],
        "gtad_val_start": 35,
    }


def tridet_hparams(dim: int) -> dict:
    """TriDet needs sgp_mlp_dim / lr scaled with input dim (default assumes I3D 2048-d)."""
    base = train_hparams(dim)
    if dim <= 384:
        return {
            **base,
            "max_epoch": 45,
            "warmup_epoch": 8,
            "val_start_epoch": 32,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "sgp_mlp_dim": 256,
            "out_channels": 512,
        }
    if dim <= 512:
        return {
            **base,
            "max_epoch": 45,
            "warmup_epoch": 8,
            "val_start_epoch": 32,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "sgp_mlp_dim": 384,
            "out_channels": 512,
        }
    if dim <= 768:
        return {
            **base,
            "max_epoch": 45,
            "warmup_epoch": 6,
            "val_start_epoch": 35,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "sgp_mlp_dim": 512,
            "out_channels": 512,
        }
    if dim <= 896:
        return {
            **base,
            "max_epoch": 50,
            "warmup_epoch": 7,
            "val_start_epoch": 38,
            "batch_size": 2,
            "lr_adamw": 8e-5,
            "sgp_mlp_dim": 640,
            "out_channels": 512,
        }
    if dim <= 1280:
        return {
            **base,
            "max_epoch": 52,
            "warmup_epoch": 8,
            "val_start_epoch": 40,
            "batch_size": 1,
            "lr_adamw": 8e-5,
            "sgp_mlp_dim": 768,
            "out_channels": 512,
        }
    if dim <= 1536:
        return {
            **base,
            "max_epoch": 55,
            "warmup_epoch": 8,
            "val_start_epoch": 42,
            "batch_size": 1,
            "lr_adamw": 5e-5,
            "sgp_mlp_dim": 768,
            "out_channels": 512,
        }
    if dim <= 2048:
        return {
            **base,
            "max_epoch": 60,
            "warmup_epoch": 8,
            "val_start_epoch": 45,
            "batch_size": 1,
            "lr_adamw": 5e-5,
            "sgp_mlp_dim": 1024,
            "out_channels": 512,
        }
    return {
        **base,
        "max_epoch": 60,
        "warmup_epoch": 8,
        "val_start_epoch": 45,
        "batch_size": 1,
        "lr_adamw": 5e-5,
        "sgp_mlp_dim": 1024,
        "out_channels": 512,
    }


def _gtad_groups(dim: int) -> int:
    for g in (32, 16, 8, 4):
        if dim % g == 0:
            return g
    return 1


def _fmt_lr(v: float) -> str:
    return f"{v:.0e}".replace("e-0", "e-")


def model_training_lines(model_key: str, dim: int) -> dict[str, str]:
    hp = train_hparams(dim)
    if model_key == "tridet":
        hp = tridet_hparams(dim)
    bs = hp["batch_size"]
    solver = (
        f"solver = dict("
        f"train=dict(batch_size={bs}, num_workers=2), "
        f"val=dict(batch_size=1, num_workers=1), "
        f"test=dict(batch_size=1, num_workers=1)"
    )
    if model_key != "gtad":
        solver += ", clip_grad_norm=1, ema=True"
    else:
        solver += ", clip_grad_norm=0"
    solver += ")"

    if model_key == "gtad":
        ms = hp["gtad_milestones"]
        optimizer = (
            f"optimizer = dict(type='Adam', lr={_fmt_lr(hp['lr_adam'])}, weight_decay=1e-4)"
        )
        scheduler = (
            f"scheduler = dict(type='MultiStepLR', max_epoch={hp['max_epoch']}, "
            f"milestones={ms}, gamma=0.1)"
        )
        workflow = (
            f"workflow = dict(logging_interval=20, checkpoint_interval=1, "
            f"val_loss_interval=1, val_eval_interval=1, val_start_epoch={hp['gtad_val_start']})"
        )
    else:
        optimizer = (
            f"optimizer = dict(type='AdamW', lr={_fmt_lr(hp['lr_adamw'])}, "
            f"weight_decay={hp['weight_decay']}, paramwise=True)"
        )
        scheduler = (
            f"scheduler = dict(type='LinearWarmupCosineAnnealingLR', "
            f"warmup_epoch={hp['warmup_epoch']}, max_epoch={hp['max_epoch']})"
        )
        workflow = (
            f"workflow = dict(logging_interval=20, checkpoint_interval=1, "
            f"val_loss_interval=1, val_eval_interval=1, val_start_epoch={hp['val_start_epoch']})"
        )

    if model_key == "actionformer":
        pdrop = 0.2 if dim >= 768 else 0.1
        projection = f"projection=dict(in_channels={dim}, input_pdrop={pdrop}),"
    elif model_key == "tridet":
        projection = (
            f"projection=dict(in_channels={dim}, out_channels={hp['out_channels']}, "
            f"sgp_mlp_dim={hp['sgp_mlp_dim']}),"
        )
    elif model_key == "temporalmaxer":
        projection = f"projection=dict(in_channels={dim}),"
    else:  # gtad
        projection = (
            f"projection=dict(in_channels={dim}, num_convs=2, "
            f"conv_cfg=dict(groups={_gtad_groups(dim)})),"
        )

    base_models = {
        "actionformer": "../../_base_/models/actionformer.py",
        "tridet": "../../_base_/models/tridet.py",
        "temporalmaxer": "../../_base_/models/temporalmaxer.py",
        "gtad": "../../_base_/models/gtad.py",
    }
    return {
        "base_model": base_models[model_key],
        "projection": projection,
        "extra_model": "roi_head=dict(proposal_head=dict(num_classes=11))," if model_key == "gtad" else "",
        "solver": solver,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "workflow": workflow,
        "hparams": hp,
    }


MODEL_KEYS = ("actionformer", "tridet", "temporalmaxer", "gtad")


def feature_paths(protocol: str, feat_key: str) -> list[str]:
    feat = FEATURE_SETS[feat_key]
    if feat["feat_type"] == "rgb":
        return [f"data/beiyu/features/rgb/r2plus1d_{protocol}_stride4/"]
    if feat["feat_type"] == "multimodal":
        paths = [f"data/beiyu/features/rgb/r2plus1d_{protocol}_stride4/"]
        for stream in feat["streams"]:
            if stream == "rgb":
                continue
            paths.append(f"data/beiyu/features/skeleton/protogcn_{stream}_{protocol}_stride4/")
        return paths
    return [
        f"data/beiyu/features/skeleton/protogcn_{s}_{protocol}_stride4/"
        for s in feat["streams"]
    ]


def dataset_base_py(protocol: str, data_paths: list[str]) -> str:
    ann = f"data/beiyu/annotations/{protocol}/anno.json"
    if len(data_paths) == 1:
        data_path_line = f'data_path = "{data_paths[0]}"'
        load_feats = 'dict(type="LoadFeats", feat_format="npy")'
    else:
        paths_repr = ",\n    ".join(f'"{p}"' for p in data_paths)
        data_path_line = f"data_path = [\n    {paths_repr},\n]"
        fmts = ", ".join('"npy"' for _ in data_paths)
        load_feats = f'dict(type="LoadFeats", feat_format=[{fmts}])'

    lines = [
        'dataset_type = "BeiyuSlidingDataset"',
        f'annotation_path = "{ann}"',
        'class_map = "data/beiyu/annotations/category_idx.txt"',
        'block_list = "data/beiyu/annotations/missing_features_blocklist.txt"',
        data_path_line,
        "window_size = 256",
        "feature_stride = 4",
        "sample_stride = 1",
        "offset_frames = 0",
        "",
        f"_load = {load_feats}",
        "_common = dict(",
        "    type=dataset_type,",
        "    ann_file=annotation_path,",
        "    block_list=block_list,",
        "    class_map=class_map,",
        "    data_path=data_path,",
        "    filter_gt=False,",
        "    feature_stride=feature_stride,",
        "    sample_stride=sample_stride,",
        "    window_size=window_size,",
        "    offset_frames=offset_frames,",
        ")",
        "",
        "dataset = dict(",
        "    train=dict(",
        "        **_common,",
        '        subset_name="training",',
        "        window_overlap_ratio=0.25,",
        "        pipeline=[",
        "            _load,",
        '            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),',
        '            dict(type="SlidingWindowTrunc", with_mask=True),',
        '            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),',
        '            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),',
        "        ],",
        "    ),",
        "    val=dict(",
        "        **_common,",
        '        subset_name="validation",',
        "        window_overlap_ratio=0.25,",
        "        pipeline=[",
        "            _load,",
        '            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),',
        '            dict(type="SlidingWindowTrunc", with_mask=True),',
        '            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),',
        '            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),',
        "        ],",
        "    ),",
        "    test=dict(",
        "        **_common,",
        '        subset_name="testing",',
        "        test_mode=True,",
        "        window_overlap_ratio=0.5,",
        "        pipeline=[",
        "            _load,",
        '            dict(type="ConvertToTensor", keys=["feats"]),',
        '            dict(type="SlidingWindowTrunc", with_mask=True),',
        '            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),',
        '            dict(type="Collect", inputs="feats", keys=["masks"]),',
        "        ],",
        "    ),",
        ")",
        "",
        "evaluation = dict(",
        '    type="mAP",',
        '    subset="validation",',
        "    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],",
        "    ground_truth_filename=annotation_path,",
        ")",
    ]
    return "\n".join(lines)


def render_config(feat_key: str, protocol: str, model_key: str) -> str:
    feat = FEATURE_SETS[feat_key]
    dim = feat["dim"]
    model = model_training_lines(model_key, dim)
    hp = model["hparams"]
    paths = feature_paths(protocol, feat_key)
    dataset_py = dataset_base_py(protocol, paths)
    work_dir = f"exps/beiyu/benchmark/{protocol}/{feat_key}_{protocol}_{model_key}"

    lines = [
        f"# Auto-generated: {feat['title']} | {protocol} | {model_key}",
        f"# Task group: {feat['task']}  in_channels={dim}",
    ]
    if model_key == "tridet":
        lines.append(
            f"# train schedule: max_epoch={hp['max_epoch']} val_start={hp['val_start_epoch']} "
            f"batch={hp['batch_size']} sgp_mlp_dim={hp['sgp_mlp_dim']} lr={_fmt_lr(hp['lr_adamw'])}"
        )
    elif model_key == "gtad":
        lines.append(
            f"# train schedule: max_epoch={hp['max_epoch']} val_start={hp['gtad_val_start']} "
            f"batch={hp['batch_size']} milestones={hp['gtad_milestones']}"
        )
    else:
        lines.append(
            f"# train schedule: max_epoch={hp['max_epoch']} val_start={hp['val_start_epoch']} "
            f"batch={hp['batch_size']}"
        )
    lines.extend(["", f'_base_ = ["{model["base_model"]}"]', "", dataset_py, "", "model = dict(", f"    {model['projection']}",])
    if model_key != "gtad":
        lines.append(f"    rpn_head=dict(num_classes={NUM_CLASSES}),")
    if model["extra_model"]:
        lines.append(f"    {model['extra_model']}")
    lines.extend(
        [
            ")",
            "",
            model["solver"],
            model["optimizer"],
            model["scheduler"],
            "",
            "inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)",
            "post_processing = dict(",
        ]
    )
    if model_key == "gtad":
        lines.extend(
            [
                "    external_cls=dict(",
                "        type='StandardClassifier',",
                f"        path='data/beiyu/classifiers/{feat_key}_{protocol}.json',",
                "        topk=11,",
                "        apply_softmax=False,",
                "    ),",
            ]
        )
    lines.extend(
        [
            "    nms=dict(",
            "        use_soft_nms=True,",
            "        sigma=0.5,",
            "        max_seg_num=2000,",
            "        iou_threshold=0.1,",
            "        min_score=0.001,",
            "        multiclass=True,",
            "        voting_thresh=0.7,",
            "    ),",
            "    save_dict=True,",
            ")",
            "",
            model["workflow"],
            f'work_dir = "{work_dir}"',
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for feat_key in FEATURE_SETS:
        for protocol in PROTOCOLS:
            for model_key in MODEL_KEYS:
                name = f"{feat_key}_{protocol}_{model_key}.py"
                rel_cfg = f"configs/beiyu/benchmark/{name}"
                content = render_config(feat_key, protocol, model_key)
                hp = train_hparams(FEATURE_SETS[feat_key]["dim"])
                out_path = OUT_DIR / name
                entry = {
                    "config": rel_cfg,
                    "task": FEATURE_SETS[feat_key]["task"],
                    "feature": feat_key,
                    "protocol": protocol,
                    "model": model_key,
                    "dim": FEATURE_SETS[feat_key]["dim"],
                    "max_epoch": hp["max_epoch"],
                    "batch_size": hp["batch_size"],
                    "work_dir": f"exps/beiyu/benchmark/{protocol}/{feat_key}_{protocol}_{model_key}",
                }
                if model_key == "tridet":
                    td = tridet_hparams(FEATURE_SETS[feat_key]["dim"])
                    entry["sgp_mlp_dim"] = td["sgp_mlp_dim"]
                    entry["lr_adamw"] = td["lr_adamw"]
                manifest.append(entry)
                if not args.dry_run:
                    out_path.write_text(content + "\n", encoding="utf-8")

    if not args.dry_run:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated {len(manifest)} configs -> {OUT_DIR.relative_to(OPENTAD_ROOT)}")
    print(f"Manifest -> {MANIFEST_PATH.relative_to(OPENTAD_ROOT)}")
    if args.dry_run:
        print("(dry-run, files not written)")


if __name__ == "__main__":
    main()
