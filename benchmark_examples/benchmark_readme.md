# BadmintonSkill-23 Benchmark Code

Evaluation utilities, dataloader examples, config templates, and CoM-M2U core for the anonymous additional.

## Dependencies

```bash
pip install numpy pandas opencv-python
```

Full training requires external installs: **MMAction2**, **PYSKL**, **OpenTAD** (see `models/README.md`).

## Quick Start

From additional root:

```bash
python benchmark/dataloader_examples/run_all_loaders.py
python benchmark/classification_evaluation.py
python benchmark/tad_evaluation.py
```

## Layout

```
benchmark/
├── classification_evaluation.py
├── tad_evaluation.py
├── benchmark_utils.py
├── paths_example.py
├── dataloader_examples/
├── config_examples/
├── example_predictions/
└── models/                  # adapters + CoM-M2U + config templates
```

## Dataloader Examples

| Script | Output |
|--------|--------|
| `load_rgb_sample.py` | `(T, H, W, 3)` uint8 |
| `load_skeleton_sample.py` | `(T, 17, 3)` COCO-17 |
| `load_imu_sample.py` | `(T, 30)` float32 |
| `load_classification_annotation.py` | clip + stroke timing fields |
| `load_tad_annotation.py` | segments for `V*` video IDs |

## Evaluation

- **Classification:** top-1 accuracy, mean class accuracy
- **TAD:** mAP @ tIoU {0.3, 0.4, 0.5, 0.6, 0.7}

Demo predictions in `example_predictions/` achieve perfect scores on bundled samples.

## TAD Protocol IDs

Official splits list **295** class-agnostic IDs `V000001`–`V000295`. Action labels appear only in ground-truth annotations, not in protocol filenames.

## Third-Party Licenses

MMAction2, PYSKL, and OpenTAD remain under Apache-2.0 upstream. This package does not redistribute their full source trees.
