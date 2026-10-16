# CoM-M2U: Complementary-Modality Fine-Tuning

**Complementary Modality to Unimodal (CoM-M2U)** — multimodal train, unimodal test for trimmed stroke **classification**.

> This is distinct from **TAD multimodal fusion** (RGB + skeleton 4-stream + IMU5 feature concat → ActionFormer), which lives under `../../tad/config_examples/rgb_skel_4stream_imu5_*.py`.

## Method summary

| Stage | Description |
|-------|-------------|
| 0 | Sample-aligned RGB + Skeleton (4-stream) + IMU, 11 classes, XSUB / XACQ |
| 1 | Export offline logits from best unimodal teachers (`export_logits.py`) |
| 2 | Leave-one-out teacher: e.g. RGB student ← confidence-fused Skeleton + IMU logits |
| 3 | Two-stage full fine-tune: freeze backbone → unfreeze with lower backbone LR |
| Test | **Single modality only** — no teacher at inference |

Loss (`com_m2u/loss.py`):

```
L = CE + λ · w · T² · KL(q_teacher || q_student)
w = 1 + γ · max(0, p_teacher(y) − p_student(y))
```

See `docs/loss.md` and `docs/method.md` for full derivation.

## Layout

```
multimodal/
├── README.md
├── run_com_m2u.sh
├── com_m2u/
│   ├── loss.py                  # Core CoM-M2U loss
│   ├── constants.py             # Env-based paths
│   ├── export_logits.py         # Stage 1: offline teacher logits
│   ├── finetune_rgb_simple.py   # RGB student (PyTorch / MMAction2)
│   ├── finetune_imu_simple.py   # IMU student (TensorFlow)
│   ├── finetune_skeleton_train.py  # Skeleton student (PYSKL ProtoGCN)
│   ├── infer_skeleton.py        # Single-GPU skeleton logits export
│   └── refusion_skeleton.py     # Re-fuse improved 4-stream checkpoints
├── docs/
│   ├── method.md
│   ├── loss.md
│   └── core_code_and_results.md
└── results/                     # Published JSON summaries (optional)
    ├── xsub/
    └── xacq/
```

## Environment

Set before running (no absolute host paths in code):

```bash
export BADMINTONSKILL23_PROJECT_ROOT=/path/to/project   # video/, training/, skeleton/
export BADMINTONSKILL23_COM_M2U_OUT=example/work/com_m2u
export MMACTION2_ROOT=../rgb/mmaction2                  # vendored in additional
export PYSKL_ROOT=../skeleton/pyskl
export MMACTION_PYTHON=python
export PYSKL_PYTHON=python
```

## Pipeline

```bash
cd benchmark/models/classification/multimodal
export PYTHONPATH=$PWD:$PYTHONPATH

# Stage 1: export offline logits
python -m com_m2u.export_logits --protocol xsub --modality all

# Stage 2: fine-tune each modality (recommended scripts)
python -m com_m2u.finetune_imu_simple --protocol xsub
python -m com_m2u.finetune_rgb_simple --protocol xsub --device cuda:0
python -m com_m2u.finetune_skeleton_train --protocol xsub --stream j --device cuda:0

# Or use orchestrator (edit python env vars inside first):
bash run_com_m2u.sh xsub all
```

## Baselines (test Acc)

| Protocol | RGB | IMU | Skeleton (4-stream) |
|----------|-----|-----|---------------------|
| XSUB | 0.5596 | 0.5827 | 0.6784 |
| XACQ | 0.5963 | 0.6153 | 0.7766 |

CoM-M2U improves each modality over its unimodal baseline while keeping **unimodal inference**.

## Related: TAD multimodal

For **temporal action detection**, multimodal fusion is early feature concatenation:

```
RGB R2+1D (512) + Skel j/b/jm/bm (4×384) + IMU5 Inception (256)
→ concat @ stride-4 → 2304-d → ActionFormer / TriDet / GTAD / TemporalMaxer
```

Configs: `benchmark/models/tad/config_examples/rgb_skel_4stream_imu5_*.py` (8 configs: 2 protocols × 4 detectors).
