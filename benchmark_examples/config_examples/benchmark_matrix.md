# BadmintonSkill-23 Full Benchmark Matrix

Index of TAD and classification experiments. Model code is bundled under `benchmark/models/`.

## TAD Experiments (96 total)

### Feature-based TAD (72 configs)

Detectors: ActionFormer, TriDet, GTAD, TemporalMaxer × protocols xsub/xacq.

| Task | Feature key | Input dim |
|------|-------------|-----------|
| 1_j / 1_b | `skel_j`, `skel_b` | 384 |
| 2 | `skel_jb` | 768 |
| 3 | `skel_4stream` | 1536 |
| 4 | `rgb` | 512 |
| 5_* | `rgb_skel_*` | 896–2048 |

Example config: `models/tad/config_examples/rgb_xsub_actionformer.py`

### Multimodal TAD — RGB + Skeleton 4-stream + IMU5 (8 configs)

Input dim **2304** = 512 + 4×384 + 256.

Example: `models/tad/config_examples/rgb_skel_4stream_imu5_xsub_actionformer.py`

Feature paths (example):

```
example/BadmintonSkill23/tad/features/rgb/r2plus1d_{protocol}_stride4/
example/BadmintonSkill23/tad/features/skeleton/protogcn_{stream}_{protocol}_stride4/
example/BadmintonSkill23/tad/features/imu/inception5_{protocol}_stride4/
```

### IMU End-to-End TAD (16 configs)

| Variant | Channels | Dataset class |
|---------|----------|---------------|
| IMU1 e2e | 6 | `BadmintonSkill23IMUSlidingDataset` |
| IMU5 e2e | 30 | `BadmintonSkill23IMUSlidingDataset` |

Annotations: `example/BadmintonSkill23/tad/annotations/imu/{protocol}/anno.json`

---

## Classification Experiments

### IMU 5-node (26 configs)

Code: `models/classification/imu/models/` — 13 TensorFlow/Keras architectures.

### RGB (20 configs)

Code: `models/classification/rgb/mmaction2/` (vendored MMAction2) + `badmintonskill23_ext/`.

10 XSUB example configs: `models/classification/rgb/configs/*.py`

### Skeleton (12+ configs)

Code: `models/classification/skeleton/pyskl_models/` + `protogcn_models/`.

Example configs: `models/classification/skeleton/config_examples/`

---

## Path Convention

All configs use **example paths** from `benchmark/paths_example.py`:

| Variable | Default |
|----------|---------|
| `BADMINTONSKILL23_ROOT` | `example/BadmintonSkill23` |
| `BADMINTONSKILL23_EXP_ROOT` | `example/experiments/tad` |
| `BADMINTONSKILL23_WORK_ROOT` | `example/work` |

No host-specific absolute paths appear in the additional benchmark code.
