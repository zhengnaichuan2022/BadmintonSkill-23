# BadmintonSkill-23 Skeleton Classification Models

Self-contained skeleton benchmark code: **8 architectures × 4 streams × 2 protocols = 60 configs**.

See **`config_examples/SKELETON_MODELS.md`** and **`skeleton_model_manifest.csv`** for the full index.

## Layout

```
classification/skeleton/
├── pyskl/                 # Full PYSKL (models, datasets, pipelines, apis, utils)
├── protogcn/              # Full ProtoGCN native package
├── blockgcn/              # BlockGCN network + COCO-17 graph
├── config_examples/
│   ├── xsub/              # 30 training configs
│   ├── xacq/              # 30 training configs
│   └── SKELETON_MODELS.md
└── skeleton_model_manifest.csv
```

## Model Families

| Family | Backbones | Config prefix examples |
|--------|-----------|------------------------|
| STGCN | STGCN | `stgcn`, `stgcn_b`, `stgcn_jm`, `stgcn_bm` |
| STGCN++ | STGCN (adaptive) | `stgcnpp`, `stgcnpp_*` |
| CTRGCN | CTRGCN | `ctrgcn`, `ctrgcn_*` |
| MSG3D | MSG3D | `msg3d`, `msg3d_*` |
| AAGCN | AAGCN | `aagcn`, `aagcn_*` |
| PoseC3D | ResNet3dSlowOnly | `posec3d_joint`, `posec3d_limb` |
| ProtoGCN | ProtoGCN | `protogcn`, `protogcn_*` |
| BlockGCN | BlockGCN | `blockgcn`, `blockgcn_*` |

## Paths

All configs import from `benchmark/paths_example.py` — no absolute host paths.

```bash
export BADMINTONSKILL23_ROOT=example/BadmintonSkill23
```

## Regenerate

```bash
Configs in `config_examples/xsub/` and `config_examples/xacq/` are templates for PYSKL. Install PYSKL separately and point dataset paths to your release root.
```
