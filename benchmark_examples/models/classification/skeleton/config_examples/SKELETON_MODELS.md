# BadmintonSkill-23 Skeleton Classification Models

Complete inventory of skeleton benchmark models bundled in this additional.

## Summary

| Group | Architectures | Streams | Protocols | Configs |
|-------|---------------|---------|-----------|---------|
| PYSKL GCN | STGCN, STGCN++, CTRGCN, MSG3D, AAGCN | j, b, jm, bm | xsub, xacq | 40 |
| PYSKL 3D | PoseC3D | joint, limb | xsub, xacq | 4 |
| ProtoGCN | ProtoGCN | j, b, jm, bm | xsub, xacq | 8 |
| BlockGCN | BlockGCN | j, b, jm, bm | xsub, xacq | 8 |
| **Total** | **8 base architectures** | **4 streams** | **2 protocols** | **60** |

Stream codes: **j** joint, **b** bone, **jm** joint-motion, **bm** bone-motion.

## Bundled Code

```
classification/skeleton/
├── pyskl/                    # Full PYSKL framework (models + datasets + pipelines + apis)
├── protogcn/                 # Full ProtoGCN native implementation
├── blockgcn/
│   ├── blockgcn_net.py       # BlockGCN Model (topological GCN)
│   └── graph/coco.py         # COCO-17 spatial graph
├── config_examples/
│   ├── xsub/                 # 30 configs
│   └── xacq/                 # 30 configs
└── skeleton_model_manifest.csv
```

## Architecture Details

### STGCN / STGCN++ (`pyskl/models/gcns/stgcn.py`)
- Recognizer: `RecognizerGCN` + `GCNHead`
- STGCN++ uses config flags: `gcn_adaptive='init'`, `tcn_type='mstcn'`

### CTRGCN (`pyskl/models/gcns/ctrgcn.py`)
- Channel-wise topology refinement GCN

### MSG3D (`pyskl/models/gcns/msg3d.py`)
- Multi-scale spatial-temporal GCN, 384-d features

### AAGCN (`pyskl/models/gcns/aagcn.py`)
- Action-aware GCN with attention

### PoseC3D (`pyskl/models/cnns/resnet3d_slowonly.py`)
- Recognizer: `Recognizer3D` + `I3DHead`
- 3D CNN on pose heatmaps (joint or limb stream)

### ProtoGCN (`pyskl/models/gcns/protogcn_ext.py` + `protogcn/models/gcns/protogcn.py`)
- Recognizer: `RecognizerProtoGCN` + `ProtoGCNHead`
- Also trainable via native ProtoGCN trainer (150 epochs)

### BlockGCN (`blockgcn/blockgcn_net.py`)
- Recognizer: `RecognizerBlockGCN` (`pyskl/models/recognizers/recognizer_blockgcn.py`)
- Topological graph convolution with Vietoris-Rips complex

### Optional (bundled, not in main 60-config matrix)
- **SGN** — `pyskl/models/gcns/sgn.py`
- **DGSTGCN** — `pyskl/models/gcns/dgstgcn.py`

## Example Paths

All configs use `benchmark/paths_example.py`:

```python
ann_file = example/BadmintonSkill23/annotations/skeleton/xsub/badmintonskill23.pkl
work_dir = example/work/skeleton/xsub/stgcn
```

## Regenerate Configs

```bash
Use the bundled `config_examples/` templates with an external PYSKL installation.
```
