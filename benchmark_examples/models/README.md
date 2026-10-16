# BadmintonSkill-23 Model Code (Review Package)

This directory contains **dataset adapters**, **config examples**, and **CoM-M2U** core code only. Full third-party frameworks are **not** vendored here.

## Included

```
models/
├── classification/
│   ├── rgb/badmintonskill23_ext/     # MMAction2 dataset adapter
│   ├── imu/models/                   # IMU classifier definitions
│   ├── skeleton/config_examples/     # PYSKL config templates (xsub/xacq)
│   └── multimodal/com_m2u/         # CoM-M2U fine-tuning core
├── tad/
│   ├── opentad_datasets/             # OpenTAD dataset adapters
│   └── config_examples/              # TAD experiment configs
└── list_models.py
```

## External Dependencies (install separately)

| Framework | Use | License |
|-----------|-----|---------|
| [MMAction2](https://github.com/open-mmlab/mmaction2) | RGB video classification | Apache-2.0 |
| [PYSKL](https://github.com/kennymckormick/pyskl) | Skeleton recognition | Apache-2.0 |
| [OpenTAD](https://github.com/sming256/OpenTAD) | Temporal action detection | Apache-2.0 |

Document the commit or release tag used for paper experiments in your reproduction notes.

## Path Placeholders

Configs use `example/BadmintonSkill23/...` paths. Set:

```bash
export BADMINTONSKILL23_ROOT=/path/to/release/root
```

See `benchmark/paths_example.py`.

## Naming

Public dataset classes: `BadmintonSkill23VideoDataset`, `BadmintonSkill23SlidingDataset`, `BadmintonSkill23IMUSlidingDataset`.

TAD protocol video IDs in configs should reference `V000001`–`V000295` lists in `protocols/`.
