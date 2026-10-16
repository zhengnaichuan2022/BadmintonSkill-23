# Vendored MMAction2 model definitions (OpenMMLab)

This directory contains a copy of the **`mmaction`** Python package from [MMAction2](https://github.com/open-mmlab/mmaction2) (action recognition framework).

## Purpose

BadmintonSkill-23 RGB classification benchmarks use 10 MMAction2 architectures (TSN, TSM, I3D, C3D, R2+1D, SlowFast, X3D, TIN, TRN, TANet). Vendoring this code keeps the anonymous additional **self-contained** for architecture review without relying on a host-specific MMAction2 install.

## Layout

| Path | Contents |
|------|----------|
| `mmaction2/` | Full `mmaction` package (~123 model `.py` files) |
| `mmaction2_configs/_base_/models/` | Base architecture configs for 10 benchmark models |
| `badmintonskill23_ext/` | Dataset-specific extensions (`BadmintonSkill23VideoDataset`) |
| `configs/` | 10 XSUB example training configs (example paths only) |

## Usage

1. Install runtime dependencies: PyTorch, mmengine, mmcv, mmaction (or use this vendored copy on `PYTHONPATH`).
2. Set `BADMINTONSKILL23_ROOT=example/BadmintonSkill23` (see `benchmark/paths_example.py`).
3. Run training with any config under `configs/`, e.g. `tsn_badminton_r50.py`.

## License

MMAction2 is distributed under the Apache 2.0 license. See upstream repository for full terms.
