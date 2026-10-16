# Beiyu badminton — Temporal Action Detection (OpenTAD fork)

 [OpenTAD](https://github.com/sming256/OpenTAD) ，****：

-  →  + 11 
-  IMU / Video / Skeleton ：`xsub` / `xacq`
- Feature-based：RGB /  →  neck + head  TAD

## 

```
OpenTAD-main/
├── config.py                 # 、、
├── run_prepare.sh            # 
├── run_extract_features.sh   # 
├── run_train.sh / run_test.sh
├── configs/
│   ├── _base_/models/        # ActionFormer / TriDet / BMN … 
│   ├── _base_/datasets/beiyu/
│   └── beiyu/                # 
├── data/beiyu/
│   ├── annotations/          # anno.json, category_idx.txt
│   └── features/             # rgb/, skeleton/
├── tools/
│   ├── prepare_data/beiyu/
│   └── extract_features/
└── opentad/datasets/beiyu.py
```

## 

```bash
cd OpenTAD-main

# 1. （ metadata/instances.csv  OpenTAD JSON）
bash run_prepare.sh

# 2. （ dummy ； MMAction / PYSKL ）
bash run_extract_features.sh

# 3.  TAD
CUDA_VISIBLE_DEVICES=0 bash run_train.sh configs/beiyu/actionformer_rgb_xsub.py

# 4. 
bash run_test.sh configs/beiyu/actionformer_rgb_xsub.py
```

## 

|  |  |  |  |
|------|------|------|------|
| `actionformer_rgb_xsub.py` | RGB I3D 2048d | ActionFormer | xsub |
| `actionformer_rgb_xacq.py` | RGB | ActionFormer | xacq |
| `actionformer_skel_xsub.py` | Skeleton 256d | ActionFormer | xsub |
| `actionformer_rgb_skel_xsub.py` | RGB+Skel 2304d | ActionFormer | xsub |
| `tridet_rgb_xsub.py` | RGB | TriDet | xsub |

： `configs/beiyu/` ， `_base_`  model  `projection.in_channels`。

## （，）

/ **** ； **TAD **  `LoadFeats(data_path=[dir1, dir2, ...])`  channel 。

|  | （ `{video_id}.npy`） |  |
|------|-------------------------------------|------|
| RGB xsub | `features/rgb/r2plus1d_xsub_stride4/` | `[T, 512]` |
| RGB xacq | `features/rgb/r2plus1d_xacq_stride4/` | `[T, 512]` |
| Skel j | `features/skeleton/protogcn_j_{xsub,xacq}_stride4/` | `[T, 384]` |
| Skel b | `features/skeleton/protogcn_b_{xsub,xacq}_stride4/` | `[T, 384]` |
| Skel jm | `features/skeleton/protogcn_jm_{xsub,xacq}_stride4/` | `[T, 384]` |
| Skel bm | `features/skeleton/protogcn_bm_{xsub,xacq}_stride4/` | `[T, 384]` |

 **10 **（2 RGB + 8 Skeleton ×）。TAD  `data_path` ，：

- RGB-only：`data_path = "…/r2plus1d_xsub_stride4/"`
- Skel ：`data_path = [j_dir, b_dir, jm_dir, bm_dir]` →  `[T, 1536]`
- RGB+：`data_path = [rgb_dir, j_dir, b_dir, jm_dir, bm_dir]` →  `[T, 2048]`

## （10 ，）

**** TAD  checkpoint ，：

| # | Job |  |  |
|---|-----|----------|------|
| 1 | rgb xsub | `features/rgb/r2plus1d_xsub_stride4/` | `video/outputs/xsub/r2plus1d_*` |
| 2 | rgb xacq | `features/rgb/r2plus1d_xacq_stride4/` | `video/outputs/xacq/r2plus1d_*` |
| 3–6 | skel j/b/jm/bm × xsub | `features/skeleton/protogcn_{stream}_xsub_stride4/` | `skeleton/outputs/xsub/protogcn_*` |
| 7–10 | skel j/b/jm/bm × xacq | `features/skeleton/protogcn_{stream}_xacq_stride4/` | `skeleton/outputs/xacq/protogcn_*` |

```bash
cd OpenTAD-main

#  10 job（ 4 ）
bash run_extract_features_10.sh

# （ 1  GPU ）
PARALLEL=0 GPUS=0 bash run_extract_features_10.sh

# smoke： job 2 
LIMIT=2 bash run_extract_features_10.sh

# 
bash show_extract_progress.sh
```

`run_extract_features.sh` / `run_extract_features_4gpu.sh`  `run_extract_features_10.sh`。

（）：
- `MMACTION_PYTHON` — RGB， RTMLib env
- `PYSKL_PYTHON` — Skeleton， vmamba env
- `MMACTION2_ROOT` — MMAction2 
- `GPUS=0,1,2,3` —  GPU 
- `SKIP_PREPARE=1` — 
- `OVERWRITE=1` —  `.npy`

## TAD  Benchmark（32 ）

， TAD benchmark：

| Task |  |  |  |
|------|------|------|------|
| 1 | `skel_j` / `skel_b` | 384 | j、b  |
| 2 | `skel_jb` | 768 | j+b  channel  |
| 3 | `skel_4stream` | 1536 | j+b+jm+bm  |

 × **4 **（ActionFormer / TriDet / GTAD / TemporalMaxer）× **xsub / xacq** = **32 **。

```bash
cd OpenTAD-main

# 1. （）
python3 tools/generate_beiyu_tad_benchmark.py

# 2.  + test 
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_tad_benchmark.sh

# 4 （）
bash run_tad_benchmark_4gpu.sh

#  Task 
TASK=1 bash run_tad_benchmark.sh   # j + b  (16)
TASK=2 bash run_tad_benchmark.sh   # j+b (8)
TASK=3 bash run_tad_benchmark.sh   #  (8)

# 
FEATURE=skel_j PROTOCOL=xsub MODEL=actionformer bash run_tad_benchmark.sh

# 3.  → TAD_benchmark_results.md
python3 summarize_tad_benchmark.py --log-dir logs/tad_benchmark_YYYYMMDD_HHMMSS
```

：`configs/beiyu/benchmark/*.py`  
：`exps/beiyu/benchmark/{xsub,xacq}/`  
：Average-mAP @ tIoU 0.3–0.7（**test** ）

## 

- mAP @ tIoU = 0.3, 0.4, 0.5, 0.6, 0.7
-  benchmark ：TAD ** + **

## IMU  TAD（）

 IMU（Sensor06, 6D）→ **iSPL-Inception 1D backbone** → ActionFormer / TriDet / TemporalMaxer / GTAD。

###  IMU 

```bash
bash run_prepare_imu.sh
#  bash run_prepare.sh   #  video + IMU 
```

：`data/beiyu/annotations/imu/{xsub,xacq}/anno.json`（296 sessions/，IMU ）

###  / 

```bash
#  Python （ torch + mmengine + einops）
export PYTHON=/home/ubuntu/users/znc/miniconda3/envs/vmamba/bin/python

# xsub — ActionFormer
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_train.sh configs/beiyu/actionformer_imu_e2e_xsub.py
bash run_test.sh configs/beiyu/actionformer_imu_e2e_xsub.py

# xacq — ActionFormer
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_train.sh configs/beiyu/actionformer_imu_e2e_xacq.py

# TriDet / TemporalMaxer / GTAD
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_train.sh configs/beiyu/tridet_imu_e2e_xsub.py
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_train.sh configs/beiyu/temporalmaxer_imu_e2e_xsub.py
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_train.sh configs/beiyu/gtad_imu_e2e_xsub.py
# GTAD  classifier JSON：data/beiyu/classifiers/imu_{xsub,xacq}.json
# ：python3 tools/extract_features/export_imu_cls.py
```

### （dataloader + forward）

```bash
bash run_smoke_imu_e2e.sh
```

|  |  | Head |
|------|------|------|
| `actionformer_imu_e2e_xsub.py` | xsub | ActionFormer |
| `actionformer_imu_e2e_xacq.py` | xacq | ActionFormer |
| `tridet_imu_e2e_xsub.py` | xsub | TriDet |
| `tridet_imu_e2e_xacq.py` | xacq | TriDet |
| `temporalmaxer_imu_e2e_xsub.py` | xsub | TemporalMaxer |
| `temporalmaxer_imu_e2e_xacq.py` | xacq | TemporalMaxer |
| `gtad_imu_e2e_xsub.py` | xsub | GTAD + `imu_xsub.json` |
| `gtad_imu_e2e_xacq.py` | xacq | GTAD + `imu_xacq.json` |

：256×4=1024  IMU （120Hz， 8.5s）；backbone stride=4 → 256 。

## 

-  THUMOS / ActivityNet / EPIC 
- （`opentad/models/`）， neck/head
-  OpenTAD README / arXiv:2502.20361
