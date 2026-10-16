# CoM-M2U：

> Complementary Modality Distillation for Multimodal-to-Unimodal Badminton Stroke Classification  
> ：`multimodal/com_m2u/`  
> ：`multimodal/outputs/com_m2u/{xsub,xacq}/results/`

---

## 1. （）

**、**： backbone， checkpoint  **leave-one-modality-out  teacher**（ + gain ）， student ；、。

|  |  | CoM-M2U |
|--------|----------|---------|
|  |  | **** |
| Teacher |  /  | **** teacher |
| Teacher  |  | **** |
| KD  |  λ | **gain weight**（teacher ， stop-grad） |
| Student  |  | **： LR** |

---

## 2. （ ↔ ）

：[`loss.py`](loss.py)

 \(m \in \{r,s,i\}\)：

\[
\mathcal{L}^{m}
=
\mathcal{L}_{CE}^{m}
+
\lambda\,
w_m \cdot T^{2}\,
\mathrm{KL}\!\left(q_{-m}^{T}\,\|\,q_m^{T}\right)
\]

：

- \(\mathcal{A}_m = \{r,s,i\}\setminus\{m\}\)：（leave-one-out）
- \(q_{-m}^{T}\)： teacher  \(T\) 
- \(w_m = 1 + \gamma\cdot\max\!\big(0,\, p_{-m}(y)-p_m(y)\big)\)， \(w_m\) **detach**

### 2.1  teacher

```python
# loss.py: confidence_fusion_probs
p_a, p_b = softmax(z_a), softmax(z_b)
H_a, H_b = entropy(p_a), entropy(p_b)
α_a = exp(-H_a) / (exp(-H_a)+exp(-H_b))   #  → 
α_b = 1 - α_a
p_teacher = α_a * p_a + α_b * p_b
q_teacher = α_a * softmax(z_a/T) + α_b * softmax(z_b/T)
```

### 2.2 Gain  KL（ `mode="com_m2u"`）

```python
# loss.py: com_m2u_loss（）
ce = cross_entropy(z_m, y)
p_teacher, q_teacher, *_ = confidence_fusion_probs(z_a.detach(), z_b.detach(), T)
q_teacher = q_teacher.detach()                 # teacher 
kl = T*T * KL(log_softmax(z_m/T), q_teacher)  # 
gain = clamp(p_teacher[y] - p_m[y], min=0)    # teacher 
w = (1 + gamma * gain).detach()               # stop-grad
loss = ce + lambda_kd * (w * kl).mean()
```

**Ablation （）**：`ce` / `full_kd` / `comp_kd` / `conf_kd` / `com_m2u`。

### 2.3 Teacher / Student 

|  |  |  |
|------|------|------|
| Teacher logits |  `train.npz`（ `export_logits.py` ） | **detach / stop_gradient** |
| Student |  checkpoint  | backbone + head  |

Leave-one-out ：

| Student | Teacher A | Teacher B |
|---------|-----------|-----------|
| RGB | Skeleton | IMU |
| Skeleton（） | RGB | IMU |
| IMU | RGB | Skeleton |

---

## 3. （）

“”。：

1. **Stage-1（`freeze_epochs`， 3–5）**： backbone， classifier head  
2. **Stage-2**： backbone；`lr_backbone ≪ lr_head`  full FT  
3. ****： student，  

：

|  |  |  |
|------|--------|------|
|  | `loss.py` |  |
|  teacher | `export_logits.py` |  RGB/Skel/IMU logits |
| Skeleton | `finetune_skeleton_train.py` | ProtoGCN 4 ；`refusion_skeleton.py`  |
| IMU | `finetune_imu_simple.py` | ispl_inception  |
| RGB | `finetune_rgb_simple.py` | r2plus1d  |
| / | `constants.py` | baseline、ckpt、 |

 4 （j/b/jm/bm） student ，****：（）。

---

## 4. （ backbone）

|  |  |  |
|------|------|------|
| RGB | r2plus1d（Kinetics  + ） | mmaction2 |
| Skeleton | ProtoGCN 4-stream（j/b/jm/bm） | pyskl |
| IMU | ispl_inception `full_w256` | Keras/TF |
|  | `xsub` / `xacq` | Study 2，11  |

（）：\(T=4\)，\(\lambda=0.3\)，\(\gamma=2\)。

---

## 5. （Test Top-1 Acc）

：`multimodal/outputs/com_m2u/{protocol}/results/*_best_summary.json`

### 5.1 （ baseline）

|  |  | Baseline | CoM-M2U | Δ |
|------|------|----------|---------|---|
| IMU | xsub | 58.27% | **59.11%** | +0.84 |
| IMU | xacq | 61.53% | **63.22%** | +1.69 |
| Skeleton 4s | xsub | 67.84% | **68.80%** | +0.96 |
| Skeleton 4s | xacq | 77.66% | **79.61%** | +1.95 |
| RGB | xsub | 55.96% | **60.29%** | +4.33 |
| RGB | xacq | 59.63% | **59.72%** | +0.09 |

****： `improved=True`；RGB-xsub ；RGB-xacq （ Stage-1 ep01，Stage-2 ）。

### 5.2 Skeleton （ 4 ）

**xsub**（ 68.80%）：

|  |  |  |
|----|------|--------|
| j | 53.47% | 55.00% |
| b | 59.56% | 61.18% |
| jm | 63.02% | 63.64% |
| bm | 59.67% | 60.04% |

**xacq**（ 79.61%）：

|  | reload | best |
|----|---------|------|
| j | 34.22%* | 58.08% |
| b | 69.80% | 70.21% |
| jm | 68.46% | 71.80% |
| bm | 63.56% | 71.48% |

\*xacq-j  reload ， init ckpt  `protogcn_j` ；。

### 5.3 RGB 

- **xsub**：6 epoch；best@ep03 = 60.29%（> 55.96%）  
- **xacq**：12 epoch，freeze=4；best@ep01 = 59.72%（> 59.63%）；full  test  ~58%

---

## 6. （）

### 6.1 Skeleton（ xsub）

```bash
cd ${BADMINTONSKILL23_ROOT}
export PYTHONPATH=$PWD/multimodal:$PWD/skeleton/pyskl-main/pyskl-main
CUDA_VISIBLE_DEVICES=1 python -m com_m2u.finetune_skeleton_train \
  --protocol xsub --device cuda:0 --streams j,b,jm,bm \
  --epochs 12 --freeze_epochs 4 --lr_head 1e-3 --lr_backbone 1e-4
CUDA_VISIBLE_DEVICES=1 python -m com_m2u.refusion_skeleton --protocol xsub --device cuda:0
```

### 6.2 IMU

```bash
cd ${BADMINTONSKILL23_ROOT}
export PYTHONPATH=$PWD/multimodal
#  TF  python
python -m com_m2u.finetune_imu_simple \
  --protocol xsub --gpu 2 \
  --epochs 12 --freeze_epochs 4 --lr_head 3e-5 --lr_backbone 3e-6
```

### 6.3 RGB

```bash
cd ${BADMINTONSKILL23_ROOT}
export PYTHONPATH=$PWD/multimodal:$PWD/video:/home/ubuntu/users/dyc/code/mmaction2
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/users/syq/miniconda3/envs/RTMLib/bin/python -m com_m2u.finetune_rgb_simple \
  --protocol xacq --device cuda:0 \
  --epochs 12 --freeze_epochs 4 --lr_head 1e-4 --lr_backbone 1e-5 --bs 4
```

---

## 7. 

```
multimodal/com_m2u/
├── loss.py                    # ★ ： teacher + gain KD
├── constants.py               # baseline /  / 
├── export_logits.py           #  train/val/test logits
├── finetune_skeleton_train.py # ★ （）
├── refusion_skeleton.py       # 4  + 
├── finetune_imu_simple.py     # ★ IMU （）
├── finetune_rgb_simple.py     # ★ RGB （）
├── finetune_skeleton.py       # /（）
├── finetune_skeleton_fused.py #  Acc  ckpt 
├── finetune_imu.py / finetune_video.py
└── infer_skeleton.py
```

：

```
multimodal/outputs/com_m2u/
├── {xsub,xacq}/
│   ├── logits/{rgb,imu,skel_*,skeleton}/
│   ├── checkpoints/{rgb,imu,skeleton/}/
│   └── results/*_best_summary.json
└── logs/
```

---

## 8. 

- ：`multimodal/.md`、`multimodal/.md`  
- ：** + **， Method / Experiments 。

---

*：2026-07-20 ·  xsub / xacq ·  RGB + Skeleton(4s) + IMU*
