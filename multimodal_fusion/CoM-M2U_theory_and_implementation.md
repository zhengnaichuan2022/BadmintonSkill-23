# CoM-M2U：

> **Complementary Modality Distillation for Multimodal-to-Unimodal Badminton Stroke Classification**  
> 、  
> ：`multimodal/com_m2u/` · ：`multimodal/outputs/com_m2u/`

---

## 1. 

### 1.1 

（11 ）， RGB、Skeleton、IMU ，****。（ `multimodal_ensemble.py`）；：

|  |  |  |
|------|----------|------|
|  | RGB + Skeleton + IMU |  |
|  | **** |  |

### 1.2 

- **** RGB / Skeleton / IMU backbone  
- ****   
- **** teacher 

### 1.3 

 checkpoint  **leave-one-modality-out  teacher**（ + ）， student ****；。

|  |  KD /  | CoM-M2U |
|--------|----------------------|---------|
|  |  | **** |
| Teacher |  | **** |
| Teacher  |  /  | **** |
| KD  |  \(\lambda\) | ** gain \(w_m\)**（stop-grad） |
| Student |  | **， LR** |

---

## 2. 

：\(X^r\)（RGB）、\(X^s\)（Skeleton）、\(X^i\)（IMU）。  
（）：

|  |  | Backbone |  | Baseline Acc (xsub / xacq) |
|------|------|----------|------|---------------------------:|
| RGB | \(E_r\) | r2plus1d |  + Kinetics | 55.96% / 59.63% |
| Skeleton | \(E_s\) | ProtoGCN |  j / b / jm / bm | 67.84% / 77.66% |
| IMU | \(E_i\) | ispl_inception | full_w256 | 58.27% / 61.53% |

：

\[
z_r = E_r(X^r),\quad
z_s = E_s(X^s),\quad
z_i = E_i(X^i)
\]

\[
p_m = \mathrm{Softmax}(z_m),\qquad m\in\{r,s,i\}
\]

Teacher ****， \(z_k\) 。Student  checkpoint 。

Leave-one-out ：

| Student \(m\) |  \(\mathcal{A}_m\) | Teacher |
|---------------|---------------------------|---------|
| RGB | \(\{s,i\}\) | Skeleton + IMU |
| Skeleton | \(\{r,i\}\) | RGB + IMU |
| IMU | \(\{r,s\}\) | RGB + Skeleton |

Skeleton  **j/b/jm/bm **  student ；，****（）。

---

## 3. ： Teacher 

 \(m\in\{r,s,i\}\)：

\[
\mathcal{L}^{m}
=
\mathcal{L}_{\mathrm{CE}}^{m}
+
\lambda\,
\mathcal{L}_{\mathrm{CoKD}}^{m}
=
-\log p_m(y)
+
\lambda\,
w_m\cdot T^{2}\,
\mathrm{KL}\!\left(q_{-m}^{T}\,\big\|\,q_m^{T}\right)
\]

### 3.1  Teacher

 \(\mathcal{A}_m=\{r,s,i\}\setminus\{m\}\)。 \(k\in\mathcal{A}_m\)：

\[
H(p_k)=-\sum_c p_k(c)\log p_k(c)
\]

，：

\[
\alpha_k
=
\frac{\exp\!\big(-H(p_k)\big)}
{\sum_{j\in\mathcal{A}_m}\exp\!\big(-H(p_j)\big)}
\]

 teacher （ gain）：

\[
p_{-m}=\sum_{k\in\mathcal{A}_m}\alpha_k\,p_k
\]

 \(T\) ：

\[
q_{-m}^{T}
=
\sum_{k\in\mathcal{A}_m}
\alpha_k\,\mathrm{Softmax}(z_k/T),\qquad
q_m^{T}=\mathrm{Softmax}(z_m/T)
\]

 RGB student ：

\[
q_{-r}^{T}
=
\alpha_s\,\mathrm{Softmax}(z_s/T)
+
\alpha_i\,\mathrm{Softmax}(z_i/T)
\]

****：teacher ；。Teacher  **detach**，。

### 3.2  \(w_m\)

。 teacher  student ：

\[
g_m=\max\!\big(0,\,p_{-m}(y)-p_m(y)\big)
\]

\[
w_m=1+\gamma\cdot\mathrm{stopgrad}(g_m)
\]

- \(g_m=0\)：teacher  →  KD（ 1）  
- \(g_m>0\)：teacher  →   
- **stop-grad**： student  \(p_m(y)\) 

### 3.3  KD 

1. **Leave-one-out**：\(q_{-m}\) ，，。  
2. ** \(\alpha_k\)**： teacher，。  
3. **Gain \(w_m\)**： teacher ，。

### 3.4 （ `mode` ）

|  | `mode` |  |
|------|--------|----------|
| CE only | `ce` |  \(\mathcal{L}_{\mathrm{CE}}\) |
| Full KD | `full_kd` | （ logits） |
| Complementary KD | `comp_kd` | leave-one-out + **** |
| + Confidence | `conf_kd` | leave-one-out + **** |
| **CoM-M2U** | `com_m2u` | leave-one-out +  + **gain** |

---

## 4. （）

```text
┌─────────────────────────────────────────────────────────────┐
│ Stage A ·  Teacher logits（freeze  checkpoint）│
│   export_logits.py → train/val/test  rgb / skel_* / imu   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage B ·  student  leave-one-out  logits            │
│   RGB←(skel,imu)  Skel←(rgb,imu)  IMU←(rgb,skel)           │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage C ·  Student（com_m2u_loss）              │
│   Stage-1: freeze backbone， head                        │
│   Stage-2: unfreeze；lr_backbone ≪ lr_head                   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage D · ：，           │
│   Skeleton： ckpt → （refusion_skeleton） │
└─────────────────────────────────────────────────────────────┘
```

### 4.1  Teacher（`export_logits.py`）

-  ckpt， train/val/test   
-  `sample_id`  logits（`.npz`）  
-  ** teacher logits**， → 、 teacher 

### 4.2 （`loss.py`）

：`confidence_fusion_probs`、`com_m2u_loss`。

（）：

```python
z_m = student(x_m)                          # 
z_a, z_b = teacher_a.detach(), teacher_b.detach()

# 
p_a, p_b = softmax(z_a), softmax(z_b)
H_a, H_b = entropy(p_a), entropy(p_b)
α_a = exp(-H_a) / (exp(-H_a) + exp(-H_b))
α_b = 1 - α_a
p_teacher = α_a * p_a + α_b * p_b
q_teacher = α_a * softmax(z_a/T) + α_b * softmax(z_b/T)
q_teacher = q_teacher.detach()

# CE + gain-weighted KL
ce = cross_entropy(z_m, y)
kl = T*T * KL(log_softmax(z_m/T), q_teacher)   # per-sample
gain = clamp(p_teacher[y] - softmax(z_m)[y], min=0)
w = (1 + gamma * gain).detach()
loss = ce + lambda_kd * (w * kl).mean()
```

### 4.3 

“”。（RGB / Skeleton / IMU ）：

|  |  |  |  |
|------|------|----------|--------|
| Stage-1 | `epoch ≤ freeze_epochs` |  classifier head | `lr_head` |
| Stage-2 |  | backbone + head | `lr_backbone ≪ lr_head` |

：

|  |  |  |
|------|------|------|
|  | `loss.py` |  |
| Teacher  | `export_logits.py` |  logits |
| RGB | `finetune_rgb_simple.py` | r2plus1d，mmaction2 |
| Skeleton | `finetune_skeleton_train.py` | ProtoGCN  |
| Skeleton  | `refusion_skeleton.py` |  |
| IMU | `finetune_imu_simple.py` | ispl_inception，TF/Keras |
|  baseline | `constants.py` | |

### 4.4 （）

|  |  |  |
|------|------|------|
| \(T\) | 4 |  |
| \(\lambda\) | 0.3 | KD  |
| \(\gamma\) | 2 | gain  |
| `freeze_epochs` | 3–4（） | Stage-1  |
|  | `xsub` / `xacq` | Study 2 |

> `constants.py`  \(T=4,\lambda=0.5,\gamma=1.0\)；** \(T=4,\lambda=0.3,\gamma=2\)**。

---

## 5. 「」

| | （`multimodal_ensemble.py` / `.py`） | CoM-M2U |
|--|------------------------------------------------------|---------|
|  |  softmax / | ****， |
|  |  | **** |
|  |  |  ckpt |
|  |  Acc（ 77.89% / 83.70%） |  Acc |

：；CoM-M2U 。

---

## 6. （Test Top-1 Acc）

：`multimodal/outputs/com_m2u/{xsub,xacq}/results/`

|  |  | Baseline | CoM-M2U | \(\Delta\) |
|------|------|----------|---------|------------|
| IMU | xsub | 58.27% | **59.11%** | +0.84 |
| IMU | xacq | 61.53% | **63.22%** | +1.69 |
| Skeleton 4s | xsub | 67.84% | **68.80%** | +0.96 |
| Skeleton 4s | xacq | 77.66% | **79.61%** | +1.95 |
| RGB | xsub | 55.96% | **60.29%** | +4.33 |
| RGB | xacq | 59.63% | **59.72%** | +0.09 |

 baseline；RGB-xsub 。

---

## 7. 

```
multimodal/com_m2u/
├── loss.py                      # ★  + gain KD
├── constants.py                 # baseline /  / 
├── export_logits.py             #  logits
├── finetune_rgb_simple.py       # ★ RGB 
├── finetune_skeleton_train.py   # ★ Skeleton （）
├── refusion_skeleton.py         #  + 
├── finetune_imu_simple.py       # ★ IMU 
├── finetune_*.py                # /
└── CoM-M2U_.md    # 

multimodal/outputs/com_m2u/{xsub,xacq}/
├── logits/{rgb,imu,skel_*}/
├── checkpoints/{rgb,imu,skeleton}/
└── results/*_best_summary.json
```

：`multimodal/run_com_m2u.sh`  
：`multimodal/.md` · `multimodal/.md`

---

## 8. 

CoM-M2U ：

1. **\(q_{-m}^{T}\)**： soft teacher  
2. **\(\alpha_k\)**：  
3. **\(w_m\)**： teacher （stop-grad）

：** teacher logits +  student + **，，。

---

*：2026-07-22 ·  xsub / xacq ·  RGB + Skeleton(4s) + IMU*
