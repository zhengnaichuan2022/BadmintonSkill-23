# CoM-M2U 

## 1. 

 RGB、Skeleton  IMU backbone， **、** 。

：

>  RGB、Skeleton  IMU ；
> ；
> 。

：

**CoM-M2U: Complementary Modality Distillation for Multimodal-to-Unimodal Badminton Stroke Classification**

：

**、**

---

## 2. 

：

|     |        |          |
| ----- | -------- | ---------- |
| (X^r) | RGB      |     |
| (X^s) | Skeleton |      |
| (X^i) | IMU      |  |

 HR / 。，，。

---

## 3.  Checkpoint

（ backbone）：

|  |  |  | xsub Acc | xacq Acc |
| ---- | -------- | ---- | -------: | -------: |
| RGB | **r2plus1d** |  + Kinetics | 55.96% | 59.63% |
| Skeleton | **protogcn** | ** j/b/jm/bm** | 67.84% | 77.66% |
| IMU | **ispl_inception** | full_w256 | 58.27% | 61.53% |

 checkpoint：

|     |        |               |
| ----- | -------- | --------------- |
| (E_r) | RGB      | RGB logits      |
| (E_s) | Skeleton | Skeleton logits（） |
| (E_i) | IMU      | IMU logits      |

，：

[
z_r = E_r(X^r)
]

[
z_s = E_s(X^s)
]

[
z_i = E_i(X^i)
]

：

* (z_r)：RGB  logits；
* (z_s)：Skeleton  logits；
* (z_i)：IMU  logits。

：

[
p_r = Softmax(z_r)
]

[
p_s = Softmax(z_s)
]

[
p_i = Softmax(z_i)
]

---

## 4. Teacher  Student 

### 4.1 Teacher ？

Teacher ， checkpoint 。

 RGB ，teacher  RGB ， Skeleton  IMU ：

[
p_{-r} = Fusion(p_s, p_i)
]

，teacher “”。

### 4.2 Student ？

Student 。

：

|         | Student     |
| ------------- | ----------- |
| RGB only      | RGB       |
| Skeleton only | Skeleton  |
| IMU only      | IMU       |

 checkpoint， student ， checkpoint  fine-tune。

：

|  checkpoint       |                    |
| ------------------- | ----------------------- |
| RGB checkpoint      | RGB-M2U checkpoint      |
| Skeleton checkpoint | Skeleton-M2U checkpoint |
| IMU checkpoint      | IMU-M2U checkpoint      |

 student， teacher。

---

## 5. ：Leave-One-Modality-Out Complementary Teacher

：

[
RGB + Skeleton + IMU \rightarrow Student
]

 student ，。

 ** teacher**：

### 5.1 RGB Student

RGB student  RGB。

 teacher  Skeleton  IMU ：

[
p_{-r} = Fusion(p_s, p_i)
]

：

> RGB  Skeleton  IMU 。

---

### 5.2 Skeleton Student

Skeleton student  Skeleton。

 teacher  RGB  IMU ：

[
p_{-s} = Fusion(p_r, p_i)
]

：

> Skeleton  RGB  IMU 。

---

### 5.3 IMU Student

IMU student  IMU。

 teacher  RGB  Skeleton ：

[
p_{-i} = Fusion(p_r, p_s)
]

：

> IMU  RGB  Skeleton 。

---

## 6. Confidence-Aware Complementary Teacher

Teacher ，。

 (m)，：

[
H(p_m) = -\sum_c p_m(c)\log p_m(c)
]

，。

 (m)， teacher  (k \neq m) ，：

[
\alpha_k =
\frac{\exp(-H(p_k))}
{\sum_{j \neq m}\exp(-H(p_j))}
]

， teacher ：

[
p_{-m} = \sum_{k \neq m}\alpha_k p_k
]

 RGB student  teacher ：

[
p_{-r}
======

\alpha_s p_s + \alpha_i p_i
]

：

[
\alpha_s =
\frac{\exp(-H(p_s))}
{\exp(-H(p_s))+\exp(-H(p_i))}
]

[
\alpha_i =
\frac{\exp(-H(p_i))}
{\exp(-H(p_s))+\exp(-H(p_i))}
]

---

## 7. Complementary Gain Weight

。

， teacher 。：

> ， teacher 。

 (m)，：

[
g_m = \max(0, p_{-m}(y) - p_m(y))
]

：

* (p_m(y))：；
* (p_{-m}(y))： teacher ；
* (y)：。

：

[
w_m = 1 + \gamma g_m
]

 (\gamma) 。

：

*  teacher ，；
*  teacher ，。

---

## 8. 

> ** `multimodal/.md` 。** 。

 (m\in{r,s,i})：

[
\mathcal{L}^{m}
===============

-\log p_m(y)
+
\lambda
w_m
T^2
KL
\left(
q_{-m}^{T}
\parallel
q_m^{T}
\right)
]

：

[
q_m^{T}=Softmax(z_m/T)
]

[
q_{-m}^{T}
==========

\sum_{k\in\mathcal{A}_m}
\alpha_k Softmax(z_k/T)
]

[
\alpha_k
========

\frac{\exp(-H(p_k))}
{\sum_{j\in\mathcal{A}_m}\exp(-H(p_j))}
]

[
w_m
===

1+\gamma\cdot stopgrad
\left(
\max(0,p_{-m}(y)-p_m(y))
\right)
]

[
p_{-m}=\sum_{k\in\mathcal{A}_m}\alpha_k p_k
]

：

1. teacher （ `Softmax(z_k/T)`； soft score  `log(p)`  logits）；
2. teacher logits / soft label  detach，；
3. (w_m)  stop-gradient；
4. ：`T=4`, `\lambda=0.5`, `\gamma=1.0`。

---

## 9. 

### Stage 0：

：

```text
sample_id
label
rgb_input
skeleton_input
imu_input
```

。

---

### Stage 1： checkpoint 

 checkpoint ：

```text
RGB checkpoint      → rgb_logits
Skeleton checkpoint → skeleton_logits
IMU checkpoint      → imu_logits
```

：

```text
sample_id
label
rgb_logits.npy
skeleton_logits.npy
imu_logits.npy
```

：

```text
rgb_feature.npy
skeleton_feature.npy
imu_feature.npy
```

-only  logits。

---

### Stage 2： teacher

， student  teacher。

####  RGB student

```text
teacher = confidence_weighted_fusion(skeleton_logits, imu_logits)
```

：

[
p_{-r} = \alpha_s p_s + \alpha_i p_i
]

---

####  Skeleton student

```text
teacher = confidence_weighted_fusion(rgb_logits, imu_logits)
```

：

[
p_{-s} = \alpha_r p_r + \alpha_i p_i
]

---

####  IMU student

```text
teacher = confidence_weighted_fusion(rgb_logits, skeleton_logits)
```

：

[
p_{-i} = \alpha_r p_r + \alpha_s p_s
]

---

### Stage 3：Fine-tune RGB Student

：

```text
RGB student ←  RGB checkpoint
```

：

```text
RGB input
ground-truth label
Skeleton logits
IMU logits
```

：

[
\mathcal{L}^{r}
===============

\mathcal{L}*{CE}(p_r, y)
+
\lambda w_r KL(p*{-r}, p_r)
]

：

```text
RGB-M2U checkpoint
```

 RGB：

```text
RGB input → RGB-M2U → prediction
```

---

### Stage 4：Fine-tune Skeleton Student

：

```text
Skeleton student ←  Skeleton checkpoint
```

：

```text
Skeleton input
ground-truth label
RGB logits
IMU logits
```

：

[
\mathcal{L}^{s}
===============

\mathcal{L}*{CE}(p_s, y)
+
\lambda w_s KL(p*{-s}, p_s)
]

：

```text
Skeleton-M2U checkpoint
```

 Skeleton：

```text
Skeleton input → Skeleton-M2U → prediction
```

---

### Stage 5：Fine-tune IMU Student

：

```text
IMU student ←  IMU checkpoint
```

：

```text
IMU input
ground-truth label
RGB logits
Skeleton logits
```

：

[
\mathcal{L}^{i}
===============

\mathcal{L}*{CE}(p_i, y)
+
\lambda w_i KL(p*{-i}, p_i)
]

：

```text
IMU-M2U checkpoint
```

 IMU：

```text
IMU input → IMU-M2U → prediction
```

---

## 10.  / 

 teacher。

 M2U student。

### 10.1 RGB-only 

```text
Input: RGB
Model: RGB-M2U checkpoint
Output: classification prediction
```

### 10.2 Skeleton-only 

```text
Input: Skeleton
Model: Skeleton-M2U checkpoint
Output: classification prediction
```

### 10.3 IMU-only 

```text
Input: IMU
Model: IMU-M2U checkpoint
Output: classification prediction
```

。

 RGB-only ：

```text
Allowed: RGB
Not allowed: Skeleton, IMU
```

---

## 11. 

，。

### 11.1 CE Baseline

 checkpoint：

[
\mathcal{L} = \mathcal{L}_{CE}
]

：

```text
RGB → RGB baseline
Skeleton → Skeleton baseline
IMU → IMU baseline
```

---

### 11.2 Full-modal KD Baseline

 teacher：

[
p_T = Fusion(p_r, p_s, p_i)
]

 student。

：

>  KD 。

---

### 11.3 Complementary KD

 teacher：

[
p_{-m} = \sum_{k \neq m}\alpha_k p_k
]

 complementary gain weight。

：

> Leave-one-modality-out teacher  full-modal teacher。

---

### 11.4 CoM-M2U

：

[
\mathcal{L}^{m}
===============

\mathcal{L}*{CE}
+
\lambda w_m KL(p*{-m}, p_m)
]

：

* teacher ；
* teacher  confidence-aware fusion；
*  complementary gain weight。

---

## 12. 

### 12.1 、

| Training Strategy | Test RGB | Test Skeleton | Test IMU | Avg |
| ----------------- | -------: | ------------: | -------: | --: |
| Single-modal CE   |          |               |          |     |
| Full-modal KD     |          |               |          |     |
| Complementary KD  |          |               |          |     |
| **CoM-M2U**       |          |               |          |     |

：

> ，CoM-M2U  RGB-only、Skeleton-only  IMU-only 。

---

### 12.2 

| Method              | Leave-one-out Teacher | Confidence Fusion | Gain Weight | Test RGB | Test Skeleton | Test IMU |
| ------------------- | --------------------- | ----------------- | ----------- | -------: | ------------: | -------: |
| CE baseline         |                       |                   |             |          |               |          |
| Full KD             |                       | ✓                 |             |          |               |          |
| Complementary KD    | ✓                     |                   |             |          |               |          |
| + Confidence Fusion | ✓                     | ✓                 |             |          |               |          |
| **CoM-M2U**         | ✓                     | ✓                 | ✓           |          |               |          |

---

### 12.3 

| Confusing Pair       | RGB Baseline | RGB-M2U | Skeleton Baseline | Skeleton-M2U | IMU Baseline | IMU-M2U |
| -------------------- | -----------: | ------: | ----------------: | -----------: | -----------: | ------: |
| Clear vs Smash       |              |         |                   |              |              |         |
| Drop vs Net Shot     |              |         |                   |              |              |         |
| Drive vs Push        |              |         |                   |              |              |         |
| Forehand vs Backhand |              |         |                   |              |              |         |

：

> CoM-M2U ，。

---

## 13. 

|            |              |           |
| ------------- | -------------- | ------------- |
| (T)           |            | 2, 4, 6       |
| (\lambda)     | KD loss      | 0.1, 0.5, 1.0 |
| (\gamma)      | gain weight  | 0.5, 1.0, 2.0 |
| learning rate (head) | Stage-1 / Stage-2 head LR | ；Stage-2 head  backbone |
| learning rate (backbone) | Stage-2 backbone LR |  head LR  0.1  |
| freeze_epochs | Stage-1  backbone  | 3–5 |
| epochs | （） | 10–15（ 12 = 4+8） |
| batch size    |             |       |

：

```text
T = 4
lambda = 0.5
gamma = 1.0
```

。

---

## 14. 

### 14.1 Teacher 

Teacher  logits，。

 RGB student ：

```text
Skeleton logits: frozen
IMU logits: frozen
RGB student: trainable
```

---

### 14.2 Student  checkpoint  fine-tune

；。

```text
student.load_state_dict(original_unimodal_checkpoint)
```

：

1. **Stage-1（ 3–5 epoch）**： backbone， classifier head；
2. **Stage-2（）**： backbone， backbone LR +  head LR  full fine-tuning。

Teacher  logits， detach / stop_gradient，。

---

### 14.3 

 RGB-M2U ：

```text
Input: RGB only
Forbidden: Skeleton, IMU
```

、。

---

### 14.4 Teacher logits 

 teacher logits  student 。

：

```text
sample_id 
label 


```

---

### 14.5 

 logits 。

：

```text
： train split  logits
： val split 
： student 
```

---

## 15. 

### 15.1  logits

```python
for sample in train_set:
    rgb_logits = rgb_model(sample.rgb)
    skeleton_logits = skeleton_model(sample.skeleton)
    imu_logits = imu_model(sample.imu)

    save({
        "sample_id": sample.id,
        "label": sample.label,
        "rgb_logits": rgb_logits,
        "skeleton_logits": skeleton_logits,
        "imu_logits": imu_logits,
    })
```

---

### 15.2  teacher

```python
def entropy(p):
    return -(p * log(p + 1e-8)).sum(dim=-1)

def confidence_fusion(logits_a, logits_b, T=1.0):
    p_a = softmax(logits_a, dim=-1)
    p_b = softmax(logits_b, dim=-1)

    h_a = entropy(p_a)
    h_b = entropy(p_b)

    w_a = exp(-h_a)
    w_b = exp(-h_b)

    alpha_a = w_a / (w_a + w_b)
    alpha_b = w_b / (w_a + w_b)

    teacher_prob = alpha_a.unsqueeze(-1) * p_a + alpha_b.unsqueeze(-1) * p_b
    return teacher_prob
```

---

### 15.3 Fine-tune RGB Student

```python
rgb_student = load_checkpoint("rgb_checkpoint.pth")

for batch in train_loader:
    rgb = batch["rgb"]
    label = batch["label"]

    skeleton_logits = batch["skeleton_logits"]
    imu_logits = batch["imu_logits"]

    rgb_logits = rgb_student(rgb)
    rgb_prob = softmax(rgb_logits, dim=-1)

    teacher_prob = confidence_fusion(skeleton_logits, imu_logits)

    ce_loss = cross_entropy(rgb_logits, label)

    p_teacher_y = teacher_prob[range(batch_size), label]
    p_student_y = rgb_prob[range(batch_size), label]

    gain = clamp(p_teacher_y - p_student_y, min=0.0)
    weight = 1.0 + gamma * gain

    kd_loss = kl_div(
        log_softmax(rgb_logits / T, dim=-1),
        teacher_prob.detach()
    ) * (T * T)

    kd_loss = (weight * kd_loss).mean()

    loss = ce_loss + lambda_kd * kd_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

save_checkpoint(rgb_student, "rgb_m2u_checkpoint.pth")
```

Skeleton student  IMU student ， teacher 。

---

## 16. 

### 16.1 

：

```text
1.  RGB checkpoint
2.  Skeleton checkpoint
3.  IMU checkpoint
4.  split 
5.  sample_id
6.  train / val / test 
7. 
8. logits 
```

---

### 16.2 

：

```text
Step 1:  checkpoint
Step 2:  train / val split  logits
Step 3:  RGB-M2U student
Step 4:  Skeleton-M2U student
Step 5:  IMU-M2U student
Step 6:  test split  RGB-only / Skeleton-only / IMU-only 
Step 7: 
Step 8: 
Step 9: 
Step 10:  checkpoint、log、
```

---

### 16.3 

：

```text
outputs/
├── logits/
│   ├── train_rgb_logits.npy
│   ├── train_skeleton_logits.npy
│   ├── train_imu_logits.npy
│   ├── val_rgb_logits.npy
│   ├── val_skeleton_logits.npy
│   └── val_imu_logits.npy
├── checkpoints/
│   ├── rgb_m2u.pth
│   ├── skeleton_m2u.pth
│   └── imu_m2u.pth
├── results/
│   ├── main_results.csv
│   ├── ablation_results.csv
│   ├── confusion_matrix_rgb.csv
│   ├── confusion_matrix_skeleton.csv
│   └── confusion_matrix_imu.csv
└── logs/
    ├── train_rgb_m2u.log
    ├── train_skeleton_m2u.log
    └── train_imu_m2u.log
```

---

## 17. 

：

> We propose CoM-M2U, a complementary modality distillation framework for multimodal-to-unimodal badminton stroke classification. Instead of designing new modality-specific backbones, CoM-M2U leverages existing RGB, skeleton, and IMU checkpoints to construct leave-one-modality-out complementary teachers. For each target modality, the teacher is formed by confidence-aware fusion of the remaining modalities, and the resulting soft predictions are used to fine-tune the unimodal student. This allows the student model to absorb discriminative information from unavailable modalities during training, while requiring only a single modality at inference.

：

> CoM-M2U ， RGB、Skeleton  IMU checkpoint ，，，。

---

## 18. 

，：

[
\mathcal{L}^{m}
===============

\mathcal{L}*{CE}
+
\lambda w_m KL(p*{-m}, p_m)
]

：

[
p_{-m}
======

\sum_{k\neq m}
\alpha_k p_k
]

[
\alpha_k =
\frac{\exp(-H(p_k))}
{\sum_{j\neq m}\exp(-H(p_j))}
]

[
w_m = 1+\gamma \max(0,p_{-m}(y)-p_m(y))
]

：

```text
1. CE baseline
2. Full-modal KD
3. Complementary KD
4. CoM-M2U
```

：

```text
1. RGB-only accuracy
2. Skeleton-only accuracy
3. IMU-only accuracy
4. Average accuracy
```

---

## 19. 

**CoM-M2U ： checkpoint  teacher， checkpoint  fine-tune，，。**
