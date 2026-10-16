，****，，、，。

：

[
\mathcal{L}^{m}
===============

\mathcal{L}*{CE}
+
\lambda w_m \mathcal{L}*{KD}
]

：

1. **teacher  probability， logits**；
2. **teacher **；
3. **gain weight  stop-gradient，**。

---

# 1. 

 (m\in{r,s,i})， RGB / Skeleton / IMU  student，：

[
\mathcal{L}^{m}
===============

\mathcal{L}*{CE}^{m}
+
\lambda
\mathcal{L}*{CoKD}^{m}
]

：

[
\mathcal{L}_{CE}^{m}
====================

-\log p_m(y)
]

[
\mathcal{L}_{CoKD}^{m}
======================

w_m
\cdot
T^2
KL
\left(
q_{-m}^{T}
\parallel
q_m^{T}
\right)
]

：

* (m)： student；
* (p_m)：student ；
* (q_m^{T}=Softmax(z_m/T))：student  (T) ；
* (q_{-m}^{T})： teacher；
* (w_m)：；
* (\lambda)：KD loss 。

---

# 2.  teacher 

 (m)，teacher ：

[
\mathcal{A}_m={r,s,i}\setminus {m}
]

 RGB student ：

[
\mathcal{A}_r={s,i}
]

 teacher = Skeleton + IMU。

 (k\in \mathcal{A}_m)，：

[
p_k = Softmax(z_k)
]

：

[
H(p_k)
======

-\sum_c p_k(c)\log p_k(c)
]

，。：

[
\alpha_k
========

\frac{\exp(-H(p_k))}
{\sum_{j\in\mathcal{A}_m}\exp(-H(p_j))}
]

 teacher：

[
q_{-m}^{T}
==========

\sum_{k\in\mathcal{A}_m}
\alpha_k
Softmax(z_k/T)
]

 RGB student ：

[
q_{-r}^{T}
==========

\alpha_s Softmax(z_s/T)
+
\alpha_i Softmax(z_i/T)
]

，** teacher**。

---

# 3.  (w_m)

。

 student  teacher ，。

：

[
g_m
===

\max
\left(
0,
p_{-m}(y)-p_m(y)
\right)
]

：

* (p_{-m}(y))： teacher  (y) ；
* (p_m(y))： student  (y) 。

：

[
w_m
===

1+\gamma \cdot stopgrad(g_m)
]

 `stopgrad` ，：

> (w_m) ，。

 (p_m(y)) ，。

---

# 4. ： teacher 

 teacher ，。

：

[
r_m =
\mathbb{1}
[
p_{-m}(y) > p_m(y)
]
]

：

[
r_m =
\sigma
\left(
\frac{p_{-m}(y)-p_m(y)}{\tau}
\right)
]

：

[
w_m
===

1+\gamma \cdot stopgrad(r_m g_m)
]

，：

[
w_m
===

1+\gamma \cdot stopgrad
\left(
\max(0,p_{-m}(y)-p_m(y))
\right)
]

。

---

# 5. 

：

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
q_m^{T}
=======

Softmax(z_m/T)
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
H(p_k)
======

-\sum_c p_k(c)\log p_k(c)
]

[
w_m
===

1+
\gamma \cdot stopgrad
\left(
\max(0,p_{-m}(y)-p_m(y))
\right)
]

 (p_{-m})  teacher ：

[
p_{-m}
======

\sum_{k\in\mathcal{A}_m}
\alpha_k p_k
]

---

# 6. RGB / Skeleton / IMU 

## RGB Student

Teacher = Skeleton + IMU：

[
\mathcal{L}^{r}
===============

-\log p_r(y)
+
\lambda
w_r
T^2
KL
\left(
q_{-r}^{T}
\parallel
q_r^{T}
\right)
]

[
q_{-r}^{T}
==========

\alpha_s Softmax(z_s/T)
+
\alpha_i Softmax(z_i/T)
]

---

## Skeleton Student

Teacher = RGB + IMU：

[
\mathcal{L}^{s}
===============

-\log p_s(y)
+
\lambda
w_s
T^2
KL
\left(
q_{-s}^{T}
\parallel
q_s^{T}
\right)
]

[
q_{-s}^{T}
==========

\alpha_r Softmax(z_r/T)
+
\alpha_i Softmax(z_i/T)
]

---

## IMU Student

Teacher = RGB + Skeleton：

[
\mathcal{L}^{i}
===============

-\log p_i(y)
+
\lambda
w_i
T^2
KL
\left(
q_{-i}^{T}
\parallel
q_i^{T}
\right)
]

[
q_{-i}^{T}
==========

\alpha_r Softmax(z_r/T)
+
\alpha_s Softmax(z_s/T)
]

---

# 7. 

 KD。 KD ：

[
FullTeacher(RGB,Skeleton,IMU)
\rightarrow
Student
]

：

[
OtherModalities
\rightarrow
TargetModality
]

：

## ： teacher

[
q_{-m}
]

， student 。

## ：

[
w_m
===

1+\gamma \max(0,p_{-m}(y)-p_m(y))
]

 teacher  student 。

## ：

[
\alpha_k
========

\frac{\exp(-H(p_k))}
{\sum_{j}\exp(-H(p_j))}
]

。

---

# 8.  PyTorch 

：

```python
# student logits
z_m = student(x_m)

# frozen teacher logits from other modalities
z_a = batch["teacher_logits_a"].detach()
z_b = batch["teacher_logits_b"].detach()

# normal probabilities for confidence and gain
p_m = torch.softmax(z_m, dim=-1)
p_a = torch.softmax(z_a, dim=-1)
p_b = torch.softmax(z_b, dim=-1)

# entropy-based confidence weights
H_a = -(p_a * torch.log(p_a + 1e-8)).sum(dim=-1)
H_b = -(p_b * torch.log(p_b + 1e-8)).sum(dim=-1)

w_a = torch.exp(-H_a)
w_b = torch.exp(-H_b)

alpha_a = w_a / (w_a + w_b + 1e-8)
alpha_b = w_b / (w_a + w_b + 1e-8)

# teacher distribution at normal temperature
p_teacher = alpha_a.unsqueeze(-1) * p_a + alpha_b.unsqueeze(-1) * p_b

# teacher distribution at KD temperature
q_a = torch.softmax(z_a / T, dim=-1)
q_b = torch.softmax(z_b / T, dim=-1)
q_teacher = alpha_a.unsqueeze(-1) * q_a + alpha_b.unsqueeze(-1) * q_b
q_teacher = q_teacher.detach()

# student distribution at KD temperature
log_q_student = torch.log_softmax(z_m / T, dim=-1)

# CE loss
ce_loss = F.cross_entropy(z_m, label)

# complementary gain weight
p_teacher_y = p_teacher[torch.arange(label.size(0)), label]
p_student_y = p_m[torch.arange(label.size(0)), label]

gain = torch.clamp(p_teacher_y - p_student_y, min=0.0)
sample_weight = (1.0 + gamma * gain).detach()

# per-sample KL
kl_per_sample = F.kl_div(
    log_q_student,
    q_teacher,
    reduction="none"
).sum(dim=-1) * (T * T)

kd_loss = (sample_weight * kl_per_sample).mean()

loss = ce_loss + lambda_kd * kd_loss
```

---

# 9. 

：

```text
T = 4
lambda_kd = 0.5
gamma = 1.0
```

：

|        |            |
| --------- | ------------- |
| (T)       | 2, 4, 6       |
| (\lambda) | 0.1, 0.5, 1.0 |
| (\gamma)  | 0.5, 1.0, 2.0 |

---

# 10. 

：

|                   |                                                    |
| ------------------- | ---------------------------------------------------- |
| CE                  | (\mathcal{L}_{CE})                                   |
| Full KD             | (\mathcal{L}_{CE}+\lambda KL(q_T,q_m))               |
| Complementary KD    | (\mathcal{L}*{CE}+\lambda KL(q*{-m},q_m))            |
| + Confidence Fusion | (\mathcal{L}*{CE}+\lambda KL(q*{-m}^{conf},q_m))     |
| **CoM-M2U**         | (\mathcal{L}*{CE}+\lambda w_m KL(q*{-m}^{conf},q_m)) |

，。

---

# 11. 

**。**

：

[
\boxed{
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
}
]

：

1. **(q_{-m}^{T})**： teacher；
2. **(\alpha_k)**：；
3. **(w_m)**：。

-only 。
