"""CoM-M2U / IMU module (anonymous review package)."""


from __future__ import annotations

import torch
import torch.nn.functional as F


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-8).log()).sum(dim=-1)


def confidence_fusion_probs(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    T: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (p_teacher@1, q_teacher@T, alpha_a, alpha_b)."""
    p_a = torch.softmax(logits_a, dim=-1)
    p_b = torch.softmax(logits_b, dim=-1)
    H_a = _entropy(p_a)
    H_b = _entropy(p_b)
    w_a = torch.exp(-H_a)
    w_b = torch.exp(-H_b)
    denom = w_a + w_b + 1e-8
    alpha_a = w_a / denom
    alpha_b = w_b / denom

    p_teacher = alpha_a.unsqueeze(-1) * p_a + alpha_b.unsqueeze(-1) * p_b
    q_a = torch.softmax(logits_a / T, dim=-1)
    q_b = torch.softmax(logits_b / T, dim=-1)
    q_teacher = alpha_a.unsqueeze(-1) * q_a + alpha_b.unsqueeze(-1) * q_b
    return p_teacher, q_teacher, alpha_a, alpha_b


def equal_fusion_probs(
    logits_list: list[torch.Tensor],
    T: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-modal / equal-weight teacher."""
    ps = [torch.softmax(z, dim=-1) for z in logits_list]
    qs = [torch.softmax(z / T, dim=-1) for z in logits_list]
    p_t = torch.stack(ps, dim=0).mean(dim=0)
    q_t = torch.stack(qs, dim=0).mean(dim=0)
    return p_t, q_t


def com_m2u_loss(
    student_logits: torch.Tensor,
    label: torch.Tensor,
    teacher_logits_a: torch.Tensor,
    teacher_logits_b: torch.Tensor,
    T: float = 4.0,
    lambda_kd: float = 0.5,
    gamma: float = 1.0,
    mode: str = "com_m2u",
    extra_teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    CoM-M2U loss. Teacher logits MUST be offline / other-modality and are always detached
    (no gradient into teachers). Student backbone+head both train under caller’s two-stage schedule.

    modes:
      - ce: CE only
      - full_kd: CE + equal fusion of a,b,(optional student-modal teacher as extra)
      - comp_kd: leave-one-out equal fusion (no conf, no gain)
      - conf_kd: leave-one-out + confidence fusion (no gain)
      - com_m2u: leave-one-out + conf + gain weight (final)
    """
    z_m = student_logits
    ce = F.cross_entropy(z_m, label)
    stats = {"ce": float(ce.detach())}

    if mode == "ce":
        return ce, stats

    # Build teacher
    if mode == "full_kd":
        parts = [teacher_logits_a.detach(), teacher_logits_b.detach()]
        if extra_teacher_logits is not None:
            parts.append(extra_teacher_logits.detach())
        # For full KD without student modality offline logits, approximate with a+b only
        # Callers should pass extra when available (student's own frozen logits).
        p_teacher, q_teacher = equal_fusion_probs(parts, T=T)
    elif mode == "comp_kd":
        p_teacher, q_teacher = equal_fusion_probs(
            [teacher_logits_a.detach(), teacher_logits_b.detach()], T=T
        )
    else:
        # conf_kd / com_m2u
        p_teacher, q_teacher, _, _ = confidence_fusion_probs(
            teacher_logits_a.detach(), teacher_logits_b.detach(), T=T
        )

    q_teacher = q_teacher.detach()
    p_m = torch.softmax(z_m, dim=-1)
    log_q_m = F.log_softmax(z_m / T, dim=-1)

    kl_per = F.kl_div(log_q_m, q_teacher, reduction="none").sum(dim=-1) * (T * T)

    if mode == "com_m2u":
        idx = torch.arange(label.size(0), device=label.device)
        gain = torch.clamp(p_teacher[idx, label] - p_m[idx, label], min=0.0)
        w = (1.0 + gamma * gain).detach()
        kd = (w * kl_per).mean()
        stats["gain_mean"] = float(gain.mean().detach())
    else:
        kd = kl_per.mean()
        w = None

    loss = ce + lambda_kd * kd
    stats["kd"] = float(kd.detach())
    stats["loss"] = float(loss.detach())
    return loss, stats


def probs_to_logits(probs: torch.Tensor | "np.ndarray", eps: float = 1e-8) -> torch.Tensor:
    """Convert softmax scores to pseudo-logits via log."""
    if not torch.is_tensor(probs):
        probs = torch.as_tensor(probs)
    probs = probs.clamp_min(eps)
    return probs.log()
