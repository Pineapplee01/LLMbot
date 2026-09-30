"""
utils/losses.py
Loss functions: EDL, IB-EDL, SupCon, KL divergence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)) — used for EDL regularization."""
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return first + second


def r_edl_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int,
    annealing_step: int = 10,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """Regularized EDL loss with annealed KL divergence."""
    S = alpha.sum(dim=1, keepdim=True)
    y_hot = F.one_hot(labels, num_classes=alpha.shape[1]).float()
    loss_nll = (y_hot * (torch.log(S) - torch.log(alpha))).sum(dim=1).mean()
    alpha_tilde = y_hot + (1 - y_hot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, alpha.shape[1])
    annealing_coef = min(1.0, epoch / max(1, annealing_step))
    return loss_nll + kl_weight * annealing_coef * kl.mean()


class EdlLoss(nn.Module):
    """Evidential Deep Learning loss with annealed KL regularization."""

    def __init__(self, annealing_step: int = 10):
        super().__init__()
        self.annealing_step = annealing_step

    def forward(self, alpha: torch.Tensor, y: torch.Tensor, epoch_num: int, total_epochs: int = None) -> torch.Tensor:
        return r_edl_loss(alpha, y, epoch_num, self.annealing_step)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        if features.dim() < 3:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        return -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
