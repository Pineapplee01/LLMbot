"""
utils/metrics.py
Calibration and evaluation metrics for bot detection.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error. probs: [N,K], labels: [N]."""
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels).float()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.float().mean().item() * abs(correct[mask].mean().item() - confs[mask].mean().item())
    return ece


def compute_ece_from_confidence(confs: torch.Tensor, correctness: torch.Tensor, n_bins: int = 15) -> float:
    """ECE from pre-computed confidence and correctness vectors."""
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confs.size(0)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum().item() / n) * abs(correctness[mask].mean().item() - confs[mask].mean().item())
    return ece


def compute_brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Brier Score (lower is better). probs: [N,K], labels: [N]."""
    y_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - y_one_hot) ** 2).sum(dim=1).mean().item()


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative Log-Likelihood (lower is better)."""
    return F.nll_loss(torch.log(probs.clamp(min=1e-8)), labels).item()


def compute_aurc(confs: torch.Tensor, correctness: torch.Tensor) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    n = confs.size(0)
    if n == 0:
        return 0.0
    order = confs.argsort(descending=True)
    sorted_correct = correctness[order].float()
    cumulative_risk = 1.0 - sorted_correct.cumsum(0) / torch.arange(1, n + 1, dtype=torch.float)
    return cumulative_risk.mean().item()
