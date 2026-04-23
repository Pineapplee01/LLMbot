"""Evaluation metrics and comparison tables."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def evaluate(
    pred: torch.Tensor,
    prob: torch.Tensor,
    labels: torch.Tensor,
    split_idx: torch.Tensor,
    struct_feats: torch.Tensor = None,
    pred_sem: torch.Tensor = None,
) -> Dict:
    """Compute Acc, Macro-F1, and slice metrics on a given split."""
    idx = split_idx.long()
    y = labels[idx].numpy()
    p = pred[idx].numpy()
    pr = prob[idx].numpy()

    acc = float(accuracy_score(y, p))
    f1 = float(f1_score(y, p, average="macro"))

    # ECE (10 bins)
    ece = _compute_ece(pr, y, n_bins=10)

    result = {"acc": acc, "f1": f1, "ece": ece, "n": len(y)}

    # Slice metrics
    if struct_feats is not None:
        sf = struct_feats[idx]
        td = sf[:, 2].numpy()  # total_log_degree (z-normed)
        gm = sf[:, 4].numpy() > 0.5  # graph_missing

        if gm.sum() > 5:
            result["gm_f1"] = float(f1_score(y[gm], p[gm], average="macro"))
        if (~gm).sum() > 5:
            ld = td < np.percentile(td[~gm], 20) if (~gm).sum() > 0 else np.zeros_like(gm)
            if ld.sum() > 5:
                result["ld_f1"] = float(f1_score(y[ld], p[ld], average="macro"))

    # Disagreement slice
    if pred_sem is not None:
        ps = pred_sem[idx].numpy()
        disagree = p != ps  # Note: using graph pred vs sem pred
        if disagree.sum() > 5:
            result["dis_f1"] = float(f1_score(y[disagree], p[disagree], average="macro"))
            result["dis_n"] = int(disagree.sum())

    return result


def _compute_ece(prob: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece / len(labels))


def save_metrics(metrics: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
