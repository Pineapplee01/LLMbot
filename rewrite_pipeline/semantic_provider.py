"""Provide semantic expert outputs (z_sem, pred_sem, q_sem, etc.)."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "LLMbot" / "code"), str(_ROOT / "LLMbot")]:
    if _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, str(_ROOT / "LLMbot" / "code"))
sys.path.insert(0, str(_ROOT / "LLMbot"))

from config import RewriteConfig


def get_semantic_outputs(cfg: RewriteConfig, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Load or compute semantic expert outputs for all nodes.

    Returns dict with: z_sem, pred_sem, prob_sem_cal, q_sem, u_sem, logits_sem.
    """
    artifact_path = _ROOT / cfg.semantic_artifact_dir / f"seed_{cfg.seed}" / "semantic_outputs.pt"
    if artifact_path.exists():
        print(f"  Loading semantic outputs from {artifact_path}")
        return torch.load(str(artifact_path), weights_only=False)

    # Fall back: load checkpoint and run forward pass
    ckpt_path = _ROOT / cfg.semantic_ckpt_dir / f"seed_{cfg.seed}" / "semantic_expert.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No semantic artifact or checkpoint for seed {cfg.seed}. "
            f"Checked: {artifact_path}, {ckpt_path}"
        )

    print(f"  Running semantic forward pass from checkpoint {ckpt_path}")
    from models.text_model import SemanticIBEDLHead
    from utils.calibration import SemanticCalibrator

    q_final = data["q_final"]
    in_dim = q_final.size(1)

    model = SemanticIBEDLHead(in_dim=in_dim, hidden_dim=cfg.hidden_dim, num_classes=cfg.num_classes, mode="ib_edl")
    model.load_state_dict(torch.load(str(ckpt_path), weights_only=True))
    model.eval()

    with torch.no_grad():
        out = model(q_final)

    # Apply ATS calibration on valid_cal, then to all
    calibrator = SemanticCalibrator(mode="ats", device=torch.device("cpu"))
    calibrator.fit(
        logits_sem=out["logits_sem"][data["valid_cal"]],
        prob_sem=out["prob_sem"][data["valid_cal"]],
        u_sem=out["u_sem"][data["valid_cal"]],
        labels=data["labels"][data["valid_cal"]],
    )
    cal_out = calibrator.apply(
        logits_sem=out["logits_sem"],
        prob_sem=out["prob_sem"],
        u_sem=out["u_sem"],
    )

    return {
        "z_sem": out["z_sem"].detach().cpu(),
        "logits_sem": out["logits_sem"].detach().cpu(),
        "prob_sem": out["prob_sem"].detach().cpu(),
        "prob_sem_cal": cal_out["prob_sem_cal"].detach().cpu(),
        "q_sem": cal_out["q_sem"].detach().cpu(),
        "u_sem": out["u_sem"].detach().cpu(),
        "pred_sem": cal_out["prob_sem_cal"].argmax(dim=1).long().cpu(),
    }
