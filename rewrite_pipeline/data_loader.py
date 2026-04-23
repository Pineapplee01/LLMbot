"""Dataset loading, validation split, and structural features."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict

import torch
import numpy as np

# Add LLMbot paths for imports — LLMbot/ MUST come before LLMbot/code/
# because code/utils.py (file) shadows utils/ (package)
_ROOT = Path(__file__).resolve().parent.parent
_llmbot = str(_ROOT / "LLMbot")
_code = str(_ROOT / "LLMbot" / "code")
# Remove any existing entries to control order
for p in [_llmbot, _code]:
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, _code)
sys.path.insert(0, _llmbot)  # LLMbot/ must be FIRST so utils/ package wins

from data.loader import load_raw_data
from utils.misc import seed_setting, compute_directed_structural_features
from config import RewriteConfig


def split_validation(valid_idx: torch.Tensor, seed: int):
    """Replicate DualRouterTrainer validation split (40/30/30)."""
    n = int(valid_idx.numel())
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = valid_idx[torch.randperm(n, generator=gen)]
    n_ckpt = max(1, int(round(0.4 * n)))
    n_cal = max(1, int(round(0.3 * n)))
    while n_ckpt + n_cal >= n:
        if n_ckpt >= n_cal and n_ckpt > 1:
            n_ckpt -= 1
        elif n_cal > 1:
            n_cal -= 1
        else:
            break
    return perm[:n_ckpt].clone(), perm[n_ckpt:n_ckpt + n_cal].clone(), perm[n_ckpt + n_cal:].clone()


def load_dataset(cfg: RewriteConfig) -> Dict[str, torch.Tensor]:
    """Load TwiBot-20 dataset with splits, structural features, and q_final."""
    ds_path = str(_ROOT / cfg.dataset_path)
    raw = load_raw_data(ds_path, use_GNN=True)

    labels = raw["labels"]
    if labels.dim() > 1 and labels.size(1) > 1:
        labels = labels.argmax(dim=1)
    labels = labels.long().cpu()

    edge_index = raw["edge_index"].long().cpu()
    edge_type = raw.get("edge_type")
    if edge_type is not None:
        edge_type = edge_type.long().cpu()

    train_idx = raw["train_idx"].long().cpu()
    valid_idx = raw["valid_idx"].long().cpu()
    test_idx = raw["test_idx"].long().cpu()
    valid_ckpt, valid_cal, valid_router = split_validation(valid_idx, cfg.seed)

    q_final = torch.load(str(_ROOT / cfg.q_final_path), weights_only=True).float().cpu()
    num_nodes = int(q_final.size(0))

    struct_feats = compute_directed_structural_features(edge_index, num_nodes, q_final).float().cpu()

    # Class weights for imbalanced training
    train_labels = labels[train_idx]
    counts = torch.bincount(train_labels, minlength=cfg.num_classes).float()
    class_weights = (counts.sum() / (cfg.num_classes * counts.clamp(min=1))).cpu()

    return {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "labels": labels,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "valid_ckpt": valid_ckpt,
        "valid_cal": valid_cal,
        "valid_router": valid_router,
        "test_idx": test_idx,
        "q_final": q_final,
        "struct_feats": struct_feats,
        "num_nodes": num_nodes,
        "class_weights": class_weights,
    }
