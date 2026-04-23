"""Standalone graph expert (RGT) training with early stopping and calibration."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "LLMbot" / "code"), str(_ROOT / "LLMbot")]:
    if _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, str(_ROOT / "LLMbot" / "code"))
sys.path.insert(0, str(_ROOT / "LLMbot"))

from graph_rgt import GraphRGTExpert
from models.graph_model import GraphGATSCalibrator
from config import RewriteConfig


def train_graph_expert(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: Optional[torch.Tensor],
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_ckpt_idx: torch.Tensor,
    valid_cal_idx: torch.Tensor,
    struct_feats: torch.Tensor,
    cfg: RewriteConfig,
    class_weights: Optional[torch.Tensor] = None,
    stage_name: str = "graph",
    save_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """Train GraphRGTExpert on given graph, return all-node outputs + metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_relations = 1 if edge_type is None else int(edge_type.max().item()) + 1

    model = GraphRGTExpert(
        in_dim=int(node_features.size(1)),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.graph_num_layers,
        num_relations=num_relations,
        heads=cfg.graph_heads,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.graph_lr, weight_decay=cfg.weight_decay)
    cw = class_weights.to(device) if class_weights is not None else None

    x = node_features.float().to(device)
    ei = edge_index.long().to(device)
    et = None if edge_type is None else edge_type.long().to(device)
    lab = labels.long().to(device)
    tr = train_idx.long().to(device)
    va = valid_ckpt_idx.long().to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -float("inf")
    no_improve = 0

    for epoch in range(cfg.graph_epochs):
        model.train()
        out = model(x, ei, et)
        loss = F.cross_entropy(out["logits_graph"][tr], lab[tr], weight=cw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = model(x, ei, et)
        val_prob = val_out["prob_graph"][va].cpu()
        val_pred = val_prob.argmax(dim=1)
        val_f1 = f1_score(labels[valid_ckpt_idx].numpy(), val_pred.numpy(), average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"    Early stop at epoch {epoch+1} (best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)

    with torch.no_grad():
        model.eval()
        full_out = model(x, ei, et)

    h_graph = full_out["h_graph"].detach().cpu()
    logits = full_out["logits_graph"].detach().cpu()
    prob = full_out["prob_graph"].detach().cpu()
    pred = prob.argmax(dim=1).long()

    # GATS calibration
    q_graph = None
    prob_cal = prob.clone()
    if cfg.calibrate_graph:
        cal = GraphGATSCalibrator(device=torch.device("cpu"))
        cal.fit(
            logits_graph=logits[valid_cal_idx],
            prob_graph=prob[valid_cal_idx],
            struct_feats=struct_feats[valid_cal_idx],
            pred_graph=pred[valid_cal_idx],
            labels=labels[valid_cal_idx],
        )
        cal_out = cal.apply(
            logits_graph=logits,
            prob_graph=prob,
            struct_feats=struct_feats,
            pred_graph=pred,
        )
        prob_cal = cal_out["prob_graph_cal"]
        q_graph = cal_out["q_graph"].squeeze(-1)

    # Metrics on test
    test_idx_np = labels.numpy()  # placeholder, actual test metrics computed by evaluator

    result = {
        "h_graph": h_graph,
        "logits_graph": logits,
        "prob_graph": prob,
        "prob_graph_cal": prob_cal,
        "pred_graph": pred,
        "q_graph": q_graph if q_graph is not None else prob.max(dim=1).values,
        "best_val_f1": best_val_f1,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"{stage_name}_expert.pt")
        torch.save(result, save_dir / f"{stage_name}_outputs.pt")

    return result
