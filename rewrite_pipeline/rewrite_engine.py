"""Graph rewrite engine — wraps existing rewrite_graph() with ablation modes."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT / "LLMbot" / "code"), str(_ROOT / "LLMbot")]:
    if _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, str(_ROOT / "LLMbot" / "code"))
sys.path.insert(0, str(_ROOT / "LLMbot"))

from data.graph_rewrite import (
    rewrite_graph,
    compute_trigger_scores,
    build_relation_priors,
    build_edge_pair_features,
)
from utils.misc import compute_directed_structural_features
from config import RewriteConfig


def compute_rewrite(
    data: Dict[str, torch.Tensor],
    semantic_outputs: Dict[str, torch.Tensor],
    raw_graph_outputs: Dict[str, torch.Tensor],
    cfg: RewriteConfig,
) -> Dict:
    """Execute graph rewrite based on config mode. Returns rewritten graph + stats."""

    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    labels = data["labels"]
    num_nodes = data["num_nodes"]

    if cfg.rewrite_mode == "none":
        return {
            "edge_index": edge_index,
            "edge_type": edge_type,
            "stats": {"mode": "none", "edges_before": edge_index.size(1), "edges_after": edge_index.size(1),
                       "pruned": 0, "added": 0, "triggered": 0},
        }

    # Compute trigger
    pred_sem = semantic_outputs["pred_sem"].cpu()
    pred_graph_raw = raw_graph_outputs["pred_graph"].cpu()
    q_sem = semantic_outputs["q_sem"].cpu()
    u_sem = semantic_outputs["u_sem"].cpu()
    q_graph = raw_graph_outputs["q_graph"].cpu()

    if cfg.trigger_mode == "random":
        n = num_nodes
        trigger_mask = torch.zeros(n, dtype=torch.bool)
        rng = np.random.RandomState(cfg.seed)
        k = int(0.15 * n)
        trigger_mask[rng.choice(n, k, replace=False)] = True
        disagreement_mask = trigger_mask
    else:
        disagreement_mask = pred_sem.ne(pred_graph_raw)
        use_disagree = cfg.trigger_mode in ("disagreement", "full")
        trigger_info = compute_trigger_scores(
            q_sem=q_sem, u_sem=u_sem, q_graph=q_graph,
            mode="disagreement_local" if use_disagree else "calib_prune",
            threshold=cfg.trigger_threshold,
            disagreement_mask=disagreement_mask if use_disagree else None,
            use_q_graph_trigger=True,
        )
        trigger_mask = trigger_info["trigger_mask"]

    # Determine budgets based on mode
    drop_in = cfg.drop_budget_in if cfg.rewrite_mode in ("prune_only", "prune_add") else 0
    drop_out = cfg.drop_budget_out if cfg.rewrite_mode in ("prune_only", "prune_add") else 0
    add_in = cfg.add_budget_in if cfg.rewrite_mode in ("add_only", "prune_add") else 0
    add_out = cfg.add_budget_out if cfg.rewrite_mode in ("add_only", "prune_add") else 0

    if cfg.rewrite_mode == "random_add":
        # First run guided rewrite to get edge count, then do random add
        guided = _run_guided_rewrite(data, semantic_outputs, raw_graph_outputs, cfg,
                                      trigger_mask, disagreement_mask, drop_in=0, drop_out=0,
                                      add_in=cfg.add_budget_in, add_out=cfg.add_budget_out)
        n_guided_added = guided["stats"].get("added", 0)
        return _random_add(data, trigger_mask, n_guided_added, cfg)

    result = _run_guided_rewrite(data, semantic_outputs, raw_graph_outputs, cfg,
                                  trigger_mask, disagreement_mask,
                                  drop_in, drop_out, add_in, add_out)
    return result


def _run_guided_rewrite(data, semantic_outputs, raw_graph_outputs, cfg,
                         trigger_mask, disagreement_mask,
                         drop_in, drop_out, add_in, add_out) -> Dict:
    """Call the existing rewrite_graph() function."""
    q_sem = semantic_outputs["q_sem"]
    u_sem = semantic_outputs["u_sem"]
    prob_sem_cal = semantic_outputs["prob_sem_cal"]
    pred_sem = semantic_outputs["pred_sem"]
    q_graph = raw_graph_outputs["q_graph"]

    feature_source = semantic_outputs.get("z_sem", data["q_final"])

    payload = rewrite_graph(
        data_dict={"edge_index": data["edge_index"], "edge_type": data["edge_type"],
                    "labels": data["labels"], "train_idx": data["train_idx"]},
        candidate_features=feature_source.float().cpu(),
        q_sem=q_sem, u_sem=u_sem, prob_sem_cal=prob_sem_cal, pred_sem=pred_sem,
        q_graph=q_graph, struct_feats=data["struct_feats"],
        train_idx=data["train_idx"],
        mode="disagreement_local",
        trigger_threshold=cfg.trigger_threshold,
        keep_threshold=cfg.keep_threshold,
        add_threshold=cfg.add_threshold,
        candidate_topk=cfg.candidate_topk,
        max_candidates=cfg.max_candidates,
        drop_budget_in=drop_in, drop_budget_out=drop_out,
        add_budget_in=add_in, add_budget_out=add_out,
        relation_aware=cfg.relation_aware,
        direction_aware=cfg.direction_aware,
        use_q_graph_trigger=True,
        disagreement_mask=disagreement_mask,
    )

    new_ei = payload["data_dict"]["edge_index"]
    new_et = payload["data_dict"].get("edge_type")
    gs = payload.get("graph_stats", {})

    return {
        "edge_index": new_ei,
        "edge_type": new_et,
        "stats": {
            "mode": cfg.rewrite_mode,
            "edges_before": int(data["edge_index"].size(1)),
            "edges_after": int(new_ei.size(1)),
            "pruned": int(gs.get("num_pruned_edges", 0)),
            "added": int(gs.get("num_added_edges", 0)),
            "triggered": int(gs.get("num_triggered_nodes", 0)),
        },
    }


def _random_add(data, trigger_mask, n_add, cfg) -> Dict:
    """Add n_add random edges to triggered nodes (densification control)."""
    ei = data["edge_index"].numpy()
    n = data["num_nodes"]
    triggered_nodes = torch.where(trigger_mask)[0].numpy()

    if len(triggered_nodes) == 0 or n_add == 0:
        return {
            "edge_index": data["edge_index"],
            "edge_type": data["edge_type"],
            "stats": {"mode": "random_add", "edges_before": ei.shape[1],
                       "edges_after": ei.shape[1], "pruned": 0, "added": 0,
                       "triggered": len(triggered_nodes)},
        }

    # Build existing edge set for dedup
    existing = set(zip(ei[0].tolist(), ei[1].tolist()))
    rng = np.random.RandomState(cfg.seed + 1000)
    new_src, new_dst = [], []
    added = 0
    max_attempts = n_add * 10

    for _ in range(max_attempts):
        if added >= n_add:
            break
        s = rng.choice(triggered_nodes)
        d = rng.randint(0, n)
        if s == d or (s, d) in existing:
            continue
        new_src.append(s)
        new_dst.append(d)
        existing.add((s, d))
        added += 1

    if added > 0:
        add_ei = torch.tensor([new_src, new_dst], dtype=torch.long)
        new_edge_index = torch.cat([data["edge_index"], add_ei], dim=1)
        add_et = torch.zeros(added, dtype=torch.long)
        new_edge_type = torch.cat([data["edge_type"], add_et]) if data["edge_type"] is not None else None
    else:
        new_edge_index = data["edge_index"]
        new_edge_type = data["edge_type"]

    return {
        "edge_index": new_edge_index,
        "edge_type": new_edge_type,
        "stats": {"mode": "random_add", "edges_before": int(data["edge_index"].size(1)),
                   "edges_after": int(new_edge_index.size(1)), "pruned": 0, "added": added,
                   "triggered": len(triggered_nodes)},
    }
