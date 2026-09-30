from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_building import _idx_tensor, _score_logits, resolve_g0_feature_bundle
from utils import save_stage_artifacts, write_json, write_text, write_torch


@dataclass(frozen=True)
class DignnDualViewConfig:
    topology_hidden_dim: int = 64
    attribute_hidden_dim: int = 64
    view_dim: int = 64
    fusion_dim: int = 64
    topology_n_layers: int = 2
    attribute_n_layers: int = 2
    n_relations: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    reconstruction_weight: float = 0.20
    mi_weight: float = 0.05
    semantic_anchor_weight: float = 0.10
    support_weight: float = 0.20
    temperature: float = 0.20

    @property
    def harmful_support_weight(self) -> float:
        return float(self.support_weight)


DignnStructuralConfig = DignnDualViewConfig


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    n_layers = max(int(n_layers), 1)
    layers = []
    dims = [int(input_dim)] + [int(hidden_dim)] * max(n_layers - 1, 0) + [int(output_dim)]
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


def build_directed_topology_features(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    num_nodes: int,
    n_relations: int,
) -> torch.Tensor:
    edge_index = edge_index.detach().cpu().long()
    edge_type = edge_type.detach().cpu().long()
    num_nodes = int(num_nodes)
    n_relations = max(int(n_relations), 1)

    src = edge_index[0]
    dst = edge_index[1]
    in_deg = torch.bincount(dst, minlength=num_nodes).float()
    out_deg = torch.bincount(src, minlength=num_nodes).float()
    total_deg = in_deg + out_deg
    degree_balance = (out_deg - in_deg) / total_deg.clamp_min(1.0)

    in_rel = torch.zeros((num_nodes, n_relations), dtype=torch.float32)
    out_rel = torch.zeros((num_nodes, n_relations), dtype=torch.float32)
    for relation_id in range(n_relations):
        rel_mask = edge_type == relation_id
        if rel_mask.any():
            rel_src = src[rel_mask]
            rel_dst = dst[rel_mask]
            in_rel[:, relation_id].scatter_add_(0, rel_dst, torch.ones_like(rel_dst, dtype=torch.float32))
            out_rel[:, relation_id].scatter_add_(0, rel_src, torch.ones_like(rel_src, dtype=torch.float32))

    in_rel_dist = in_rel / in_rel.sum(dim=1, keepdim=True).clamp_min(1.0)
    out_rel_dist = out_rel / out_rel.sum(dim=1, keepdim=True).clamp_min(1.0)
    relation_skew = torch.abs(in_rel_dist - out_rel_dist).sum(dim=1)
    relation_mix = in_rel + out_rel
    relation_mix_dist = relation_mix / relation_mix.sum(dim=1, keepdim=True).clamp_min(1.0)
    relation_entropy = -(relation_mix_dist * torch.log(relation_mix_dist.clamp_min(1e-8))).sum(dim=1)

    features = torch.cat(
        [
            torch.log1p(in_deg).unsqueeze(1),
            torch.log1p(out_deg).unsqueeze(1),
            torch.log1p(total_deg).unsqueeze(1),
            degree_balance.unsqueeze(1),
            relation_skew.unsqueeze(1),
            relation_entropy.unsqueeze(1),
            torch.log1p(in_rel),
            torch.log1p(out_rel),
        ],
        dim=1,
    )
    return features.contiguous()


def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_t = p.detach().cpu().float() if torch.is_tensor(p) else torch.tensor(p, dtype=torch.float32)
    q_t = q.detach().cpu().float() if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
    p_t = p_t.clamp_min(eps)
    p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(eps)
    q_t = q_t.clamp_min(eps)
    q_t = q_t / q_t.sum(dim=-1, keepdim=True).clamp_min(eps)
    mix = 0.5 * (p_t + q_t)
    kl_pm = torch.sum(p_t * (torch.log(p_t) - torch.log(mix.clamp_min(eps))), dim=-1)
    kl_qm = torch.sum(q_t * (torch.log(q_t) - torch.log(mix.clamp_min(eps))), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def differentiable_js_divergence_from_logits(
    logits_p: torch.Tensor,
    probs_q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    p = torch.softmax(logits_p, dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = probs_q.to(logits_p.device).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    mix = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(mix.clamp_min(eps))), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(mix.clamp_min(eps))), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


class DignnDualViewProxy(nn.Module):
    def __init__(self, topology_input_dim: int, attribute_input_dim: int, config: DignnDualViewConfig):
        super().__init__()
        self.config = config
        self.topology_encoder = _build_mlp(
            topology_input_dim,
            int(config.topology_hidden_dim),
            int(config.view_dim),
            int(config.topology_n_layers),
            float(config.dropout),
        )
        self.attribute_encoder = _build_mlp(
            attribute_input_dim,
            int(config.attribute_hidden_dim),
            int(config.view_dim),
            int(config.attribute_n_layers),
            float(config.dropout),
        )
        self.attention_proj = nn.Linear(int(config.view_dim), int(config.fusion_dim))
        self.attention_vector = nn.Linear(int(config.fusion_dim), 1, bias=False)
        self.topology_classifier = nn.Linear(int(config.view_dim), 2)
        self.attribute_classifier = nn.Linear(int(config.view_dim), 2)
        self.fusion_classifier = nn.Linear(int(config.view_dim), 2)
        self.topology_decoder = nn.Linear(int(config.view_dim), topology_input_dim)
        self.attribute_decoder = nn.Linear(int(config.view_dim), attribute_input_dim)
        self.mi_topology_proj = nn.Linear(int(config.view_dim), int(config.view_dim), bias=False)
        self.mi_attribute_proj = nn.Linear(int(config.view_dim), int(config.view_dim), bias=False)
        self.dropout = nn.Dropout(float(config.dropout))
        self.activation = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.attention_proj,
            self.attention_vector,
            self.topology_classifier,
            self.attribute_classifier,
            self.fusion_classifier,
            self.topology_decoder,
            self.attribute_decoder,
            self.mi_topology_proj,
            self.mi_attribute_proj,
        ):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward_outputs(
        self,
        topology_features: torch.Tensor,
        attribute_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        topo_repr = self.dropout(self.topology_encoder(topology_features))
        attr_repr = self.dropout(self.attribute_encoder(attribute_features))
        topo_score = self.attention_vector(torch.tanh(self.attention_proj(topo_repr)))
        attr_score = self.attention_vector(torch.tanh(self.attention_proj(attr_repr)))
        attn_logits = torch.cat([topo_score, attr_score], dim=-1)
        attention_weights = torch.softmax(attn_logits, dim=-1)
        fused_repr = attention_weights[:, :1] * topo_repr + attention_weights[:, 1:2] * attr_repr
        fused_repr = self.dropout(fused_repr)

        topo_logits = self.topology_classifier(topo_repr)
        attr_logits = self.attribute_classifier(attr_repr)
        fused_logits = self.fusion_classifier(fused_repr)

        topo_recon = self.topology_decoder(topo_repr)
        attr_recon = self.attribute_decoder(attr_repr)
        topo_recon_error = torch.mean((topo_recon - topology_features).pow(2), dim=-1)
        attr_recon_error = torch.mean((attr_recon - attribute_features).pow(2), dim=-1)
        node_support = torch.exp(-0.5 * (topo_recon_error + attr_recon_error))

        return {
            "topology_logits": topo_logits,
            "attribute_logits": attr_logits,
            "fused_logits": fused_logits,
            "topology_prob": torch.softmax(topo_logits, dim=-1),
            "attribute_prob": torch.softmax(attr_logits, dim=-1),
            "fused_prob": torch.softmax(fused_logits, dim=-1),
            "pred": torch.softmax(fused_logits, dim=-1).argmax(dim=1),
            "topology_repr": topo_repr,
            "attribute_repr": attr_repr,
            "fused_repr": fused_repr,
            "attention_weights": attention_weights,
            "topology_recon": topo_recon,
            "attribute_recon": attr_recon,
            "topology_recon_error": topo_recon_error,
            "attribute_recon_error": attr_recon_error,
            "node_support": node_support,
        }

    def mutual_exclusive_loss(self, topology_repr: torch.Tensor, attribute_repr: torch.Tensor) -> torch.Tensor:
        topo = F.normalize(self.mi_topology_proj(topology_repr), p=2, dim=-1, eps=1e-12)
        attr = F.normalize(self.mi_attribute_proj(attribute_repr), p=2, dim=-1, eps=1e-12)
        pos = (topo * attr).sum(dim=-1)
        perm = torch.roll(torch.arange(attr.size(0), device=attr.device), shifts=1)
        neg = (topo * attr[perm]).sum(dim=-1)
        return F.softplus(pos - neg).mean()


def reconstruction_loss(
    outputs: Dict[str, torch.Tensor],
    topology_features: torch.Tensor,
    attribute_features: torch.Tensor,
) -> torch.Tensor:
    topo_loss = F.mse_loss(outputs["topology_recon"], topology_features)
    attr_loss = F.mse_loss(outputs["attribute_recon"], attribute_features)
    return 0.5 * (topo_loss + attr_loss)


def harmful_edge_score(delta_conflict: float, reconstruction_support: float, support_weight: float) -> float:
    return float(delta_conflict) + float(support_weight) * (1.0 - float(reconstruction_support))


def fit_dignn_structural_encoder(
    *,
    topology_features: torch.Tensor,
    attribute_features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    valid_idx: torch.Tensor,
    test_idx: torch.Tensor,
    semantic_prob: Optional[torch.Tensor],
    device: torch.device,
    config: Optional[DignnDualViewConfig] = None,
    score_logits_fn=None,
) -> Dict[str, object]:
    if config is None:
        config = DignnDualViewConfig()
    if score_logits_fn is None:
        raise ValueError("fit_dignn_structural_encoder requires score_logits_fn for split metrics.")

    topology_features = topology_features.detach().cpu().float()
    attribute_features = attribute_features.detach().cpu().float()
    labels = labels.detach().cpu().long()
    if topology_features.ndim != 2 or attribute_features.ndim != 2:
        raise ValueError("Topology and attribute features must be 2D tensors.")
    if topology_features.size(0) != attribute_features.size(0) or topology_features.size(0) != labels.numel():
        raise ValueError("Topology features, attribute features, and labels must share the same number of nodes.")

    model = DignnDualViewProxy(int(topology_features.size(1)), int(attribute_features.size(1)), config).to(device)
    topo_dev = topology_features.to(device)
    attr_dev = attribute_features.to(device)
    labels_dev = labels.to(device)
    train_idx_dev = train_idx.to(device)
    valid_idx_dev = valid_idx.to(device)
    test_idx_dev = test_idx.to(device)
    semantic_prob_dev = semantic_prob.to(device) if semantic_prob is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_state = None
    best_metrics = None
    history = []
    wait = 0

    for epoch in range(int(config.max_epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward_outputs(topo_dev, attr_dev)

        ce_loss = F.cross_entropy(outputs["fused_logits"][train_idx_dev], labels_dev[train_idx_dev])
        recon_loss = reconstruction_loss(outputs, topo_dev, attr_dev)
        mi_loss = model.mutual_exclusive_loss(outputs["topology_repr"], outputs["attribute_repr"])
        if semantic_prob_dev is not None:
            attr_anchor_loss = F.kl_div(
                F.log_softmax(outputs["attribute_logits"], dim=-1),
                semantic_prob_dev.clamp_min(1e-8),
                reduction="batchmean",
            )
        else:
            attr_anchor_loss = outputs["fused_logits"].new_tensor(0.0)
        loss = (
            ce_loss
            + float(config.reconstruction_weight) * recon_loss
            + float(config.mi_weight) * mi_loss
            + float(config.semantic_anchor_weight) * attr_anchor_loss
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_outputs = model.forward_outputs(topo_dev, attr_dev)
        valid_metrics = score_logits_fn(eval_outputs["fused_logits"].detach().cpu(), labels.detach().cpu(), valid_idx.cpu())
        valid_metrics["epoch"] = int(epoch)
        history.append(
            {
                "epoch": int(epoch),
                "loss": float(loss.detach().cpu().item()),
                "ce_loss": float(ce_loss.detach().cpu().item()),
                "reconstruction_loss": float(recon_loss.detach().cpu().item()),
                "mi_loss": float(mi_loss.detach().cpu().item()),
                "attr_anchor_loss": float(attr_anchor_loss.detach().cpu().item()),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
                "valid_loss": float(valid_metrics["loss"]),
            }
        )
        if best_metrics is None or valid_metrics["macro_f1"] > best_metrics["macro_f1"] or (
            valid_metrics["macro_f1"] == best_metrics["macro_f1"] and valid_metrics["loss"] < best_metrics["loss"]
        ):
            best_metrics = valid_metrics
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(config.patience):
                break

    if best_state is None:
        raise RuntimeError("DIGNN-style dual-view proxy did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        final_outputs = model.forward_outputs(topo_dev, attr_dev)

    outputs_cpu = {key: value.detach().cpu() for key, value in final_outputs.items()}
    return {
        "model": model,
        "outputs": outputs_cpu,
        "checkpoint": {key: value.detach().cpu().clone() for key, value in best_state.items()},
        "fit_summary": {
            "max_epochs": int(config.max_epochs),
            "patience": int(config.patience),
            "best_epoch": int(best_metrics.get("epoch", -1) if best_metrics else -1),
            "history": history,
        },
        "metrics": {
            "train": score_logits_fn(final_outputs["fused_logits"].detach().cpu(), labels.detach().cpu(), train_idx.cpu()),
            "valid": score_logits_fn(final_outputs["fused_logits"].detach().cpu(), labels.detach().cpu(), valid_idx.cpu()),
            "test": score_logits_fn(final_outputs["fused_logits"].detach().cpu(), labels.detach().cpu(), test_idx.cpu()),
        },
        "node_support": final_outputs["node_support"].detach().cpu(),
        "model_config": {
            "type": "dignn_style_dual_view_proxy",
            **asdict(config),
            "checkpoint_selection": {
                "primary": "validation_macro_f1",
                "tie_breaker": "validation_loss",
            },
        },
    }


def _score_all(labels, pred):
    from sklearn.metrics import accuracy_score, f1_score

    labels_np = np.asarray(labels)
    pred_np = np.asarray(pred)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1_score(labels_np, pred_np, average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np, pred_np, average="binary", zero_division=0)),
        "count": int(labels_np.shape[0]),
    }


def _split_metrics_from_outputs(outputs, labels_np, valid_mask, test_mask):
    pred = outputs.get("pred")
    pred_np = pred.detach().cpu().numpy().astype(np.int64) if torch.is_tensor(pred) else np.asarray(pred, dtype=np.int64)
    return {
        "valid": _score_all(labels_np[valid_mask], pred_np[valid_mask]) if valid_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
        "test": _score_all(labels_np[test_mask], pred_np[test_mask]) if test_mask.any() else {"accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0},
        "pred": pred_np,
    }


def _degree_arrays(edge_index, num_nodes):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    src = edge_index_cpu[0]
    dst = edge_index_cpu[1]
    out_degree = torch.bincount(src, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
    in_degree = torch.bincount(dst, minlength=int(num_nodes)).cpu().numpy().astype(np.int64)
    return in_degree, out_degree


def _remove_edge_at_position(edge_index, edge_type, edge_pos):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
    keep_mask[int(edge_pos)] = False
    return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()


def _pruned_graph_from_removed_positions(edge_index, edge_type, removed_positions):
    edge_index_cpu = edge_index.detach().cpu().long() if torch.is_tensor(edge_index) else torch.tensor(edge_index, dtype=torch.long)
    edge_type_cpu = edge_type.detach().cpu().long() if torch.is_tensor(edge_type) else torch.tensor(edge_type, dtype=torch.long)
    if not removed_positions:
        return edge_index_cpu.contiguous(), edge_type_cpu.contiguous()
    keep_mask = torch.ones(edge_index_cpu.size(1), dtype=torch.bool)
    keep_mask[torch.tensor(sorted(int(pos) for pos in removed_positions), dtype=torch.long)] = False
    return edge_index_cpu[:, keep_mask].contiguous(), edge_type_cpu[keep_mask].contiguous()


def _fit_router(stage_runner, semantic_prob, structural_prob):
    valid_idx = np.asarray(stage_runner.data["valid_idx"], dtype=np.int64)
    test_idx = np.asarray(stage_runner.data["test_idx"], dtype=np.int64)
    budget = float(
        getattr(stage_runner.args, "local_dignn_conflict_router_budget", None)
        or getattr(stage_runner.args, "conflict_router_budget", 0.10)
        or 0.10
    )
    if not (0.0 < budget < 1.0):
        raise ValueError("--local_dignn_conflict_router_budget must be in (0, 1) for local_dignn_conflict_refine_diag.")
    conflict_score = js_divergence_from_probs(structural_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
    valid_scores = conflict_score[valid_idx]
    if valid_scores.size == 0:
        raise RuntimeError("local_dignn_conflict_refine_diag requires non-empty valid conflict scores.")
    routed_count_valid = max(int(round(valid_scores.size * budget)), 1)
    ranked_valid = np.sort(valid_scores)[::-1]
    threshold = float(ranked_valid[min(routed_count_valid - 1, ranked_valid.size - 1)])
    routed_mask = conflict_score >= (threshold - 1e-12)
    return {
        "conflict_score": conflict_score,
        "threshold": threshold,
        "routed_mask": routed_mask,
        "budget": budget,
        "routed_count_valid_target": int(routed_count_valid),
        "test_conflict_summary": {
            "mean": float(conflict_score[test_idx].mean()) if test_idx.size else 0.0,
            "median": float(np.median(conflict_score[test_idx])) if test_idx.size else 0.0,
            "max": float(conflict_score[test_idx].max()) if test_idx.size else 0.0,
        },
    }


def _run_structural_forward(model, device, topology_features, attribute_features):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward_outputs(topology_features.to(device), attribute_features.to(device))
    payload = {key: value.detach().cpu() for key, value in outputs.items()}
    return payload


def run_local_dignn_conflict_refine_diag(stage_runner, stage_dir: Path, base_bundle: Dict[str, object]) -> Dict[str, object]:
    context = stage_runner.ensure_backbone_context()
    gnn_outputs = context["gnn_outputs"]
    base_pred = gnn_outputs.get("pred")
    if base_pred is None:
        raise RuntimeError("local_dignn_conflict_refine_diag requires frozen_g0 outputs with pred.")

    edge_index, edge_type = stage_runner._active_graph_tensors()
    labels_t = torch.tensor(stage_runner.labels, dtype=torch.long)
    labels_np = labels_t.numpy()
    semantic_bundle = resolve_g0_feature_bundle(stage_runner.args, stage_runner.data)
    semantic_raw = semantic_bundle["raw_features"].detach().cpu().float()
    semantic_norm = F.normalize(semantic_raw, p=2, dim=1, eps=1e-12)
    degree_guard_enabled = not bool(
        getattr(stage_runner.args, "local_dignn_conflict_disable_degree_guard", False)
        or getattr(stage_runner.args, "conflict_disable_degree_guard", False)
    )
    topk_per_bucket = max(
        int(
            getattr(stage_runner.args, "local_dignn_conflict_topk_per_bucket", None)
            or getattr(stage_runner.args, "conflict_topk_per_bucket", 1)
            or 1
        ),
        1,
    )

    lm_head = stage_runner._build_or_load_lm_only_head()
    semantic_prob = lm_head["outputs"].get("prob_cal", lm_head["outputs"]["prob"]).detach().cpu().float()
    topology_features = build_directed_topology_features(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=int(labels_t.numel()),
        n_relations=int(getattr(stage_runner.args, "n_relations", 2)),
    ).float()
    config = DignnStructuralConfig()
    structural_bundle = fit_dignn_structural_encoder(
        topology_features=topology_features,
        attribute_features=semantic_raw,
        labels=labels_t,
        train_idx=_idx_tensor(stage_runner.data["train_idx"]),
        valid_idx=_idx_tensor(stage_runner.data["valid_idx"]),
        test_idx=_idx_tensor(stage_runner.data["test_idx"]),
        semantic_prob=semantic_prob,
        device=stage_runner.device,
        config=config,
        score_logits_fn=_score_logits,
    )
    structural_outputs = structural_bundle["outputs"]
    structural_prob = structural_outputs["fused_prob"].detach().cpu().float()
    structural_pred = structural_outputs["pred"].detach().cpu().long()
    node_support_cpu = structural_bundle["node_support"].detach().cpu().float()

    router_bundle = _fit_router(stage_runner, semantic_prob, structural_prob)
    conflict_score = router_bundle["conflict_score"]
    routed_mask = router_bundle["routed_mask"]
    routing_threshold = float(router_bundle["threshold"])
    routed_budget = float(router_bundle["budget"])

    edge_index_cpu = edge_index.detach().cpu().long()
    edge_type_cpu = edge_type.detach().cpu().long()
    src_np = edge_index_cpu[0].numpy().astype(np.int64)
    dst_np = edge_index_cpu[1].numpy().astype(np.int64)
    rel_np = edge_type_cpu.numpy().astype(np.int64)
    num_nodes = int(labels_t.numel())
    base_in_degree, base_out_degree = _degree_arrays(edge_index_cpu, num_nodes)

    routed_node_ids = np.flatnonzero(routed_mask)
    routed_node_ids = np.asarray(sorted(routed_node_ids.tolist(), key=lambda node_id: (-float(conflict_score[int(node_id)]), int(node_id))), dtype=np.int64)
    incident_edges: List[List[int]] = [[] for _ in range(num_nodes)]
    for edge_pos, (u, v) in enumerate(zip(src_np.tolist(), dst_np.tolist())):
        incident_edges[int(u)].append(int(edge_pos))
        if int(v) != int(u):
            incident_edges[int(v)].append(int(edge_pos))

    edge_after_cache = {}
    bucket_rows = {}
    candidate_rows = []
    for target_id in routed_node_ids.tolist():
        for edge_pos in incident_edges[int(target_id)]:
            u = int(src_np[edge_pos])
            v = int(dst_np[edge_pos])
            relation_id = int(rel_np[edge_pos])
            if int(target_id) == v:
                target_role = "incoming"
            elif int(target_id) == u:
                target_role = "outgoing"
            else:
                continue
            bucket_key = (int(target_id), int(relation_id), str(target_role))
            rows = bucket_rows.setdefault(bucket_key, [])
            if edge_pos not in edge_after_cache:
                cf_edge_index, cf_edge_type = _remove_edge_at_position(edge_index_cpu, edge_type_cpu, edge_pos)
                cf_topology = build_directed_topology_features(
                    edge_index=cf_edge_index,
                    edge_type=cf_edge_type,
                    num_nodes=num_nodes,
                    n_relations=int(getattr(stage_runner.args, "n_relations", 2)),
                ).float()
                cf_outputs = _run_structural_forward(structural_bundle["model"], stage_runner.device, cf_topology, semantic_raw)
                cf_prob = cf_outputs["fused_prob"].detach().cpu().float()
                cf_conflict = js_divergence_from_probs(cf_prob, semantic_prob).detach().cpu().numpy().astype(np.float32)
                edge_after_cache[int(edge_pos)] = {
                    "conflict": cf_conflict,
                    "node_support_mean": float(cf_outputs["node_support"].mean().item()) if "node_support" in cf_outputs else 0.0,
                }
            conflict_before = float(conflict_score[int(target_id)])
            conflict_after = float(edge_after_cache[int(edge_pos)]["conflict"][int(target_id)])
            reconstruction_support = float(node_support_cpu[int(target_id)].item()) if node_support_cpu.numel() else 0.0
            delta_conflict = float(conflict_before - conflict_after)
            score = harmful_edge_score(delta_conflict, reconstruction_support, float(config.harmful_support_weight))
            row = {
                "target_id": int(target_id),
                "edge": [u, v],
                "relation": int(relation_id),
                "target_role": str(target_role),
                "edge_pos": int(edge_pos),
                "semantic_cosine": float((semantic_norm[u] * semantic_norm[v]).sum().item()),
                "conflict_before": conflict_before,
                "conflict_after": conflict_after,
                "delta_conflict": delta_conflict,
                "reconstruction_support": reconstruction_support,
                "harmful_score": float(score),
                "selected": False,
                "selection_reason": None,
                "guard_blocked": False,
            }
            rows.append(row)
            candidate_rows.append(row)

    conflict_removed_positions = set()
    conflict_bucket_budget = {}
    conflict_in_degree = base_in_degree.copy()
    conflict_out_degree = base_out_degree.copy()
    duplicate_attempts = 0
    for bucket_key in sorted(bucket_rows.keys(), key=lambda item: (item[0], item[1], item[2])):
        rows = bucket_rows.get(bucket_key, [])
        positive_rows = sorted(
            [row for row in rows if float(row["delta_conflict"]) > 0.0],
            key=lambda row: (-float(row["harmful_score"]), -float(row["delta_conflict"]), int(row["edge_pos"])),
        )
        selected_count = 0
        for row in positive_rows:
            if selected_count >= topk_per_bucket:
                break
            edge_pos = int(row["edge_pos"])
            u, v = row["edge"]
            if edge_pos in conflict_removed_positions:
                row["selection_reason"] = "duplicate_already_selected"
                duplicate_attempts += 1
                continue
            if degree_guard_enabled and (conflict_out_degree[int(u)] <= 1 or conflict_in_degree[int(v)] <= 1):
                row["guard_blocked"] = True
                row["selection_reason"] = "degree_guard_blocked"
                continue
            conflict_removed_positions.add(edge_pos)
            if degree_guard_enabled:
                conflict_out_degree[int(u)] -= 1
                conflict_in_degree[int(v)] -= 1
            row["selected"] = True
            row["selection_reason"] = "topk_positive_delta_conflict_with_reconstruction_support"
            selected_count += 1
        conflict_bucket_budget[bucket_key] = int(selected_count)

    for row in candidate_rows:
        if row["selection_reason"] is None:
            if float(row["delta_conflict"]) <= 0.0:
                row["selection_reason"] = "non_positive_delta_conflict"
            else:
                row["selection_reason"] = "not_topk_positive_delta_conflict"

    rng = np.random.default_rng(int(stage_runner.seed))
    rand_removed_positions = set()
    rand_in_degree = base_in_degree.copy()
    rand_out_degree = base_out_degree.copy()
    for bucket_key in sorted(bucket_rows.keys(), key=lambda item: (item[0], item[1], item[2])):
        desired = int(conflict_bucket_budget.get(bucket_key, 0))
        if desired <= 0:
            continue
        rows = list(bucket_rows.get(bucket_key, []))
        if not rows:
            continue
        order = rng.permutation(len(rows))
        picked = 0
        for row_idx in order.tolist():
            row = rows[int(row_idx)]
            edge_pos = int(row["edge_pos"])
            u, v = row["edge"]
            if edge_pos in rand_removed_positions:
                continue
            if degree_guard_enabled and (rand_out_degree[int(u)] <= 1 or rand_in_degree[int(v)] <= 1):
                continue
            rand_removed_positions.add(edge_pos)
            if degree_guard_enabled:
                rand_out_degree[int(u)] -= 1
                rand_in_degree[int(v)] -= 1
            picked += 1
            if picked >= desired:
                break

    conflict_edge_index, conflict_edge_type = _pruned_graph_from_removed_positions(edge_index_cpu, edge_type_cpu, conflict_removed_positions)
    rand_edge_index, rand_edge_type = _pruned_graph_from_removed_positions(edge_index_cpu, edge_type_cpu, rand_removed_positions)
    conflict_zero_in, conflict_zero_out = _degree_arrays(conflict_edge_index, num_nodes)
    rand_zero_in, rand_zero_out = _degree_arrays(rand_edge_index, num_nodes)

    local_conflict_context = stage_runner._rerun_frozen_g0_with_edge_override(stage_dir, "local_dignn_conflict", conflict_edge_index, conflict_edge_type)
    local_rand_context = stage_runner._rerun_frozen_g0_with_edge_override(stage_dir, "local_dignn_rand", rand_edge_index, rand_edge_type)

    base_metrics = _split_metrics_from_outputs(context["frozen_g0"]["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)
    conflict_metrics = _split_metrics_from_outputs(local_conflict_context["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)
    rand_metrics = _split_metrics_from_outputs(local_rand_context["outputs"], labels_np, stage_runner.val_mask, stage_runner.test_mask)

    selected_by_relation_role = {}
    selected_by_bucket = {}
    for bucket_key, count in conflict_bucket_budget.items():
        relation_role_key = f"{bucket_key[1]}::{bucket_key[2]}"
        selected_by_relation_role[relation_role_key] = int(selected_by_relation_role.get(relation_role_key, 0) + count)
        selected_by_bucket[f"{bucket_key[0]}::{bucket_key[1]}::{bucket_key[2]}"] = int(count)

    routed_rows = []
    p_sem_np = semantic_prob.detach().cpu().numpy()
    p_struct_np = structural_prob.detach().cpu().numpy()
    base_pred_np = base_pred.detach().cpu().numpy().astype(np.int64)
    for split_name, idx_mask in (("train", stage_runner.train_mask), ("valid", stage_runner.val_mask), ("test", stage_runner.test_mask)):
        for node_id in np.flatnonzero(idx_mask).tolist():
            routed_rows.append(
                {
                    "node_id": int(node_id),
                    "split": str(split_name),
                    "conflict_score": float(conflict_score[int(node_id)]),
                    "routing_threshold": float(routing_threshold),
                    "routed": bool(routed_mask[int(node_id)]),
                    "p_sem": [float(item) for item in p_sem_np[int(node_id)].tolist()],
                    "p_struct": [float(item) for item in p_struct_np[int(node_id)].tolist()],
                    "base_pred": int(base_pred_np[int(node_id)]),
                    "base_correct": bool(base_pred_np[int(node_id)] == int(labels_np[int(node_id)])),
                }
            )

    selected_rows_sorted = sorted(
        candidate_rows,
        key=lambda row: (
            int(row["target_id"]),
            int(row["relation"]),
            str(row["target_role"]),
            -float(row["harmful_score"]),
            -float(row["delta_conflict"]),
            int(row["edge_pos"]),
        ),
    )
    evidence_rows = [
        {
            "target_id": int(row["target_id"]),
            "edge": [int(row["edge"][0]), int(row["edge"][1])],
            "relation": int(row["relation"]),
            "target_role": str(row["target_role"]),
            "edge_pos": int(row["edge_pos"]),
            "semantic_cosine": float(row["semantic_cosine"]),
            "conflict_before": float(row["conflict_before"]),
            "conflict_after": float(row["conflict_after"]),
            "delta_conflict": float(row["delta_conflict"]),
            "reconstruction_support": float(row["reconstruction_support"]),
            "harmful_score": float(row["harmful_score"]),
            "removed_from_propagation": True,
            "evidence_role": "camouflage_conflict_evidence",
        }
        for row in selected_rows_sorted
        if bool(row["selected"])
    ]

    metrics_payload = {
        "contract": "local_dignn_conflict_refine_diag_metrics_v1",
        "routing_mode": "dignn_style_dual_view_conflict_plus_reconstruction",
        "paper_faithful_dignn_style": True,
        "official_code_verified": False,
        "routed_budget": routed_budget,
        "topk_per_bucket": int(topk_per_bucket),
        "base": {"valid": base_metrics["valid"], "test": base_metrics["test"]},
        "local_dignn_conflict": {
            "valid": conflict_metrics["valid"],
            "test": conflict_metrics["test"],
            "delta_vs_base": {
                "valid_accuracy": float(conflict_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                "valid_macro_f1": float(conflict_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                "test_accuracy": float(conflict_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                "test_macro_f1": float(conflict_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
            },
        },
        "local_dignn_rand": {
            "valid": rand_metrics["valid"],
            "test": rand_metrics["test"],
            "delta_vs_base": {
                "valid_accuracy": float(rand_metrics["valid"]["accuracy"] - base_metrics["valid"]["accuracy"]),
                "valid_macro_f1": float(rand_metrics["valid"]["macro_f1"] - base_metrics["valid"]["macro_f1"]),
                "test_accuracy": float(rand_metrics["test"]["accuracy"] - base_metrics["test"]["accuracy"]),
                "test_macro_f1": float(rand_metrics["test"]["macro_f1"] - base_metrics["test"]["macro_f1"]),
            },
        },
        "structural_view": {
            "train": structural_bundle["metrics"]["train"],
            "valid": structural_bundle["metrics"]["valid"],
            "test": structural_bundle["metrics"]["test"],
            "fit_summary": structural_bundle["fit_summary"],
            "node_support_mean": float(node_support_cpu.mean().item()) if node_support_cpu.numel() else 0.0,
            "model_config": structural_bundle["model_config"],
        },
        "routing": {
            "routing_threshold_valid": float(routing_threshold),
            "routed_counts": {
                "train": int(routed_mask[stage_runner.train_mask].sum()),
                "valid": int(routed_mask[stage_runner.val_mask].sum()),
                "test": int(routed_mask[stage_runner.test_mask].sum()),
                "all": int(routed_mask.sum()),
            },
            "routed_count_valid_target": int(router_bundle["routed_count_valid_target"]),
            "test_conflict_summary": router_bundle["test_conflict_summary"],
        },
        "selection": {
            "candidate_edge_count": int(len(candidate_rows)),
            "positive_delta_conflict_count": int(sum(1 for row in candidate_rows if float(row["delta_conflict"]) > 0.0)),
            "positive_delta_conflict_ratio": float(sum(1 for row in candidate_rows if float(row["delta_conflict"]) > 0.0) / max(len(candidate_rows), 1)),
            "selected_conflict_edge_count": int(len(conflict_removed_positions)),
            "selected_random_edge_count": int(len(rand_removed_positions)),
            "guard_blocked_count": int(sum(1 for row in candidate_rows if bool(row["guard_blocked"]))),
            "duplicate_selected_attempts": int(duplicate_attempts),
            "selected_edges_per_relation_role": selected_by_relation_role,
            "selected_edges_per_bucket": selected_by_bucket,
            "local_conflict_zero_in_nodes_after": int((conflict_zero_in == 0).sum()),
            "local_conflict_zero_out_nodes_after": int((conflict_zero_out == 0).sum()),
            "local_rand_zero_in_nodes_after": int((rand_zero_in == 0).sum()),
            "local_rand_zero_out_nodes_after": int((rand_zero_out == 0).sum()),
            "conflict_before_mean": float(np.mean(conflict_score)) if conflict_score.size else 0.0,
            "conflict_after_mean_selected_targets": float(np.mean([row["conflict_after"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
            "selected_semantic_cosine_mean": float(np.mean([row["semantic_cosine"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
            "selected_reconstruction_support_mean": float(np.mean([row["reconstruction_support"] for row in selected_rows_sorted if bool(row["selected"])])) if evidence_rows else 0.0,
        },
    }

    manifest = {
        "contract": "local_dignn_conflict_refine_diag_v1",
        "status": "completed",
        "stage": "local_dignn_conflict_refine_diag",
        "routing_mode": "dignn_style_dual_view_conflict_plus_reconstruction",
        "paper_faithful_dignn_style": True,
        "paper_faithful_dig_in_gnn_style_local_selection": True,
        "official_code_verified": False,
        "diagnostic_only": True,
        "candidate_scope": "1hop_incident_edges",
        "bucket_definition": "(relation, target_role)",
        "target_roles": ["incoming", "outgoing"],
        "topk_per_bucket": int(topk_per_bucket),
        "routed_budget": float(routed_budget),
        "routing_threshold_valid": float(routing_threshold),
        "structural_guard": "no_zero_in/no_zero_out" if degree_guard_enabled else "disabled",
        "structural_view": structural_bundle["model_config"],
        "semantic_view": {"source": lm_head.get("source", "lm_only_head"), "manifest": lm_head["manifest"]},
        "dual_view_proxy": {
            "topology_features": "directed_degree_relation_statistics",
            "attribute_features": "raw_semantic_embedding",
            "fusion": "attention_weighted_sum",
            "mi_objective": "topology_attribute_mutual_exclusive_proxy",
        },
        "conflict_definition": "JS(p_struct(node;G), p_sem(node))",
        "counterfactual_definition": "delta_conflict = conflict_before - conflict_after_single_edge_delete",
        "harmful_edge_scoring": "delta_conflict + support_weight*(1-reconstruction_support)",
        "test_labels_used_for_training": False,
        "test_labels_used_for_threshold": False,
        "oracle_labels_used": False,
        "frozen_g0_dir": str(context["frozen_g0"]["dir"]),
        "frozen_g0_manifest": context["frozen_g0"]["manifest"],
        "graph_source": context["graph_bundle"]["source"],
        "graph_source_root": context["graph_bundle"]["source_root"],
        "local_dignn_conflict_frozen_g0_dir": str(local_conflict_context["dir"]),
        "local_dignn_rand_frozen_g0_dir": str(local_rand_context["dir"]),
        "local_dignn_conflict_edge_index_path": str(stage_dir / "local_dignn_conflict_edge_index.pt"),
        "local_dignn_conflict_edge_type_path": str(stage_dir / "local_dignn_conflict_edge_type.pt"),
        "local_dignn_rand_edge_index_path": str(stage_dir / "local_dignn_rand_edge_index.pt"),
        "local_dignn_rand_edge_type_path": str(stage_dir / "local_dignn_rand_edge_type.pt"),
        "conflict_evidence_edges_path": str(stage_dir / "conflict_evidence_edges.jsonl"),
    }
    stage_runner._write_stage_manifest(stage_dir, manifest)
    write_json(stage_dir / "metrics.json", metrics_payload)
    write_text(
        stage_dir / "selected_edges.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in selected_rows_sorted) + ("\n" if selected_rows_sorted else ""),
    )
    write_text(
        stage_dir / "routed_nodes.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in routed_rows) + ("\n" if routed_rows else ""),
    )
    write_text(
        stage_dir / "conflict_evidence_edges.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in evidence_rows) + ("\n" if evidence_rows else ""),
    )
    write_torch(stage_dir / "local_dignn_conflict_edge_index.pt", conflict_edge_index)
    write_torch(stage_dir / "local_dignn_conflict_edge_type.pt", conflict_edge_type)
    write_torch(stage_dir / "local_dignn_rand_edge_index.pt", rand_edge_index)
    write_torch(stage_dir / "local_dignn_rand_edge_type.pt", rand_edge_type)
    save_stage_artifacts(
        stage_dir,
        {
            "view_outputs.pt": {
                "semantic_prob": semantic_prob,
                "structural_prob": structural_prob,
                "structural_pred": structural_pred,
                "conflict_score": torch.tensor(conflict_score, dtype=torch.float32),
                "routed_mask": torch.tensor(routed_mask, dtype=torch.bool),
                "routing_threshold": torch.tensor([routing_threshold], dtype=torch.float32),
                "topology_features": topology_features,
                "attribute_features": semantic_raw,
                "topology_repr": structural_outputs["topology_repr"],
                "attribute_repr": structural_outputs["attribute_repr"],
                "fused_repr": structural_outputs["fused_repr"],
                "attention_weights": structural_outputs["attention_weights"],
                "node_support": node_support_cpu,
            },
            **base_bundle,
        },
    )
    write_text(
        stage_dir / "notes.md",
        "\n".join(
            [
                "# DIGNN-style local conflict propagation refiner",
                "",
                "This stage trains a DIGNN-style structural encoder with disentangled shared/private views and edge reconstruction support.",
                "Routed hard nodes are selected by high JS divergence between structural and semantic posteriors.",
                "Candidate edges are target-centered 1-hop incident edges, bucketed by (relation, target_role).",
                "Propagation validation is performed by retraining frozen_g0 on the edited graph only.",
                "Removed conflict edges are preserved as evidence-ready provenance for later local evidence graph consumption.",
            ]
        )
        + "\n",
    )
    return {
        "stage": "local_dignn_conflict_refine_diag",
        "stage_dir": str(stage_dir),
        "metrics": metrics_payload,
    }
