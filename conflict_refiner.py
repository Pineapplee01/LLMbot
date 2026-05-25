from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    relation_mix = (in_rel + out_rel)
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
