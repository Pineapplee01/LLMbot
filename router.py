import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.tensor(np.asarray(value), dtype=torch.float32)


def fit_reliability_temperature(logits, labels, min_temperature=0.5, max_temperature=5.0, step=0.1):
    logits_t = _to_float_tensor(logits)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if logits_t.ndim != 2 or labels_t.numel() != int(logits_t.shape[0]) or labels_t.numel() == 0:
        return {
            "temperature": 1.0,
            "best_nll": None,
            "source": "identity_fallback_invalid_input",
        }

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.arange(float(min_temperature), float(max_temperature) + 1e-8, float(step)):
        scaled_probs = F.softmax(logits_t / float(temperature), dim=1)
        nll = -torch.log(scaled_probs[torch.arange(labels_t.numel()), labels_t] + 1e-8).mean().item()
        if nll < best_nll:
            best_nll = float(nll)
            best_temperature = float(temperature)
    return {
        "temperature": float(best_temperature),
        "best_nll": float(best_nll),
        "source": "grid_search_nll",
    }


def apply_temperature_scaled_probs(logits, temperature):
    logits_t = _to_float_tensor(logits)
    safe_temperature = max(float(temperature), 1e-6)
    return F.softmax(logits_t / safe_temperature, dim=1)


class GlanceReliabilityRouterMLP(nn.Module):
    """Reliability-first router for selecting base-unreliable nodes."""

    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        bottleneck_dim = int(bottleneck_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        normalized = self.input_norm(x)
        logits = self.net(normalized).squeeze(-1)
        prob = torch.sigmoid(logits)
        return logits, prob


class SelectiveNetResidualRouter(nn.Module):
    """SelectiveNet-style residual-error router.

    The model keeps a shared feature trunk and exposes:
    - prediction head: predicts P(base_wrong | x)
    - selection head: predicts whether the node should remain in-coverage
    - auxiliary head: stabilizes representation learning on all samples
    """

    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        bottleneck_dim = int(bottleneck_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.prediction_head = nn.Linear(bottleneck_dim, 1)
        self.selection_head = nn.Linear(bottleneck_dim, 1)
        self.auxiliary_head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x):
        normalized = self.input_norm(x)
        hidden = self.trunk(normalized)
        prediction_logits = self.prediction_head(hidden).squeeze(-1)
        selection_logits = self.selection_head(hidden).squeeze(-1)
        auxiliary_logits = self.auxiliary_head(hidden).squeeze(-1)
        prediction_prob = torch.sigmoid(prediction_logits)
        selection_prob = torch.sigmoid(selection_logits)
        auxiliary_prob = torch.sigmoid(auxiliary_logits)
        combined_score = selection_prob * prediction_prob
        return {
            "prediction_logits": prediction_logits,
            "prediction_prob": prediction_prob,
            "selection_logits": selection_logits,
            "selection_prob": selection_prob,
            "auxiliary_logits": auxiliary_logits,
            "auxiliary_prob": auxiliary_prob,
            "combined_score": combined_score,
        }


def selectivenet_selective_loss(
    prediction_prob,
    selection_prob,
    targets,
    *,
    coverage_target,
    lambda_coverage=32.0,
    eps=1e-6,
):
    prediction_prob = prediction_prob.float().reshape(-1)
    selection_prob = selection_prob.float().reshape(-1)
    targets = targets.float().reshape(-1)
    if prediction_prob.numel() == 0:
        zero = prediction_prob.sum() * 0.0
        return {
            "loss": zero,
            "empirical_coverage": zero.detach(),
            "selective_risk": zero.detach(),
            "coverage_penalty": zero.detach(),
        }
    bce = F.binary_cross_entropy(
        prediction_prob.clamp(min=eps, max=1.0 - eps),
        targets,
        reduction="none",
    )
    empirical_coverage = selection_prob.mean()
    selective_risk = (bce * selection_prob).sum() / selection_prob.sum().clamp_min(eps)
    coverage_gap = torch.clamp(float(coverage_target) - empirical_coverage, min=0.0)
    coverage_penalty = coverage_gap.pow(2)
    total_loss = selective_risk + float(lambda_coverage) * coverage_penalty
    return {
        "loss": total_loss,
        "empirical_coverage": empirical_coverage.detach(),
        "selective_risk": selective_risk.detach(),
        "coverage_penalty": coverage_penalty.detach(),
    }


def _safe_mean(values):
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    return numerator / np.clip(denominator, 1.0, None)


def _build_directional_graph_statistics(num_nodes, edge_index, edge_type, q_probs, pred_class):
    total_degree = np.zeros(num_nodes, dtype=np.float32)
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_rel_counts = np.zeros((num_nodes, 2), dtype=np.float32)
    out_rel_counts = np.zeros((num_nodes, 2), dtype=np.float32)
    soft_homophily = np.zeros(num_nodes, dtype=np.float32)
    in_homophily = np.zeros(num_nodes, dtype=np.float32)
    out_homophily = np.zeros(num_nodes, dtype=np.float32)
    in_disagreement = np.zeros(num_nodes, dtype=np.float32)
    out_disagreement = np.zeros(num_nodes, dtype=np.float32)
    reciprocity = np.zeros(num_nodes, dtype=np.float32)
    relative_degree = np.ones(num_nodes, dtype=np.float32)

    if edge_index is None:
        return {
            "total_degree": total_degree,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "in_rel_counts": in_rel_counts,
            "out_rel_counts": out_rel_counts,
            "soft_homophily": soft_homophily,
            "in_homophily": in_homophily,
            "out_homophily": out_homophily,
            "in_disagreement": in_disagreement,
            "out_disagreement": out_disagreement,
            "reciprocity": reciprocity,
            "relative_degree": relative_degree,
        }

    edge_index_t = _to_float_tensor(edge_index).long()
    if edge_index_t.ndim != 2 or int(edge_index_t.shape[0]) != 2:
        raise ValueError("router feature builder expects edge_index shaped [2, num_edges].")
    src = edge_index_t[0].numpy().astype(np.int64)
    dst = edge_index_t[1].numpy().astype(np.int64)
    if edge_type is None:
        rel = np.full(src.shape[0], -1, dtype=np.int64)
    else:
        rel = _to_float_tensor(edge_type).long().numpy().astype(np.int64).reshape(-1)
        if rel.shape[0] != src.shape[0]:
            raise ValueError("router feature builder expects edge_type aligned with edge_index.")

    incoming = [[] for _ in range(num_nodes)]
    outgoing = [[] for _ in range(num_nodes)]
    edge_pairs = set()

    for edge_src, edge_dst, edge_rel in zip(src.tolist(), dst.tolist(), rel.tolist()):
        if edge_src < 0 or edge_src >= num_nodes or edge_dst < 0 or edge_dst >= num_nodes:
            continue
        outgoing[edge_src].append(edge_dst)
        incoming[edge_dst].append(edge_src)
        total_degree[edge_src] += 1.0
        total_degree[edge_dst] += 1.0
        out_degree[edge_src] += 1.0
        in_degree[edge_dst] += 1.0
        if edge_rel in (0, 1):
            out_rel_counts[edge_src, edge_rel] += 1.0
            in_rel_counts[edge_dst, edge_rel] += 1.0
        edge_pairs.add((edge_src, edge_dst))

    for node_idx in range(num_nodes):
        incoming_idx = np.asarray(incoming[node_idx], dtype=np.int64)
        outgoing_idx = np.asarray(outgoing[node_idx], dtype=np.int64)
        all_neighbors = np.concatenate([incoming_idx, outgoing_idx], axis=0)
        if all_neighbors.size > 0:
            neighbor_degree = total_degree[all_neighbors]
            relative_degree[node_idx] = float(
                np.mean(np.sqrt(total_degree[node_idx] + 1.0) / np.sqrt(neighbor_degree + 1.0))
            )
            neighbor_probs = q_probs[all_neighbors]
            soft_homophily[node_idx] = float((q_probs[node_idx] * neighbor_probs.mean(axis=0)).sum())
        if incoming_idx.size > 0:
            incoming_probs = q_probs[incoming_idx]
            in_homophily[node_idx] = float((q_probs[node_idx] * incoming_probs.mean(axis=0)).sum())
            in_disagreement[node_idx] = float(np.mean(pred_class[incoming_idx] != pred_class[node_idx]))
        if outgoing_idx.size > 0:
            outgoing_probs = q_probs[outgoing_idx]
            out_homophily[node_idx] = float((q_probs[node_idx] * outgoing_probs.mean(axis=0)).sum())
            out_disagreement[node_idx] = float(np.mean(pred_class[outgoing_idx] != pred_class[node_idx]))
            reciprocal_edges = sum(1 for dst_idx in outgoing_idx.tolist() if (dst_idx, node_idx) in edge_pairs)
            reciprocity[node_idx] = float(reciprocal_edges / max(outgoing_idx.size, 1))

    return {
        "total_degree": total_degree,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_rel_counts": in_rel_counts,
        "out_rel_counts": out_rel_counts,
        "soft_homophily": np.clip(soft_homophily, 0.0, 1.0),
        "in_homophily": np.clip(in_homophily, 0.0, 1.0),
        "out_homophily": np.clip(out_homophily, 0.0, 1.0),
        "in_disagreement": np.clip(in_disagreement, 0.0, 1.0),
        "out_disagreement": np.clip(out_disagreement, 0.0, 1.0),
        "reciprocity": np.clip(reciprocity, 0.0, 1.0),
        "relative_degree": np.nan_to_num(relative_degree, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32),
    }


def build_reliability_router_feature_bundle(
    *,
    logits_gnn,
    z_gnn,
    original_node_features,
    q_probs,
    uncertainty,
    edge_index=None,
    edge_type=None,
    calibration_temperature=1.0,
):
    logits_t = _to_float_tensor(logits_gnn)
    z_gnn_t = _to_float_tensor(z_gnn)
    node_features_t = _to_float_tensor(original_node_features)
    q_probs_t = _to_float_tensor(q_probs)
    uncertainty_t = _to_float_tensor(uncertainty).reshape(-1)

    num_nodes = int(z_gnn_t.shape[0])
    if int(logits_t.shape[0]) != num_nodes:
        raise ValueError("router feature builder requires logits aligned with z_gnn.")
    if int(node_features_t.shape[0]) != num_nodes:
        raise ValueError("router feature builder requires original node features aligned with z_gnn.")
    if int(q_probs_t.shape[0]) != num_nodes:
        raise ValueError("router feature builder requires q_probs aligned with z_gnn.")
    if int(uncertainty_t.numel()) != num_nodes:
        raise ValueError("router feature builder requires one uncertainty value per node.")

    calibrated_probs = apply_temperature_scaled_probs(logits_t, calibration_temperature)
    calibrated_probs_np = calibrated_probs.numpy().astype(np.float32)
    pred_class = calibrated_probs_np.argmax(axis=1).astype(np.int64)
    prob_margin = (calibrated_probs_np[:, 1] - calibrated_probs_np[:, 0]).astype(np.float32)
    pred_prob = np.max(calibrated_probs_np, axis=1).astype(np.float32)
    entropy = (
        -(calibrated_probs_np * np.log(np.clip(calibrated_probs_np, 1e-6, 1.0))).sum(axis=1)
    ).astype(np.float32)
    raw_logit_margin = (logits_t[:, 1] - logits_t[:, 0]).detach().cpu().numpy().astype(np.float32)

    graph_stats = _build_directional_graph_statistics(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_type=edge_type,
        q_probs=q_probs_t.numpy().astype(np.float32),
        pred_class=pred_class,
    )

    total_degree = graph_stats["total_degree"]
    in_degree = graph_stats["in_degree"]
    out_degree = graph_stats["out_degree"]
    in_rel_counts = graph_stats["in_rel_counts"]
    out_rel_counts = graph_stats["out_rel_counts"]
    out_in_ratio = _safe_ratio(out_degree + 1.0, in_degree + 1.0)
    in_out_ratio = _safe_ratio(in_degree + 1.0, out_degree + 1.0)

    has_in = (in_degree > 0).astype(np.float32)
    has_out = (out_degree > 0).astype(np.float32)
    is_isolated = (total_degree == 0).astype(np.float32)
    is_sparse_total = (total_degree <= 1).astype(np.float32)
    is_sparse_in = (in_degree <= 1).astype(np.float32)
    is_sparse_out = (out_degree <= 1).astype(np.float32)

    z_gnn_np = z_gnn_t.numpy().astype(np.float32)
    node_features_np = node_features_t.numpy().astype(np.float32)
    z_gnn_norm = np.linalg.norm(z_gnn_np, axis=1).astype(np.float32)
    z_gnn_mean = z_gnn_np.mean(axis=1).astype(np.float32)
    z_gnn_std = z_gnn_np.std(axis=1).astype(np.float32)
    node_feature_norm = np.linalg.norm(node_features_np, axis=1).astype(np.float32)
    node_feature_mean = node_features_np.mean(axis=1).astype(np.float32)
    node_feature_std = node_features_np.std(axis=1).astype(np.float32)

    feature_columns = [
        raw_logit_margin.reshape(-1, 1),
        prob_margin.reshape(-1, 1),
        pred_prob.reshape(-1, 1),
        entropy.reshape(-1, 1),
        uncertainty_t.numpy().astype(np.float32).reshape(-1, 1),
        graph_stats["soft_homophily"].reshape(-1, 1),
        graph_stats["in_homophily"].reshape(-1, 1),
        graph_stats["out_homophily"].reshape(-1, 1),
        graph_stats["in_disagreement"].reshape(-1, 1),
        graph_stats["out_disagreement"].reshape(-1, 1),
        graph_stats["reciprocity"].reshape(-1, 1),
        np.log1p(total_degree).reshape(-1, 1).astype(np.float32),
        np.log1p(in_degree).reshape(-1, 1).astype(np.float32),
        np.log1p(out_degree).reshape(-1, 1).astype(np.float32),
        graph_stats["relative_degree"].reshape(-1, 1),
        np.log1p(in_rel_counts[:, 0]).reshape(-1, 1).astype(np.float32),
        np.log1p(in_rel_counts[:, 1]).reshape(-1, 1).astype(np.float32),
        np.log1p(out_rel_counts[:, 0]).reshape(-1, 1).astype(np.float32),
        np.log1p(out_rel_counts[:, 1]).reshape(-1, 1).astype(np.float32),
        out_in_ratio.reshape(-1, 1).astype(np.float32),
        in_out_ratio.reshape(-1, 1).astype(np.float32),
        has_in.reshape(-1, 1),
        has_out.reshape(-1, 1),
        is_isolated.reshape(-1, 1),
        is_sparse_total.reshape(-1, 1),
        is_sparse_in.reshape(-1, 1),
        is_sparse_out.reshape(-1, 1),
        z_gnn_norm.reshape(-1, 1),
        z_gnn_mean.reshape(-1, 1),
        z_gnn_std.reshape(-1, 1),
        node_feature_norm.reshape(-1, 1),
        node_feature_mean.reshape(-1, 1),
        node_feature_std.reshape(-1, 1),
    ]
    features = np.concatenate(feature_columns, axis=1).astype(np.float32)

    feature_names = [
        "raw_logit_margin",
        "calibrated_prob_margin",
        "calibrated_pred_prob",
        "calibrated_entropy",
        "mc_dropout_uncertainty",
        "soft_local_homophily",
        "in_soft_homophily",
        "out_soft_homophily",
        "in_prediction_disagreement",
        "out_prediction_disagreement",
        "reciprocity_ratio",
        "log_total_degree",
        "log_in_degree",
        "log_out_degree",
        "relative_degree",
        "log_in_rel0_count",
        "log_in_rel1_count",
        "log_out_rel0_count",
        "log_out_rel1_count",
        "out_in_ratio",
        "in_out_ratio",
        "has_in_neighbors",
        "has_out_neighbors",
        "is_isolated",
        "is_sparse_total",
        "is_sparse_in",
        "is_sparse_out",
        "z_gnn_l2_norm",
        "z_gnn_mean",
        "z_gnn_std",
        "node_feature_l2_norm",
        "node_feature_mean",
        "node_feature_std",
    ]

    return {
        "features": features,
        "feature_names": feature_names,
        "full_feature_bundle": {
            "feature_family": "reliability_first_social_router_v1",
            "calibrated_confidence": True,
            "relation_aware_social_features": True,
            "temperature_scaling": {
                "temperature": float(calibration_temperature),
                "source": "validation_nll_grid_search",
            },
            "feature_groups": {
                "confidence": [
                    "raw_logit_margin",
                    "calibrated_prob_margin",
                    "calibrated_pred_prob",
                    "calibrated_entropy",
                ],
                "uncertainty": ["mc_dropout_uncertainty"],
                "social_structure": [
                    "soft_local_homophily",
                    "in_soft_homophily",
                    "out_soft_homophily",
                    "in_prediction_disagreement",
                    "out_prediction_disagreement",
                    "reciprocity_ratio",
                    "log_total_degree",
                    "log_in_degree",
                    "log_out_degree",
                    "relative_degree",
                    "log_in_rel0_count",
                    "log_in_rel1_count",
                    "log_out_rel0_count",
                    "log_out_rel1_count",
                    "out_in_ratio",
                    "in_out_ratio",
                    "has_in_neighbors",
                    "has_out_neighbors",
                    "is_isolated",
                    "is_sparse_total",
                    "is_sparse_in",
                    "is_sparse_out",
                ],
                "representation_summaries": [
                    "z_gnn_l2_norm",
                    "z_gnn_mean",
                    "z_gnn_std",
                    "node_feature_l2_norm",
                    "node_feature_mean",
                    "node_feature_std",
                ],
            },
            "feature_dimension": int(features.shape[1]),
        },
    }
