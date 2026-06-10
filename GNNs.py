import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, HypergraphConv, RGCNConv

try:
    import dhg
    from dhg.nn import HGNNConv as DHG_HGNNConv
except Exception:
    dhg = None
    DHG_HGNNConv = None

from hypergnn import build_batch_local_knn_hypergraph, build_dynamic_hypergraph
from RGT import RGTLayer
from SimpleHGN import SimpleHGNConv


def _cfg(model_config, *names, default=None):
    for name in names:
        if name in model_config:
            return model_config[name]
    if default is not None:
        return default
    raise KeyError(f"Missing config keys {names}")


class BaseGraphBackbone(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = dict(model_config)
        self.hidden_dim = _cfg(model_config, "gnn_hidden_dim", "hidden_dim")
        self.n_layers = _cfg(model_config, "gnn_n_layers", "n_layers")
        self.n_relations = _cfg(model_config, "n_relations", default=2)
        self.dropout = nn.Dropout(float(_cfg(model_config, "dropout", default=0.4)))
        self.activation = self._build_activation(
            str(_cfg(model_config, "activation", default="leakyrelu")).lower()
        )
        self.linear_out = nn.Linear(self.hidden_dim, 2)

    def _build_activation(self, name):
        if name == "leakyrelu":
            return nn.LeakyReLU()
        if name == "relu":
            return nn.ReLU()
        if name == "elu":
            return nn.ELU()
        raise ValueError('Please choose activation function from "leakyrelu", "relu" or "elu".')

    def forward_outputs(self, x, edge_index, edge_type):
        hidden = self.encode(x, edge_index, edge_type)
        logits = self.linear_out(hidden)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "aux_features": {},
        }

    def forward(self, x, edge_index, edge_type):
        return self.forward_outputs(x, edge_index, edge_type)["logits"]


def _as_long_tensor_list(values):
    return [torch.tensor(item, dtype=torch.long) for item in values]


def _cpu_scalar_dict(payload):
    result = {}
    for key, value in dict(payload or {}).items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[key] = value.detach().cpu().item()
            else:
                result[key] = value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


def _dhg_hypergraph_from_incidence(hyperedge_index, num_nodes, device):
    if hyperedge_index is None or int(hyperedge_index.numel()) == 0:
        return None
    incidence_cpu = hyperedge_index.detach().cpu().long()
    if incidence_cpu.dim() != 2 or int(incidence_cpu.size(0)) != 2:
        raise ValueError("DHG incidence conversion expects hyperedge_index shaped [2, incidence_count].")
    hyperedge_count = int(incidence_cpu[1].max().item()) + 1 if int(incidence_cpu.size(1)) else 0
    if hyperedge_count <= 0:
        return None
    e_list = [[] for _ in range(hyperedge_count)]
    for node_id, hyperedge_id in incidence_cpu.t().tolist():
        node = int(node_id)
        edge = int(hyperedge_id)
        if 0 <= node < int(num_nodes) and 0 <= edge < hyperedge_count:
            e_list[edge].append(node)
    e_list = [sorted(set(members)) for members in e_list if members]
    if not e_list:
        return None
    try:
        hg = dhg.Hypergraph(int(num_nodes), e_list)
    except TypeError:
        hg = dhg.Hypergraph(num_v=int(num_nodes), e_list=e_list)
    return hg.to(device)


class _BidirectionalCrossAttention(nn.Module):
    def __init__(self, model_dim, q_dim, k_dim, v_dim):
        super().__init__()
        self.query_matrix = nn.Linear(model_dim, q_dim)
        self.key_matrix = nn.Linear(model_dim, k_dim)
        self.value_matrix = nn.Linear(model_dim, v_dim)

    def forward(self, query, key, value):
        q = self.query_matrix(query)
        k = self.key_matrix(key)
        v = self.value_matrix(value)
        score = torch.bmm(q, k.transpose(-1, -2))
        scaled_score = score / (k.shape[-1] ** 0.5)
        return torch.bmm(F.softmax(scaled_score, dim=-1), v)


class _MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim, q_dim, k_dim, v_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [_BidirectionalCrossAttention(model_dim, q_dim, k_dim, v_dim) for _ in range(int(num_heads))]
        )
        self.projection_matrix = nn.Linear(int(num_heads) * v_dim, model_dim)

    def forward(self, query, key, value):
        heads = [head(query, key, value) for head in self.attention_heads]
        return self.projection_matrix(torch.cat(heads, dim=-1))


class _Feedforward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.linear_w1 = nn.Linear(model_dim, hidden_dim)
        self.linear_w2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_w2(self.relu(self.linear_w1(x))))


class _AddNorm(nn.Module):
    def __init__(self, model_dim, dropout_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))


class _MultiAttnLayer(nn.Module):
    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()
        q_dim = k_dim = v_dim = model_dim // num_heads
        self.attn_1 = _MultiHeadAttention(num_heads, model_dim, q_dim, k_dim, v_dim)
        self.add_norm_1 = _AddNorm(model_dim, dropout_rate)
        self.add_norm_2 = _AddNorm(model_dim, dropout_rate)
        self.ff = _Feedforward(model_dim, hidden_dim, dropout_rate)

    def forward(self, query_modality, modality_a):
        attn_output = self.add_norm_1(query_modality, lambda q: self.attn_1(q, modality_a, modality_a))
        return self.add_norm_2(attn_output, self.ff)


class _MultiAttn(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            [_MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(int(num_layers))]
        )

    def forward(self, query_modality, modality_a):
        for layer in self.layers:
            query_modality = layer(query_modality, modality_a)
        return query_modality


class _MultiAttnModel(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()
        self.multiattn_low = _MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_high = _MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)

    def forward(self, low_features, high_features):
        f_l = self.multiattn_low(low_features, high_features)
        f_h = self.multiattn_high(high_features, low_features)
        return f_l, f_h


def _init_hyperscan_detector(module, model_config, hidden_dim):
    style = str(
        _cfg(
            model_config,
            "graph_second_view_fusion",
            "hyperscan_detector_style",
            default="residual",
        )
        or "residual"
    ).lower()
    if style == "multiattn":
        style = "original_cross_attention"
    if style not in {"residual", "original_cross_attention"}:
        raise ValueError("--graph_second_view_fusion must be one of {residual, multiattn}.")
    module.hyperscan_detector_style = style
    module.graph_second_view_fusion = "multiattn" if style == "original_cross_attention" else "residual"
    if style == "original_cross_attention":
        detector_heads = 4
        if int(hidden_dim) % detector_heads != 0:
            raise ValueError("original_cross_attention detector requires hidden_dim to be divisible by 4.")
        module.detector_multiattn = _MultiAttnModel(
            num_layers=4,
            model_dim=int(hidden_dim),
            num_heads=detector_heads,
            hidden_dim=int(hidden_dim) * 2,
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        module.detector_activation = nn.LeakyReLU()
        module.linear_out = nn.Linear(int(hidden_dim) * 2, 2)
    else:
        module.fusion_linear = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))


def _apply_hyperscan_detector(module, x_low, x_high, incident_scale):
    if module.hyperscan_detector_style == "original_cross_attention":
        fused_low, fused_high = module.detector_multiattn(x_low.unsqueeze(1), x_high.unsqueeze(1))
        hidden = torch.cat((fused_low.squeeze(1), fused_high.squeeze(1)), dim=-1)
        hidden = module.detector_activation(hidden)
        return hidden, module.linear_out(hidden)

    delta = module.fusion_linear(torch.cat([x_low, x_high], dim=1))
    delta = module.activation(delta)
    delta = module.dropout(delta)
    hidden = x_low + delta * incident_scale
    return hidden, module.linear_out(hidden)


def mhlgc_mask_node_features(x, mask_probability=0.0):
    p = float(mask_probability or 0.0)
    if p <= 0.0:
        return x
    if p >= 1.0:
        raise ValueError("mhlgc feature mask probability must be < 1.0.")
    keep = (torch.rand(x.shape, dtype=x.dtype, device=x.device) >= p).to(dtype=x.dtype)
    return x * keep / max(1.0 - p, 1e-12)


def mhlgc_mask_edges(edge_index, edge_type, mask_probability=0.0):
    p = float(mask_probability or 0.0)
    if p <= 0.0 or edge_index is None or int(edge_index.size(1)) == 0:
        return edge_index, edge_type
    if p >= 1.0:
        raise ValueError("mhlgc edge mask probability must be < 1.0.")
    edge_count = int(edge_index.size(1))
    keep = torch.rand(edge_count, device=edge_index.device) >= p
    if not bool(keep.any().item()):
        keep[torch.randint(edge_count, (1,), device=edge_index.device)] = True
    masked_edge_index = edge_index[:, keep].contiguous()
    if edge_type is None:
        return masked_edge_index, edge_type
    return masked_edge_index, edge_type.view(-1)[keep].contiguous()


def mhlgc_select_borderline_anchors(labels, fraud_scores, positive_label=1, anchors_per_batch=1, anchor_mask=None):
    labels = labels.detach().view(-1)
    positive_mask = labels == int(positive_label)
    if anchor_mask is not None:
        positive_mask = positive_mask & anchor_mask.to(labels.device).bool().view(-1)
    positive_idx = torch.nonzero(positive_mask, as_tuple=False).view(-1)
    if positive_idx.numel() == 0:
        return positive_idx
    anchors_per_batch = max(int(anchors_per_batch or 1), 1)
    if fraud_scores is None:
        return positive_idx[:anchors_per_batch]
    scores = fraud_scores.detach().view(-1)[positive_idx]
    order = torch.argsort(scores, descending=False)
    return positive_idx[order[: min(anchors_per_batch, int(order.numel()))]]


def mhlgc_llm_guided_contrastive_loss(
    origin_embeddings,
    augmented_embeddings,
    labels,
    fraud_scores=None,
    semantic_embeddings=None,
    positive_label=1,
    anchors_per_batch=1,
    beta=1.0,
    gamma=0.5,
    temperature=1.0,
    negative_mask=None,
):
    """MH-LGC-style hardness-aware InfoNCE for routed hard-node training.

    This implements the loss boundary from Ou et al.: the LLM is used only as
    a semantic guide for hard-negative weighting, while the optimized
    embeddings are the GNN embeddings from original and augmented views.
    """
    if origin_embeddings.dim() != 2 or augmented_embeddings.dim() != 2:
        raise ValueError("origin_embeddings and augmented_embeddings must be 2-D tensors.")
    if origin_embeddings.shape != augmented_embeddings.shape:
        raise ValueError("origin_embeddings and augmented_embeddings must have the same shape.")
    labels = labels.to(origin_embeddings.device).view(-1)
    if int(labels.numel()) != int(origin_embeddings.size(0)):
        raise ValueError("labels must align with origin_embeddings rows.")
    temperature = float(temperature or 1.0)
    if temperature <= 0.0:
        raise ValueError("mhlgc temperature must be > 0.")
    gamma = min(max(float(gamma or 0.0), 0.0), 1.0)
    beta = float(beta or 0.0)
    semantic_embeddings_clean = None
    anchor_mask = None
    if semantic_embeddings is not None and gamma > 0.0:
        semantic_embeddings_clean = torch.nan_to_num(
            semantic_embeddings.to(origin_embeddings.device).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if (
            semantic_embeddings_clean.dim() != 2
            or int(semantic_embeddings_clean.size(0)) != int(origin_embeddings.size(0))
        ):
            raise ValueError("semantic_embeddings must be shaped [batch_nodes, semantic_dim].")
        anchor_mask = semantic_embeddings_clean.norm(dim=1) > 0.0

    anchors = mhlgc_select_borderline_anchors(
        labels=labels,
        fraud_scores=fraud_scores,
        positive_label=positive_label,
        anchors_per_batch=anchors_per_batch,
        anchor_mask=anchor_mask,
    )
    if negative_mask is None:
        negative_mask = labels != int(positive_label)
    else:
        negative_mask = negative_mask.to(origin_embeddings.device).bool().view(-1)
    negative_idx = torch.nonzero(negative_mask, as_tuple=False).view(-1)
    zero = origin_embeddings.sum() * 0.0
    if anchors.numel() == 0 or negative_idx.numel() == 0:
        return zero, {
            "mhlgc_anchor_count": int(anchors.numel()),
            "mhlgc_anchor_candidate_count": int(anchor_mask.sum().item()) if anchor_mask is not None else int((labels == int(positive_label)).sum().item()),
            "mhlgc_negative_count": int(negative_idx.numel()),
            "mhlgc_active": False,
        }

    z_anchor = origin_embeddings[anchors]
    z_positive = augmented_embeddings[anchors]
    z_negative = origin_embeddings[negative_idx]
    pos_logits = torch.sum(z_anchor * z_positive, dim=1) / temperature
    neg_logits = torch.matmul(z_anchor, z_negative.transpose(0, 1)) / temperature

    structure_sim = torch.matmul(
        z_anchor.detach(),
        z_negative.detach().transpose(0, 1),
    ) / max(int(z_anchor.size(1)), 1)
    if semantic_embeddings_clean is not None and gamma > 0.0:
        g_anchor = semantic_embeddings_clean[anchors]
        g_negative = semantic_embeddings_clean[negative_idx]
        semantic_sim = torch.matmul(
            g_anchor.detach(),
            g_negative.detach().transpose(0, 1),
        ) / max(int(g_anchor.size(1)), 1)
        hardness = (1.0 - gamma) * structure_sim + gamma * semantic_sim
    else:
        hardness = structure_sim

    weights = F.softmax(beta * hardness, dim=1)
    log_weight = torch.log(weights.clamp_min(1e-12))
    log_neg = torch.logsumexp(neg_logits + log_weight, dim=1)
    loss = F.softplus(log_neg - pos_logits).mean()
    return loss, {
        "mhlgc_anchor_count": int(anchors.numel()),
        "mhlgc_anchor_candidate_count": int(anchor_mask.sum().item()) if anchor_mask is not None else int((labels == int(positive_label)).sum().item()),
        "mhlgc_negative_count": int(negative_idx.numel()),
        "mhlgc_active": True,
        "mhlgc_beta": float(beta),
        "mhlgc_gamma": float(gamma),
        "mhlgc_temperature": float(temperature),
    }


class FACNConv(nn.Module):
    """Attention-weighted relational convolution used as a BotRGCN-style proxy."""

    def __init__(self, in_channels, out_channels, num_relations=2, temperature=0.01, rcm=True, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.temperature = temperature
        self.rcm = rcm
        self.dropout = dropout

        self.root_weight = nn.Linear(in_channels, out_channels)
        self.weight = nn.Parameter(torch.empty(num_relations, in_channels, out_channels))
        self.att = nn.Parameter(torch.empty(num_relations, out_channels * 2, 1))
        if rcm:
            self.relation_weight = nn.Linear(num_relations * out_channels, num_relations * out_channels)
        self.relu = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.root_weight.weight)
        nn.init.zeros_(self.root_weight.bias)
        nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.att)
        if self.rcm:
            nn.init.kaiming_uniform_(self.relation_weight.weight)
            nn.init.zeros_(self.relation_weight.bias)

    def forward(self, x, edge_index, edge_type):
        root_x = self.root_weight(x)
        src, dst = edge_index
        relation_outputs = []

        for relation_id in range(self.num_relations):
            mask = edge_type == relation_id
            if not mask.any():
                relation_outputs.append(torch.zeros_like(root_x))
                continue

            src_r = src[mask]
            dst_r = dst[mask]
            x_r = x @ self.weight[relation_id]
            att_input = torch.cat([x_r[dst_r], x_r[src_r]], dim=-1)
            att_score = att_input @ self.att[relation_id]

            if self.training:
                eps = torch.rand_like(att_score).clamp(1e-6, 1 - 1e-6)
                att_score = att_score + torch.log(eps / (1 - eps))

            att_score = torch.tanh(att_score / self.temperature)
            att_score = F.dropout(att_score, p=self.dropout, training=self.training)

            degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, dst_r, torch.ones_like(dst_r, dtype=x.dtype))
            degree = degree.clamp(min=1)

            msg = x_r[src_r] * att_score / degree[dst_r].unsqueeze(-1)
            out_r = torch.zeros_like(root_x)
            out_r.index_add_(0, dst_r, msg)
            relation_outputs.append(out_r)

        if self.rcm:
            total = torch.cat(relation_outputs, dim=1)
            relation_weight = self.relation_weight(total)
            relation_weight = self.relu(relation_weight)
            relation_weight = relation_weight.view(-1, self.num_relations, self.out_channels)
            relation_weight = torch.softmax(relation_weight, dim=1)
            total = total.view(-1, self.num_relations, self.out_channels)
            return torch.sum(total * relation_weight, dim=1) + root_x

        return torch.sum(torch.stack(relation_outputs, dim=0), dim=0) + root_x


class BotRGCN(BaseGraphBackbone):
    """BotRGCN-style proxy backbone aligned to the current unified LM input surface."""

    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        self.linear_in = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.PReLU(self.hidden_dim),
        )
        self.conv1 = FACNConv(self.hidden_dim, self.hidden_dim, num_relations=self.n_relations, dropout=self.dropout.p)
        self.conv2 = FACNConv(self.hidden_dim, self.hidden_dim, num_relations=self.n_relations, dropout=self.dropout.p)
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(self, x, edge_index, edge_type):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        x = self.linear_in(x)
        x = self.conv1(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class RGCN(BaseGraphBackbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class RGCNHyperScanProxy(BaseGraphBackbone):
    """RGCN relation view plus a routed-node dynamic similarity hypergraph branch."""

    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        branch_cfg = dict(model_config.get("dynamic_similarity_branch", {}) or {})
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hyper_conv1 = HypergraphConv(self.hidden_dim + input_dim, self.hidden_dim, use_attention=False)
        self.hyper_conv2 = HypergraphConv(self.hidden_dim, self.hidden_dim, use_attention=False)
        _init_hyperscan_detector(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_candidate_scope = str(branch_cfg.get("candidate_scope", "undirected_relation_1hop"))
        self.dynamic_similarity_metric = str(branch_cfg.get("similarity_metric", "cosine"))
        self.dynamic_feature_source = str(branch_cfg.get("feature_source", "relation_node_repr_plus_g0_input"))
        self.dynamic_center_source = str(branch_cfg.get("center_source", "labeled_prefix"))
        self.dynamic_routed_nodes_path = str(branch_cfg.get("routed_nodes_path", "") or "")
        self.dynamic_routed_nodes_split = str(branch_cfg.get("routed_nodes_split", "all") or "all")
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": self.dynamic_candidate_scope,
            "similarity_metric": self.dynamic_similarity_metric,
            "feature_source": self.dynamic_feature_source,
            "center_source": self.dynamic_center_source,
            "routed_nodes_path": self.dynamic_routed_nodes_path,
            "routed_nodes_split": self.dynamic_routed_nodes_split,
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "pyg",
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
        }

    def _encode_relation_view(self, x, edge_index, edge_type):
        hidden = self.linear_in(x)
        hidden = self.dropout(hidden)
        for conv in self.convs:
            hidden = conv(hidden, edge_index, edge_type)
            hidden = self.activation(hidden)
        hidden = self.linear_pool(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def _build_dynamic_hypergraph(self, feature_tensor):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
        )

    def forward_outputs(self, x, edge_index, edge_type):
        x_low = self._encode_relation_view(x, edge_index, edge_type)
        x_new = torch.cat([x_low, x], dim=1)
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new)
        if hyperedge_index is None:
            x_high = torch.zeros_like(x_low)
            incident_scale = torch.zeros((x_low.size(0), 1), dtype=x_low.dtype, device=x_low.device)
        else:
            x_high = self.hyper_conv1(x_new, hyperedge_index)
            x_high = self.activation(x_high)
            x_high = self.dropout(x_high)
            x_high = self.hyper_conv2(x_high, hyperedge_index)
            x_high = self.activation(x_high)
            x_high = self.dropout(x_high)
            incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)

        hidden, logits = _apply_hyperscan_detector(self, x_low, x_high, incident_scale)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
            },
        }


class RGCNHyperScanNodeInputProxy(BaseGraphBackbone):
    """HyperScan-style backbone with paper-inspired tweet/num/cat node preprocessing."""

    def __init__(self, model_config):
        super().__init__(model_config)
        branch_cfg = dict(model_config.get("dynamic_similarity_branch", {}) or {})
        self.tweet_dim = int(_cfg(model_config, "tweet_dim"))
        self.num_prop_dim = int(_cfg(model_config, "num_prop_dim"))
        self.cat_prop_dim = int(_cfg(model_config, "cat_prop_dim"))
        if self.tweet_dim <= 0 or self.num_prop_dim <= 0 or self.cat_prop_dim <= 0:
            raise ValueError("RGCNHyperScanNodeInputProxy requires positive tweet/num/cat dims.")

        tweet_hidden = int(self.hidden_dim // 2)
        num_hidden = int(self.hidden_dim // 4)
        cat_hidden = int(self.hidden_dim - tweet_hidden - num_hidden)
        self.linear_relu_tweet = nn.Sequential(nn.Linear(self.tweet_dim, tweet_hidden), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(self.num_prop_dim, num_hidden), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(self.cat_prop_dim, cat_hidden), nn.LeakyReLU())
        self.linear_input = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hyper_conv1 = HypergraphConv(self.hidden_dim * 2, self.hidden_dim, use_attention=False)
        self.hyper_conv2 = HypergraphConv(self.hidden_dim, self.hidden_dim, use_attention=False)
        _init_hyperscan_detector(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": str(branch_cfg.get("feature_source", "relation_node_repr_plus_node_input_dynamic_forward")),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "pyg",
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "hyperscan_meta_tweet_proxy",
        }

    def _encode_node_input(self, x):
        tweet = x[:, : self.tweet_dim]
        num_prop = x[:, self.tweet_dim : self.tweet_dim + self.num_prop_dim]
        cat_prop = x[:, self.tweet_dim + self.num_prop_dim : self.tweet_dim + self.num_prop_dim + self.cat_prop_dim]
        x_in = torch.cat(
            [
                self.linear_relu_tweet(tweet),
                self.linear_relu_num_prop(num_prop),
                self.linear_relu_cat_prop(cat_prop),
            ],
            dim=1,
        )
        x_rel = self.activation(self.linear_input(x_in))
        return x_in, x_rel

    def _encode_relation_view(self, x_rel, edge_index, edge_type):
        hidden = self.dropout(x_rel)
        for conv in self.convs:
            hidden = conv(hidden, edge_index, edge_type)
            hidden = self.activation(hidden)
        hidden = self.linear_pool(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def _build_dynamic_hypergraph(self, feature_tensor):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
        )

    def forward_outputs(self, x, edge_index, edge_type):
        x_in, x_rel = self._encode_node_input(x)
        x_low = self._encode_relation_view(x_rel, edge_index, edge_type)
        x_new = torch.cat([x_low, x_in], dim=1)
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new)
        if hyperedge_index is None:
            x_high = torch.zeros_like(x_low)
            incident_scale = torch.zeros((x_low.size(0), 1), dtype=x_low.dtype, device=x_low.device)
        else:
            x_high = self.hyper_conv1(x_new, hyperedge_index)
            x_high = self.activation(x_high)
            x_high = self.dropout(x_high)
            x_high = self.hyper_conv2(x_high, hyperedge_index)
            x_high = self.activation(x_high)
            x_high = self.dropout(x_high)
            incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)

        hidden, logits = _apply_hyperscan_detector(self, x_low, x_high, incident_scale)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
            },
        }


class RGCNHyperScanNodeInputDHG(BaseGraphBackbone):
    """HyperScan-style node-input proxy with DHG HGNN over configured second-view incidence."""

    def __init__(self, model_config):
        super().__init__(model_config)
        if dhg is None or DHG_HGNNConv is None:
            raise ImportError("RGCNHyperScanNodeInputDHG requires dhg to be installed.")
        branch_cfg = dict(model_config.get("dynamic_similarity_branch", {}) or {})
        self.tweet_dim = int(_cfg(model_config, "tweet_dim"))
        self.num_prop_dim = int(_cfg(model_config, "num_prop_dim"))
        self.cat_prop_dim = int(_cfg(model_config, "cat_prop_dim"))
        if self.tweet_dim <= 0 or self.num_prop_dim <= 0 or self.cat_prop_dim <= 0:
            raise ValueError("RGCNHyperScanNodeInputDHG requires positive tweet/num/cat dims.")

        tweet_hidden = int(self.hidden_dim // 2)
        num_hidden = int(self.hidden_dim // 4)
        cat_hidden = int(self.hidden_dim - tweet_hidden - num_hidden)
        self.linear_relu_tweet = nn.Sequential(nn.Linear(self.tweet_dim, tweet_hidden), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(self.num_prop_dim, num_hidden), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(self.cat_prop_dim, cat_hidden), nn.LeakyReLU())
        self.linear_input = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hgnn_layer1 = DHG_HGNNConv(self.hidden_dim * 2, self.hidden_dim, use_bn=False, drop_rate=float(_cfg(model_config, "dropout", default=0.4)))
        self.hgnn_layer2 = DHG_HGNNConv(self.hidden_dim, self.hidden_dim, use_bn=False, is_last=True)
        _init_hyperscan_detector(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": str(branch_cfg.get("feature_source", "relation_node_repr_plus_node_input_dynamic_forward")),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "dhg",
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "hyperscan_meta_tweet_proxy",
        }

    def _encode_node_input(self, x):
        tweet = x[:, : self.tweet_dim]
        num_prop = x[:, self.tweet_dim : self.tweet_dim + self.num_prop_dim]
        cat_prop = x[:, self.tweet_dim + self.num_prop_dim : self.tweet_dim + self.num_prop_dim + self.cat_prop_dim]
        x_in = torch.cat(
            [
                self.linear_relu_tweet(tweet),
                self.linear_relu_num_prop(num_prop),
                self.linear_relu_cat_prop(cat_prop),
            ],
            dim=1,
        )
        x_rel = self.activation(self.linear_input(x_in))
        return x_in, x_rel

    def _encode_relation_view(self, x_rel, edge_index, edge_type):
        hidden = self.dropout(x_rel)
        for conv in self.convs:
            hidden = conv(hidden, edge_index, edge_type)
            hidden = self.activation(hidden)
        hidden = self.linear_pool(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def _build_dynamic_hypergraph(self, feature_tensor):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
        )

    def forward_outputs(self, x, edge_index, edge_type):
        x_in, x_rel = self._encode_node_input(x)
        x_low = self._encode_relation_view(x_rel, edge_index, edge_type)
        x_new = torch.cat([x_low, x_in], dim=1)
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new)
        hg = _dhg_hypergraph_from_incidence(hyperedge_index, x_new.size(0), x_new.device)
        if hg is None:
            x_high = torch.zeros_like(x_low)
        else:
            x_mid = self.hgnn_layer1(x_new, hg)
            x_high = self.hgnn_layer2(x_mid, hg)
        if incident_mask is None:
            incident_mask = torch.zeros(x_low.size(0), dtype=torch.bool, device=x_low.device)
        incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)
        hidden, logits = _apply_hyperscan_detector(self, x_low, x_high, incident_scale)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
            },
        }


class RGCNHyperScanDHGProxy(BaseGraphBackbone):
    """Semantic-only HyperScan-style backbone with DHG HGNN over configured second-view incidence."""

    def __init__(self, model_config):
        super().__init__(model_config)
        if dhg is None or DHG_HGNNConv is None:
            raise ImportError("RGCNHyperScanDHGProxy requires dhg to be installed.")
        input_dim = _cfg(model_config, "lm_input_dim")
        branch_cfg = dict(model_config.get("dynamic_similarity_branch", {}) or {})
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(self.hidden_dim, self.hidden_dim, self.n_relations) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hgnn_layer1 = DHG_HGNNConv(self.hidden_dim + input_dim, self.hidden_dim, use_bn=False, drop_rate=float(_cfg(model_config, "dropout", default=0.4)))
        self.hgnn_layer2 = DHG_HGNNConv(self.hidden_dim, self.hidden_dim, use_bn=False, is_last=True)
        _init_hyperscan_detector(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": str(branch_cfg.get("feature_source", "relation_node_repr_plus_g0_input_dynamic_forward")),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "dhg",
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "semantic_embedding",
        }

    def _encode_relation_view(self, x, edge_index, edge_type):
        hidden = self.linear_in(x)
        hidden = self.dropout(hidden)
        for conv in self.convs:
            hidden = conv(hidden, edge_index, edge_type)
            hidden = self.activation(hidden)
        hidden = self.linear_pool(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden

    def _build_dynamic_hypergraph(self, feature_tensor):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
        )

    def forward_outputs(self, x, edge_index, edge_type):
        x_low = self._encode_relation_view(x, edge_index, edge_type)
        x_new = torch.cat([x_low, x], dim=1)
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new)
        hg = _dhg_hypergraph_from_incidence(hyperedge_index, x_new.size(0), x_new.device)
        if hg is None:
            x_high = torch.zeros_like(x_low)
        else:
            x_mid = self.hgnn_layer1(x_new, hg)
            x_high = self.hgnn_layer2(x_mid, hg)
        if incident_mask is None:
            incident_mask = torch.zeros(x_low.size(0), dtype=torch.bool, device=x_low.device)
        incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)
        hidden, logits = _apply_hyperscan_detector(self, x_low, x_high, incident_scale)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
            },
        }


class SimpleHGN(BaseGraphBackbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim", default=768)
        edge_res = float(_cfg(model_config, "SimpleHGN_att_res", default=0.2))
        self.convs = nn.ModuleList([])
        for layer_idx in range(self.n_layers):
            if layer_idx == 0:
                self.convs.append(SimpleHGNConv(input_dim, self.hidden_dim, self.n_relations, 32, beta=edge_res))
            elif layer_idx == self.n_layers - 1:
                self.convs.append(
                    SimpleHGNConv(
                        self.hidden_dim,
                        self.hidden_dim,
                        self.n_relations,
                        32,
                        beta=edge_res,
                        final_layer=True,
                    )
                )
            else:
                self.convs.append(SimpleHGNConv(self.hidden_dim, self.hidden_dim, self.n_relations, 32, beta=edge_res))
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(self, x, edge_index, edge_type):
        att_res = None
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx == 0:
                x, att_res = conv(x, edge_index, edge_type)
            else:
                x, att_res = conv(x, edge_index, edge_type, pre_alpha=att_res)
            x = self.dropout(x)
        x = self.linear_pool(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class HGT(BaseGraphBackbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        heads = int(_cfg(model_config, "att_heads", default=8))
        self.metadata = (["user"], [("user", "follower", "user"), ("user", "following", "user")])
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [HGTConv(self.hidden_dim, self.hidden_dim, self.metadata, heads) for _ in range(self.n_layers)]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def prepare_data_for_HGT(self, x, edge_index, edge_type):
        x_dict = {self.metadata[0][0]: x}
        edge_index_dict = {}
        for relation_idx, relation in enumerate(self.metadata[1]):
            edge_index_dict[relation] = edge_index[:, edge_type == relation_idx]
        return x_dict, edge_index_dict

    def encode(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        x_dict, edge_index_dict = self.prepare_data_for_HGT(x, edge_index, edge_type)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict[self.metadata[0][0]] = self.activation(self.dropout(x_dict[self.metadata[0][0]]))
        x = self.linear_pool(x_dict[self.metadata[0][0]])
        x = self.dropout(x)
        x = self.activation(x)
        return x


class RGT(BaseGraphBackbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        att_heads = int(_cfg(model_config, "att_heads", default=8))
        semantic_heads = int(_cfg(model_config, "RGT_semantic_heads", default=8))
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList(
            [
                RGTLayer(self.n_relations, self.hidden_dim, self.hidden_dim, att_heads, semantic_heads, dropout=self.dropout.p)
                for _ in range(self.n_layers)
            ]
        )
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def prepare_data_for_RGT(self, x, edge_index, edge_type):
        edge_index_list = []
        for relation_idx in range(self.n_relations):
            edge_index_list.append(edge_index[:, edge_type == relation_idx])
        return x, edge_index_list

    def encode(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x, edge_index_list = self.prepare_data_for_RGT(x, edge_index, edge_type)
        for conv in self.convs:
            x = conv(x, edge_index_list)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GATv2Bot(BaseGraphBackbone):
    def __init__(self, model_config):
        super().__init__(model_config)
        input_dim = _cfg(model_config, "lm_input_dim")
        self.heads = int(_cfg(model_config, "att_heads", default=8))
        self.linear_in = nn.Linear(input_dim, self.hidden_dim)
        self.convs = nn.ModuleList([])
        for _ in range(self.n_layers):
            layer_convs = nn.ModuleList(
                [
                    GATv2Conv(
                        self.hidden_dim,
                        self.hidden_dim // self.heads,
                        heads=self.heads,
                        dropout=self.dropout.p,
                        add_self_loops=False,
                    )
                    for _ in range(self.n_relations)
                ]
            )
            self.convs.append(layer_convs)
        self.linear_pool = nn.Linear(self.hidden_dim, self.hidden_dim)

    def encode(self, x, edge_index, edge_type):
        x = self.linear_in(x)
        x = self.dropout(x)
        for layer_convs in self.convs:
            rel_outputs = []
            for relation_idx in range(self.n_relations):
                ei_r = edge_index[:, edge_type == relation_idx]
                rel_outputs.append(layer_convs[relation_idx](x, ei_r))
            x = torch.stack(rel_outputs, dim=0).mean(dim=0)
            x = self.activation(x)
        x = self.linear_pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
