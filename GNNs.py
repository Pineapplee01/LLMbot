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
    export_relation_x_new = False

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

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        hidden = self.encode(x, edge_index, edge_type)
        logits = self.linear_out(hidden)
        outputs = {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "aux_features": {},
        }
        if self.export_relation_x_new:
            x_new = torch.cat([hidden, x], dim=1)
            outputs.update(
                {
                    "x_low": hidden,
                    "x_new": x_new,
                    "x_new_base": x_new,
                }
            )
        return outputs

    def forward(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        return self.forward_outputs(
            x,
            edge_index,
            edge_type,
            construct_x=construct_x,
            hyperedge_mask_probability=hyperedge_mask_probability,
            second_view_repair_delta=second_view_repair_delta,
            second_view_repair_mask=second_view_repair_mask,
            routed_multiview_bundle=routed_multiview_bundle,
            routed_highpass_bundle=routed_highpass_bundle,
            batch_node_ids=batch_node_ids,
        )["logits"]


def _as_long_tensor_list(values):
    return [torch.tensor(item, dtype=torch.long) for item in values]


def _dynamic_branch_policy_config(branch_cfg):
    raw_quotas = branch_cfg.get("candidate_policy_quotas", {}) or {}
    quotas = {}
    if isinstance(raw_quotas, dict):
        for key, value in raw_quotas.items():
            try:
                quotas[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return {
        "candidate_policy": str(branch_cfg.get("candidate_policy", "default") or "default"),
        "candidate_policy_quotas": quotas,
    }


def _as_long_tensor(values):
    if values is None:
        return None
    return values.detach().cpu().long().view(-1) if torch.is_tensor(values) else torch.tensor(values, dtype=torch.long).view(-1)


def _dynamic_branch_policy_stats(branch_cfg):
    config = _dynamic_branch_policy_config(branch_cfg)
    return {
        "candidate_policy": config["candidate_policy"],
        "candidate_policy_quotas": dict(config["candidate_policy_quotas"]),
        "candidate_role_contract": str(branch_cfg.get("candidate_role_contract", "none") or "none"),
    }


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


class ConstructAdaptiveChannelMixing(nn.Module):
    """Node-wise adaptive identity/low/high mixing for construct-complete graph branches."""

    def __init__(self, model_dim, dropout_rate=0.4):
        super().__init__()
        self.model_dim = int(model_dim)
        context_dim = int(model_dim) * 6
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, int(model_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(model_dim), 3),
        )
        self.mix_norm = nn.LayerNorm(int(model_dim))

    def forward(self, x_identity, x_low, x_high):
        if x_identity.shape != x_low.shape or x_identity.shape != x_high.shape:
            raise ValueError(
                "construct_acm requires matching identity/low/high shapes: "
                f"got {tuple(x_identity.shape)}, {tuple(x_low.shape)}, and {tuple(x_high.shape)}."
            )
        context = torch.cat(
            [
                x_identity,
                x_low,
                x_high,
                x_identity - x_low,
                x_identity - x_high,
                x_low - x_high,
            ],
            dim=-1,
        )
        gate = torch.softmax(self.gate_mlp(context), dim=-1)
        mixed = gate[:, 0:1] * x_identity + gate[:, 1:2] * x_low + gate[:, 2:3] * x_high
        mixed = self.mix_norm(mixed)
        return mixed, gate


class HyperScanAdaptiveLowHighFusion(nn.Module):
    """FAGCN-style node-wise adaptive mixing over relation/high-order detector views.

    This module does not reimplement spectral filtering. It keeps HyperScan's
    relation-view and hypergraph-view branches intact, then learns a node-wise
    low/high mixture over the cross-attended detector tokens.
    """

    def __init__(self, model_dim, dropout_rate=0.4):
        super().__init__()
        self.model_dim = int(model_dim)
        context_dim = int(model_dim) * 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, int(model_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(model_dim), 2),
        )
        self.mix_norm = nn.LayerNorm(int(model_dim))

    def forward(self, low_hidden, high_hidden):
        if low_hidden.shape != high_hidden.shape:
            raise ValueError(
                "adaptive low/high fusion requires matching detector tokens: "
                f"got {tuple(low_hidden.shape)} and {tuple(high_hidden.shape)}."
            )
        context = torch.cat(
            [
                low_hidden,
                high_hidden,
                low_hidden - high_hidden,
                low_hidden * high_hidden,
            ],
            dim=-1,
        )
        gate = torch.softmax(self.gate_mlp(context), dim=-1)
        mixed = gate[:, 0:1] * low_hidden + gate[:, 1:2] * high_hidden
        mixed = self.mix_norm(mixed)
        return mixed, gate


class RoutedHighPassCorrection(nn.Module):
    """Routed-node low/high-pass correction over a chosen representation space."""

    def __init__(self, model_dim, dropout_rate=0.4):
        super().__init__()
        self.model_dim = int(model_dim)
        context_dim = int(model_dim) * 3 + 6
        self.self_proj = nn.Linear(int(model_dim), int(model_dim))
        self.low_proj = nn.Linear(int(model_dim), int(model_dim))
        self.high_proj = nn.Linear(int(model_dim), int(model_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, int(model_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(model_dim), 3),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(model_dim), 2),
        )
        self.edge_role_head = nn.Sequential(
            nn.Linear(int(model_dim) * 3, int(model_dim)),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(int(model_dim), 1),
        )

    def forward(self, hidden, low_agg, high_agg, scalar_features, routed_mask, mode="adaptive"):
        mode = str(mode or "adaptive").strip().lower()
        if mode not in {"low_only", "high_only", "adaptive"}:
            raise ValueError("--routed_highpass_mode must be one of {off, low_only, high_only, adaptive}.")
        if scalar_features is None:
            scalar_features = hidden.new_zeros((int(hidden.size(0)), 6))
        scalar_features = scalar_features.to(device=hidden.device, dtype=hidden.dtype)
        if scalar_features.dim() != 2 or int(scalar_features.size(0)) != int(hidden.size(0)):
            raise ValueError("routed_highpass scalar_features must be shaped [num_nodes, feature_dim].")
        if int(scalar_features.size(1)) < 6:
            pad = hidden.new_zeros((int(hidden.size(0)), 6 - int(scalar_features.size(1))))
            scalar_features = torch.cat([scalar_features, pad], dim=1)
        elif int(scalar_features.size(1)) > 6:
            scalar_features = scalar_features[:, :6]
        routed_mask = routed_mask.to(device=hidden.device).bool().view(-1)
        if int(routed_mask.numel()) != int(hidden.size(0)):
            raise ValueError("routed_highpass routed_mask must align with hidden rows.")

        context = torch.cat([hidden, low_agg, high_agg, scalar_features], dim=1)
        learned_gate = torch.softmax(self.gate_mlp(context), dim=1)
        if mode == "low_only":
            gate = torch.zeros_like(learned_gate)
            gate[:, 1] = 1.0
        elif mode == "high_only":
            gate = torch.zeros_like(learned_gate)
            gate[:, 2] = 1.0
        else:
            gate = learned_gate

        self_channel = self.self_proj(hidden)
        low_channel = self.low_proj(low_agg)
        high_channel = self.high_proj(high_agg)
        corrected_hidden = (
            gate[:, 0:1] * self_channel
            + gate[:, 1:2] * low_channel
            + gate[:, 2:3] * high_channel
        )
        delta_logits = self.correction_head(corrected_hidden)
        delta_logits = delta_logits * routed_mask.to(dtype=hidden.dtype).unsqueeze(1)
        return delta_logits, {
            "gate": gate,
            "learned_gate": learned_gate,
            "routed_mask": routed_mask,
            "delta_logits": delta_logits,
            "corrected_hidden": corrected_hidden,
        }


class RoutedMultiViewTransformer(nn.Module):
    """Routed-node multi-view refiner over detector-space tokens.

    The module consumes per-node view tokens already aligned to the detector
    hidden size, optionally performs bidirectional MultiAttn between relation
    and semantic halves, then runs a lightweight Transformer encoder over the
    view sequence and returns a routed-node delta.
    """

    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate, num_layers=2, bidirectional_multiattn=True):
        super().__init__()
        self.model_dim = int(model_dim)
        self.bidirectional_multiattn = bool(bidirectional_multiattn)
        self.dropout = nn.Dropout(dropout_rate)
        self.view_type_embeddings = nn.Embedding(8, self.model_dim)
        self.sequence_pos_embeddings = nn.Embedding(8, self.model_dim)
        if self.bidirectional_multiattn:
            self.cross_view_multiattn = _MultiAttnModel(
                num_layers=max(int(num_layers), 1),
                model_dim=self.model_dim,
                num_heads=int(num_heads),
                hidden_dim=int(hidden_dim),
                dropout_rate=float(dropout_rate),
            )
        else:
            self.cross_view_multiattn = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout_rate),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, self.model_dim)
        self.output_gate = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.model_dim, 1),
        )

    def forward(self, view_tokens, token_mask=None, relation_token_count=4, semantic_token_count=2):
        if view_tokens.dim() != 3:
            raise ValueError(f"Expected routed multi-view tokens shaped [batch, tokens, dim], got {tuple(view_tokens.shape)}.")
        batch_size, token_count, model_dim = view_tokens.shape
        if int(model_dim) != self.model_dim:
            raise ValueError(
                f"RoutedMultiViewTransformer expected token dim {self.model_dim}, got {int(model_dim)}."
            )
        device = view_tokens.device
        relation_token_count = int(max(min(int(relation_token_count), token_count), 0))
        semantic_token_count = int(max(min(int(semantic_token_count), token_count - relation_token_count), 0))
        type_ids = torch.arange(token_count, device=device, dtype=torch.long).clamp_max(7).unsqueeze(0).expand(batch_size, -1)
        pos_ids = torch.arange(token_count, device=device, dtype=torch.long).clamp_max(7).unsqueeze(0).expand(batch_size, -1)
        tokens = view_tokens + self.view_type_embeddings(type_ids) + self.sequence_pos_embeddings(pos_ids)
        tokens = self.dropout(tokens)
        if self.cross_view_multiattn is not None and relation_token_count > 0 and semantic_token_count > 0:
            relation_tokens = tokens[:, :relation_token_count, :]
            semantic_tokens = tokens[:, relation_token_count : relation_token_count + semantic_token_count, :]
            fused_relation, fused_semantic = self.cross_view_multiattn(relation_tokens, semantic_tokens)
            tokens = torch.cat(
                [
                    fused_relation,
                    fused_semantic,
                    tokens[:, relation_token_count + semantic_token_count :, :],
                ],
                dim=1,
            )
        padding_mask = None
        if token_mask is not None:
            token_mask = token_mask.to(device=device).bool()
            if token_mask.shape != (batch_size, token_count):
                raise ValueError(
                    "token_mask must align with view_tokens rows: "
                    f"expected {(batch_size, token_count)}, got {tuple(token_mask.shape)}."
                )
            padding_mask = ~token_mask
            # Ensure self token stays visible if upstream mask is malformed.
            padding_mask[:, 0] = False
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        pooled = encoded[:, 0, :]
        pooled = self.output_norm(pooled)
        delta = self.output_projection(pooled)
        gate = torch.sigmoid(self.output_gate(torch.cat([pooled, view_tokens[:, 0, :]], dim=-1)))
        return gate * delta, {
            "token_count": int(token_count),
            "bidirectional_multiattn": bool(self.bidirectional_multiattn),
        }


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
    elif style == "multiattn_adaptive":
        style = "original_cross_attention_adaptive"
    if style not in {"residual", "original_cross_attention", "original_cross_attention_adaptive", "construct_acm"}:
        raise ValueError(
            "--graph_second_view_fusion must be one of "
            "{residual, multiattn, multiattn_adaptive, construct_acm}."
        )
    module.hyperscan_detector_style = style
    if style == "original_cross_attention":
        module.graph_second_view_fusion = "multiattn"
    elif style == "original_cross_attention_adaptive":
        module.graph_second_view_fusion = "multiattn_adaptive"
    elif style == "construct_acm":
        module.graph_second_view_fusion = "construct_acm"
    else:
        module.graph_second_view_fusion = "residual"
    module.hyperscan_branch_hidden_dim = int(hidden_dim)
    module.graph_second_view_consumer_scope = str(
        _cfg(model_config, "graph_second_view_consumer_scope", default="all_nodes") or "all_nodes"
    ).strip().lower()
    module.graph_second_view_nonconsumer_fallback = str(
        _cfg(model_config, "graph_second_view_nonconsumer_fallback", default="low_only") or "low_only"
    ).strip().lower()
    module.graph_second_view_risk_gate_mode = str(
        _cfg(model_config, "graph_second_view_risk_gate_mode", default="linear_sigmoid") or "linear_sigmoid"
    ).strip().lower()
    module.second_view_selective_consumer_enabled = module.graph_second_view_consumer_scope in {
        "routed_only",
        "risk_gated_all_nodes",
    }
    module.second_view_risk_scores = None
    if style in {"original_cross_attention", "original_cross_attention_adaptive"}:
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
        module.detector_adaptive_lowhigh = (
            HyperScanAdaptiveLowHighFusion(
                model_dim=int(hidden_dim),
                dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
            )
            if style == "original_cross_attention_adaptive"
            else None
        )
        module.detector_activation = nn.LeakyReLU()
        module.linear_out = nn.Linear(int(hidden_dim) * 2, 2)
        module.hyperscan_detector_hidden_dim = int(hidden_dim) * 2
    elif style == "construct_acm":
        module.detector_adaptive_lowhigh = None
        module.detector_construct_acm = ConstructAdaptiveChannelMixing(
            model_dim=int(hidden_dim),
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        module.detector_identity_projector = nn.Identity()
        module.linear_out = nn.Linear(int(hidden_dim), 2)
        module.hyperscan_detector_hidden_dim = int(hidden_dim)
    else:
        module.detector_adaptive_lowhigh = None
        module.fusion_linear = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        module.hyperscan_detector_hidden_dim = int(hidden_dim)
    if style == "residual":
        module.second_view_risk_gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        module.second_view_risk_gate_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    else:
        module.second_view_risk_gate_scale = None
        module.second_view_risk_gate_bias = None


def _init_routed_highpass_correction(module, model_config, hidden_dim):
    cfg = dict(model_config.get("routed_highpass", {}) or {})
    mode = str(cfg.get("mode", "off") or "off").strip().lower()
    target = str(cfg.get("target", "logits") or "logits").strip().lower()
    module.routed_highpass_mode = mode
    module.routed_highpass_target = target
    module.routed_highpass_enabled = mode != "off"
    if mode not in {"off", "low_only", "high_only", "adaptive"}:
        raise ValueError("--routed_highpass_mode must be one of {off, low_only, high_only, adaptive}.")
    if target not in {"logits", "x_high"}:
        raise ValueError("--routed_highpass_target must be one of {logits, x_high}.")
    if not module.routed_highpass_enabled:
        module.routed_highpass_correction = None
        return
    detector_hidden_dim = int(getattr(module, "hyperscan_detector_hidden_dim", hidden_dim) or hidden_dim)
    correction_dim = int(hidden_dim) if target == "x_high" else int(detector_hidden_dim)
    module.routed_highpass_correction = RoutedHighPassCorrection(
        model_dim=correction_dim,
        dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
    )


def _apply_hyperscan_detector(module, x_low, x_high, incident_scale, x_identity=None):
    if module.hyperscan_detector_style in {"original_cross_attention", "original_cross_attention_adaptive"}:
        fused_low, fused_high = module.detector_multiattn(x_low.unsqueeze(1), x_high.unsqueeze(1))
        fused_low = fused_low.squeeze(1)
        fused_high = fused_high.squeeze(1)
        if module.hyperscan_detector_style == "original_cross_attention_adaptive":
            mixed, _ = module.detector_adaptive_lowhigh(fused_low, fused_high)
            hidden = torch.cat((mixed, fused_high), dim=-1)
            hidden = module.detector_activation(hidden)
            logits = module.linear_out(hidden)
            return hidden, logits
        hidden = torch.cat((fused_low, fused_high), dim=-1)
        hidden = module.detector_activation(hidden)
        return hidden, module.linear_out(hidden)
    if module.hyperscan_detector_style == "construct_acm":
        if x_identity is None or not torch.is_tensor(x_identity):
            raise ValueError("construct_acm detector requires construct identity features.")
        if getattr(module, "detector_identity_projector", None) is not None:
            x_identity = module.detector_identity_projector(x_identity)
        hidden, _ = module.detector_construct_acm(x_identity, x_low, x_high)
        return hidden, module.linear_out(hidden)

    delta = module.fusion_linear(torch.cat([x_low, x_high], dim=1))
    delta = module.activation(delta)
    delta = module.dropout(delta)
    hidden = x_low + delta * incident_scale
    return hidden, module.linear_out(hidden)


def _resolve_second_view_risk_gate(module, x_low, batch_node_ids=None):
    consumer_scope = str(getattr(module, "graph_second_view_consumer_scope", "all_nodes") or "all_nodes")
    num_nodes = int(x_low.size(0))
    default_gate = torch.ones((num_nodes,), dtype=x_low.dtype, device=x_low.device)
    default_risk = torch.zeros((num_nodes,), dtype=x_low.dtype, device=x_low.device)
    if consumer_scope != "risk_gated_all_nodes":
        return default_gate, default_risk

    gate_mode = str(getattr(module, "graph_second_view_risk_gate_mode", "linear_sigmoid") or "linear_sigmoid").strip().lower()
    if gate_mode != "linear_sigmoid":
        raise ValueError("risk_gated_all_nodes currently supports only graph_second_view_risk_gate_mode=linear_sigmoid.")

    risk_scores = getattr(module, "second_view_risk_scores", None)
    if risk_scores is None:
        raise ValueError(
            "graph_second_view_consumer_scope risk_gated_all_nodes requires an attached full-graph risk vector."
        )
    if not torch.is_tensor(risk_scores):
        risk_scores = torch.as_tensor(risk_scores, dtype=x_low.dtype, device=x_low.device)
    else:
        risk_scores = risk_scores.to(device=x_low.device, dtype=x_low.dtype).view(-1)

    if batch_node_ids is None:
        if int(risk_scores.numel()) != num_nodes:
            raise ValueError(
                "risk_gated_all_nodes without batch_node_ids requires risk vector rows to match the current node rows."
            )
        risk_batch = risk_scores
    else:
        if not torch.is_tensor(batch_node_ids):
            batch_node_ids = torch.as_tensor(batch_node_ids, dtype=torch.long, device=x_low.device)
        else:
            batch_node_ids = batch_node_ids.to(device=x_low.device, dtype=torch.long).view(-1)
        if int(batch_node_ids.numel()) != num_nodes:
            raise ValueError(
                "risk_gated_all_nodes expects batch_node_ids to align with the current node rows."
            )
        max_id = int(batch_node_ids.max().item()) if int(batch_node_ids.numel()) > 0 else -1
        if max_id >= int(risk_scores.numel()):
            raise ValueError(
                "risk_gated_all_nodes batch node ids exceed the attached full-graph risk vector length."
            )
        risk_batch = risk_scores[batch_node_ids]

    gate = torch.sigmoid(
        module.second_view_risk_gate_scale.to(dtype=x_low.dtype)
        * risk_batch
        + module.second_view_risk_gate_bias.to(dtype=x_low.dtype)
    )
    return gate, risk_batch


def _low_only_detector_view(module, x_low):
    style = str(getattr(module, "hyperscan_detector_style", "residual") or "residual")
    if style in {"original_cross_attention", "original_cross_attention_adaptive"}:
        zero_high = torch.zeros_like(x_low)
        hidden = torch.cat((x_low, zero_high), dim=-1)
        hidden = module.detector_activation(hidden)
        logits = module.linear_out(hidden)
        return hidden, logits
    if style == "construct_acm":
        hidden = x_low
        return hidden, module.linear_out(hidden)
    zero_high = torch.zeros_like(x_low)
    delta = module.fusion_linear(torch.cat([x_low, zero_high], dim=1))
    delta = module.activation(delta)
    delta = module.dropout(delta)
    hidden = x_low + (delta * 0.0)
    return hidden, module.linear_out(hidden)


def _apply_selective_second_view_consumption(module, x_low, x_high, incident_scale, batch_node_ids=None, x_identity=None):
    consumer_scope = str(getattr(module, "graph_second_view_consumer_scope", "all_nodes") or "all_nodes")
    if consumer_scope == "risk_gated_all_nodes":
        if str(getattr(module, "hyperscan_detector_style", "residual") or "residual").strip().lower() != "residual":
            raise ValueError("risk_gated_all_nodes currently supports only residual second-view detector fusion.")
        gate, risk_batch = _resolve_second_view_risk_gate(module, x_low, batch_node_ids=batch_node_ids)
        hidden_hnn, logits_hnn = _apply_hyperscan_detector(
            module,
            x_low,
            x_high,
            incident_scale * gate.unsqueeze(-1),
            x_identity=x_identity,
        )
        return {
            "hidden": hidden_hnn,
            "logits": logits_hnn,
            "hidden_hnn": hidden_hnn,
            "logits_hnn": logits_hnn,
            "hidden_lowonly": hidden_hnn,
            "logits_lowonly": logits_hnn,
            "consumer_mask": torch.ones((x_low.size(0),), dtype=torch.bool, device=x_low.device),
            "consumer_gate": gate,
            "consumer_risk": risk_batch,
        }

    hidden_hnn, logits_hnn = _apply_hyperscan_detector(module, x_low, x_high, incident_scale, x_identity=x_identity)
    if consumer_scope != "routed_only":
        consumer_mask = torch.ones((x_low.size(0),), dtype=torch.bool, device=x_low.device)
        return {
            "hidden": hidden_hnn,
            "logits": logits_hnn,
            "hidden_hnn": hidden_hnn,
            "logits_hnn": logits_hnn,
            "hidden_lowonly": hidden_hnn,
            "logits_lowonly": logits_hnn,
            "consumer_mask": consumer_mask,
            "consumer_gate": torch.ones((x_low.size(0),), dtype=x_low.dtype, device=x_low.device),
            "consumer_risk": torch.zeros((x_low.size(0),), dtype=x_low.dtype, device=x_low.device),
        }

    fallback = str(getattr(module, "graph_second_view_nonconsumer_fallback", "low_only") or "low_only")
    if fallback != "low_only":
        raise ValueError("Selective second-view consumption currently supports only low_only fallback.")
    hidden_lowonly, logits_lowonly = _low_only_detector_view(module, x_low)
    if batch_node_ids is None:
        consumer_mask = torch.zeros((x_low.size(0),), dtype=torch.bool, device=x_low.device)
    else:
        if not torch.is_tensor(batch_node_ids):
            batch_node_ids = torch.as_tensor(batch_node_ids, dtype=torch.long, device=x_low.device)
        else:
            batch_node_ids = batch_node_ids.to(device=x_low.device, dtype=torch.long).view(-1)
        routed_ids = getattr(module, "_dynamic_batch_local_routed_node_ids", None)
        if routed_ids is None or int(routed_ids.numel()) == 0:
            consumer_mask = torch.zeros((x_low.size(0),), dtype=torch.bool, device=x_low.device)
        else:
            routed_ids = routed_ids.to(device=x_low.device, dtype=torch.long).view(-1)
            consumer_mask = (batch_node_ids.unsqueeze(1) == routed_ids.unsqueeze(0)).any(dim=1)
    hidden = torch.where(consumer_mask.unsqueeze(-1), hidden_hnn, hidden_lowonly)
    logits = torch.where(consumer_mask.unsqueeze(-1), logits_hnn, logits_lowonly)
    return {
        "hidden": hidden,
        "logits": logits,
        "hidden_hnn": hidden_hnn,
        "logits_hnn": logits_hnn,
        "hidden_lowonly": hidden_lowonly,
        "logits_lowonly": logits_lowonly,
        "consumer_mask": consumer_mask,
        "consumer_gate": consumer_mask.to(dtype=x_low.dtype),
        "consumer_risk": torch.zeros((x_low.size(0),), dtype=x_low.dtype, device=x_low.device),
    }


def _apply_routed_highpass_correction(module, hidden, logits, routed_highpass_bundle):
    stats = {
        "enabled": bool(getattr(module, "routed_highpass_enabled", False)),
        "applied": False,
        "mode": str(getattr(module, "routed_highpass_mode", "off")),
        "target": str(getattr(module, "routed_highpass_target", "logits")),
        "routed_node_count": 0,
        "gate_self_mean": 0.0,
        "gate_low_mean": 0.0,
        "gate_high_mean": 0.0,
    }
    if not bool(getattr(module, "routed_highpass_enabled", False)):
        return hidden, logits, stats, None
    if str(getattr(module, "routed_highpass_target", "logits")) != "logits":
        return hidden, logits, stats, None
    if routed_highpass_bundle is None or module.routed_highpass_correction is None:
        return hidden, logits, stats, None
    low_agg = routed_highpass_bundle.get("low_agg")
    high_agg = routed_highpass_bundle.get("high_agg")
    routed_mask = routed_highpass_bundle.get("routed_mask")
    if not torch.is_tensor(low_agg) or not torch.is_tensor(high_agg) or not torch.is_tensor(routed_mask):
        return hidden, logits, stats, None
    low_agg = low_agg.to(device=hidden.device, dtype=hidden.dtype)
    high_agg = high_agg.to(device=hidden.device, dtype=hidden.dtype)
    scalar_features = routed_highpass_bundle.get("scalar_features")
    if scalar_features is not None and not torch.is_tensor(scalar_features):
        scalar_features = torch.as_tensor(scalar_features)
    if scalar_features is not None:
        scalar_features = scalar_features.to(device=hidden.device, dtype=hidden.dtype)
    routed_mask = routed_mask.to(device=hidden.device).bool().view(-1)
    if low_agg.shape != hidden.shape or high_agg.shape != hidden.shape:
        raise ValueError(
            "routed_highpass low_agg/high_agg must match detector hidden shape: "
            f"expected {tuple(hidden.shape)}, got {tuple(low_agg.shape)} and {tuple(high_agg.shape)}."
        )
    delta_logits, aux = module.routed_highpass_correction(
        hidden=hidden,
        low_agg=low_agg,
        high_agg=high_agg,
        scalar_features=scalar_features,
        routed_mask=routed_mask,
        mode=str(getattr(module, "routed_highpass_mode", "adaptive")),
    )
    refined_logits = logits + delta_logits
    gate = aux["gate"]
    active_count = int(routed_mask.sum().detach().cpu().item())
    if active_count > 0:
        active_gate = gate[routed_mask]
        stats.update(
            {
                "applied": True,
                "routed_node_count": active_count,
                "gate_self_mean": float(active_gate[:, 0].detach().mean().cpu().item()),
                "gate_low_mean": float(active_gate[:, 1].detach().mean().cpu().item()),
                "gate_high_mean": float(active_gate[:, 2].detach().mean().cpu().item()),
            }
        )
    return hidden, refined_logits, stats, aux


def _apply_routed_highpass_x_high_correction(module, x_high, routed_highpass_bundle):
    stats = {
        "enabled": bool(getattr(module, "routed_highpass_enabled", False)),
        "applied": False,
        "mode": str(getattr(module, "routed_highpass_mode", "off")),
        "target": str(getattr(module, "routed_highpass_target", "logits")),
        "routed_node_count": 0,
        "gate_self_mean": 0.0,
        "gate_low_mean": 0.0,
        "gate_high_mean": 0.0,
    }
    if not bool(getattr(module, "routed_highpass_enabled", False)):
        return x_high, stats, None
    if str(getattr(module, "routed_highpass_target", "logits")) != "x_high":
        return x_high, stats, None
    if routed_highpass_bundle is None or module.routed_highpass_correction is None:
        return x_high, stats, None
    low_agg = routed_highpass_bundle.get("low_agg")
    high_agg = routed_highpass_bundle.get("high_agg")
    routed_mask = routed_highpass_bundle.get("routed_mask")
    if not torch.is_tensor(low_agg) or not torch.is_tensor(high_agg) or not torch.is_tensor(routed_mask):
        return x_high, stats, None
    low_agg = low_agg.to(device=x_high.device, dtype=x_high.dtype)
    high_agg = high_agg.to(device=x_high.device, dtype=x_high.dtype)
    scalar_features = routed_highpass_bundle.get("scalar_features")
    if scalar_features is not None and not torch.is_tensor(scalar_features):
        scalar_features = torch.as_tensor(scalar_features)
    if scalar_features is not None:
        scalar_features = scalar_features.to(device=x_high.device, dtype=x_high.dtype)
    routed_mask = routed_mask.to(device=x_high.device).bool().view(-1)
    if low_agg.shape != x_high.shape or high_agg.shape != x_high.shape:
        raise ValueError(
            "routed_highpass x_high low_agg/high_agg must match x_high shape: "
            f"expected {tuple(x_high.shape)}, got {tuple(low_agg.shape)} and {tuple(high_agg.shape)}."
        )
    _, aux = module.routed_highpass_correction(
        hidden=x_high,
        low_agg=low_agg,
        high_agg=high_agg,
        scalar_features=scalar_features,
        routed_mask=routed_mask,
        mode=str(getattr(module, "routed_highpass_mode", "adaptive")),
    )
    corrected_hidden = aux.get("corrected_hidden")
    if not torch.is_tensor(corrected_hidden):
        return x_high, stats, aux
    routed_gate = routed_mask.to(dtype=x_high.dtype).unsqueeze(1)
    refined_x_high = x_high + routed_gate * (corrected_hidden - x_high)
    gate = aux["gate"]
    active_count = int(routed_mask.sum().detach().cpu().item())
    if active_count > 0:
        active_gate = gate[routed_mask]
        stats.update(
            {
                "applied": True,
                "routed_node_count": active_count,
                "gate_self_mean": float(active_gate[:, 0].detach().mean().cpu().item()),
                "gate_low_mean": float(active_gate[:, 1].detach().mean().cpu().item()),
                "gate_high_mean": float(active_gate[:, 2].detach().mean().cpu().item()),
            }
        )
    return refined_x_high, stats, aux


class NeighborOnlyRGCNConv(nn.Module):
    """Relation-aware neighbor-only aggregation without root/self mixing."""

    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_relations = int(num_relations)
        self.relation_linears = nn.ModuleList(
            [nn.Linear(self.in_channels, self.out_channels, bias=False) for _ in range(self.num_relations)]
        )

    def forward(self, x, edge_index, edge_type):
        if edge_index is None or int(edge_index.numel()) == 0:
            return x.new_zeros((int(x.size(0)), self.out_channels))
        if edge_type is None:
            edge_type = torch.zeros((int(edge_index.size(1)),), dtype=torch.long, device=edge_index.device)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        out = x.new_zeros((int(x.size(0)), self.out_channels))
        degree = x.new_zeros((int(x.size(0)), 1))
        for rel_id, linear in enumerate(self.relation_linears):
            rel_mask = edge_type == int(rel_id)
            if not bool(rel_mask.any().item()):
                continue
            rel_src = src[rel_mask]
            rel_dst = dst[rel_mask]
            rel_msg = linear(x[rel_src])
            out.index_add_(0, rel_dst, rel_msg)
            degree.index_add_(
                0,
                rel_dst,
                torch.ones((int(rel_dst.numel()), 1), dtype=out.dtype, device=out.device),
            )
        return out / degree.clamp_min(1.0)


class SelfLowHighGate(nn.Module):
    """Node-wise adaptive self/low/high mixing used by dual-space encoders."""

    def __init__(self, hidden_dim, dropout_rate=0.4):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.hidden_dim = hidden_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(hidden_dim, 3),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, self_hidden, low_hidden, high_hidden):
        context = torch.cat(
            [
                self_hidden,
                low_hidden,
                high_hidden,
                self_hidden - low_hidden,
                self_hidden - high_hidden,
            ],
            dim=-1,
        )
        gate = torch.softmax(self.gate_mlp(context), dim=-1)
        mixed = (
            gate[:, 0:1] * self_hidden
            + gate[:, 1:2] * low_hidden
            + gate[:, 2:3] * high_hidden
        )
        return self.norm(mixed), gate


class DualSpaceRelationEncoder(nn.Module):
    """Strict H2GCN/FAGCN-style relation encoder over a chosen graph-input space."""

    def __init__(self, input_dim, hidden_dim, num_relations, dropout_rate=0.4, activation=None, output_prefix="construct"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation if activation is not None else nn.LeakyReLU()
        self.dropout = nn.Dropout(float(dropout_rate))
        self.output_prefix = str(output_prefix or "construct")
        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.rel_conv1 = NeighborOnlyRGCNConv(self.hidden_dim, self.hidden_dim, num_relations=num_relations)
        self.rel_conv2 = NeighborOnlyRGCNConv(self.hidden_dim, self.hidden_dim, num_relations=num_relations)
        self.gate1 = SelfLowHighGate(self.hidden_dim, dropout_rate=dropout_rate)
        self.gate2 = SelfLowHighGate(self.hidden_dim, dropout_rate=dropout_rate)
        self.out_linear = nn.Linear(self.hidden_dim * 6, self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, edge_index, edge_type):
        z0 = self.dropout(self.activation(self.linear_in(x)))
        low1 = self.rel_conv1(z0, edge_index, edge_type)
        high1 = z0 - low1
        state1, gate1 = self.gate1(z0, low1, high1)
        state1 = self.dropout(self.activation(state1))

        low2 = self.rel_conv2(state1, edge_index, edge_type)
        high2 = state1 - low2
        state2, gate2 = self.gate2(state1, low2, high2)
        state2 = self.dropout(self.activation(state2))

        x_low = self.out_linear(torch.cat([z0, low1, high1, low2, high2, state2], dim=-1))
        x_low = self.out_norm(self.activation(x_low))
        x_low = self.dropout(x_low)
        prefix = self.output_prefix
        return {
            "x_low": x_low,
            f"x_ego_{prefix}": z0,
            f"x_rel_low1_{prefix}": low1,
            f"x_rel_high1_{prefix}": high1,
            f"x_rel_state1_{prefix}": state1,
            f"x_rel_low2_{prefix}": low2,
            f"x_rel_high2_{prefix}": high2,
            f"x_rel_state2_{prefix}": state2,
            f"gate1_{prefix}": gate1,
            f"gate2_{prefix}": gate2,
        }


class DualSpaceHighOrderEncoder(nn.Module):
    """Strict construct-space high-order encoder with self/low/high separation."""

    def __init__(
        self,
        construct_feature_dim,
        hidden_dim,
        dropout_rate=0.4,
        activation=None,
        hypergraph_backend="pyg",
        use_bn=False,
    ):
        super().__init__()
        self.construct_feature_dim = int(construct_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation if activation is not None else nn.LeakyReLU()
        self.dropout = nn.Dropout(float(dropout_rate))
        self.hypergraph_backend = str(hypergraph_backend or "pyg").strip().lower()
        if self.hypergraph_backend not in {"pyg", "dhg"}:
            raise ValueError("DualSpaceHighOrderEncoder hypergraph_backend must be one of {pyg, dhg}.")
        self.use_bn = bool(use_bn)
        self.hg_input_linear = nn.Linear(self.construct_feature_dim, self.hidden_dim)
        if self.hypergraph_backend == "dhg":
            if dhg is None or DHG_HGNNConv is None:
                raise ImportError("DualSpaceHighOrderEncoder with hypergraph_backend=dhg requires dhg to be installed.")
            self.hyper_conv1 = DHG_HGNNConv(
                self.hidden_dim,
                self.hidden_dim,
                use_bn=self.use_bn,
                drop_rate=float(dropout_rate),
            )
            self.hyper_conv2 = DHG_HGNNConv(
                self.hidden_dim,
                self.hidden_dim,
                use_bn=self.use_bn,
                is_last=True,
            )
        else:
            self.hyper_conv1 = HypergraphConv(self.hidden_dim, self.hidden_dim, use_attention=False)
            self.hyper_conv2 = HypergraphConv(self.hidden_dim, self.hidden_dim, use_attention=False)
        self.gate1 = SelfLowHighGate(self.hidden_dim, dropout_rate=dropout_rate)
        self.gate2 = SelfLowHighGate(self.hidden_dim, dropout_rate=dropout_rate)
        self.hg_input_norm = nn.LayerNorm(self.hidden_dim)

    def build_hg_base(self, x_new_construct):
        x_hg_base = self.hg_input_linear(x_new_construct)
        x_hg_base = self.hg_input_norm(self.activation(x_hg_base))
        x_hg_base = self.dropout(x_hg_base)
        return x_hg_base

    def forward(self, x_new_construct, hyperedge_index, incident_mask, prepared_construct=None):
        if prepared_construct is None:
            x_hg_base = self.build_hg_base(x_new_construct)
        else:
            x_hg_base = prepared_construct["x_hg_base"]

        if hyperedge_index is None or int(hyperedge_index.numel()) == 0:
            zero = torch.zeros_like(x_hg_base)
            incident_scale = torch.zeros((int(x_hg_base.size(0)), 1), dtype=x_hg_base.dtype, device=x_hg_base.device)
            return {
                "x_hg_base": x_hg_base,
                "x_high_low1_construct": zero,
                "x_high_high1_construct": zero,
                "x_high_state1_construct": zero,
                "x_high_low2_construct": zero,
                "x_high_high2_construct": zero,
                "x_high_construct": zero,
                "x_high": zero,
                "incident_scale": incident_scale,
                "gate1_construct": None,
                "gate2_construct": None,
            }

        dhg_hg = None
        if self.hypergraph_backend == "dhg":
            dhg_hg = _dhg_hypergraph_from_incidence(hyperedge_index, x_hg_base.size(0), x_hg_base.device)
            if dhg_hg is None:
                zero = torch.zeros_like(x_hg_base)
                incident_scale = torch.zeros((int(x_hg_base.size(0)), 1), dtype=x_hg_base.dtype, device=x_hg_base.device)
                return {
                    "x_hg_base": x_hg_base,
                    "x_high_low1_construct": zero,
                    "x_high_high1_construct": zero,
                    "x_high_state1_construct": zero,
                    "x_high_low2_construct": zero,
                    "x_high_high2_construct": zero,
                    "x_high_construct": zero,
                    "x_high": zero,
                    "incident_scale": incident_scale,
                    "gate1_construct": None,
                    "gate2_construct": None,
                }

        hg_low1 = self.hyper_conv1(x_hg_base, dhg_hg if dhg_hg is not None else hyperedge_index)
        hg_low1 = self.dropout(self.activation(hg_low1))
        hg_high1 = x_hg_base - hg_low1
        hg_state1, gate1 = self.gate1(x_hg_base, hg_low1, hg_high1)
        hg_state1 = self.dropout(self.activation(hg_state1))

        hg_low2 = self.hyper_conv2(hg_state1, dhg_hg if dhg_hg is not None else hyperedge_index)
        hg_low2 = self.dropout(self.activation(hg_low2))
        hg_high2 = hg_state1 - hg_low2
        x_high_construct, gate2 = self.gate2(hg_state1, hg_low2, hg_high2)
        x_high_construct = self.dropout(self.activation(x_high_construct))
        incident_scale = incident_mask.to(dtype=x_high_construct.dtype).unsqueeze(-1)
        return {
            "x_hg_base": x_hg_base,
            "x_high_low1_construct": hg_low1,
            "x_high_high1_construct": hg_high1,
            "x_high_state1_construct": hg_state1,
            "x_high_low2_construct": hg_low2,
            "x_high_high2_construct": hg_high2,
            "x_high_construct": x_high_construct,
            "x_high": x_high_construct,
            "incident_scale": incident_scale,
            "gate1_construct": gate1,
            "gate2_construct": gate2,
        }


class ConstructNodeInputEncoder(nn.Module):
    """Paper-style tweet/num/cat preprocessing that defines x_in_construct."""

    def __init__(self, tweet_dim, num_prop_dim, cat_prop_dim, hidden_dim, dropout_rate=0.4):
        super().__init__()
        self.tweet_dim = int(tweet_dim)
        self.num_prop_dim = int(num_prop_dim)
        self.cat_prop_dim = int(cat_prop_dim)
        self.hidden_dim = int(hidden_dim)
        if self.tweet_dim <= 0 or self.num_prop_dim <= 0 or self.cat_prop_dim <= 0:
            raise ValueError("ConstructNodeInputEncoder requires positive tweet/num/cat dims.")
        tweet_hidden = int(self.hidden_dim // 2)
        num_hidden = int(self.hidden_dim // 4)
        cat_hidden = int(self.hidden_dim - tweet_hidden - num_hidden)
        self.linear_relu_tweet = nn.Sequential(nn.Linear(self.tweet_dim, tweet_hidden), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(self.num_prop_dim, num_hidden), nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(self.cat_prop_dim, cat_hidden), nn.LeakyReLU())
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward(self, x):
        tweet = x[:, : self.tweet_dim]
        num_prop = x[:, self.tweet_dim : self.tweet_dim + self.num_prop_dim]
        cat_prop = x[:, self.tweet_dim + self.num_prop_dim : self.tweet_dim + self.num_prop_dim + self.cat_prop_dim]
        x_in_construct = torch.cat(
            [
                self.linear_relu_tweet(tweet),
                self.linear_relu_num_prop(num_prop),
                self.linear_relu_cat_prop(cat_prop),
            ],
            dim=1,
        )
        return self.dropout(x_in_construct)


def _gate_mean_stats(gate, prefix):
    if not torch.is_tensor(gate):
        return {
            f"{prefix}_self_mean": 0.0,
            f"{prefix}_low_mean": 0.0,
            f"{prefix}_high_mean": 0.0,
        }
    return {
        f"{prefix}_self_mean": float(gate[:, 0].detach().mean().cpu().item()),
        f"{prefix}_low_mean": float(gate[:, 1].detach().mean().cpu().item()),
        f"{prefix}_high_mean": float(gate[:, 2].detach().mean().cpu().item()),
    }


def _second_view_feature_source(branch_cfg):
    feature_source = str(branch_cfg.get("feature_source", "x_low_plus_x_in_dynamic_forward"))
    if "node_repr" in feature_source:
        raise ValueError(
            "HyperScan-style dynamic second-view feature_source must describe x_new=cat(x_low,x_in), "
            "not final node_repr. Use relation_overlap_knn_repr_prefit_augment for the explicit legacy proxy."
        )
    return feature_source


def _init_routed_multiview_refiner(module, model_config, hidden_dim):
    routed_cfg = dict(model_config.get("routed_multiview_refiner", {}) or {})
    module.routed_multiview_refiner_enabled = bool(routed_cfg.get("enabled", False))
    module.routed_multiview_relation_token_count = int(routed_cfg.get("relation_token_count", 4))
    module.routed_multiview_semantic_token_count = int(routed_cfg.get("semantic_token_count", 2))
    module.routed_multiview_bidirectional = bool(routed_cfg.get("bidirectional_multiattn", True))
    detector_hidden_dim = int(getattr(module, "hyperscan_detector_hidden_dim", hidden_dim) or hidden_dim)
    branch_hidden_dim = int(getattr(module, "hyperscan_branch_hidden_dim", hidden_dim) or hidden_dim)
    semantic_input_dim = int(routed_cfg.get("semantic_input_dim", detector_hidden_dim) or detector_hidden_dim)
    module.routed_multiview_model_dim = detector_hidden_dim
    module.routed_multiview_branch_hidden_dim = branch_hidden_dim
    module.routed_multiview_semantic_input_dim = semantic_input_dim
    if not module.routed_multiview_refiner_enabled:
        module.routed_multiview_refiner = None
        module.routed_multiview_view_proj = None
        module.routed_multiview_semantic_proj = None
        return
    num_heads = int(routed_cfg.get("num_heads", 4))
    if int(detector_hidden_dim) % max(num_heads, 1) != 0:
        raise ValueError("routed multi-view refiner requires hidden_dim to be divisible by num_heads.")
    module.routed_multiview_refiner = RoutedMultiViewTransformer(
        model_dim=int(detector_hidden_dim),
        num_heads=num_heads,
        hidden_dim=int(routed_cfg.get("ff_hidden_dim", int(detector_hidden_dim) * 2)),
        dropout_rate=float(routed_cfg.get("dropout", _cfg(model_config, "dropout", default=0.4))),
        num_layers=int(routed_cfg.get("num_layers", 2)),
        bidirectional_multiattn=module.routed_multiview_bidirectional,
    )
    if int(branch_hidden_dim) == int(detector_hidden_dim):
        module.routed_multiview_view_proj = nn.Identity()
    else:
        module.routed_multiview_view_proj = nn.Linear(int(branch_hidden_dim), int(detector_hidden_dim))
    if int(semantic_input_dim) == int(detector_hidden_dim):
        module.routed_multiview_semantic_proj = nn.Identity()
    else:
        module.routed_multiview_semantic_proj = nn.Linear(int(semantic_input_dim), int(detector_hidden_dim))


def _build_detector_space_view_tokens(module, hidden, x_low, x_high, routed_multiview_bundle):
    rows = routed_multiview_bundle.get("rows") or []
    if not rows:
        return None, None, None, None
    target_node_ids = routed_multiview_bundle.get("target_node_ids")
    if target_node_ids is None:
        target_node_ids = [int(item.get("node_id", -1)) for item in rows]
    semantic_guide = routed_multiview_bundle.get("semantic_embeddings")
    view_projector = getattr(module, "routed_multiview_view_proj", None)
    semantic_projector = getattr(module, "routed_multiview_semantic_proj", None)
    semantic_input_dim = int(getattr(module, "routed_multiview_semantic_input_dim", hidden.size(-1)))
    token_rows = []
    token_masks = []
    valid_node_ids = []
    max_tokens = 6
    for node_id, row in zip(target_node_ids, rows):
        node_id = int(node_id)
        if node_id < 0 or node_id >= int(hidden.size(0)) or not isinstance(row, dict):
            continue
        token_list = [hidden[node_id]]
        mask_list = [True]
        has_auxiliary_context = False
        for key, source_tensor in (
            ("following_view", x_low),
            ("follower_view", x_low),
            ("mutual_view", x_low),
            ("semantic_knn_view", x_high),
        ):
            view = row.get(key) if isinstance(row, dict) else None
            members = list((view or {}).get("members") or [])
            member_ids = [int(item.get("node_id", -1)) for item in members if 0 <= int(item.get("node_id", -1)) < int(source_tensor.size(0))]
            if member_ids:
                member_tensor = source_tensor[torch.tensor(member_ids, device=source_tensor.device, dtype=torch.long)]
                member_token = member_tensor.mean(dim=0)
                if view_projector is not None:
                    member_token = view_projector(member_token.unsqueeze(0)).squeeze(0)
                token_list.append(member_token)
                mask_list.append(True)
                has_auxiliary_context = True
            else:
                token_list.append(torch.zeros_like(hidden[node_id]))
                mask_list.append(False)
        if semantic_guide is not None and 0 <= node_id < int(semantic_guide.size(0)):
            semantic_token = semantic_guide[node_id].to(device=hidden.device, dtype=hidden.dtype)
            if int(semantic_token.numel()) != int(semantic_input_dim):
                raise ValueError(
                    "routed semantic guide dimension mismatch: "
                    f"expected {int(semantic_input_dim)}, got {int(semantic_token.numel())}."
                )
            if semantic_projector is None:
                semantic_token = semantic_token.view_as(hidden[node_id])
            else:
                semantic_token = semantic_projector(semantic_token.unsqueeze(0)).squeeze(0)
            semantic_nonzero = bool(torch.norm(semantic_token, p=2).item() > 0.0)
            token_list.append(semantic_token)
            mask_list.append(semantic_nonzero)
            has_auxiliary_context = has_auxiliary_context or semantic_nonzero
        else:
            token_list.append(torch.zeros_like(hidden[node_id]))
            mask_list.append(False)
        if not has_auxiliary_context:
            continue
        token_rows.append(torch.stack(token_list[:max_tokens], dim=0))
        token_masks.append(torch.tensor(mask_list[:max_tokens], device=hidden.device, dtype=torch.bool))
        valid_node_ids.append(node_id)
    if not token_rows:
        return None, None, None, None
    return (
        torch.stack(token_rows, dim=0),
        torch.stack(token_masks, dim=0),
        torch.tensor(valid_node_ids, device=hidden.device, dtype=torch.long),
        max_tokens,
    )


def _apply_routed_multiview_refinement(module, hidden, x_low, x_high, logits, routed_multiview_bundle):
    stats = {
        "enabled": bool(getattr(module, "routed_multiview_refiner_enabled", False)),
        "applied": False,
        "routed_node_count": 0,
        "token_count": 0,
    }
    if not bool(getattr(module, "routed_multiview_refiner_enabled", False)):
        return hidden, logits, stats
    if routed_multiview_bundle is None or module.routed_multiview_refiner is None:
        return hidden, logits, stats
    token_tensor, token_mask, routed_node_ids, token_count = _build_detector_space_view_tokens(
        module,
        hidden,
        x_low,
        x_high,
        routed_multiview_bundle,
    )
    if token_tensor is None:
        return hidden, logits, stats
    delta, refiner_stats = module.routed_multiview_refiner(
        token_tensor,
        token_mask=token_mask,
        relation_token_count=int(getattr(module, "routed_multiview_relation_token_count", 4)),
        semantic_token_count=int(getattr(module, "routed_multiview_semantic_token_count", 2)),
    )
    refined_hidden = hidden.clone()
    refined_hidden[routed_node_ids] = refined_hidden[routed_node_ids] + delta
    refined_logits = module.linear_out(refined_hidden)
    stats.update(
        {
            "applied": True,
            "routed_node_count": int(routed_node_ids.numel()),
            "token_count": int(token_count),
            "bidirectional_multiattn": bool(refiner_stats.get("bidirectional_multiattn", False)),
        }
    )
    return refined_hidden, refined_logits, stats


def _apply_second_view_repair(x_new, repair_delta=None, repair_mask=None):
    if repair_delta is None:
        return x_new, {
            "second_view_repair_active": False,
            "second_view_repair_nodes": 0,
            "second_view_repair_delta_norm_mean": 0.0,
        }
    if not torch.is_tensor(repair_delta):
        repair_delta = torch.as_tensor(repair_delta, device=x_new.device, dtype=x_new.dtype)
    repair_delta = repair_delta.to(device=x_new.device, dtype=x_new.dtype)
    if repair_delta.shape != x_new.shape:
        raise ValueError(
            "second_view_repair_delta must match x_new shape: "
            f"expected {tuple(x_new.shape)}, got {tuple(repair_delta.shape)}."
        )
    if repair_mask is None:
        repair_mask = repair_delta.norm(dim=1) > 0.0
    else:
        if not torch.is_tensor(repair_mask):
            repair_mask = torch.as_tensor(repair_mask, device=x_new.device)
        repair_mask = repair_mask.to(device=x_new.device).bool().view(-1)
        if int(repair_mask.numel()) != int(x_new.size(0)):
            raise ValueError(
                "second_view_repair_mask must align with x_new rows: "
                f"expected {int(x_new.size(0))}, got {int(repair_mask.numel())}."
            )
    repair_gate = repair_mask.to(dtype=x_new.dtype).unsqueeze(-1)
    repaired_x_new = x_new + repair_gate * repair_delta
    active_nodes = int(repair_mask.sum().item())
    delta_norm_mean = 0.0
    if active_nodes > 0:
        delta_norm_mean = float(repair_delta[repair_mask].norm(dim=1).mean().detach().cpu().item())
    return repaired_x_new, {
        "second_view_repair_active": bool(active_nodes > 0),
        "second_view_repair_nodes": active_nodes,
        "second_view_repair_delta_norm_mean": delta_norm_mean,
    }


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


def mhlgc_mask_hyperedge_membership(
    hyperedge_index,
    incident_mask=None,
    num_nodes=None,
    mask_probability=0.0,
):
    p = float(mask_probability or 0.0)
    stats = {
        "hyperedge_membership_mask_probability": p,
        "hyperedge_membership_mask_applied": False,
    }
    if hyperedge_index is None:
        return hyperedge_index, incident_mask, stats
    if p <= 0.0:
        return hyperedge_index, incident_mask, stats
    if p >= 1.0:
        raise ValueError("mhlgc hyperedge membership mask probability must be < 1.0.")
    if hyperedge_index.dim() != 2 or int(hyperedge_index.size(0)) != 2:
        raise ValueError("hyperedge_index must be shaped [2, incidence_count].")

    if num_nodes is None:
        if incident_mask is not None:
            num_nodes = int(incident_mask.numel())
        elif int(hyperedge_index.numel()) > 0:
            num_nodes = int(hyperedge_index[0].max().item()) + 1
        else:
            num_nodes = 0
    num_nodes = int(num_nodes)

    incidence_count_before = int(hyperedge_index.size(1))
    keep = torch.rand(incidence_count_before, device=hyperedge_index.device) >= p
    if incidence_count_before > 0 and not bool(keep.any().item()):
        keep[torch.randint(incidence_count_before, (1,), device=hyperedge_index.device)] = True
    masked_hyperedge_index = hyperedge_index[:, keep].contiguous()
    if int(masked_hyperedge_index.numel()) == 0:
        masked_hyperedge_index = hyperedge_index.new_empty((2, 0))

    masked_incident_mask = torch.zeros(num_nodes, dtype=torch.bool, device=hyperedge_index.device)
    if int(masked_hyperedge_index.numel()) > 0 and num_nodes > 0:
        masked_incident_mask[masked_hyperedge_index[0].long().unique()] = True

    stats.update(
        {
            "hyperedge_membership_mask_applied": True,
            "hyperedge_membership_incidence_before": incidence_count_before,
            "hyperedge_membership_incidence_after": int(masked_hyperedge_index.size(1)),
            "hyperedge_membership_drop_count": int(incidence_count_before - int(masked_hyperedge_index.size(1))),
            "hyperedge_membership_nodes_after": int(masked_incident_mask.sum().item()),
        }
    )
    return masked_hyperedge_index, masked_incident_mask, stats


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
    negative_count=0,
    negative_mask=None,
    anchor_mask=None,
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
    negative_count = max(int(negative_count or 0), 0)
    semantic_embeddings_clean = None
    semantic_anchor_mask = None
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
        semantic_anchor_mask = semantic_embeddings_clean.norm(dim=1) > 0.0
    if anchor_mask is not None:
        anchor_mask = anchor_mask.to(origin_embeddings.device).bool().view(-1)
        if int(anchor_mask.numel()) != int(origin_embeddings.size(0)):
            raise ValueError("anchor_mask must align with origin_embeddings rows.")
        if semantic_anchor_mask is not None:
            anchor_mask = anchor_mask & semantic_anchor_mask
    else:
        anchor_mask = semantic_anchor_mask

    anchors = mhlgc_select_borderline_anchors(
        labels=labels,
        fraud_scores=fraud_scores,
        positive_label=positive_label,
        anchors_per_batch=anchors_per_batch,
        anchor_mask=anchor_mask,
    )
    anchor_candidate_mask = labels == int(positive_label)
    if anchor_mask is not None:
        anchor_candidate_mask = anchor_candidate_mask & anchor_mask
    anchor_candidate_count = int(anchor_candidate_mask.sum().item())
    if negative_mask is None:
        negative_mask = labels != int(positive_label)
    else:
        negative_mask = negative_mask.to(origin_embeddings.device).bool().view(-1)
    negative_idx = torch.nonzero(negative_mask, as_tuple=False).view(-1)
    zero = origin_embeddings.sum() * 0.0
    if anchors.numel() == 0 or negative_idx.numel() == 0:
        return zero, {
            "mhlgc_anchor_count": int(anchors.numel()),
            "mhlgc_anchor_candidate_count": int(anchor_candidate_count),
            "mhlgc_negative_count": 0,
            "mhlgc_negative_candidate_count": int(negative_idx.numel()),
            "mhlgc_negative_count_per_anchor": int(negative_count),
            "mhlgc_negative_selection": "hard_topk" if negative_count > 0 else "all",
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

    negative_candidate_count = int(negative_idx.numel())
    if negative_count > 0 and negative_candidate_count > negative_count:
        _, selected_negative_pos = torch.topk(
            hardness.detach(),
            k=int(negative_count),
            dim=1,
            largest=True,
        )
        neg_logits = torch.gather(neg_logits, dim=1, index=selected_negative_pos)
        hardness = torch.gather(hardness, dim=1, index=selected_negative_pos)

    weights = F.softmax(beta * hardness, dim=1)
    log_weight = torch.log(weights.clamp_min(1e-12))
    log_neg = torch.logsumexp(neg_logits + log_weight, dim=1)
    loss = F.softplus(log_neg - pos_logits).mean()
    return loss, {
        "mhlgc_anchor_count": int(anchors.numel()),
        "mhlgc_anchor_candidate_count": int(anchor_candidate_count),
        "mhlgc_negative_count": int(neg_logits.numel()),
        "mhlgc_negative_candidate_count": int(negative_candidate_count),
        "mhlgc_negative_count_per_anchor": int(negative_count),
        "mhlgc_negative_selection": "hard_topk" if negative_count > 0 else "all",
        "mhlgc_active": True,
        "mhlgc_beta": float(beta),
        "mhlgc_gamma": float(gamma),
        "mhlgc_temperature": float(temperature),
    }


def routed_same_node_contrastive_loss(low_embeddings, high_embeddings, temperature=0.2):
    """Same-node low/high InfoNCE over routed eligible nodes only."""
    if low_embeddings.dim() != 2 or high_embeddings.dim() != 2:
        raise ValueError("low_embeddings and high_embeddings must be 2-D tensors.")
    if low_embeddings.shape != high_embeddings.shape:
        raise ValueError("low_embeddings and high_embeddings must have the same shape.")
    node_count = int(low_embeddings.size(0))
    temperature = float(temperature or 0.2)
    zero = low_embeddings.sum() * 0.0
    if node_count < 2:
        return zero, {
            "active": False,
            "family": "low_high",
            "eligible_count": int(node_count),
            "anchor_count": 0,
            "positive_count": 0,
            "negative_count": 0,
        }
    if temperature <= 0.0:
        raise ValueError("routed contrast temperature must be > 0.")
    z_low = F.normalize(low_embeddings, dim=1, eps=1e-12)
    z_high = F.normalize(high_embeddings, dim=1, eps=1e-12)
    logits = torch.matmul(z_low, z_high.transpose(0, 1)) / temperature
    labels = torch.arange(node_count, device=logits.device)
    loss_lh = F.cross_entropy(logits, labels)
    loss_hl = F.cross_entropy(logits.transpose(0, 1), labels)
    loss = 0.5 * (loss_lh + loss_hl)
    return loss, {
        "active": True,
        "family": "low_high",
        "eligible_count": int(node_count),
        "anchor_count": int(node_count),
        "positive_count": int(node_count),
        "negative_count": int(node_count * max(node_count - 1, 0)),
    }


def routed_frozen_alignment_loss(frozen_embeddings, target_embeddings):
    """Stop-grad alignment from frozen semantic teacher to routed detector view."""
    if frozen_embeddings.dim() != 2 or target_embeddings.dim() != 2:
        raise ValueError("frozen_embeddings and target_embeddings must be 2-D tensors.")
    if frozen_embeddings.shape != target_embeddings.shape:
        raise ValueError("frozen_embeddings and target_embeddings must have the same shape.")
    node_count = int(target_embeddings.size(0))
    zero = target_embeddings.sum() * 0.0
    if node_count < 1:
        return zero, {
            "active": False,
            "family": "frozen_control",
            "eligible_count": 0,
        }
    teacher = F.normalize(frozen_embeddings.detach(), dim=1, eps=1e-12)
    student = F.normalize(target_embeddings, dim=1, eps=1e-12)
    loss = 1.0 - torch.sum(teacher * student, dim=1).mean()
    return loss, {
        "active": True,
        "family": "frozen_control",
        "eligible_count": int(node_count),
    }


def routed_supervised_contrastive_loss(embeddings, labels, temperature=0.2):
    """BotSCL-style same-class positive, different-class negative contrast."""
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be 2-D.")
    labels = labels.view(-1).to(embeddings.device)
    if int(labels.numel()) != int(embeddings.size(0)):
        raise ValueError("labels must align with embeddings rows.")
    node_count = int(embeddings.size(0))
    temperature = float(temperature or 0.2)
    zero = embeddings.sum() * 0.0
    if node_count < 2:
        return zero, {
            "active": False,
            "family": "supcon_class",
            "eligible_count": int(node_count),
            "anchor_count": 0,
            "positive_pair_count": 0,
            "negative_pair_count": 0,
        }
    if temperature <= 0.0:
        raise ValueError("routed contrast temperature must be > 0.")
    z = F.normalize(embeddings, dim=1, eps=1e-12)
    similarity = torch.matmul(z, z.transpose(0, 1)) / temperature
    diag_mask = torch.eye(node_count, device=similarity.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~diag_mask)
    negative_mask = ~labels.unsqueeze(0).eq(labels.unsqueeze(1))
    valid_anchor_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    valid_anchor_idx = torch.nonzero(valid_anchor_mask, as_tuple=False).view(-1)
    if valid_anchor_idx.numel() == 0:
        return zero, {
            "active": False,
            "family": "supcon_class",
            "eligible_count": int(node_count),
            "anchor_count": 0,
            "positive_pair_count": int(positive_mask.sum().item()),
            "negative_pair_count": int(negative_mask.sum().item()),
        }
    logits = similarity[valid_anchor_idx]
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~diag_mask[valid_anchor_idx]).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_mask_valid = positive_mask[valid_anchor_idx].to(logits.dtype)
    positive_counts = positive_mask_valid.sum(dim=1)
    loss = -((positive_mask_valid * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)).mean()
    return loss, {
        "active": True,
        "family": "supcon_class",
        "eligible_count": int(node_count),
        "anchor_count": int(valid_anchor_idx.numel()),
        "positive_pair_count": int(positive_mask_valid.sum().item()),
        "negative_pair_count": int(negative_mask[valid_anchor_idx].sum().item()),
    }


def routed_bot_edge_mask_human_contrastive_loss(
    anchor_embeddings,
    positive_embeddings,
    negative_embeddings,
    temperature=0.2,
):
    """Anchor routed bots, use edge-masked same-node positives, and routed humans as negatives."""
    if anchor_embeddings.dim() != 2 or positive_embeddings.dim() != 2 or negative_embeddings.dim() != 2:
        raise ValueError("anchor_embeddings, positive_embeddings, and negative_embeddings must be 2-D.")
    if anchor_embeddings.shape != positive_embeddings.shape:
        raise ValueError("anchor_embeddings and positive_embeddings must have the same shape.")
    if int(anchor_embeddings.size(1)) != int(negative_embeddings.size(1)):
        raise ValueError("negative_embeddings must match anchor embedding width.")
    anchor_count = int(anchor_embeddings.size(0))
    negative_count = int(negative_embeddings.size(0))
    temperature = float(temperature or 0.2)
    zero = anchor_embeddings.sum() * 0.0
    if anchor_count < 1 or negative_count < 1:
        return zero, {
            "active": False,
            "family": "bot_edge_mask_human",
            "anchor_count": int(anchor_count),
            "positive_count": int(anchor_count),
            "negative_count": int(negative_count),
        }
    if temperature <= 0.0:
        raise ValueError("routed contrast temperature must be > 0.")
    z_anchor = F.normalize(anchor_embeddings, dim=1, eps=1e-12)
    z_positive = F.normalize(positive_embeddings, dim=1, eps=1e-12)
    z_negative = F.normalize(negative_embeddings, dim=1, eps=1e-12)
    pos_logits = torch.sum(z_anchor * z_positive, dim=1, keepdim=True) / temperature
    neg_logits = torch.matmul(z_anchor, z_negative.transpose(0, 1)) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    targets = torch.zeros(anchor_count, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    return loss, {
        "active": True,
        "family": "bot_edge_mask_human",
        "anchor_count": int(anchor_count),
        "positive_count": int(anchor_count),
        "negative_count": int(anchor_count * negative_count),
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

    export_relation_x_new = True

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
    export_relation_x_new = True

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


class RGCNH2FAGDualSpace(BaseGraphBackbone):
    """Construct-complete relation backbone over construct-space low-order propagation."""

    export_relation_x_new = False
    backbone_contract_version = "construct_complete_v2"

    def __init__(self, model_config):
        super().__init__(model_config)
        construct_input_dim = int(_cfg(model_config, "construct_input_dim"))
        if construct_input_dim <= 0:
            raise ValueError("construct-complete dualspace backbones require positive construct_input_dim.")
        self.construct_input_dim = construct_input_dim
        self.construct_identity_projector = nn.Linear(self.construct_input_dim, self.hidden_dim)
        self.construct_relation_encoder = DualSpaceRelationEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_relations=self.n_relations,
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
            activation=self.activation,
            output_prefix="construct",
        )

    def _encode_construct_identity(self, construct_x):
        if construct_x is None or not torch.is_tensor(construct_x):
            raise ValueError(
                "construct-complete dualspace backbones require construct_x from the construct representation space."
            )
        x_in_construct = self.construct_identity_projector(construct_x)
        x_in_construct = self.activation(x_in_construct)
        x_in_construct = self.dropout(x_in_construct)
        return x_in_construct

    def _build_construct_relation_view(self, construct_x, edge_index, edge_type):
        x_in_construct = self._encode_construct_identity(construct_x)
        rel = self.construct_relation_encoder(x_in_construct, edge_index, edge_type)
        x_low_construct = rel["x_low"]
        return x_in_construct, x_low_construct, rel

    def _relation_aux_features(self, rel):
        aux = {
            "enabled": True,
            "space": "construct_representation",
        }
        aux.update(_gate_mean_stats(rel.get("gate1_construct"), "gate1"))
        aux.update(_gate_mean_stats(rel.get("gate2_construct"), "gate2"))
        return aux

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        if second_view_repair_delta is not None or second_view_repair_mask is not None:
            raise ValueError("construct-complete relation-only dualspace backbones do not support repair-aware second-view deltas.")
        if routed_multiview_bundle is not None:
            raise ValueError("construct-complete relation-only dualspace backbones do not support routed multiview refinement.")
        if routed_highpass_bundle is not None:
            raise ValueError("construct-complete relation-only dualspace backbones do not support routed high-pass correction.")
        if float(hyperedge_mask_probability or 0.0) != 0.0:
            raise ValueError("construct-complete relation-only dualspace backbones do not support hyperedge masking.")

        x_in_construct, x_low_construct, rel = self._build_construct_relation_view(construct_x, edge_index, edge_type)
        logits = self.linear_out(x_low_construct)
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": x_low_construct,
            "fused_x": x_low_construct,
            "x_in_construct": x_in_construct,
            "x_low": x_low_construct,
            "x_low_construct": x_low_construct,
            "x_ego_construct": rel["x_ego_construct"],
            "x_rel_low1_construct": rel["x_rel_low1_construct"],
            "x_rel_high1_construct": rel["x_rel_high1_construct"],
            "x_rel_state1_construct": rel["x_rel_state1_construct"],
            "x_rel_low2_construct": rel["x_rel_low2_construct"],
            "x_rel_high2_construct": rel["x_rel_high2_construct"],
            "x_rel_state2_construct": rel["x_rel_state2_construct"],
            "aux_features": {
                "construct_relation_encoder": self._relation_aux_features(rel),
            },
            "backbone_contract_version": self.backbone_contract_version,
        }


class RGCNH2FAGDualSpaceHyperScan(RGCNH2FAGDualSpace):
    """Construct-complete HyperScan branch with construct-space low/high graph views."""

    def __init__(self, model_config):
        super().__init__(model_config)
        branch_cfg = dict(model_config.get("dynamic_similarity_branch", {}) or {})
        self.graph_second_view_hypergraph_backend = str(
            _cfg(model_config, "graph_second_view_hypergraph_backend", default="pyg") or "pyg"
        ).strip().lower()
        if self.graph_second_view_hypergraph_backend not in {"pyg", "dhg"}:
            raise ValueError("--graph_second_view_hypergraph_backend must be one of {pyg, dhg}.")
        self.graph_second_view_use_bn = bool(_cfg(model_config, "graph_second_view_use_bn", default=False))
        self.clean_encoder = DualSpaceHighOrderEncoder(
            construct_feature_dim=self.hidden_dim * 2,
            hidden_dim=self.hidden_dim,
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
            activation=self.activation,
            hypergraph_backend=self.graph_second_view_hypergraph_backend,
            use_bn=self.graph_second_view_use_bn,
        )
        _init_hyperscan_detector(self, model_config, self.hidden_dim)
        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_candidate_scope = str(branch_cfg.get("candidate_scope", "undirected_relation_1hop"))
        self.dynamic_similarity_metric = str(branch_cfg.get("similarity_metric", "cosine"))
        self.dynamic_feature_source = str(branch_cfg.get("feature_source", "hyperscan_clean_representation"))
        self.dynamic_center_source = str(branch_cfg.get("center_source", "labeled_prefix"))
        self.dynamic_routed_nodes_path = str(branch_cfg.get("routed_nodes_path", "") or "")
        self.dynamic_routed_nodes_split = str(branch_cfg.get("routed_nodes_split", "all") or "all")
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        self._dynamic_batch_local_routed_node_ids = _as_long_tensor(branch_cfg.get("batch_local_routed_node_ids", []))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        center_candidate_role_ids = list(branch_cfg.get("center_candidate_role_ids", []) or [])
        self._dynamic_center_candidate_role_ids = (
            _as_long_tensor_list(center_candidate_role_ids) if center_candidate_role_ids else None
        )
        center_candidate_keep_mask = list(branch_cfg.get("center_candidate_keep_mask", []) or [])
        self._dynamic_center_candidate_keep_mask = (
            [torch.tensor(item, dtype=torch.bool) for item in center_candidate_keep_mask]
            if center_candidate_keep_mask
            else None
        )
        self.dynamic_candidate_policy_config = _dynamic_branch_policy_config(branch_cfg)
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
            "hypergraph_backend": self.graph_second_view_hypergraph_backend,
            "hypergraph_use_bn": self.graph_second_view_use_bn,
            "fusion": self.graph_second_view_fusion,
            "representation_roles": "construct_complete_v2",
            **_dynamic_branch_policy_stats(branch_cfg),
        }

    def _apply_construct_detector(self, x_in_construct, x_low_construct, x_high_construct, incident_scale):
        detector_gate = None
        if self.graph_second_view_fusion == "construct_acm":
            x_identity = self.detector_identity_projector(x_in_construct)
            hidden, detector_gate = self.detector_construct_acm(x_identity, x_low_construct, x_high_construct)
            logits = self.linear_out(hidden)
            return hidden, logits, detector_gate
        hidden, logits = _apply_hyperscan_detector(
            self,
            x_low_construct,
            x_high_construct,
            incident_scale,
            x_identity=x_in_construct,
        )
        return hidden, logits, detector_gate

    def _highorder_aux_features(self, high):
        aux = {
            "enabled": True,
            "space": "construct_representation",
        }
        aux.update(_gate_mean_stats(high.get("gate1_construct"), "gate1"))
        aux.update(_gate_mean_stats(high.get("gate2_construct"), "gate2"))
        return aux

    def _build_dynamic_hypergraph(self, feature_tensor, batch_node_ids=None):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
                node_ids=batch_node_ids,
                routed_node_ids=self._dynamic_batch_local_routed_node_ids,
                **self.dynamic_candidate_policy_config,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
            center_candidate_role_ids=self._dynamic_center_candidate_role_ids,
            center_candidate_keep_mask=self._dynamic_center_candidate_keep_mask,
            **self.dynamic_candidate_policy_config,
        )

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        if construct_x is None or not torch.is_tensor(construct_x):
            raise ValueError("construct-complete dualspace hyperscan backbones require construct_x from the construct representation.")
        if second_view_repair_delta is not None or second_view_repair_mask is not None:
            raise ValueError("construct-complete dualspace hyperscan backbones do not support MH-LGC repair-aware second-view deltas in v1.")
        if routed_multiview_bundle is not None:
            raise ValueError("construct-complete dualspace hyperscan backbones do not support routed multiview refinement in v1.")
        if routed_highpass_bundle is not None:
            raise ValueError("construct-complete dualspace hyperscan backbones do not support routed high-pass correction in v1.")
        if float(hyperedge_mask_probability or 0.0) != 0.0:
            raise ValueError("construct-complete dualspace hyperscan backbones do not support hyperedge masking in v1.")

        x_in_construct, x_low_construct, rel = self._build_construct_relation_view(construct_x, edge_index, edge_type)
        x_new_construct = torch.cat([x_low_construct, x_in_construct], dim=1)
        x_hg_base = self.clean_encoder.build_hg_base(x_new_construct)
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(
            x_new_construct,
            batch_node_ids=batch_node_ids,
        )
        high = self.clean_encoder(
            x_new_construct,
            hyperedge_index,
            incident_mask,
            prepared_construct={
                "x_hg_base": x_hg_base,
            },
        )
        hidden, logits, detector_gate = self._apply_construct_detector(
            x_in_construct,
            x_low_construct,
            high["x_high_construct"],
            high["incident_scale"],
        )
        aux = {
            "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
            "construct_relation_encoder": self._relation_aux_features(rel),
            "construct_highorder_encoder": self._highorder_aux_features(high),
        }
        if torch.is_tensor(detector_gate):
            aux["construct_acm_detector"] = _gate_mean_stats(detector_gate, "detector")
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "x_in_construct": x_in_construct,
            "x_low": x_low_construct,
            "x_low_construct": x_low_construct,
            "x_ego_construct": rel["x_ego_construct"],
            "x_rel_low1_construct": rel["x_rel_low1_construct"],
            "x_rel_high1_construct": rel["x_rel_high1_construct"],
            "x_rel_state1_construct": rel["x_rel_state1_construct"],
            "x_rel_low2_construct": rel["x_rel_low2_construct"],
            "x_rel_high2_construct": rel["x_rel_high2_construct"],
            "x_rel_state2_construct": rel["x_rel_state2_construct"],
            "x_new": x_new_construct,
            "x_new_construct": x_new_construct,
            "x_hg_base": high["x_hg_base"],
            "x_high_low1_construct": high["x_high_low1_construct"],
            "x_high_high1_construct": high["x_high_high1_construct"],
            "x_high_state1_construct": high["x_high_state1_construct"],
            "x_high_low2_construct": high["x_high_low2_construct"],
            "x_high_high2_construct": high["x_high_high2_construct"],
            "x_high_construct": high["x_high_construct"],
            "x_high": high["x_high_construct"],
            "aux_features": aux,
            "backbone_contract_version": self.backbone_contract_version,
        }


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
        _init_routed_multiview_refiner(self, model_config, self.hidden_dim)
        _init_routed_highpass_correction(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_candidate_scope = str(branch_cfg.get("candidate_scope", "undirected_relation_1hop"))
        self.dynamic_similarity_metric = str(branch_cfg.get("similarity_metric", "cosine"))
        self.dynamic_feature_source = _second_view_feature_source(branch_cfg)
        self.dynamic_center_source = str(branch_cfg.get("center_source", "labeled_prefix"))
        self.dynamic_routed_nodes_path = str(branch_cfg.get("routed_nodes_path", "") or "")
        self.dynamic_routed_nodes_split = str(branch_cfg.get("routed_nodes_split", "all") or "all")
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        self._dynamic_batch_local_routed_node_ids = _as_long_tensor(branch_cfg.get("batch_local_routed_node_ids", []))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        center_candidate_role_ids = list(branch_cfg.get("center_candidate_role_ids", []) or [])
        if center_candidate_role_ids and len(center_candidate_role_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate role ids must match candidate ids.")
        self._dynamic_center_candidate_role_ids = (
            _as_long_tensor_list(center_candidate_role_ids) if center_candidate_role_ids else None
        )
        center_candidate_keep_mask = list(branch_cfg.get("center_candidate_keep_mask", []) or [])
        if center_candidate_keep_mask and len(center_candidate_keep_mask) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate keep mask must match candidate ids.")
        self._dynamic_center_candidate_keep_mask = (
            [torch.tensor(item, dtype=torch.bool) for item in center_candidate_keep_mask]
            if center_candidate_keep_mask
            else None
        )
        self.dynamic_candidate_policy_config = _dynamic_branch_policy_config(branch_cfg)
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
            **_dynamic_branch_policy_stats(branch_cfg),
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

    def _build_dynamic_hypergraph(self, feature_tensor, batch_node_ids=None):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
                node_ids=batch_node_ids,
                routed_node_ids=self._dynamic_batch_local_routed_node_ids,
                **self.dynamic_candidate_policy_config,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
            center_candidate_role_ids=self._dynamic_center_candidate_role_ids,
            center_candidate_keep_mask=self._dynamic_center_candidate_keep_mask,
            **self.dynamic_candidate_policy_config,
        )

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        x_low = self._encode_relation_view(x, edge_index, edge_type)
        x_new_base = torch.cat([x_low, x], dim=1)
        x_new, repair_stats = _apply_second_view_repair(
            x_new_base,
            repair_delta=second_view_repair_delta,
            repair_mask=second_view_repair_mask,
        )
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new, batch_node_ids=batch_node_ids)
        hyperedge_index, incident_mask, mask_stats = mhlgc_mask_hyperedge_membership(
            hyperedge_index,
            incident_mask=incident_mask,
            num_nodes=int(x_new.size(0)),
            mask_probability=hyperedge_mask_probability,
        )
        branch_stats = dict(branch_stats or {})
        branch_stats.update(mask_stats)
        branch_stats.update(repair_stats)
        if hyperedge_index is None or int(hyperedge_index.numel()) == 0:
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

        x_high_base = x_high
        x_high, highpass_xhigh_stats, highpass_xhigh_aux = _apply_routed_highpass_x_high_correction(
            self,
            x_high,
            routed_highpass_bundle,
        )
        detector_views = _apply_selective_second_view_consumption(
            self,
            x_low,
            x_high,
            incident_scale,
            batch_node_ids=batch_node_ids,
        )
        hidden = detector_views["hidden"]
        logits = detector_views["logits"]
        hidden, logits, routed_stats = _apply_routed_multiview_refinement(
            self,
            hidden,
            x_low,
            x_high,
            logits,
            routed_multiview_bundle,
        )
        hidden, logits, highpass_stats, highpass_aux = _apply_routed_highpass_correction(
            self,
            hidden,
            logits,
            routed_highpass_bundle,
        )
        if highpass_xhigh_aux is not None:
            highpass_stats = highpass_xhigh_stats
            highpass_aux = highpass_xhigh_aux
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "fused_x_hnn": detector_views["hidden_hnn"],
            "fused_x_lowonly": detector_views["hidden_lowonly"],
            "logits_hnn": detector_views["logits_hnn"],
            "logits_lowonly": detector_views["logits_lowonly"],
            "highorder_consumer_mask": detector_views["consumer_mask"],
            "highorder_consumer_gate": detector_views["consumer_gate"],
            "highorder_risk_score": detector_views["consumer_risk"],
            "x_low": x_low,
            "x_high": x_high,
            "x_high_base": x_high_base,
            "x_new": x_new,
            "x_new_base": x_new_base,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
                "routed_multiview_refiner": _cpu_scalar_dict(routed_stats),
                "routed_highpass": _cpu_scalar_dict(highpass_stats),
            },
            "routed_highpass_aux": highpass_aux,
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
        _init_routed_multiview_refiner(self, model_config, self.hidden_dim)
        _init_routed_highpass_correction(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        self._dynamic_batch_local_routed_node_ids = _as_long_tensor(branch_cfg.get("batch_local_routed_node_ids", []))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        center_candidate_role_ids = list(branch_cfg.get("center_candidate_role_ids", []) or [])
        if center_candidate_role_ids and len(center_candidate_role_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate role ids must match candidate ids.")
        self._dynamic_center_candidate_role_ids = (
            _as_long_tensor_list(center_candidate_role_ids) if center_candidate_role_ids else None
        )
        center_candidate_keep_mask = list(branch_cfg.get("center_candidate_keep_mask", []) or [])
        if center_candidate_keep_mask and len(center_candidate_keep_mask) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate keep mask must match candidate ids.")
        self._dynamic_center_candidate_keep_mask = (
            [torch.tensor(item, dtype=torch.bool) for item in center_candidate_keep_mask]
            if center_candidate_keep_mask
            else None
        )
        self.dynamic_candidate_policy_config = _dynamic_branch_policy_config(branch_cfg)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": _second_view_feature_source(branch_cfg),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "pyg",
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "hyperscan_meta_tweet_proxy",
            **_dynamic_branch_policy_stats(branch_cfg),
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

    def _build_dynamic_hypergraph(self, feature_tensor, batch_node_ids=None):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
                node_ids=batch_node_ids,
                routed_node_ids=self._dynamic_batch_local_routed_node_ids,
                **self.dynamic_candidate_policy_config,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
            center_candidate_role_ids=self._dynamic_center_candidate_role_ids,
            center_candidate_keep_mask=self._dynamic_center_candidate_keep_mask,
            **self.dynamic_candidate_policy_config,
        )

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        x_in, x_rel = self._encode_node_input(x)
        x_low = self._encode_relation_view(x_rel, edge_index, edge_type)
        x_new_base = torch.cat([x_low, x_in], dim=1)
        x_new, repair_stats = _apply_second_view_repair(
            x_new_base,
            repair_delta=second_view_repair_delta,
            repair_mask=second_view_repair_mask,
        )
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new, batch_node_ids=batch_node_ids)
        hyperedge_index, incident_mask, mask_stats = mhlgc_mask_hyperedge_membership(
            hyperedge_index,
            incident_mask=incident_mask,
            num_nodes=int(x_new.size(0)),
            mask_probability=hyperedge_mask_probability,
        )
        branch_stats = dict(branch_stats or {})
        branch_stats.update(mask_stats)
        branch_stats.update(repair_stats)
        if hyperedge_index is None or int(hyperedge_index.numel()) == 0:
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

        x_high_base = x_high
        x_high, highpass_xhigh_stats, highpass_xhigh_aux = _apply_routed_highpass_x_high_correction(
            self,
            x_high,
            routed_highpass_bundle,
        )
        detector_views = _apply_selective_second_view_consumption(
            self,
            x_low,
            x_high,
            incident_scale,
            batch_node_ids=batch_node_ids,
        )
        hidden = detector_views["hidden"]
        logits = detector_views["logits"]
        hidden, logits, routed_stats = _apply_routed_multiview_refinement(
            self,
            hidden,
            x_low,
            x_high,
            logits,
            routed_multiview_bundle,
        )
        hidden, logits, highpass_stats, highpass_aux = _apply_routed_highpass_correction(
            self,
            hidden,
            logits,
            routed_highpass_bundle,
        )
        if highpass_xhigh_aux is not None:
            highpass_stats = highpass_xhigh_stats
            highpass_aux = highpass_xhigh_aux
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "fused_x_hnn": detector_views["hidden_hnn"],
            "fused_x_lowonly": detector_views["hidden_lowonly"],
            "logits_hnn": detector_views["logits_hnn"],
            "logits_lowonly": detector_views["logits_lowonly"],
            "highorder_consumer_mask": detector_views["consumer_mask"],
            "highorder_consumer_gate": detector_views["consumer_gate"],
            "highorder_risk_score": detector_views["consumer_risk"],
            "x_low": x_low,
            "x_high": x_high,
            "x_high_base": x_high_base,
            "x_new": x_new,
            "x_new_base": x_new_base,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
                "routed_multiview_refiner": _cpu_scalar_dict(routed_stats),
                "routed_highpass": _cpu_scalar_dict(highpass_stats),
            },
            "routed_highpass_aux": highpass_aux,
        }


class RGCNH2FAGDualSpaceNodeInput(RGCNH2FAGDualSpace):
    """Construct-complete relation backbone with Hyperscan-style nodeinput preprocessing."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.construct_node_input_encoder = ConstructNodeInputEncoder(
            tweet_dim=int(_cfg(model_config, "tweet_dim")),
            num_prop_dim=int(_cfg(model_config, "num_prop_dim")),
            cat_prop_dim=int(_cfg(model_config, "cat_prop_dim")),
            hidden_dim=self.hidden_dim,
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        self.construct_identity_projector = None

    def _encode_construct_identity(self, construct_x):
        if construct_x is None or not torch.is_tensor(construct_x):
            raise ValueError(
                "construct-complete dualspace nodeinput backbones require construct_x ordered as tweet|num|cat."
            )
        return self.construct_node_input_encoder(construct_x)


class RGCNH2FAGDualSpaceHyperScanNodeInput(RGCNH2FAGDualSpaceHyperScan):
    """Construct-complete HyperScan graph branch with Hyperscan-style nodeinput preprocessing."""

    def __init__(self, model_config):
        super().__init__(model_config)
        self.construct_node_input_encoder = ConstructNodeInputEncoder(
            tweet_dim=int(_cfg(model_config, "tweet_dim")),
            num_prop_dim=int(_cfg(model_config, "num_prop_dim")),
            cat_prop_dim=int(_cfg(model_config, "cat_prop_dim")),
            hidden_dim=self.hidden_dim,
            dropout_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        self.construct_identity_projector = None
        self.dynamic_branch_static_stats["node_input_family"] = "hyperscan_meta_tweet_proxy"

    def _encode_construct_identity(self, construct_x):
        if construct_x is None or not torch.is_tensor(construct_x):
            raise ValueError(
                "construct-complete dualspace hyperscan nodeinput backbones require construct_x ordered as tweet|num|cat."
            )
        return self.construct_node_input_encoder(construct_x)


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
        graph_second_view_use_bn = bool(_cfg(model_config, "graph_second_view_use_bn", default=False))
        self.hgnn_layer1 = DHG_HGNNConv(
            self.hidden_dim * 2,
            self.hidden_dim,
            use_bn=graph_second_view_use_bn,
            drop_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        self.hgnn_layer2 = DHG_HGNNConv(
            self.hidden_dim,
            self.hidden_dim,
            use_bn=graph_second_view_use_bn,
            is_last=True,
        )
        _init_hyperscan_detector(self, model_config, self.hidden_dim)
        _init_routed_multiview_refiner(self, model_config, self.hidden_dim)
        _init_routed_highpass_correction(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        self._dynamic_batch_local_routed_node_ids = _as_long_tensor(branch_cfg.get("batch_local_routed_node_ids", []))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        center_candidate_role_ids = list(branch_cfg.get("center_candidate_role_ids", []) or [])
        if center_candidate_role_ids and len(center_candidate_role_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate role ids must match candidate ids.")
        self._dynamic_center_candidate_role_ids = (
            _as_long_tensor_list(center_candidate_role_ids) if center_candidate_role_ids else None
        )
        center_candidate_keep_mask = list(branch_cfg.get("center_candidate_keep_mask", []) or [])
        if center_candidate_keep_mask and len(center_candidate_keep_mask) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate keep mask must match candidate ids.")
        self._dynamic_center_candidate_keep_mask = (
            [torch.tensor(item, dtype=torch.bool) for item in center_candidate_keep_mask]
            if center_candidate_keep_mask
            else None
        )
        self.dynamic_candidate_policy_config = _dynamic_branch_policy_config(branch_cfg)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": _second_view_feature_source(branch_cfg),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "dhg",
            "hypergraph_use_bn": graph_second_view_use_bn,
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "hyperscan_meta_tweet_proxy",
            **_dynamic_branch_policy_stats(branch_cfg),
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

    def _build_dynamic_hypergraph(self, feature_tensor, batch_node_ids=None):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
                node_ids=batch_node_ids,
                routed_node_ids=self._dynamic_batch_local_routed_node_ids,
                **self.dynamic_candidate_policy_config,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
            center_candidate_role_ids=self._dynamic_center_candidate_role_ids,
            center_candidate_keep_mask=self._dynamic_center_candidate_keep_mask,
            **self.dynamic_candidate_policy_config,
        )

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        x_in, x_rel = self._encode_node_input(x)
        x_low = self._encode_relation_view(x_rel, edge_index, edge_type)
        x_new_base = torch.cat([x_low, x_in], dim=1)
        x_new, repair_stats = _apply_second_view_repair(
            x_new_base,
            repair_delta=second_view_repair_delta,
            repair_mask=second_view_repair_mask,
        )
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new, batch_node_ids=batch_node_ids)
        hyperedge_index, incident_mask, mask_stats = mhlgc_mask_hyperedge_membership(
            hyperedge_index,
            incident_mask=incident_mask,
            num_nodes=int(x_new.size(0)),
            mask_probability=hyperedge_mask_probability,
        )
        branch_stats = dict(branch_stats or {})
        branch_stats.update(mask_stats)
        branch_stats.update(repair_stats)
        hg = _dhg_hypergraph_from_incidence(hyperedge_index, x_new.size(0), x_new.device)
        if hg is None:
            x_high = torch.zeros_like(x_low)
        else:
            x_mid = self.hgnn_layer1(x_new, hg)
            x_high = self.hgnn_layer2(x_mid, hg)
        if incident_mask is None:
            incident_mask = torch.zeros(x_low.size(0), dtype=torch.bool, device=x_low.device)
        incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)
        x_high_base = x_high
        x_high, highpass_xhigh_stats, highpass_xhigh_aux = _apply_routed_highpass_x_high_correction(
            self,
            x_high,
            routed_highpass_bundle,
        )
        detector_views = _apply_selective_second_view_consumption(
            self,
            x_low,
            x_high,
            incident_scale,
            batch_node_ids=batch_node_ids,
        )
        hidden = detector_views["hidden"]
        logits = detector_views["logits"]
        hidden, logits, routed_stats = _apply_routed_multiview_refinement(
            self,
            hidden,
            x_low,
            x_high,
            logits,
            routed_multiview_bundle,
        )
        hidden, logits, highpass_stats, highpass_aux = _apply_routed_highpass_correction(
            self,
            hidden,
            logits,
            routed_highpass_bundle,
        )
        if highpass_xhigh_aux is not None:
            highpass_stats = highpass_xhigh_stats
            highpass_aux = highpass_xhigh_aux
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "fused_x_hnn": detector_views["hidden_hnn"],
            "fused_x_lowonly": detector_views["hidden_lowonly"],
            "logits_hnn": detector_views["logits_hnn"],
            "logits_lowonly": detector_views["logits_lowonly"],
            "highorder_consumer_mask": detector_views["consumer_mask"],
            "highorder_consumer_gate": detector_views["consumer_gate"],
            "highorder_risk_score": detector_views["consumer_risk"],
            "x_low": x_low,
            "x_high": x_high,
            "x_high_base": x_high_base,
            "x_new": x_new,
            "x_new_base": x_new_base,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
                "routed_multiview_refiner": _cpu_scalar_dict(routed_stats),
                "routed_highpass": _cpu_scalar_dict(highpass_stats),
            },
            "routed_highpass_aux": highpass_aux,
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
        graph_second_view_use_bn = bool(_cfg(model_config, "graph_second_view_use_bn", default=False))
        self.hgnn_layer1 = DHG_HGNNConv(
            self.hidden_dim + input_dim,
            self.hidden_dim,
            use_bn=graph_second_view_use_bn,
            drop_rate=float(_cfg(model_config, "dropout", default=0.4)),
        )
        self.hgnn_layer2 = DHG_HGNNConv(
            self.hidden_dim,
            self.hidden_dim,
            use_bn=graph_second_view_use_bn,
            is_last=True,
        )
        _init_hyperscan_detector(self, model_config, self.hidden_dim)
        _init_routed_multiview_refiner(self, model_config, self.hidden_dim)
        _init_routed_highpass_correction(self, model_config, self.hidden_dim)

        self.dynamic_branch_enabled = bool(branch_cfg.get("enabled", False))
        self.dynamic_hyper_k = int(branch_cfg.get("knn_k", 8))
        self.dynamic_batch_local_knn = bool(branch_cfg.get("batch_local_knn", False))
        self._dynamic_batch_local_routed_node_ids = _as_long_tensor(branch_cfg.get("batch_local_routed_node_ids", []))
        center_node_ids = list(branch_cfg.get("center_node_ids", []) or [])
        center_candidate_node_ids = list(branch_cfg.get("center_candidate_node_ids", []) or [])
        if (not self.dynamic_batch_local_knn) and len(center_node_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch center ids and candidate ids must have the same length.")
        self.register_buffer("_dynamic_center_node_ids", torch.tensor(center_node_ids, dtype=torch.long), persistent=False)
        self._dynamic_center_candidate_node_ids = _as_long_tensor_list(center_candidate_node_ids)
        center_candidate_role_ids = list(branch_cfg.get("center_candidate_role_ids", []) or [])
        if center_candidate_role_ids and len(center_candidate_role_ids) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate role ids must match candidate ids.")
        self._dynamic_center_candidate_role_ids = (
            _as_long_tensor_list(center_candidate_role_ids) if center_candidate_role_ids else None
        )
        center_candidate_keep_mask = list(branch_cfg.get("center_candidate_keep_mask", []) or [])
        if center_candidate_keep_mask and len(center_candidate_keep_mask) != len(center_candidate_node_ids):
            raise ValueError("dynamic_similarity_branch candidate keep mask must match candidate ids.")
        self._dynamic_center_candidate_keep_mask = (
            [torch.tensor(item, dtype=torch.bool) for item in center_candidate_keep_mask]
            if center_candidate_keep_mask
            else None
        )
        self.dynamic_candidate_policy_config = _dynamic_branch_policy_config(branch_cfg)
        self.dynamic_branch_static_stats = {
            "enabled": self.dynamic_branch_enabled,
            "center_count": int(len(center_node_ids)),
            "knn_k": int(self.dynamic_hyper_k),
            "candidate_scope": str(branch_cfg.get("candidate_scope", "undirected_relation_1hop")),
            "similarity_metric": str(branch_cfg.get("similarity_metric", "cosine")),
            "feature_source": _second_view_feature_source(branch_cfg),
            "center_source": str(branch_cfg.get("center_source", "labeled_prefix")),
            "routed_nodes_path": str(branch_cfg.get("routed_nodes_path", "") or ""),
            "routed_nodes_split": str(branch_cfg.get("routed_nodes_split", "all") or "all"),
            "batch_local_knn": self.dynamic_batch_local_knn,
            "hypergraph_backend": "dhg",
            "hypergraph_use_bn": graph_second_view_use_bn,
            "fusion": self.graph_second_view_fusion,
            "centers_with_relation_neighbors": int(branch_cfg.get("centers_with_relation_neighbors", 0)),
            "mean_relation_candidates_per_center": float(branch_cfg.get("mean_relation_candidates_per_center", 0.0)),
            "node_input_family": "semantic_embedding",
            **_dynamic_branch_policy_stats(branch_cfg),
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

    def _build_dynamic_hypergraph(self, feature_tensor, batch_node_ids=None):
        if self.dynamic_batch_local_knn:
            return build_batch_local_knn_hypergraph(
                feature_tensor,
                branch_enabled=self.dynamic_branch_enabled,
                knn_k=self.dynamic_hyper_k,
                static_stats=self.dynamic_branch_static_stats,
                node_ids=batch_node_ids,
                routed_node_ids=self._dynamic_batch_local_routed_node_ids,
                **self.dynamic_candidate_policy_config,
            )
        return build_dynamic_hypergraph(
            feature_tensor,
            center_node_ids=self._dynamic_center_node_ids,
            center_candidate_node_ids=self._dynamic_center_candidate_node_ids,
            branch_enabled=self.dynamic_branch_enabled,
            knn_k=self.dynamic_hyper_k,
            static_stats=self.dynamic_branch_static_stats,
            center_candidate_role_ids=self._dynamic_center_candidate_role_ids,
            center_candidate_keep_mask=self._dynamic_center_candidate_keep_mask,
            **self.dynamic_candidate_policy_config,
        )

    def forward_outputs(
        self,
        x,
        edge_index,
        edge_type,
        construct_x=None,
        hyperedge_mask_probability=0.0,
        second_view_repair_delta=None,
        second_view_repair_mask=None,
        routed_multiview_bundle=None,
        routed_highpass_bundle=None,
        batch_node_ids=None,
    ):
        x_low = self._encode_relation_view(x, edge_index, edge_type)
        x_new_base = torch.cat([x_low, x], dim=1)
        x_new, repair_stats = _apply_second_view_repair(
            x_new_base,
            repair_delta=second_view_repair_delta,
            repair_mask=second_view_repair_mask,
        )
        hyperedge_index, incident_mask, branch_stats = self._build_dynamic_hypergraph(x_new, batch_node_ids=batch_node_ids)
        hyperedge_index, incident_mask, mask_stats = mhlgc_mask_hyperedge_membership(
            hyperedge_index,
            incident_mask=incident_mask,
            num_nodes=int(x_new.size(0)),
            mask_probability=hyperedge_mask_probability,
        )
        branch_stats = dict(branch_stats or {})
        branch_stats.update(mask_stats)
        branch_stats.update(repair_stats)
        hg = _dhg_hypergraph_from_incidence(hyperedge_index, x_new.size(0), x_new.device)
        if hg is None:
            x_high = torch.zeros_like(x_low)
        else:
            x_mid = self.hgnn_layer1(x_new, hg)
            x_high = self.hgnn_layer2(x_mid, hg)
        if incident_mask is None:
            incident_mask = torch.zeros(x_low.size(0), dtype=torch.bool, device=x_low.device)
        incident_scale = incident_mask.to(dtype=x_low.dtype).unsqueeze(-1)
        x_high_base = x_high
        x_high, highpass_xhigh_stats, highpass_xhigh_aux = _apply_routed_highpass_x_high_correction(
            self,
            x_high,
            routed_highpass_bundle,
        )
        detector_views = _apply_selective_second_view_consumption(
            self,
            x_low,
            x_high,
            incident_scale,
            batch_node_ids=batch_node_ids,
        )
        hidden = detector_views["hidden"]
        logits = detector_views["logits"]
        hidden, logits, routed_stats = _apply_routed_multiview_refinement(
            self,
            hidden,
            x_low,
            x_high,
            logits,
            routed_multiview_bundle,
        )
        hidden, logits, highpass_stats, highpass_aux = _apply_routed_highpass_correction(
            self,
            hidden,
            logits,
            routed_highpass_bundle,
        )
        if highpass_xhigh_aux is not None:
            highpass_stats = highpass_xhigh_stats
            highpass_aux = highpass_xhigh_aux
        return {
            "logits": logits,
            "prob": torch.softmax(logits, dim=-1),
            "node_repr": hidden,
            "fused_x": hidden,
            "fused_x_hnn": detector_views["hidden_hnn"],
            "fused_x_lowonly": detector_views["hidden_lowonly"],
            "logits_hnn": detector_views["logits_hnn"],
            "logits_lowonly": detector_views["logits_lowonly"],
            "highorder_consumer_mask": detector_views["consumer_mask"],
            "highorder_consumer_gate": detector_views["consumer_gate"],
            "highorder_risk_score": detector_views["consumer_risk"],
            "x_low": x_low,
            "x_high": x_high,
            "x_high_base": x_high_base,
            "x_new": x_new,
            "x_new_base": x_new_base,
            "aux_features": {
                "dynamic_similarity_branch": _cpu_scalar_dict(branch_stats),
                "routed_multiview_refiner": _cpu_scalar_dict(routed_stats),
                "routed_highpass": _cpu_scalar_dict(highpass_stats),
            },
            "routed_highpass_aux": highpass_aux,
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
