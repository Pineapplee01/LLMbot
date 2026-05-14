import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, RGCNConv

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
