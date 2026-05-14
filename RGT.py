import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, num_head, out_size, hidden_size=128):
        super().__init__()
        self.num_head = num_head
        self.att_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False),
            )
            for _ in range(num_head)
        )

    def _head_output(self, attention_layer, semantic_embeddings, return_beta):
        weights = attention_layer(semantic_embeddings).mean(0)
        beta = torch.softmax(weights, dim=0)
        if return_beta:
            self.last_beta = beta.detach()
        beta = beta.expand((semantic_embeddings.shape[0],) + beta.shape)
        return (beta * semantic_embeddings).sum(1)

    def forward(self, semantic_embeddings, return_beta):
        output = self._head_output(self.att_layers[0], semantic_embeddings, return_beta)
        for attention_layer in self.att_layers[1:]:
            output += self._head_output(attention_layer, semantic_embeddings, return_beta)
        return output / self.num_head


class RGTLayer(nn.Module):
    def __init__(self, num_edge_type, in_size, out_size, layer_num_heads, semantic_head, dropout):
        super().__init__()
        self.gated = nn.Sequential(
            nn.Linear(in_size + out_size, in_size),
            nn.Sigmoid(),
        )
        self.activation = nn.ELU()
        self.gat_layers = nn.ModuleList(
            TransformerConv(
                in_channels=in_size,
                out_channels=out_size,
                heads=layer_num_heads,
                dropout=dropout,
                concat=False,
            )
            for _ in range(int(num_edge_type))
        )
        self.semantic_attention = SemanticAttention(in_size=out_size, num_head=semantic_head, out_size=out_size)

    def _relation_embedding(self, features, relation_idx, edge_index):
        transformed = self.gat_layers[relation_idx](features, edge_index.squeeze(0)).flatten(1)
        gate = self.gated(torch.cat((transformed, features), dim=1))
        return torch.tanh(transformed) * gate + features * (1 - gate)

    def forward(self, features, edge_index_list, beta=False, agg=None):
        relation_embeddings = [
            self._relation_embedding(features, relation_idx, edge_index).unsqueeze(1)
            for relation_idx, edge_index in enumerate(edge_index_list)
        ]
        semantic_embeddings = torch.cat(relation_embeddings, dim=1)

        if agg == "max":
            return semantic_embeddings.max(dim=1)[0]
        if agg == "min":
            return semantic_embeddings.min(dim=1)[0]
        if agg == "sum":
            return semantic_embeddings.sum(1)
        if agg == "mean":
            return semantic_embeddings.mean(1)
        return self.semantic_attention(semantic_embeddings, return_beta=beta)
