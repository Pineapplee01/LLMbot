import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class SemanticAttention(nn.Module):
    """
    (From Official Code)
    Learns weights to combine different relation types (Followers vs Following).
    """
    def __init__(self, in_size, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.num_head = num_head
        self.att_layers = nn.ModuleList()
        
        for i in range(num_head):
            self.att_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1, bias=False)
                )
            )
       
    def forward(self, z):
        # z shape: [Batch_Size (N), Num_Relations (R), Hidden_Dim (D)]
        multi_head_outputs = []

        for i in range(self.num_head):
            # 1. Calculate Attention Scores
            # w shape: [N, R, 1] 
            # [Fix]: 去掉了原代码中的 .mean(0)，保留 N 维度
            w = self.att_layers[i](z)
            
            # 2. Softmax over relations
            # [Fix]: dim=1 表示在 Relation 维度归一化
            beta = torch.softmax(w, dim=1) 
            
            # 3. Weighted Sum
            # beta * z -> [N, R, D]
            # sum(1) -> [N, D] (聚合不同关系的信息)
            weighted_sum = (beta * z).sum(1)
            multi_head_outputs.append(weighted_sum)

        # 4. Average Multi-head outputs
        # output shape: [N, D]
        output = sum(multi_head_outputs) / self.num_head
        
        return output

class RGTLayer(nn.Module):
    """
    (From Official Code)
    Applies TransformerConv per relation, then aggregates via SemanticAttention.
    """
    def __init__(self, num_edge_type, in_size, out_size, layer_num_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        
        # Gating Mechanism (The "Beta" in their code)
        self.gated = nn.Sequential(
            nn.Linear(in_size + out_size, in_size),
            nn.Sigmoid()
        )

        self.gat_layers = nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.gat_layers.append(
                TransformerConv(
                    in_channels=in_size, 
                    out_channels=out_size, 
                    heads=layer_num_heads, 
                    dropout=dropout, 
                    concat=False # Keep dimensions constant
                )
            )
        
        self.semantic_attention = SemanticAttention(in_size=out_size, num_head=semantic_head)

    def forward(self, features, edge_index_list):
        # 1. Process First Relation
        outputs = []
        
        # 2. Process Remaining Relations
        for i in range(len(edge_index_list)):
            # GNN Aggregation for relation i
            # 如果该关系没有边，TransformerConv 通常能处理空 edge_index，但为了安全建议在上层处理好
            u = self.gat_layers[i](features, edge_index_list[i])
            
            # Gated Residual Connection
            # a: [N, D]
            a = self.gated(torch.cat((u, features), dim=1))
            
            # Semantic Embedding for relation i
            # Formula: Tanh(New) * Gate + Old * (1-Gate)
            out_r = torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))
            outputs.append(out_r)
            
        # 2. Stack Relations
        # [Fix]: 使用 stack 替代原来的循环 cat，效率更高且不易出错
        # z shape: [N, Num_Relations, D]
        semantic_embeddings = torch.stack(outputs, dim=1)
            
        # 3. Aggregate Relations using Node-wise Attention
        return self.semantic_attention(semantic_embeddings)