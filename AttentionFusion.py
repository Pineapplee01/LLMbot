"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP


class CrossAttentionFusion(nn.Module):
    """
    Qwenbot Fusion Module: SeGA (Structure-Enhanced Graph Attention)
    """
    def __init__(self, lm_dim, gnn_dim, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Projectors
        # IMPORTANT: lm_dim here will be 4096 (Raw Qwen)
        self.lm_proj = nn.Sequential(
            nn.Linear(lm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # gnn_dim will be whatever the GNN outputs (usually 256)
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Cross-Attention (Query=Text, Key=Graph)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Gating
        self.gate_net = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, lm_features, gnn_features):
        """
        lm_features:  [Batch, 4096] (Raw Qwen)
        gnn_features: [Batch, Hidden] (GNN Output)
        """
        # A. Projection
        h_text = self.lm_proj(lm_features)    # [B, 4096] -> [B, 256]
        h_graph = self.gnn_proj(gnn_features) # [B, Hidden] -> [B, 256]
        
        # B. Prepare for Attention [Batch, Seq=1, Dim]
        query = h_text.unsqueeze(1)
        key = h_graph.unsqueeze(1)
        value = h_graph.unsqueeze(1)

        # C. Attention
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = attn_out.squeeze(1)
        
        # D. Gated Residual
        concat = torch.cat([h_text, attn_out], dim=-1)
        alpha = torch.sigmoid(self.gate_net(concat))
        
        h_fused = h_text + (alpha * attn_out)
        h_fused = self.norm(h_fused)
        h_fused = self.dropout(h_fused)

        # --- THE FIX: Normalization ---
        # Projects vectors to Unit Sphere. Struct Loss will drop from 250 -> 1.0
        norm_fusion = F.normalize(h_fused, p=2, dim=-1)
        
        # E. Classify
        logits = self.classifier(norm_fusion)
        
        # Return logits for CE Loss, norm_fusion for Structural Loss
        return logits, norm_fusion

class BiDirectionalAttentionFusion(nn.Module):
    """
    双向注意力融合模块
    同时进行LM→GNN和GNN→LM的注意力计算
    """
    def __init__(self, lm_dim, gnn_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.lm_dim = lm_dim
        self.gnn_dim = gnn_dim
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.lm_proj = nn.Linear(lm_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # LM → GNN 注意力
        self.lm_to_gnn_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # GNN → LM 注意力
        self.gnn_to_lm_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm_lm = nn.LayerNorm(hidden_dim)
        self.norm_gnn = nn.LayerNorm(hidden_dim)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 融合门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim // 2,
            out_channels=2,
            num_layers=2,
            dropout=dropout
        )
        
    def forward(self, lm_features, gnn_features):
        """
        Args:
            lm_features: (batch_size, lm_dim)
            gnn_features: (batch_size, gnn_dim)
        Returns:
            logits: (batch_size, 2)
            fused_features: (batch_size, hidden_dim)
        """
        batch_size = lm_features.size(0)
        
        # 投影
        lm_proj = self.lm_proj(lm_features).unsqueeze(1)  # (batch, 1, hidden)
        gnn_proj = self.gnn_proj(gnn_features).unsqueeze(1)  # (batch, 1, hidden)
        
        # LM → GNN 注意力
        lm_attended, _ = self.lm_to_gnn_attn(
            query=lm_proj,
            key=gnn_proj,
            value=gnn_proj
        )
        lm_attended = self.norm_lm(lm_proj + lm_attended)
        
        # GNN → LM 注意力
        gnn_attended, _ = self.gnn_to_lm_attn(
            query=gnn_proj,
            key=lm_proj,
            value=lm_proj
        )
        gnn_attended = self.norm_gnn(gnn_proj + gnn_attended)
        
        # 移除序列维度
        lm_attended = lm_attended.squeeze(1)
        gnn_attended = gnn_attended.squeeze(1)
        
        # 门控融合
        gate_input = torch.cat([lm_attended, gnn_attended], dim=-1)
        gate_value = self.gate(gate_input)
        
        fused = gate_value * lm_attended + (1 - gate_value) * gnn_attended
        fused = self.norm_fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits, fused

class SimpleAttentionFusion(nn.Module):
    """
    简单的注意力融合模块
    使用加权求和的方式融合LM和GNN特征
    """
    def __init__(self, lm_dim, gnn_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # 投影到相同维度
        self.lm_proj = nn.Linear(lm_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 分类器
        self.classifier = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim // 2,
            out_channels=2,
            num_layers=2,
            dropout=dropout
        )
        
    def forward(self, lm_features, gnn_features):
        """
        Args:
            lm_features: (batch_size, lm_dim)
            gnn_features: (batch_size, gnn_dim)
        Returns:
            logits: (batch_size, 2)
            fused_features: (batch_size, hidden_dim)
        """
        # 投影
        lm_proj = self.lm_proj(lm_features)  # (batch, hidden)
        gnn_proj = self.gnn_proj(gnn_features)  # (batch, hidden)
        
        # 计算注意力权重
        concat_features = torch.cat([lm_proj, gnn_proj], dim=-1)
        attn_weights = self.attention(concat_features)  # (batch, 2)
        
        # 加权融合
        fused = attn_weights[:, 0:1] * lm_proj + attn_weights[:, 1:2] * gnn_proj
        
        # 分类
        logits = self.classifier(fused)
        
        return logits, fused

class AdaptiveAttentionFusion(nn.Module):
    """
    自适应注意力融合模块
    根据样本特性动态调整LM和GNN的融合权重
    """
    def __init__(self, lm_dim, gnn_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 特征投影
        self.lm_proj = nn.Linear(lm_dim, hidden_dim)
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 分类器
        self.classifier = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim // 2,
            out_channels=2,
            num_layers=2,
            dropout=dropout
        )
        
    def forward(self, lm_features, gnn_features):
        """
        Args:
            lm_features: (batch_size, lm_dim)
            gnn_features: (batch_size, gnn_dim)
        Returns:
            logits: (batch_size, 2)
            fused_features: (batch_size, hidden_dim)
            importance_weights: (batch_size, 2) LM和GNN的重要性权重
        """
        # 投影
        lm_proj = self.lm_proj(lm_features)
        gnn_proj = self.gnn_proj(gnn_features)
        
        # 堆叠为序列: (batch, 2, hidden)
        stacked = torch.stack([lm_proj, gnn_proj], dim=1)
        
        # 自注意力
        attn_out, attn_weights = self.self_attention(
            query=stacked,
            key=stacked,
            value=stacked
        )
        
        # 残差连接
        attn_out = self.norm(stacked + attn_out)
        
        # 评估特征重要性
        concat_for_importance = torch.cat([lm_proj, gnn_proj], dim=-1)
        importance_weights = self.importance_net(concat_for_importance)  # (batch, 2)
        
        # 加权求和
        weighted_lm = importance_weights[:, 0:1].unsqueeze(1) * attn_out[:, 0, :]
        weighted_gnn = importance_weights[:, 1:2].unsqueeze(1) * attn_out[:, 1, :]
        fused = weighted_lm + weighted_gnn
        
        # 分类
        logits = self.classifier(fused)
        
        return logits, fused, importance_weights
