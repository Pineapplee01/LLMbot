"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
    [Expert Modified] Added numerical stability and divide-by-zero protection.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [batch_size, dim] or [batch_size, n_views, dim]
            labels: [batch_size]
            mask: [batch_size, batch_size]
        """
        device = features.device

        # 1. 维度适配：如果输入是 [Batch, Dim]，自动升维到 [Batch, 1, Dim]
        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        
        # 2. 构造 Mask
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # 将多视图解绑并拼接：[Batch * Views, Dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # 3. 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 4. 数值稳定性处理 (Log-Sum-Exp Trick)
        # 减去最大值防止 exp 溢出
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 5. 排除自身 (Self-Contrast masking)
        # 将对角线（自己对比自己）的位置设为极小值或在 mask 中剔除
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 6. 计算 Log-Prob
        exp_logits = torch.exp(logits) * logits_mask
        # sum(1) 是分母：所有负样本 + 除自己外的正样本 的指数和
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6) # 加 1e-6 防止 log(0)

        # 7. 计算 Mean Log-Likelihood
        # 分母是每个样本拥有的正样本数量 (Batch内同类数量 - 1)
        # ### Expert Fix: 增加 1e-6 防止除以 0 (当 Batch 内该类只有一个样本时)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # 8. Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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
        nn.init.constant_(self.gate_net.bias, 2.0)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # 128维通常对对比学习效果最好
        )
        
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
        
        if self.training:
            prob = torch.rand(1).item()
            
            # 30% 概率：彻底丢弃文本，纯靠 GNN (逼它学图)
            if prob < 0.15:
                h_text = torch.zeros_like(h_text)
            
            # 30% 概率：彻底丢弃图，纯靠文本 (保持文本能力)
            elif prob < 0.30:
                h_graph = torch.zeros_like(h_graph)

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

        contrast_feat = self.contrastive_head(h_fused)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        logits = self.classifier(h_fused)
        
        # Return logits for CE Loss, 
        return logits, contrast_feat, alpha

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
