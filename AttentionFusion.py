"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn.models import MLP
from torch_geometric.utils import scatter

class EvidentialGraphHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.evidence_layer = nn.Linear(input_dim, num_classes)
        
        # 可学习的缩放参数
        self.evidence_scale = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, x, consistency=None):
        """
        x: [Batch, Dim]
        consistency: [Batch, 1] 结构一致性分数 (预计算好)
        """
        # 1. 特征变换
        h = F.relu(self.proj(x))
        
        # 2. 原始证据 (Learnable Scale)
        raw_evidence = F.softplus(self.evidence_layer(h)) * F.softplus(self.evidence_scale)
        
        # 3. 结构一致性校准
        if consistency is not None:
            # consistency 越低(冲突)，证据越被抑制
            evidence = raw_evidence * consistency 
        else:
            evidence = raw_evidence
            
        # 4. EDL 计算
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = 2.0 / S
        
        # 5. 计算期望概率和 Logits (用于辅助 Loss)
        probs = alpha / S
        logits = torch.log(probs + 1e-9)
        
        return logits, alpha, uncertainty, probs

class VariationalTextAdapter(nn.Module):
    """
    [FIXED] Variational Information Bottleneck with Proper Uncertainty
    
    Key Fixes:
    1. Clamp logvar to prevent numerical instability
    2. Use predictive entropy (MC dropout) instead of avg variance
    3. Initialize logvar bias to reasonable values
    4. Normalize uncertainty to [0, 1] range
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3, num_classes=2):
        super().__init__()
        
        # Encoder with LayerNorm for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VIB Latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Stability: Initialize logvar to low variance
        nn.init.constant_(self.fc_logvar.bias, -1.0)
        self.register_buffer('logvar_min', torch.tensor(-10.0))
        self.register_buffer('logvar_max', torch.tensor(0.0))

        self.classifier = nn.Linear(latent_dim, num_classes)

        # Evidence head for DST uncertainty
        self.evidence_layer = nn.Linear(latent_dim, num_classes)
        self.evidence_scale = nn.Parameter(torch.tensor(2.0))
        
        # Decoder (improved)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, num_samples=10):
        
        # VIB Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar_raw = self.fc_logvar(h)
        logvar = torch.clamp(logvar_raw, self.logvar_min, self.logvar_max)
        std = torch.exp(0.5 * logvar)
        
        # Reparameterization
        if self.training:
            z_samples = mu.unsqueeze(1) + std.unsqueeze(1) * torch.randn(x.size(0), num_samples, mu.size(1), device=x.device)
            logits_samples = self.classifier(z_samples) 
            logits = torch.mean(logits_samples, dim=1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        else:
            z = mu
            logits = self.classifier(z)
            kl_loss = torch.tensor(0.0, device=x.device)
        
        # DST Evidence Process
        evidence = F.softplus(logits) * F.softplus(self.evidence_scale)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = 2.0 / S
        
        return logits, mu, alpha, uncertainty, kl_loss

class ReliabilityAwareFusion(nn.Module):
    def __init__(self, lm_dim, gnn_dim, hidden_dim, dropout=0.3):
        super().__init__()
        
        # 1. Text Expert (VIB)
        self.vib = VariationalTextAdapter(lm_dim, hidden_dim, hidden_dim, dropout)
        
        # 2. GNN Projection
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1) # Optional: slight noise to prevent overfitting
        )

        self.gnn_evidence_head = EvidentialGraphHead(hidden_dim, num_classes=2)
        
        # 3. Meta Features & Gate
        self.meta_dim = 5  
        self.meta_bn = nn.BatchNorm1d(self.meta_dim)
        
        # Gate Network
        self.gate_net = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def compute_conflict_aware_metadata(self, prob_lm, prob_gnn, u_text, homophily, degree):
        
        # JSD Calculation
        m = 0.5 * (prob_lm + prob_gnn)
        kl_lm = F.kl_div(m.clamp(min=1e-7).log(), prob_lm.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
        kl_gnn = F.kl_div(m.clamp(min=1e-7).log(), prob_gnn.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
        jsd = 0.5 * (kl_lm + kl_gnn)
        
        # Graph Entropy
        entropy_gnn = -torch.sum(prob_gnn * torch.log(prob_gnn + 1e-9), dim=1, keepdim=True) / 0.693
        conf_lm = prob_lm.max(dim=1, keepdim=True).values
        conf_gnn = prob_gnn.max(dim=1, keepdim=True).values
        conf_gap = conf_lm - conf_gnn
        
        # Construct Feature Vector (Dim = 5)
        meta_features = torch.cat([
            u_text, entropy_gnn, homophily, jsd, conf_gap
        ], dim=1)
        
        return meta_features

    def forward(self, lm_emb, gnn_emb, homophily, degree, consistency, **kwargs):
        # 1. Text Forward
        logits_lm, z_text, alpha_text, uncertainty_text, kl_loss_tensor = self.vib(lm_emb)
        probs_lm = F.softmax(logits_lm, dim=1)
        
        # 2. GNN Forward
        z_gnn = self.gnn_proj(gnn_emb)

        logits_gnn, alpha_gnn, u_graph_evidential, probs_gnn = self.gnn_evidence_head(z_gnn, consistency)
        
        # 3. Meta Features
        with torch.no_grad():
            meta_features = self.compute_conflict_aware_metadata(
                probs_lm.detach(), 
                probs_gnn.detach(), 
                uncertainty_text.detach(), 
                homophily, 
                degree
            )

        if self.training and meta_features.shape[0] > 1:
            meta_features = self.meta_bn(meta_features)
        
        # 4. Gate Calculation
        beta = self.gate_net(meta_features)
        
        # 5. Fusion
        z_fused = beta * z_text + (1 - beta) * z_gnn
        z_fused = self.norm(z_fused + self.dropout(z_fused))
        logits = self.classifier(z_fused)
        
        return {
            "logits": logits,
            "logits_lm": logits_lm,
            "logits_gnn": logits_gnn,     # 用于辅助分类 Loss
            "probs_gnn": probs_gnn,       # 用于分析
            "gate_value": beta,
            "u_text": uncertainty_text,
            "u_graph": u_graph_evidential,# 真实的结构感知不确定性
            "meta_features": meta_features,
            "alpha_text": alpha_text,
            "alpha_gnn": alpha_gnn,       # 用于 EDL Loss
            "kl_loss": kl_loss_tensor
        }

    # 保持 get_structure_consistency_loss 不变...
    def get_structure_consistency_loss(self, gate_value, homophily, u_text):
        target_graph_trust = (homophily > 0.8) | (u_text > 0.7)
        target_text_trust = (homophily < 0.4) & (u_text < 0.2)
        loss = torch.tensor(0.0, device=gate_value.device)
        if target_graph_trust.any():
            loss += (gate_value[target_graph_trust] ** 2).mean()
        if target_text_trust.any():
            loss += ((1 - gate_value[target_text_trust]) ** 2).mean()
        return loss * 0.1