"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn.models import MLP

class EvidentialGraphHead(nn.Module):
    """
    [NEW] 图模态的证据分类头 (替换原有的 Linear + Softmax)
    """
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        # Evidence Layer: 输出非负的证据
        self.evidence_layer = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # 1. 特征变换
        x = F.relu(self.proj(x))
        
        # 2. 获取证据 (Softplus + 缩放)
        # 缩放因子 10.0 用于防止证据过小，与文本模态保持一致
        evidence = F.softplus(self.evidence_layer(x)) * 10.0
        
        # 3. 计算 Dirichlet 参数
        alpha = evidence + 1
        
        # 4. 计算不确定性
        S = torch.sum(alpha, dim=1, keepdim=True)
        K = alpha.shape[1]
        uncertainty = K / S
        
        # 5. 计算用于分类的 Logits (基于期望概率)
        prob = alpha / S
        logits = torch.log(prob + 1e-9)
        
        return logits, alpha, uncertainty

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

        # Evidence head for DST uncertainty
        self.evidence_layer = nn.Linear(latent_dim, num_classes)
        
        # Decoder (improved)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        
        # VIB Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar_raw = self.fc_logvar(h)
        logvar = torch.clamp(logvar_raw, self.logvar_min, self.logvar_max)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        # DST Evidence Process
        evidence_scale = 10.0
        evidence = F.softplus(self.evidence_layer(z)) 
        alpha = evidence * evidence_scale + 1

        # Epistemic Uncertainty
        S = torch.sum(alpha, dim=1, keepdim=True)
        K = alpha.shape[1]
        uncertainty = K / S

        uncertainty = uncertainty ** 0.5

        # Auxiliary Outputs
        probs = alpha / S
        logits = torch.log(probs + 1e-9)

        # VIB KL Loss (Regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        return logits, z, alpha, uncertainty, kl_loss

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

        self.gnn_classifier = nn.Linear(hidden_dim, 2)
        self.gnn_evidence_head = EvidentialGraphHead(hidden_dim, num_classes=2)
        
        # 3. Meta Features & Gate
        # [Critical] 确保这里的维度与下方 compute_conflict_aware_metadata 返回的一致
        # 当前我们使用 5 个特征: u_text, entropy_gnn, homophily, jsd, conf_gap
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

    def compute_conflict_aware_metadata(self, logits_lm, logits_gnn, u_text, homophily, degree):
        prob_lm = F.softmax(logits_lm, dim=1)
        prob_gnn = F.softmax(logits_gnn, dim=1)
        
        # JSD Calculation
        m = 0.5 * (prob_lm + prob_gnn)
        kl_lm = F.kl_div(m.clamp(min=1e-7).log(), prob_lm.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
        kl_gnn = F.kl_div(m.clamp(min=1e-7).log(), prob_gnn.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
        jsd = 0.5 * (kl_lm + kl_gnn)
        
        # Graph Entropy
        entropy_gnn = -torch.sum(prob_gnn * torch.log(prob_gnn + 1e-9), dim=1, keepdim=True) / 0.693
        
        # Confidence Gap
        conf_lm = prob_lm.max(dim=1, keepdim=True).values
        conf_gnn = prob_gnn.max(dim=1, keepdim=True).values
        conf_gap = conf_lm - conf_gnn
        
        # Construct Feature Vector (Dim = 5)
        meta_features = torch.cat([
            u_text,          # Text Uncertainty
            entropy_gnn,     # Graph Uncertainty
            homophily,       # Structural Reliability
            jsd,             # Conflict
            conf_gap         # Relative Confidence
        ], dim=1)
        
        return meta_features

    def forward(self, lm_emb, gnn_emb, homophily, degree, **kwargs):
        # 1. Text Forward
        logits_lm, z_text, alpha_text, uncertainty_text, kl_loss_tensor = self.vib(lm_emb)
        
        # 2. GNN Forward
        z_gnn = self.gnn_proj(gnn_emb)
        logits_gnn, alpha_gnn, u_graph_evidential = self.gnn_evidence_head(z_gnn)
        
        # 3. Meta Features
        with torch.no_grad():
            meta_features = self.compute_conflict_aware_metadata(
                logits_lm.detach(), 
                logits_gnn.detach(), 
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
            "logits_gnn": logits_gnn,
            "gate_value": beta,
            "u_text": uncertainty_text,
            "u_graph": u_graph_evidential, # [FIXED] 返回真实值供分析
            "meta_features": meta_features,
            "alpha_text": alpha_text,
            "alpha_gnn": alpha_gnn,        # [NEW] 必须返回给 Trainer 计算 Loss
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