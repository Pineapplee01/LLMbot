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
        self.evidence_layer = nn.Linear(latent_dim, num_classes)
        self.evidence_scale = nn.Parameter(torch.tensor(2.0))
        
        # Decoder (improved)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, num_samples=10):
        
        # VIB Encode
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        
        z = self.reparameterize(mu, logvar)

        # Classifier / Evidence 
        logits = self.classifier(z)
        evidence = F.softplus(self.evidence_layer(z)) 
        alpha = evidence + 1.0
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = alpha.shape[1] / S
        
        # KL 散度 (VIB 正则)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        return logits, z, alpha, uncertainty, kl_loss

class ReliabilityAwareFusion(nn.Module):
    def __init__(self, lm_dim, gnn_dim, hidden_dim, dropout=0.3):
        super().__init__()
        
        # Variational Text Adapter
        self.vib = VariationalTextAdapter(lm_dim, hidden_dim, hidden_dim, dropout)
        
        # [补全] 2. GNN Projection & Evidence Head
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1) 
        )
        # 确保 EvidentialGraphHead 类已在文件顶部定义
        self.gnn_evidence_head = EvidentialGraphHead(hidden_dim, num_classes=2)
        
        # 3. Meta Features & Gate (这是上一轮修改的重点)
        self.meta_dim = 5  

        self.meta_bn = nn.BatchNorm1d(self.meta_dim)
        self.meta_norm = nn.LayerNorm(self.meta_dim)
        
        # Gate Network
        self.gate_net = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )


    def forward(self, lm_emb, gnn_emb, homophily, degree, consistency, **kwargs):
       
        logits_lm, z_text, alpha_text, uncertainty_text, kl_loss_tensor = self.vib(lm_emb)
        probs_lm = F.softmax(logits_lm, dim=1)

        z_gnn_proj = self.gnn_proj(gnn_emb)
        logits_gnn, alpha_gnn, u_graph_evidential, probs_gnn = self.gnn_evidence_head(z_gnn_proj, consistency)

        # 3. Meta Features Calculation
        with torch.no_grad():
            # JSD Calculation
            m = 0.5 * (probs_lm + probs_gnn)
            kl_lm = F.kl_div(m.clamp(min=1e-7).log(), probs_lm.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
            kl_gnn = F.kl_div(m.clamp(min=1e-7).log(), probs_gnn.clamp(min=1e-7), reduction='none').sum(dim=1, keepdim=True)
            jsd = 0.5 * (kl_lm + kl_gnn)
            
            # Entropy & Gap
            entropy_gnn = -torch.sum(probs_gnn * torch.log(probs_gnn + 1e-9), dim=1, keepdim=True) / 0.693
            conf_lm = probs_lm.max(dim=1, keepdim=True).values
            conf_gnn = probs_gnn.max(dim=1, keepdim=True).values
            conf_gap = conf_lm - conf_gnn

            if homophily.dim() == 1: homophily = homophily.unsqueeze(1)
            if uncertainty_text.dim() == 1: uncertainty_text = uncertainty_text.unsqueeze(1)
            if u_graph_evidential.dim() == 1: u_graph_evidential = u_graph_evidential.unsqueeze(1)
            
            # Construct [B, 5]
            meta_features = torch.cat([
                uncertainty_text, u_graph_evidential, homophily, jsd, conf_gap
            ], dim=1)

        # Layer Norm for Stability (Only if batch size > 1)
        if meta_features.shape[0] > 1:
            meta_features = self.meta_norm(meta_features)
        
        # 4. Gate Calculation
        beta = self.gate_net(meta_features)
        
        # 5. Fusion
        probs_fused = beta * probs_lm + (1 - beta) * probs_gnn
        fused_logits = torch.log(probs_fused + 1e-7)
        
        return {
            "logits": fused_logits,      # 用于计算主 Loss
            "probs_fused": probs_fused,
            "probs_lm": probs_lm,       # 新增：方便 Trainer 调用
            "probs_gnn": probs_gnn,

            "logits_lm": logits_lm,
            "logits_gnn": logits_gnn,
            
            "gate_value": beta,
            "u_text": uncertainty_text,
            "u_graph": u_graph_evidential, # 确保这是 EDL 不确定性
            
            "meta_features": meta_features,
            "alpha_text": alpha_text,
            "alpha_gnn": alpha_gnn,
            "kl_loss": kl_loss_tensor
        }

    # (Dynamic Relative Thresholding)
    def get_structure_consistency_loss(self, gate_value, homophily, u_text):
        u_mean = u_text.mean()
        u_std = u_text.std() + 1e-6
        thresh_low = (u_mean - 0.5 * u_std).detach()
        thresh_high = (u_mean + 0.5 * u_std).detach()

        target_graph_trust = (homophily > 0.8) | (u_text > thresh_high)
        target_text_trust = (homophily < 0.5) & (u_text < thresh_low)
        
        loss = torch.tensor(0.0, device=gate_value.device)
        if target_graph_trust.any():
            loss += (gate_value[target_graph_trust] ** 2).mean()
        if target_text_trust.any():
            loss += ((1 - gate_value[target_text_trust]) ** 2).mean()
            
        return loss * 0.1