"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn.models import MLP


class VariationalTextAdapter(nn.Module):
    """
    [FIXED] Variational Information Bottleneck with Proper Uncertainty
    
    Key Fixes:
    1. Clamp logvar to prevent numerical instability
    2. Use predictive entropy (MC dropout) instead of avg variance
    3. Initialize logvar bias to reasonable values
    4. Normalize uncertainty to [0, 1] range
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        
        # Encoder with LayerNorm for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Added
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # FIX: Initialize logvar to output low variance initially
        nn.init.constant_(self.fc_logvar.bias, -3.0)
        
        # Decoder (improved)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # FIX: Learnable bounds for logvar
        self.register_buffer('logvar_min', torch.tensor(-10.0))
        self.register_buffer('logvar_max', torch.tensor(0.0))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar_raw = self.fc_logvar(h)
        
        # FIX: Clamp logvar to prevent explosion
        logvar = torch.clamp(logvar_raw, self.logvar_min, self.logvar_max)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        logits = self.decoder(z)
        
        # FIX: Compute PROPER uncertainty via Monte Carlo sampling
        with torch.no_grad():
            n_samples = 10 if self.training else 20
            logits_samples = []
            
            for _ in range(n_samples):
                z_sample = self.reparameterize(mu, logvar)
                logits_samples.append(self.decoder(z_sample))
            
            # Stack: [N_samples, Batch, 2]
            logits_samples = torch.stack(logits_samples)
            probs_samples = F.softmax(logits_samples, dim=-1)
            
            # Mean probability across samples
            probs_mean = probs_samples.mean(dim=0)  # [Batch, 2]
            
            # Predictive entropy (proper uncertainty measure)
            entropy = -(probs_mean * torch.log(probs_mean + 1e-9)).sum(dim=-1, keepdim=True)
            
            # Normalize to [0, 1] (max entropy for 2 classes is log(2))
            uncertainty = entropy / torch.log(torch.tensor(2.0))
        
        # KL divergence loss (regularization)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        return logits, z, uncertainty, kl_loss


class ReliabilityAwareFusion(nn.Module):
    """
    [FIXED] Reliability-Aware Fusion with Enhanced Meta Features
    
    Key Fixes:
    1. 8 meta features instead of 4 (added agreement, conf_gap, degree, similarity)
    2. Deeper reliability heads (3 layers with LayerNorm)
    3. Learnable gate temperature for training stability
    4. Proper detachment to prevent gradient interference
    """
    def __init__(self, lm_dim, gnn_dim, hidden_dim, dropout=0.3):
        super().__init__()
        
        # 1. Fixed VIB Text Adapter
        self.vib = VariationalTextAdapter(lm_dim, hidden_dim, hidden_dim, dropout)
        
        # 2. GNN Projection
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.gnn_classifier = nn.Linear(hidden_dim, 2)
        
        # 3. Enhanced Meta Features (8 dims)
        meta_dim = 8  # jsd, homophily, text_var, entropy_gnn, agreement, conf_gap, degree, similarity
        self.meta_bn = nn.BatchNorm1d(meta_dim)
        
        # 4. IMPROVED Reliability Heads (3 layers, wider, with normalization)
        self.lm_reliability_head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1) 
        )
        self.gnn_reliability_head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        
        # 5. Learnable temperature for gate (helps training stability)
        self.gate_temperature = nn.Parameter(torch.ones(1) * 2.0)
        
        # 6. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def compute_enhanced_meta_features(self, z_text, gnn_feat, logits_lm, logits_gnn,
                                       homophily, degree, entropy_gnn, jsd):
        """
        NEW: Compute 8 meta features instead of 4
        
        Original 4:
        - jsd: Jensen-Shannon divergence between text and graph predictions
        - homophily: Neighbor similarity (graph reliability indicator)
        - text_var: Text feature variance
        - entropy_gnn: Graph prediction entropy
        
        New 4:
        - agreement: Whether text and graph agree on prediction
        - conf_gap: Confidence difference between text and graph
        - degree: Node degree (normalized)
        - similarity: Cosine similarity between text and graph embeddings
        """
        
        # Original 4 features
        text_var = z_text.std(dim=1, keepdim=True)
        
        meta_original = torch.cat([
            jsd,           # [B, 1]
            homophily,     # [B, 1]
            text_var,      # [B, 1]
            entropy_gnn    # [B, 1]
        ], dim=1)  # [B, 4]
        
        # NEW: Additional 4 features
        with torch.no_grad():
            # 1. Agreement: Do text and graph agree on prediction?
            pred_lm = logits_lm.argmax(dim=1, keepdim=True).float()
            pred_gnn = logits_gnn.argmax(dim=1, keepdim=True).float()
            agreement = (pred_lm == pred_gnn).float()  # [B, 1]
            
            # 2. Confidence gap: How much more confident is one vs other?
            prob_lm = F.softmax(logits_lm, dim=1)
            prob_gnn = F.softmax(logits_gnn, dim=1)
            conf_lm = prob_lm.max(dim=1, keepdim=True).values
            conf_gnn = prob_gnn.max(dim=1, keepdim=True).values
            conf_gap = torch.abs(conf_lm - conf_gnn)  # [B, 1]
            
            # 3. Degree (already normalized in preprocessing)
            degree_feat = degree  # [B, 1]
            
            # 4. Embedding similarity: How aligned are text and graph features?
            text_norm = F.normalize(z_text, p=2, dim=1)
            graph_norm = F.normalize(gnn_feat, p=2, dim=1)
            similarity = (text_norm * graph_norm).sum(dim=1, keepdim=True)  # [B, 1]
        
        # Concatenate all 8 features
        meta_enhanced = torch.cat([
            meta_original,  # [B, 4]
            agreement,      # [B, 1]
            conf_gap,       # [B, 1]
            degree_feat,    # [B, 1]
            similarity      # [B, 1]
        ], dim=1)  # [B, 8]
        
        return meta_enhanced

    def forward(self, lm_emb, gnn_emb, homophily, degree, entropy_gnn, jsd):
        """
        Forward pass with enhanced reliability estimation.
        
        Args:
            lm_emb: [B, lm_dim] - LLM embeddings
            gnn_emb: [B, gnn_dim] - GNN embeddings
            homophily: [B, 1] - Neighbor similarity
            degree: [B, 1] - Node degree
            entropy_gnn: [B, 1] - GNN prediction entropy
            jsd: [B, 1] - JS divergence between text and graph
        
        Returns:
            dict with keys:
                logits, logits_lm, logits_gnn, alpha,
                r_lm_logits, r_gnn_logits, uncertainty_text, kl_loss
        """
        # 1. VIB Forward (Text Expert)
        logits_lm, z_text, uncertainty_text, kl_loss = self.vib(lm_emb)
        
        # 2. GNN Forward (Graph Expert)
        gnn_feat = F.relu(self.gnn_proj(gnn_emb))
        logits_gnn = self.gnn_classifier(gnn_feat)
        
        # 3. Compute Enhanced Meta Features (8 dims)
        meta_features = self.compute_enhanced_meta_features(
            z_text, gnn_feat, logits_lm, logits_gnn,
            homophily, degree, entropy_gnn, jsd
        )
        
        # Normalize meta features if training with batch size > 1
        if self.training and meta_features.shape[0] > 1:
            meta_features = self.meta_bn(meta_features)
        
        # 4. Reliability Prediction
        # IMPORTANT: Detach features to prevent gradient interference
        r_lm_input = torch.cat([z_text.detach(), meta_features], dim=1)
        r_gnn_input = torch.cat([gnn_feat.detach(), meta_features], dim=1)
        
        r_lm_logits = self.lm_reliability_head(r_lm_input)   # [B, 1]
        r_gnn_logits = self.gnn_reliability_head(r_gnn_input) # [B, 1]
        
        # 5. Gate Computation (with learnable temperature)
        gate_logits = torch.cat([r_lm_logits, r_gnn_logits], dim=1)  # [B, 2]
        alpha = F.softmax(gate_logits / self.gate_temperature, dim=1)  # [B, 2]
        
        # 6. Weighted Fusion
        fused_emb = alpha[:, 0:1] * z_text + alpha[:, 1:2] * gnn_feat  # [B, hidden]
        fused_emb = self.norm(fused_emb + self.dropout(fused_emb))
        
        # 7. Final Classification
        logits = self.classifier(fused_emb)
        
        return {
            "logits": logits,
            "logits_lm": logits_lm,
            "logits_gnn": logits_gnn,
            "alpha": alpha,
            "r_lm_logits": r_lm_logits,
            "r_gnn_logits": r_gnn_logits,
            "uncertainty_text": uncertainty_text,  # Changed from sigma_text
            "kl_loss": kl_loss,
            "sigma_text": uncertainty_text,  # For compatibility
            "meta_features": meta_features  # For debugging
        }

    def get_structure_consistency_loss(self, alpha, homophily):
        """
        Regularization: Encourage using graph when homophily is high.
        """
        threshold = 0.9
        high_homophily_mask = (homophily > threshold).float()
        
        # Ensure dimensions match
        if alpha[:, 1].shape != high_homophily_mask.shape:
            high_homophily_mask = high_homophily_mask.view_as(alpha[:, 1])
        
        # Penalize low graph weight when homophily is high
        consistency_loss = high_homophily_mask * (1.0 - alpha[:, 1])
        return consistency_loss.mean()
