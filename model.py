"""
model.py
RACE-Bot-D3F Model Components

Modules:
  - TextCandidateEvidence     (ablation: multilayer evidence routing)
  - TextFinalSemanticHead     (main: final-layer calibrated text branch)
  - GraphEvidenceEncoder      (RGCN + LayerNorm + residual)
  - GraphReliabilityHead      (graph-native structural reliability)
  - D3FFusion                 (reliability-aware routing + residual T)
  - RACEBotD3F                (full model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from utils import jsd_probs, weighted_belief_fusion


# ── [ablation] Multilayer evidence routing ────────────────────────────────────
class TextCandidateEvidence(nn.Module):
    def __init__(self, layer_ids=('22', '30', '36'), in_dim=4096,
                 hid_dim=256, num_classes=2, topk=2):
        super().__init__()
        self.layer_ids = layer_ids
        self.topk = topk
        self.num_classes = num_classes
        self.adapters = nn.ModuleDict({
            l: nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(), nn.Dropout(0.1))
            for l in layer_ids
        })
        self.evidential_head = nn.Linear(hid_dim, num_classes)

    def forward(self, layer_feats, layer_prior,
                lambda_u=1.0, lambda_d=0.5, lambda_g=0.5):
        B = layer_feats[self.layer_ids[0]].size(0)
        alphas, probs, us = {}, {}, {}
        for l in self.layer_ids:
            z_l = self.adapters[l](layer_feats[l])
            e_l = F.softplus(self.evidential_head(z_l))
            alpha_l = e_l + 1.0
            S_l = alpha_l.sum(dim=1, keepdim=True)
            alphas[l] = alpha_l
            probs[l] = alpha_l / S_l
            us[l] = self.num_classes / S_l

        p_bar = torch.stack(list(probs.values()), dim=0).mean(dim=0)
        risks = []
        for l in self.layer_ids:
            r_l = lambda_u * us[l] + lambda_d * jsd_probs(probs[l], p_bar) - lambda_g * layer_prior[l]
            risks.append(r_l)
        risks_tensor = torch.cat(risks, dim=1)
        pi_raw = F.softmax(-risks_tensor, dim=1)
        topk_vals, topk_idx = torch.topk(pi_raw, k=self.topk, dim=1)
        mask = torch.zeros_like(pi_raw).scatter_(1, topk_idx, 1.0)
        pi_masked = pi_raw * mask
        pi_norm = pi_masked / (pi_masked.sum(dim=1, keepdim=True) + 1e-8)

        u_r = torch.zeros((B, 1), device=p_bar.device)
        for idx, l in enumerate(self.layer_ids):
            u_r += pi_norm[:, idx:idx+1] * jsd_probs(probs[l], p_bar)

        alpha_list = [alphas[l] for l in self.layer_ids]
        weight_list = [pi_norm[:, i:i+1] for i in range(len(self.layer_ids))]
        alpha_t = weighted_belief_fusion(alpha_list, weight_list, num_classes=self.num_classes)
        return {"alpha_text": alpha_t, "u_route": u_r, "route_weight": pi_norm}


# ── Final-layer text branch ──────────────────────────────────────────────────
class TextFinalSemanticHead(nn.Module):
    """
    Final-layer embedding → semantic classification + uncertainty estimation.
    Decouples semantic head from uncertainty adapter.
    """
    def __init__(self, in_dim=4096, hid_dim=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.semantic_adapter = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.LayerNorm(hid_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hid_dim, num_classes)
        self.uncertainty_adapter = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.LayerNorm(hid_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.temperature_head = nn.Linear(hid_dim, 1)
        self.concentration_head = nn.Linear(hid_dim, 1)

    def forward(self, final_feat, cfg_text=None):
        cfg_text = cfg_text or {}
        min_temp = float(cfg_text.get('min_temp', 1.0))
        temp_scale = float(cfg_text.get('temp_scale', 1.0))
        conc_scale = float(cfg_text.get('concentration_scale', 1.0))

        h_sem = self.semantic_adapter(final_feat)
        logits = self.classifier(h_sem)

        h_unc = self.uncertainty_adapter(final_feat)
        temp = min_temp + temp_scale * F.softplus(self.temperature_head(h_unc))
        logits_cal = logits / temp
        probs = F.softmax(logits_cal, dim=1)

        concentration = conc_scale * F.softplus(self.concentration_head(h_unc))
        alpha_t = probs * concentration + 1.0
        u_text = self.num_classes / alpha_t.sum(dim=1, keepdim=True)

        return {
            "alpha_text": alpha_t, "logits_text": logits,
            "temperature_text": temp, "concentration_text": concentration,
            "u_text": u_text,
        }


# ── Graph evidence encoder (RGCN + LN + residual) ────────────────────────────
class GraphEvidenceEncoder(nn.Module):
    """Two-layer RGCN with LayerNorm, residual connection, configurable params."""
    def __init__(self, in_dim, hid_dim, num_classes=2,
                 num_relations=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid_dim)
        self.ln0 = nn.LayerNorm(hid_dim)
        self.conv1 = RGCNConv(hid_dim, hid_dim, num_relations=num_relations)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.conv2 = RGCNConv(hid_dim, hid_dim, num_relations=num_relations)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.head = nn.Linear(hid_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, seed_idx):
        h = F.relu(self.ln0(self.proj(x)))
        h_res = h                                       # residual anchor
        h = F.relu(self.ln1(self.conv1(h, edge_index, edge_type)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.ln2(self.conv2(h, edge_index, edge_type)) + h_res
        h = F.relu(h)
        h_seed = h[seed_idx]
        e_g = F.softplus(self.head(h_seed))
        return e_g + 1.0


# ── Graph reliability head ────────────────────────────────────────────────────
class GraphReliabilityHead(nn.Module):
    """
    Graph-native structural reliability estimator.
    Input: [in_log_deg, out_log_deg, total_log_deg, neighbor_deg_var, graph_missing]
    All features are pure topology — no text embedding dependency.
    Output: r_graph in [0, 1]
    """
    def __init__(self, in_dim=5, hid_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, struct_feats):
        return self.net(struct_feats)


# ── D3F Fusion (reliability-aware routing) ────────────────────────────────────
class D3FFusion(nn.Module):
    """
    Dependency-Discounted Directional Fusion with reliability-aware routing.

    Key design: r_graph directly gates graph trust and graph evidence,
    not just post-fusion temperature. This makes the method a true
    reliability-aware routing mechanism rather than heuristic smoothing.
    """
    def __init__(self, num_classes=2, lambda_dep=0.7, tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dep = nn.Parameter(torch.tensor(lambda_dep))
        self.tau = nn.Parameter(torch.tensor(tau))
        # Residual temperature smoothing parameters
        self.a1 = nn.Parameter(torch.tensor(1.0))   # conflict
        self.a2 = nn.Parameter(torch.tensor(1.0))   # u_text
        self.a3 = nn.Parameter(torch.tensor(1.0))   # 1 - r_graph
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, alpha_t, alpha_g, u_text, r_graph):
        """
        Args:
            alpha_t : [B, K]  text Dirichlet params
            alpha_g : [B, K]  graph Dirichlet params
            u_text  : [B, 1]  text uncertainty
            r_graph : [B, 1]  graph reliability in [0, 1]
        """
        eps = 1e-8
        e_t = alpha_t - 1.0
        e_g = alpha_g - 1.0

        S_t = alpha_t.sum(dim=1, keepdim=True)
        S_g = alpha_g.sum(dim=1, keepdim=True)
        p_t = alpha_t / S_t
        p_g = alpha_g / S_g
        u_t = self.num_classes / S_t
        u_g = self.num_classes / S_g

        # (a) Sample-level dependency coefficient
        O_tg = (p_t * p_g).sum(dim=1, keepdim=True)
        rho = torch.clamp(self.lambda_dep * O_tg, min=0.0, max=1.0)

        # (b) Dependency discount on evidence
        e_t_d = (1.0 - 0.5 * rho) * e_t
        e_g_d = r_graph * (1.0 - 0.5 * rho) * e_g   # r_graph gates evidence

        # (c) Reliability-aware trust weighting
        trust_t = (1.0 - u_t) * p_t
        trust_g = r_graph * (1.0 - u_g) * p_g        # r_graph gates trust
        w_t = trust_t / (trust_t + trust_g + eps)
        w_g = 1.0 - w_t

        # (d) Fused evidence
        e_f = 2.0 * (w_t * e_t_d + w_g * e_g_d)

        # (e) Conflict
        c = jsd_probs(p_t, p_g)

        # (f) Residual temperature smoothing
        T = 1.0 + self.tau * torch.sigmoid(
            self.a1 * c + self.a2 * u_text + self.a3 * (1.0 - r_graph) + self.b
        )
        alpha_f = (e_f / T) + 1.0

        return {
            "alpha_fused": alpha_f,
            "overlap": O_tg,
            "rho": rho,
            "conflict": c,
            "temperature": T,
        }


# ── Full model ────────────────────────────────────────────────────────────────
class RACEBotD3F(nn.Module):
    """
    RACE-Bot-D3F: Risk-Aware Calibrated Evidence with
    Dependency-Discounted Directional Fusion.
    """
    def __init__(self, in_dim=4096, hid_dim=256, num_classes=2,
                 num_relations=2, dropout=0.1,
                 lambda_dep=0.7, tau=1.0,
                 struct_feat_dim=5):
        super().__init__()
        self.text_branch = TextFinalSemanticHead(
            in_dim, hid_dim, num_classes, dropout,
        )
        self.graph_branch = GraphEvidenceEncoder(
            in_dim, hid_dim, num_classes, num_relations, dropout,
        )
        self.reliability_head = GraphReliabilityHead(
            in_dim=struct_feat_dim, hid_dim=32,
        )
        self.fusion = D3FFusion(num_classes, lambda_dep, tau)

    def forward(self, batch, cfg_text=None):
        bsz = batch.batch_size
        seed_idx = torch.arange(bsz, device=batch.x.device)

        # 1. Text branch (final-layer only)
        text_out = self.text_branch(batch.q_final[:bsz], cfg_text)
        alpha_t = text_out["alpha_text"]
        u_text = text_out["u_text"]

        # 2. Graph branch
        alpha_g = self.graph_branch(
            batch.x, batch.edge_index, batch.edge_type, seed_idx,
        )

        # 3. Graph reliability from structural features
        r_graph = self.reliability_head(batch.struct_feats[:bsz])

        # 4. Reliability-aware fusion
        fusion_out = self.fusion(alpha_t, alpha_g, u_text, r_graph)

        fusion_out.update({
            "alpha_text": alpha_t,
            "alpha_graph": alpha_g,
            "u_text": u_text,
            "r_graph": r_graph,
            "u_fused": self.text_branch.num_classes
                       / fusion_out["alpha_fused"].sum(dim=1, keepdim=True),
        })
        return fusion_out
