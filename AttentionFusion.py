"""
Attention-based Fusion Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

class OrthogonalityLoss(nn.Module):
    """
    强制 Private 特征与 Shared 特征正交，保证它们学到不一样的知识。
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, shared, private):
        
        # Normalize
        shared = F.normalize(shared, dim=1)
        private = F.normalize(private, dim=1)
        
        # Correlation matrix
        correlation = torch.mm(shared.t(), private) 
        
        # Minimize the Frobenius norm of correlation
        loss = torch.norm(correlation, p='fro')
        return loss

class DisentangledMetaFusion(nn.Module):
    """
    [SeGA v4.0] Disentangled Fusion with Metadata-Aware Gating
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, meta_dim=3, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # 1. 共享空间投影 (Shared Projectors)
        self.lm_shared = nn.Sequential(
            nn.Linear(lm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.gnn_shared = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. 私有空间投影 (Private Projectors)
        # GNN 的私有特征 (拓扑结构)
        self.gnn_private = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Metadata 增强门控 (Context-Aware Gating)
        # Input: [Shared_LM, Shared_GNN, Metadata]
        self.gate_input_dim = hidden_dim * 2 + meta_dim
        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.gate_temperature = nn.Parameter(torch.ones(1)) # Learnable temperature
        
        # 4. 主分类器 (使用 Shared + Private)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Shared_Mix + Private_GNN
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # 5. 辅助分类头 (Auxiliary Heads)
        # 确保 Private GNN 特征具有判别力
        self.aux_gnn_head = nn.Linear(hidden_dim, 2)
        
        # SupCon Head
        self.supcon_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, lm_features, gnn_features, metadata=None):
        # 处理 Metadata
        if metadata is None:
            # 默认填充 0
            metadata = torch.zeros(lm_features.size(0), 3, device=lm_features.device)
            
        # A. 投影
        h_lm_s = self.lm_shared(lm_features)
        h_gnn_s = self.gnn_shared(gnn_features)
        h_gnn_p = self.gnn_private(gnn_features)
        
        # B. 门控决策
        gate_in = torch.cat([h_lm_s, h_gnn_s, metadata], dim=-1)
        gate_logits = self.gate_net(gate_in) / self.gate_temperature
        alpha = torch.sigmoid(gate_logits) # Alpha -> Trust LM Shared
        
        # C. 共享空间融合
        h_shared = alpha * h_lm_s + (1 - alpha) * h_gnn_s
        
        # D. 最终特征拼接 (Shared + Private)
        h_final = torch.cat([h_shared, h_gnn_p], dim=-1)
        
        # E. 输出
        logits = self.classifier(h_final)
        
        # F. 辅助输出
        aux_gnn_logits = self.aux_gnn_head(h_gnn_p)
        z_supcon = F.normalize(self.supcon_head(h_final), dim=1)
        
        return {
            "logits": logits,
            "logits_gnn_aux": aux_gnn_logits,
            "shared_gnn": h_gnn_s,
            "private_gnn": h_gnn_p,
            "shared_lm": h_lm_s, # 用于 Ortho loss 计算 (可选)
            "alpha": alpha,
            "z_supcon": z_supcon
        }

class DegreeAwareConfidenceFusion(nn.Module):
    """
    [SeGA v3.0] Degree-Aware Confidence Fusion Mechanism.
    Integrates Node Degree information to dynamically weight LLM vs GNN experts.
    (Inspired by LGB's MoE gating strategy).
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # 1. Feature Projection
        self.lm_proj = nn.Sequential(
            nn.Linear(lm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Degree-Aware Gating Network
        # Input: [LM_Feature; GNN_Feature; Degree_Score]
        # Degree_Score is a scalar (log-normalized degree)
        self.gate_input_dim = hidden_dim * 2 + 1
        
        self.gate_net = nn.Sequential(
            nn.Linear(self.gate_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Output scalar alpha
        )
        
        # Learnable temperature for sharpening the gate
        self.gate_temperature = nn.Parameter(torch.ones(1))
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 4. SupCon Head (for contrastive learning)
        self.supcon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, lm_features, gnn_features, degree=None):
        """
        Args:
            lm_features: (batch_size, lm_dim)
            gnn_features: (batch_size, gnn_dim)
            degree: (batch_size, 1) Normalized log-degree of nodes.
        """
        # A. Projection
        h_lm = self.lm_proj(lm_features)
        h_gnn = self.gnn_proj(gnn_features)
        
        # B. Handle Degree Input
        if degree is None:
            # Fallback if degree not provided (though Trainer should provide it)
            # Default to 0.5 (neutral structural confidence)
            degree = torch.ones(h_lm.size(0), 1, device=h_lm.device) * 0.5
            
        # C. Compute Gate Alpha
        # Concatenate features and degree
        combined = torch.cat([h_lm, h_gnn, degree], dim=-1)
        
        # Calculate raw logits and scale by temperature
        gate_logits = self.gate_net(combined) / self.gate_temperature
        
        # Alpha -> 1 means trust LM more (Logic: High sparsity/Low degree -> Trust LM)
        # Alpha -> 0 means trust GNN more
        alpha = torch.sigmoid(gate_logits)
        
        # D. Weighted Fusion
        h_fused = alpha * h_lm + (1 - alpha) * h_gnn
        
        # E. Task Outputs
        logits = self.classifier(h_fused)
        z_supcon = F.normalize(self.supcon_head(h_fused), dim=1)
        
        return logits, z_supcon, alpha

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    

class ConfidenceFusion(nn.Module):
    """
    [2025 Optimized] Confidence-Aware Semantic Alignment
    拒绝盲目的邻居平滑，专注于特征的可信度加权与类内聚类。
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # 1. 对齐投影 (Alignment Projectors)
        # 将 LLM 和 GNN 映射到同一个 Metric Space
        self.lm_proj = nn.Sequential(
            nn.Linear(lm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 置信度门控 (Confidence Gating Network)
        # 输入：[LM_proj, GNN_proj] -> 输出：Scalar alpha (0~1)
        # 决定：当前样本更应该相信文本还是相信图？
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
        
        # 3. 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # 4. 辅助：用于 SupCon 的投影头 (Projection Head)
        self.supcon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128) # 压缩到低维进行对比学习
        )

    def forward(self, lm_emb, gnn_emb):
        # A. 投影到公共空间
        h_lm = self.lm_proj(lm_emb)      # [B, 512]
        h_gnn = self.gnn_proj(gnn_emb)   # [B, 512]
        
        # B. 计算置信度 alpha
        # alpha close to 1: Trust LLM more
        # alpha close to 0: Trust GNN more
        combined = torch.cat([h_lm, h_gnn], dim=-1)
        alpha = self.confidence_net(combined) # [B, 1]
        
        # C. 动态加权融合 (Dynamic Weighted Fusion)
        # 不使用简单的相加，而是使用凸组合
        h_fused = alpha * h_lm + (1 - alpha) * h_gnn
        
        # D. 分类
        logits = self.classifier(h_fused)
        
        # E. 生成对比学习特征 (Normalize for Cosine Similarity)
        # 我们对 h_fused 进行约束，确保融合后的特征在类内紧凑
        z_supcon = F.normalize(self.supcon_head(h_fused), dim=1)
        
        return logits, z_supcon, alpha

class AlignAndEnhanceFusion(nn.Module):
    """
    [2025 SOTA Design] 跨模态对齐投影 + 分类增强架构
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, projection_dim=512, dropout=0.3):
        super().__init__()
        
        # --- 第一阶段：跨模态对齐投影器 (Alignment Projector) ---
        # 负责将 LLM 语义空间 映射到 GNN 结构空间
        self.lm_projector = nn.Sequential(
            nn.Linear(lm_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Dropout(dropout)
        )
        
        # GNN 侧投影（保持对称性，利于对比学习）
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # --- 第二阶段：自适应门控融合 ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid()
        )

        # --- 第三阶段：分类增强头 ---
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.BatchNorm1d(projection_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 2)
        )

    def forward(self, lm_features, gnn_features):
        # 1. 投影对齐
        z_lm = self.lm_projector(lm_features)      # [B, 512]
        z_gnn = self.gnn_projector(gnn_features)   # [B, 512]
        
        # 2. 门控融合 (计算残差信息)
        gate = self.fusion_gate(torch.cat([z_lm, z_gnn], dim=-1))
        fused = z_gnn + gate * (z_lm - z_gnn) # 结构为基准，文本做增量修正
        
        # 3. 分类增强
        logits = self.classifier(fused)
        
        # 返回对齐特征用于计算训练时的 Alignment Loss
        return logits, z_lm, z_gnn

class GatedModulationFusion(nn.Module):
    """
    [Expert Design] Asymmetric Bottleneck Modulation
    Philosophy: GNN is the Anchor, Text is the Modifier.
    Formula: Fused = GNN * Sigmoid(Text_Gate) + GNN
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, bottleneck_dim=64, dropout=0.3):
        super().__init__()
        
        # 1. 文本瓶颈层 (Text Bottleneck)
        # 强制将 4096 维的高维稀疏语义压缩为 64 维的核心语义
        self.text_bottleneck = nn.Sequential(
            nn.Linear(lm_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 门控生成器 (Gate Generator)
        # 将文本语义映射到 GNN 的特征空间，并生成 0-1 之间的门控信号
        self.gate_generator = nn.Sequential(
            nn.Linear(bottleneck_dim, gnn_dim),
            nn.Sigmoid() 
        )
        
        # 3. 文本残差 (Text Residual)
        # 允许部分经过筛选的文本语义直接补充进来
        self.text_residual = nn.Sequential(
            nn.Linear(bottleneck_dim, gnn_dim),
            nn.Tanh()
        )
        
        # 4. 对比学习头 (SupCon Head)
        self.contrastive_head = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim),
            nn.ReLU(),
            nn.Linear(gnn_dim, 128)
        )
        
        # 5. 分类器 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_dim // 2, 2)
        )

    def forward(self, lm_features, gnn_features):
        """
        lm_features: [B, 4096]
        gnn_features: [B, 512]
        """
        # A. 压缩文本
        t_code = self.text_bottleneck(lm_features) # [B, 64]
        
        # B. 生成门控 (Gate)
        # 每一维的数值代表：文本认为 GNN 的这个特征维度有多重要
        gate = self.gate_generator(t_code) # [B, 512]
        
        # C. 调制融合 (Modulation)
        # 核心逻辑：用文本去"重塑"图特征的形状
        # GNN * Gate: 抑制 GNN 中的噪声维度
        # Text_Residual: 补充 GNN 缺失的纯文本语义
        t_res = self.text_residual(t_code)
        fused_h = (gnn_features * gate) + (t_res * 0.5) 
        
        # D. 输出
        logits = self.classifier(fused_h)
        
        # SupCon 特征 (归一化)
        contrast_feat = self.contrastive_head(fused_h)
        contrast_feat = F.normalize(contrast_feat, dim=1)
        
        return logits, contrast_feat, gate

class ResidualFusion(nn.Module):
    """
    [Final Expert Version]: Zero-Initialized Residual Logit Ensemble
    Ensures Stage 2 starts exactly where Stage 1 ended (F1 ~0.78).
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # 1. 文本处理专家 (Text Expert)
        self.text_mlp = nn.Sequential(
            nn.Linear(lm_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim) 
        )
        
        # 2. 融合纠错模块 (Correction Module)
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim + gnn_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # 输出 Logit 的修正值
        )
        
        # 3. 门控系数 (Learnable Scalar) - 关键修改！
        # 直接初始化为 0.0，不经过 Sigmoid，允许学出负值（负修正）
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # 4. 对比学习头
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim + gnn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, lm_features, gnn_features):
        """
        Returns:
            correction_logits: The 'delta' to add to GNN logits
            contrast_feat: Combined features for SupCon
            alpha: Current weight (scalar)
        """
        # A. 提取文本特征
        h_text = self.text_mlp(lm_features) 
        
        # B. 拼接
        combined = torch.cat([gnn_features, h_text], dim=-1) 
        
        # C. 计算纠错值
        correction_logits = self.correction_head(combined)
        
        # D. 对比特征
        contrast_feat = self.contrastive_head(combined)
        contrast_feat = F.normalize(contrast_feat, dim=1)
        
        # E. 返回权重 (此时 alpha 初始为 0)
        # 我们返回 correction_logits，让 Trainer 去做: logits = GNN + alpha * correction
        return correction_logits, contrast_feat, self.alpha

class CrossAttentionFusion(nn.Module):
    """
    [Revised Architecture]: Structure-Anchored Gated Fusion
    Uses GNN features as the 'Anchor' and LLM features as 'Augmentation'.
    """
    def __init__(self, lm_dim=4096, gnn_dim=512, hidden_dim=512, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Feature Alignment (Projection)
        # Compress LLM to hidden space
        self.lm_proj = nn.Sequential(
            nn.Linear(lm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Align GNN to hidden space (if needed)
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Interaction Mechanism: GNN Querying Text
        # "Structure looks at Content"
        # We use a simplified Attention mechanism to avoid parameter explosion
        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid() # Outputs a gating weight [0, 1]
        )

        # 3. Final Classification Head
        # Input: [GNN_Original, Gated_Context] -> Concatenation for stability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), # BN is crucial for stability after fusion
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        # Contrastive Learning Head
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, lm_features, gnn_features):
        """
        Args:
            lm_features: (batch_size, lm_dim) - Frozen Qwen Embeddings
            gnn_features: (batch_size, gnn_dim) - Pre-trained GNN Features
        """
        
        # A. Projection & Alignment
        h_lm = self.lm_proj(lm_features)      # [B, hidden]
        h_gnn = self.gnn_proj(gnn_features)   # [B, hidden]
        
        # B. Interaction (Asymmetric)
        # We want to know: "How much does the text support the graph structure?"
        # Simple Element-wise interaction is often more robust than full Softmax Attention for 1-to-1 mapping
        interaction_raw = h_lm * h_gnn # Hadamard product capturing correlation
        
        # C. Gating
        # Concatenate both to decide importance
        combined_for_gate = torch.cat([h_gnn, h_lm], dim=-1)
        gate = self.interaction_gate(combined_for_gate) # [B, hidden]
        
        # D. Weighted Augmentation
        # The text information is filtered by the gate
        h_lm_filtered = h_lm * gate
        
        # E. Structure-Anchored Concatenation (The "Golden Path")
        # We explicitly keep h_gnn separate to guarantee the baseline performance.
        # Theoretical formula: Final = Concat(Anchor, Gate * Augment)
        fused_vector = torch.cat([h_gnn, h_lm_filtered], dim=-1) # [B, hidden*2]
        
        # F. Heads
        logits = self.classifier(fused_vector)
        
        contrast_feat = self.contrastive_head(fused_vector)
        contrast_feat = F.normalize(contrast_feat, dim=1)
        
        # Return logits, features, and the gate (for importance analysis)
        return logits, contrast_feat, gate

class AsymmetricFusion(nn.Module):
    def __init__(self, lm_dim=4096, gnn_dim=256, fusion_dim=512): # 提升 fusion_dim
        super().__init__()
        
        # 1. 维度适配（缓解信息漏斗）
        # 不直接压到 256，而是先降到 512 或 1024，保留更多语义
        self.lm_proj = nn.Sequential(
            nn.Linear(lm_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        # GNN 维度通常较小，可以升维或保持
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # 2. 单向强交互 (GNN Query Text)
        # 我们只保留一条最关键的 Attention 路径，减少参数量和训练难度
        # "图结构觉得这段文本哪里可疑？"
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        
        # 3. 门控融合 (Gated Fusion)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        
        # 4. 最终分类器 (使用 Concat 保底)
        # 输入维度是 fusion_dim * 2 (原始GNN + 交互后的Text)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2)
        )
        
        # 独立的对比学习头
        self.contrastive_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 128)
        )

    def forward(self, lm_raw, gnn_raw):
        # A. 投影
        h_text = self.lm_proj(lm_raw).unsqueeze(1)    # [B, 1, 512]
        h_graph = self.gnn_proj(gnn_raw).unsqueeze(1) # [B, 1, 512]
        
        # B. Attention: GNN 查 Text
        # Q=Graph, K=Text, V=Text
        # 提取出"与图结构相关的文本特征"
        aligned_text, _ = self.cross_attn(query=h_graph, key=h_text, value=h_text)
        
        # C. 门控残差
        # 动态决定是使用"对齐后的文本"还是"原始文本"(这里简化为只用对齐后的，防止噪声)
        # 但我们把 GNN 的原始特征保留，作为"锚点"
        
        h_graph_squeeze = h_graph.squeeze(1)
        aligned_text_squeeze = aligned_text.squeeze(1)
        
        # D. 拼接 (Concat) - 采纳 Mentor A 的建议作为物理保底
        # 我们拼接 [原始GNN, 对齐后的Text]
        # 这样即便 Attention 失败，分类器至少能看到原始 GNN，保证 F1 不低于 0.78
        final_vec = torch.cat([h_graph_squeeze, aligned_text_squeeze], dim=-1) # [B, 1024]
        
        logits = self.classifier(final_vec)
        
        # E. Contrastive Feature
        contrast_feat = self.contrastive_head(final_vec)
        contrast_feat = F.normalize(contrast_feat, dim=1)
        
        return logits, contrast_feat

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
