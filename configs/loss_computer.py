"""
Loss Computer: 集中管理所有损失函数计算
将损失计算逻辑从 Trainer 中解耦
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import EdlLoss


class LossComputer:
    """统一损失计算器"""
    
    def __init__(self, config, device):
        """
        Args:
            config: LossConfig 对象
            device: torch.device
        """
        self.config = config
        self.device = device
        
        # 基础损失函数
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.edl_loss = EdlLoss()
        
    def compute_fusion_loss(self, out, labels, h_gnn, epoch, homophily=None):
        """
        计算融合训练的完整损失
        
        Args:
            out: fusion model 的输出字典，包含:
                - logits: 融合预测
                - logits_lm: 文本分支预测
                - alpha_text: 文本 EDL alpha
                - alpha_gnn: 图 EDL alpha (可选)
                - u_text: 文本不确定性
                - u_graph: 图不确定性
                - gate_value: 门控值
                - kl_loss: VIB KL 散度
            labels: 真实标签
            h_gnn: GNN 特征 (用于计算 graph auxiliary loss)
            epoch: 当前 epoch (用于 KL warmup 和 EDL)
            homophily: 节点同质性 (可选，用于 gate loss)
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典 (用于日志)
        """
        loss_dict = {}
        
        # 1. 主分类损失
        loss_dict['cls'] = self.criterion_cls(out['logits'], labels)
        
        # 2. 辅助分类损失 (Text)
        loss_dict['text_aux'] = self.criterion_cls(
            out['logits_lm'], labels
        ) * self.config.text_aux_weight
        
        # 3. 辅助分类损失 (Graph)
        # 假设 fusion 有 gnn_classifier 或使用外部分类器
        if 'logits_gnn' in out:
            logits_gnn = out['logits_gnn']
        else:
            # 需要外部传入 gnn.classifier
            logits_gnn = None
            
        if logits_gnn is not None:
            loss_dict['graph_aux'] = self.criterion_cls(
                logits_gnn, labels
            ) * self.config.graph_aux_weight
        else:
            loss_dict['graph_aux'] = torch.tensor(0.0, device=self.device)
        
        # 4. EDL 损失 (Text)
        if hasattr(self.edl_loss, 'epoch_num'):
            self.edl_loss.epoch_num = epoch
        loss_dict['edl_text'] = self.edl_loss(
            out['alpha_text'], labels
        ) * self.config.edl_text_weight
        
        # 5. EDL 损失 (Graph)
        if 'alpha_gnn' in out:
            loss_dict['edl_graph'] = self.edl_loss(
                out['alpha_gnn'], labels
            ) * self.config.edl_graph_weight
        else:
            loss_dict['edl_graph'] = torch.tensor(0.0, device=self.device)
        
        # 6. VIB KL 损失 (带 warmup)
        kl_weight = self._get_kl_weight(epoch)
        loss_dict['vib'] = out['kl_loss'] * kl_weight
        
        # 7. AvUC 损失
        u_fused = out['gate_value'] * out['u_text'] + \
                  (1 - out['gate_value']) * out['u_graph']
        loss_dict['avuc'] = self._compute_avuc_loss(
            out['logits'], labels, u_fused
        ) * self.config.lambda_avuc
        
        # 8. Gate 一致性损失
        if homophily is not None:
            loss_dict['gate'] = self._compute_gate_loss(
                out['gate_value'], homophily, out['u_text']
            ) * self.config.gate_weight
        else:
            loss_dict['gate'] = torch.tensor(0.0, device=self.device)
        
        # 9. 不确定性边界损失
        loss_dict['margin'] = self._compute_margin_loss(
            out['logits_lm'], labels, out['u_text']
        )
        
        # 总损失
        total_loss = sum(loss_dict.values())
        
        return total_loss, loss_dict
    
    def compute_vib_loss(self, logits, alpha, u_text, kl_loss, labels, epoch):
        """
        VIB 预训练损失
        
        Args:
            logits: 文本分类 logits
            alpha: EDL alpha
            u_text: 不确定性
            kl_loss: VIB KL 散度
            labels: 真实标签
            epoch: 当前 epoch
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失
        """
        loss_dict = {}
        
        # 分类损失
        loss_dict['cls'] = self.criterion_cls(logits, labels)
        
        # EDL 损失
        if hasattr(self.edl_loss, 'epoch_num'):
            self.edl_loss.epoch_num = epoch
        loss_dict['edl'] = self.edl_loss(alpha, labels)
        
        # VIB KL 损失 (warmup)
        kl_weight = self._get_vib_kl_weight(epoch)
        loss_dict['vib'] = kl_loss * kl_weight
        
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
    
    def compute_gnn_loss(self, logits, alpha, labels, epoch):
        """
        GNN 预训练损失
        
        Args:
            logits: GNN 分类 logits
            alpha: EDL alpha
            labels: 真实标签
            epoch: 当前 epoch
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失
        """
        loss_dict = {}
        
        # 分类损失
        loss_dict['cls'] = self.criterion_cls(logits, labels)
        
        # EDL 损失
        if hasattr(self.edl_loss, 'epoch_num'):
            self.edl_loss.epoch_num = epoch
        loss_dict['edl'] = self.edl_loss(alpha, labels) * 0.2
        
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
    
    def _get_kl_weight(self, epoch):
        """KL 散度权重 warmup (用于融合训练)"""
        if epoch < self.config.kl_warmup_start:
            return 0.0
        progress = min(1.0, (epoch - self.config.kl_warmup_start) / 
                      self.config.kl_warmup_duration)
        return self.config.lambda_struct * progress
    
    def _get_vib_kl_weight(self, epoch):
        """VIB KL 散度权重 warmup (用于 VIB 预训练)"""
        if epoch < self.config.kl_warmup_start:
            return 0.0
        progress = min(1.0, (epoch - self.config.kl_warmup_start) / 
                      self.config.kl_warmup_duration)
        return self.config.kl_base_weight * progress
    
    def _compute_avuc_loss(self, logits, labels, uncertainty):
        """
        计算 Accuracy vs Uncertainty Calibration (AvUC) 损失
        目标: 正确样本低不确定性, 错误样本高不确定性
        """
        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        
        # 确保维度匹配
        if uncertainty.dim() == 1:
            uncertainty = uncertainty.unsqueeze(1)
        if correct.dim() == 1:
            correct = correct.unsqueeze(1)
        
        # 数值稳定性
        uncertainty = torch.clamp(uncertainty, min=1e-6, max=1.0 - 1e-6)
        
        # AvUC = -log(E[AC + IU])
        # AC: Accuracy * Certainty
        # IU: Inaccuracy * Uncertainty
        ac = correct * (1 - uncertainty)
        iu = (1 - correct) * uncertainty
        
        loss = -torch.log(torch.mean(ac + iu) + 1e-10)
        return loss
    
    def _compute_margin_loss(self, logits, labels, uncertainty):
        """
        不确定性边界损失
        正确样本: u < u_correct_upper
        错误样本: u > u_wrong_lower
        """
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        if uncertainty.dim() > 1:
            uncertainty = uncertainty.squeeze()
        
        mask_correct = (preds == labels)
        
        loss = torch.tensor(0.0, device=self.device)
        
        # 正确样本: 惩罚过高的不确定性
        if mask_correct.sum() > 0:
            loss += F.relu(
                uncertainty[mask_correct] - self.config.u_correct_upper
            ).mean()
        
        # 错误样本: 惩罚过低的不确定性
        if (~mask_correct).sum() > 0:
            loss += F.relu(
                self.config.u_wrong_lower - uncertainty[~mask_correct]
            ).mean()
        
        return loss
    
    def _compute_gate_loss(self, gate_value, homophily, u_text):
        """
        门控一致性损失
        高同质性 + 低文本不确定性 → 高 gate (信任文本)
        低同质性 + 高文本不确定性 → 低 gate (信任图)
        
        注: 具体实现取决于 fusion model 的 get_structure_consistency_loss
        这里返回占位符
        """
        # 简化实现: MSE(gate, 1 - u_text * (1 - homophily))
        # 实际应该调用 fusion.get_structure_consistency_loss
        target_gate = 1 - u_text * (1 - homophily)
        loss = F.mse_loss(gate_value, target_gate)
        return loss
