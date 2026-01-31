"""
Refactored QwenPrecomputedTrainer
重构后的训练器: 精简、模块化、易扩展

核心改进:
1. 配置对象替代散列参数
2. 损失计算解耦到 LossComputer
3. 批次处理解耦到 BatchProcessor
4. 移除重复代码
5. 清晰的职责边界

Author: Refactored from QwenPrecomputedTrainer.py
Date: 2026-01-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.utils import degree, scatter

from configs.trainer_config import TrainerConfig
from training.loss_computer import LossComputer
from data.batch_processor import BatchProcessor
from utils import EdlLoss


class QwenPrecomputedTrainerRefactored:
    """
    重构后的训练器 (精简版)
    
    职责:
    1. 训练循环编排
    2. 模型状态管理 (冻结/解冻)
    3. 优化器管理
    4. 检查点保存/加载
    5. 日志记录
    
    不再负责:
    - 数据加载 (交给 DataModule)
    - 损失计算 (交给 LossComputer)
    - 批次处理 (交给 BatchProcessor)
    - 诊断分析 (交给 DiagnosticAnalyzer)
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        gnn_model: nn.Module,
        fusion_model: nn.Module,
        precomputed_embeddings: torch.Tensor,
        data_dict: dict,
        dataloader,
        val_loader,
        test_loader
    ):
        """
        Args:
            config: TrainerConfig 配置对象
            gnn_model: GNN 模型
            fusion_model: Fusion 模型
            precomputed_embeddings: 预计算嵌入 [N, D]
            data_dict: 包含 labels, edge_index, edge_type 等
            dataloader: 训练集 DataLoader
            val_loader: 验证集 DataLoader
            test_loader: 测试集 DataLoader
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Models
        self.gnn = gnn_model.to(self.device)
        self.fusion = fusion_model.to(self.device)
        self.embeddings = precomputed_embeddings.to('cpu')  # 内存优化
        
        # Data
        self.data_dict = data_dict
        self.labels = self._prepare_labels(data_dict['labels'])
        self.edge_index = data_dict['edge_index'].to(self.device)
        self.edge_type = data_dict.get('edge_type')
        if self.edge_type is not None:
            self.edge_type = self.edge_type.to(self.device)
        
        # DataLoaders
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 辅助模块
        self.loss_computer = LossComputer(config.loss, self.device)
        self.batch_processor = BatchProcessor(
            self.embeddings, self.gnn, self.device
        )
        
        # 预计算特征
        self.homophily = self._precompute_homophily()
        self.node_degrees = self._precompute_degrees()
        
        # 优化器 (延迟初始化)
        self.optimizer = None
        
        # 检查点路径
        self.ckpt_dir = Path('checkpoints') / config.exp_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_stage = None
        self.best_metric = 0.0
        
    def _prepare_labels(self, labels):
        """准备标签: one-hot → class index"""
        labels = labels.to(self.device)
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)
        return labels.long()
    
    def _precompute_homophily(self):
        """预计算节点同质性 (占位符)"""
        return torch.zeros(self.labels.size(0), 1).to(self.device)
    
    def _precompute_degrees(self):
        """预计算节点度数 (标准化)"""
        row, col = self.edge_index
        deg = degree(col, self.labels.size(0), dtype=torch.float)
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        return deg.unsqueeze(1).to(self.device)
    
    # =========================================================================
    # 阶段 1: Text Expert (VIB) 预训练
    # =========================================================================
    
    def pretrain_text_expert(self, epochs=None):
        """
        预训练文本专家 (VIB)
        冻结: GNN, Gate, Classifier
        训练: fusion.vib
        """
        epochs = epochs or self.config.training.pretrain_text_epochs
        print(f"\n{'='*60}")
        print(f"Stage 1 | Text Expert (VIB) Pre-training ({epochs} epochs)")
        print(f"{'='*60}\n")
        
        self.current_stage = 'text_vib'
        
        # 冻结所有模块, 除了 VIB
        self._freeze_module(self.gnn, True)
        self._freeze_module(self.fusion.gnn_proj, True)
        self._freeze_module(self.fusion.gate_net, True)
        self._freeze_module(self.fusion.classifier, True)
        self._freeze_module(self.fusion.norm, True)
        self._freeze_module(self.fusion.gnn_evidence_head, True)
        self._freeze_module(self.fusion.vib, False)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.fusion.parameters()),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._train_epoch_vib(epoch)
            val_acc = self._validate_text_only()
            
            print(f"VIB Ep {epoch:02d} | Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_metrics['acc']:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            print(f"U_Corr: {train_metrics['u_corr']:.4f} | "
                  f"U_Wrong: {train_metrics['u_wrong']:.4f} | "
                  f"Gap: {train_metrics['u_gap']:.4f}")
            
            if self.config.use_wandb:
                self._log_to_wandb('vib', epoch, train_loss, train_metrics, 
                                   {'val_acc': val_acc})
            
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint('best_text_vib.pt')
        
        print(f"Text Expert Finished. Best Val Acc: {best_acc:.4f}\n")
        self._freeze_module(self.fusion.vib, True)
    
    def _train_epoch_vib(self, epoch):
        """VIB 训练单个 epoch"""
        self.fusion.vib.train()
        total_loss = 0.0
        steps = 0
        
        correct = 0
        total = 0
        u_correct_list = []
        u_wrong_list = []
        
        for batch in self.dataloader:
            # 解包批次 (简化: 假设是 tuple)
            if isinstance(batch, (list, tuple)):
                nodes, labels = batch
                nodes = nodes.to(self.device)
                labels = labels.to(self.device)
            else:
                raise NotImplementedError("Only tuple batch supported in this version")
            
            self.optimizer.zero_grad()
            
            # VIB 前向
            h_lm = self.embeddings[nodes].to(self.device)
            logits, _, alpha, u_text, kl_loss = self.fusion.vib(h_lm)
            
            # 损失计算
            loss, loss_dict = self.loss_computer.compute_vib_loss(
                logits, alpha, u_text, kl_loss, labels, epoch
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            # 统计
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                is_correct = (preds == labels)
                correct += is_correct.sum().item()
                total += labels.size(0)
                
                u_val = u_text.detach()
                if u_val.dim() > 1:
                    u_val = u_val.view(-1)
                
                u_correct_list.extend(u_val[is_correct].tolist())
                u_wrong_list.extend(u_val[~is_correct].tolist())
        
        # 计算指标
        avg_loss = total_loss / steps
        train_acc = correct / total
        u_corr = np.mean(u_correct_list) if u_correct_list else 0
        u_wrong = np.mean(u_wrong_list) if u_wrong_list else 0
        
        metrics = {
            'acc': train_acc,
            'u_corr': u_corr,
            'u_wrong': u_wrong,
            'u_gap': u_wrong - u_corr,
            'u_corr_std': np.std(u_correct_list) if u_correct_list else 0,
            'u_wrong_std': np.std(u_wrong_list) if u_wrong_list else 0
        }
        
        return avg_loss, metrics
    
    def _validate_text_only(self):
        """验证文本专家性能"""
        self.fusion.vib.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    nodes, labels = batch
                else:
                    nodes = batch
                    labels = self.labels[nodes]
                
                nodes = nodes.to(self.device)
                h_lm = self.embeddings[nodes].to(self.device)
                
                logits, _, _, _, _ = self.fusion.vib(h_lm)
                preds.append(logits.argmax(dim=1).cpu())
                targets.append(labels.cpu())
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        return accuracy_score(targets, preds)
    
    # =========================================================================
    # 阶段 2: GNN 预训练
    # =========================================================================
    
    def pretrain_gnn(self, epochs=None):
        """
        预训练 GNN + Evidence Head
        冻结: Fusion (除 gnn_proj 和 gnn_evidence_head)
        训练: GNN, gnn_proj, gnn_evidence_head
        """
        epochs = epochs or self.config.training.pretrain_gnn_epochs
        print(f"\n{'='*60}")
        print(f"Stage 2 | GNN Pre-training ({epochs} epochs)")
        print(f"{'='*60}\n")
        
        self.current_stage = 'gnn'
        
        # 冻结设置
        self._freeze_module(self.fusion, True)
        self._freeze_module(self.gnn, False)
        self._freeze_module(self.fusion.gnn_proj, False)
        self._freeze_module(self.fusion.gnn_evidence_head, False)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, 
                   list(self.gnn.parameters()) + 
                   list(self.fusion.gnn_proj.parameters()) +
                   list(self.fusion.gnn_evidence_head.parameters())),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        
        best_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch_gnn(epoch)
            val_metrics = self._validate_gnn()
            
            print(f"GNN Ep {epoch:02d} | Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"U_Gap: {val_metrics['u_gap']:.4f}")
            
            if self.config.use_wandb:
                self._log_to_wandb('gnn', epoch, train_loss, {}, val_metrics)
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self._save_checkpoint('best_gnn.pt')
        
        print(f"GNN Pre-training Finished. Best Val F1: {best_f1:.4f}\n")
    
    def _train_epoch_gnn(self, epoch):
        """GNN 训练单个 epoch"""
        self.gnn.train()
        self.fusion.gnn_proj.train()
        self.fusion.gnn_evidence_head.train()
        
        total_loss = 0.0
        steps = 0
        
        for batch in self.dataloader:
            # 简化: 假设全图批次
            if isinstance(batch, (list, tuple)):
                nodes, labels = batch
                nodes = nodes.to(self.device)
                labels = labels.to(self.device)
            else:
                raise NotImplementedError()
            
            self.optimizer.zero_grad()
            
            # GNN 前向 (全图)
            h_all = self._forward_gnn_full()
            h_gnn = h_all[nodes]
            
            # 一致性 (简化: 占位符)
            consistency = torch.ones(nodes.size(0), 1).to(self.device)
            
            # Evidence Head
            z_gnn = self.fusion.gnn_proj(h_gnn)
            logits, alpha, u, probs = self.fusion.gnn_evidence_head(
                z_gnn, consistency
            )
            
            # 损失
            loss, loss_dict = self.loss_computer.compute_gnn_loss(
                logits, alpha, labels, epoch
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        return total_loss / steps
    
    def _forward_gnn_full(self):
        """全图 GNN 前向"""
        x = self.embeddings.to(self.device)
        if self.edge_type is not None:
            h = self.gnn(x, self.edge_index, self.edge_type)
        else:
            h = self.gnn(x, self.edge_index)
        
        if isinstance(h, tuple):
            h = h[0]
        return h
    
    def _validate_gnn(self):
        """验证 GNN 性能"""
        self.gnn.eval()
        self.fusion.gnn_proj.eval()
        self.fusion.gnn_evidence_head.eval()
        
        preds_list, targets_list = [], []
        u_correct_list, u_wrong_list = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    nodes, labels = batch
                    nodes = nodes.to(self.device)
                    labels = labels.to(self.device)
                else:
                    raise NotImplementedError()
                
                h_all = self._forward_gnn_full()
                h_gnn = h_all[nodes]
                
                consistency = torch.ones(nodes.size(0), 1).to(self.device)
                z_gnn = self.fusion.gnn_proj(h_gnn)
                logits, _, u, _ = self.fusion.gnn_evidence_head(z_gnn, consistency)
                
                preds = logits.argmax(1)
                is_correct = (preds == labels)
                
                u_correct_list.extend(u[is_correct].cpu().tolist())
                u_wrong_list.extend(u[~is_correct].cpu().tolist())
                preds_list.append(preds.cpu())
                targets_list.append(labels.cpu())
        
        all_preds = torch.cat(preds_list).numpy()
        all_targets = torch.cat(targets_list).numpy()
        
        metrics = {
            'f1': f1_score(all_targets, all_preds, average='macro'),
            'u_corr': np.mean(u_correct_list) if u_correct_list else 0,
            'u_wrong': np.mean(u_wrong_list) if u_wrong_list else 0,
            'u_gap': (np.mean(u_wrong_list) if u_wrong_list else 0) - 
                     (np.mean(u_correct_list) if u_correct_list else 0)
        }
        
        return metrics
    
    # =========================================================================
    # 阶段 3: 融合训练
    # =========================================================================
    
    def train_fusion(self, epochs=None):
        """
        融合训练 (所有模块联合)
        """
        epochs = epochs or self.config.training.epochs
        print(f"\n{'='*60}")
        print(f"Stage 3 | Fusion Training ({epochs} epochs)")
        print(f"{'='*60}\n")
        
        self.current_stage = 'fusion'
        
        # 解冻所有模块
        self._freeze_module(self.gnn, False)
        self._freeze_module(self.fusion, False)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters()) + list(self.fusion.parameters()),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        
        best_f1 = 0.0
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch_fusion(epoch)
            val_metrics = self.evaluate('val')
            
            print(f"Fusion Ep {epoch:02d} | Loss: {train_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.4f}")
            
            if self.config.use_wandb:
                self._log_to_wandb('fusion', epoch, train_loss, {}, val_metrics)
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self._save_checkpoint('best_fusion.pt')
        
        print(f"Fusion Training Finished. Best Val F1: {best_f1:.4f}\n")
        return best_f1
    
    def _train_epoch_fusion(self, epoch):
        """融合训练单个 epoch"""
        self.gnn.train()
        self.fusion.train()
        
        total_loss = 0.0
        steps = 0
        
        for batch in self.dataloader:
            # 简化: 全图批次
            if isinstance(batch, (list, tuple)):
                nodes, labels = batch
                nodes = nodes.to(self.device)
                labels = labels.to(self.device)
            else:
                raise NotImplementedError()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            h_all = self._forward_gnn_full()
            h_gnn = h_all[nodes]
            h_lm = self.embeddings[nodes].to(self.device)
            
            # 辅助特征
            homophily = self.homophily[nodes]
            degrees = self.node_degrees[nodes]
            consistency = torch.ones(nodes.size(0), 1).to(self.device)
            
            # Fusion
            out = self.fusion(h_lm, h_gnn, homophily, degrees, consistency)
            
            # 损失
            loss, loss_dict = self.loss_computer.compute_fusion_loss(
                out, labels, h_gnn, epoch, homophily
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn.parameters()) + list(self.fusion.parameters()),
                max_norm=self.config.training.grad_clip
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        return total_loss / steps
    
    # =========================================================================
    # 评估
    # =========================================================================
    
    def evaluate(self, split='val'):
        """
        标准评估
        
        Args:
            split: 'train', 'val', 'test'
            
        Returns:
            metrics: {'f1': float, 'acc': float}
        """
        self.gnn.eval()
        self.fusion.eval()
        
        if split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        else:
            loader = self.dataloader
        
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    nodes, labels = batch
                    nodes = nodes.to(self.device)
                    labels = labels.to(self.device)
                else:
                    raise NotImplementedError()
                
                h_all = self._forward_gnn_full()
                h_gnn = h_all[nodes]
                h_lm = self.embeddings[nodes].to(self.device)
                
                homophily = self.homophily[nodes]
                degrees = self.node_degrees[nodes]
                consistency = torch.ones(nodes.size(0), 1).to(self.device)
                
                out = self.fusion(h_lm, h_gnn, homophily, degrees, consistency)
                logits = out['logits']
                
                preds.append(logits.argmax(1).cpu())
                targets.append(labels.cpu())
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        
        return {
            'f1': f1_score(targets, preds, average='macro'),
            'acc': accuracy_score(targets, preds)
        }
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    
    def _freeze_module(self, module, freeze=True):
        """冻结/解冻模块参数"""
        for param in module.parameters():
            param.requires_grad = not freeze
    
    def _save_checkpoint(self, filename):
        """保存检查点"""
        ckpt_path = self.ckpt_dir / filename
        state = {
            'gnn_state_dict': self.gnn.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'stage': self.current_stage
        }
        torch.save(state, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        ckpt_path = self.ckpt_dir / filename
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return
        
        state = torch.load(ckpt_path, map_location=self.device)
        self.gnn.load_state_dict(state['gnn_state_dict'])
        self.fusion.load_state_dict(state['fusion_state_dict'])
        print(f"Checkpoint loaded: {ckpt_path}")
    
    def _log_to_wandb(self, stage, epoch, loss, train_metrics, val_metrics):
        """记录到 wandb"""
        if not self.config.use_wandb:
            return
        
        log_dict = {
            f'{stage}/epoch': epoch,
            f'{stage}/train_loss': loss
        }
        
        for k, v in train_metrics.items():
            log_dict[f'{stage}/train_{k}'] = v
        
        for k, v in val_metrics.items():
            log_dict[f'{stage}/{k}'] = v
        
        wandb.log(log_dict)
    
    @property
    def wandb_active(self):
        """动态检查 wandb 状态"""
        return wandb.run is not None
