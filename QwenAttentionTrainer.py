"""
Trainer for Qwen + Attention Fusion
使用注意力机制融合LM和GNN，替代知识蒸馏
"""
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from dataloader import build_LM_dataloader, build_GNN_dataloader
from pathlib import Path
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR


class QwenAttentionTrainer:
    """
    Qwen + Attention Fusion 训练器
    与原始的知识蒸馏方法不同，这里使用注意力机制端到端训练
    """
    def __init__(
        self,
        qwen_model,
        gnn_model,
        fusion_model,
        tokenizer,
        device,
        epochs,
        lr_lm,
        lr_gnn,
        lr_fusion,
        weight_decay,
        warmup_ratio,
        max_length,
        batch_size_lm,
        batch_size_gnn,
        train_idx,
        valid_idx,
        test_idx,
        hard_labels,
        user_seq,
        edge_index,
        edge_type,
        ckpt_filepath,
        run,
        train_mode='joint',  # 'joint', 'sequential', 'alternating'
        eval_patience=20
    ):
        self.qwen_model = qwen_model
        self.gnn_model = gnn_model
        self.fusion_model = fusion_model
        self.tokenizer = tokenizer
        self.device = device
        self.epochs = epochs
        self.lr_lm = lr_lm
        self.lr_gnn = lr_gnn
        self.lr_fusion = lr_fusion
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_length = max_length
        self.batch_size_lm = batch_size_lm
        self.batch_size_gnn = batch_size_gnn
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.hard_labels = hard_labels
        self.user_seq = user_seq
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.ckpt_filepath = Path(ckpt_filepath)
        self.run = run
        self.train_mode = train_mode
        self.eval_patience = eval_patience
        
        self.criterion = CrossEntropyLoss()
        self.best_valid_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 构建优化器
        self._build_optimizers()
        
    def _build_optimizers(self):
        """构建优化器和学习率调度器"""
        if self.train_mode == 'joint':
            # 联合训练：一个优化器优化所有参数
            all_params = (
                list(self.qwen_model.parameters()) +
                list(self.gnn_model.parameters()) +
                list(self.fusion_model.parameters())
            )
            self.optimizer = torch.optim.AdamW(
                all_params,
                lr=self.lr_lm,
                weight_decay=self.weight_decay
            )
            total_steps = len(self.train_idx) // self.batch_size_lm * self.epochs
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(total_steps * self.warmup_ratio),
                num_training_steps=total_steps
            )
        else:
            # 分别优化：为LM、GNN、Fusion创建独立优化器
            self.optimizer_lm = torch.optim.AdamW(
                self.qwen_model.parameters(),
                lr=self.lr_lm,
                weight_decay=self.weight_decay
            )
            self.optimizer_gnn = torch.optim.AdamW(
                self.gnn_model.parameters(),
                lr=self.lr_gnn,
                weight_decay=self.weight_decay
            )
            self.optimizer_fusion = torch.optim.AdamW(
                self.fusion_model.parameters(),
                lr=self.lr_fusion,
                weight_decay=self.weight_decay
            )
            
            total_steps = len(self.train_idx) // self.batch_size_lm * self.epochs
            self.scheduler_lm = get_cosine_schedule_with_warmup(
                self.optimizer_lm,
                num_warmup_steps=int(total_steps * self.warmup_ratio),
                num_training_steps=total_steps
            )
            self.scheduler_gnn = CosineAnnealingLR(
                self.optimizer_gnn,
                T_max=self.epochs
            )
            self.scheduler_fusion = CosineAnnealingLR(
                self.optimizer_fusion,
                T_max=self.epochs
            )
    
    def batch_to_tensor(self, batch):
        """将文本batch转换为tensor"""
        texts, labels = batch
        # 注意：add_special_tokens=False 因为文本中已经包含了DESCRIPTION:, METADATA:, TWEET:等标记
        # 这与原始LMBot项目的tokenize策略保持一致
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False  # 与原项目保持一致
        )
        return (
            {k: v.to(self.device) for k, v in tokenized.items()},
            labels.to(self.device)
        )
    
    def extract_all_embeddings(self):
        """提取所有节点的LM embeddings"""
        self.qwen_model.eval()
        all_embeddings = []
        
        dataloader_config = {'batch_size': self.batch_size_lm}
        all_idx = torch.arange(len(self.user_seq))
        loader = build_LM_dataloader(
            dataloader_config, all_idx, self.user_seq,
            self.hard_labels, mode='infer'
        )
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                tokenized, _ = self.batch_to_tensor(batch)
                embeddings, _ = self.qwen_model(tokenized)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)
    
    def train_epoch_joint(self, epoch):
        """联合训练模式：同时优化LM、GNN和Fusion"""
        self.qwen_model.train()
        self.gnn_model.train()
        self.fusion_model.train()
        
        total_loss = 0
        num_batches = 0
        
        # 首先提取所有LM embeddings用于GNN
        print("Extracting LM embeddings for GNN...")
        with torch.no_grad():
            all_lm_embeddings = self.extract_all_embeddings()
        
        # 构建GNN数据加载器
        dataloader_config_gnn = {
            'batch_size': self.batch_size_gnn,
            'n_layers': self.gnn_model.n_layers
        }
        gnn_loader = build_GNN_dataloader(
            dataloader_config_gnn,
            self.train_idx,
            all_lm_embeddings,
            self.hard_labels,
            self.edge_index,
            self.edge_type,
            mode='train'
        )
        
        # 训练
        for batch in tqdm(gnn_loader, desc=f"Epoch {epoch}"):
            batch = batch.to(self.device)
            batch_idx = batch.n_id[:batch.batch_size]
            
            # 获取LM特征（需要重新forward以获取梯度）
            texts = [self.user_seq[idx.item()] for idx in batch_idx]
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=False
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Forward LM (保留梯度)
            self.qwen_model.eval()  # 可选：冻结LM的BN/Dropout
            lm_embeddings_batch, _ = self.qwen_model(tokenized)
            lm_embeddings_batch = lm_embeddings_batch.detach().requires_grad_(True)
            
            # Forward GNN
            gnn_out = self.gnn_model(batch.x, batch.edge_index, batch.edge_type)
            gnn_features = gnn_out[:batch.batch_size]
            
            # Attention Fusion
            logits, _ = self.fusion_model(lm_embeddings_batch, gnn_features)
            
            # 计算损失
            labels = batch.labels[:batch.batch_size]
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.qwen_model.parameters()) +
                list(self.gnn_model.parameters()) +
                list(self.fusion_model.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch_sequential(self, epoch):
        """顺序训练模式：先训练LM，再训练GNN，最后训练Fusion"""
        # Phase 1: 训练LM
        print("Phase 1: Training LM...")
        self.qwen_model.train()
        self.gnn_model.eval()
        self.fusion_model.eval()
        
        dataloader_config_lm = {'batch_size': self.batch_size_lm}
        lm_loader = build_LM_dataloader(
            dataloader_config_lm,
            self.train_idx,
            self.user_seq,
            self.hard_labels,
            mode='train'
        )
        
        for batch in tqdm(lm_loader, desc=f"Epoch {epoch} - LM"):
            tokenized, labels = self.batch_to_tensor(batch)
            _, logits = self.qwen_model(tokenized)
            loss = self.criterion(logits, labels)
            
            self.optimizer_lm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qwen_model.parameters(), max_norm=1.0)
            self.optimizer_lm.step()
            self.scheduler_lm.step()
        
        # Phase 2: 训练GNN
        print("Phase 2: Training GNN...")
        self.qwen_model.eval()
        self.gnn_model.train()
        
        with torch.no_grad():
            all_lm_embeddings = self.extract_all_embeddings()
        
        dataloader_config_gnn = {
            'batch_size': self.batch_size_gnn,
            'n_layers': self.gnn_model.n_layers
        }
        gnn_loader = build_GNN_dataloader(
            dataloader_config_gnn,
            self.train_idx,
            all_lm_embeddings,
            self.hard_labels,
            self.edge_index,
            self.edge_type,
            mode='train'
        )
        
        for batch in tqdm(gnn_loader, desc=f"Epoch {epoch} - GNN"):
            batch = batch.to(self.device)
            gnn_out = self.gnn_model(batch.x, batch.edge_index, batch.edge_type)
            logits = gnn_out[:batch.batch_size]
            labels = batch.labels[:batch.batch_size]
            loss = self.criterion(logits, labels)
            
            self.optimizer_gnn.zero_grad()
            loss.backward()
            self.optimizer_gnn.step()
        
        self.scheduler_gnn.step()
        
        # Phase 3: 训练Fusion
        print("Phase 3: Training Fusion...")
        self.qwen_model.eval()
        self.gnn_model.eval()
        self.fusion_model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(gnn_loader, desc=f"Epoch {epoch} - Fusion"):
            batch = batch.to(self.device)
            batch_idx = batch.n_id[:batch.batch_size]
            
            with torch.no_grad():
                # 获取LM特征
                texts = [self.user_seq[idx.item()] for idx in batch_idx]
                tokenized = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                    add_special_tokens=False
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                lm_embeddings_batch, _ = self.qwen_model(tokenized)
                
                # 获取GNN特征
                gnn_out = self.gnn_model(batch.x, batch.edge_index, batch.edge_type)
                gnn_features = gnn_out[:batch.batch_size]
            
            # 训练Fusion
            logits, _ = self.fusion_model(lm_embeddings_batch, gnn_features)
            labels = batch.labels[:batch.batch_size]
            loss = self.criterion(logits, labels)
            
            self.optimizer_fusion.zero_grad()
            loss.backward()
            self.optimizer_fusion.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler_fusion.step()
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, split='valid'):
        """评估模型"""
        self.qwen_model.eval()
        self.gnn_model.eval()
        self.fusion_model.eval()
        
        if split == 'valid':
            eval_idx = self.valid_idx
        elif split == 'test':
            eval_idx = self.test_idx
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # 提取所有embeddings
        with torch.no_grad():
            all_lm_embeddings = self.extract_all_embeddings()
        
        # 构建GNN数据加载器
        dataloader_config_gnn = {
            'batch_size': self.batch_size_gnn,
            'n_layers': self.gnn_model.n_layers
        }
        gnn_loader = build_GNN_dataloader(
            dataloader_config_gnn,
            eval_idx,
            all_lm_embeddings,
            self.hard_labels,
            self.edge_index,
            self.edge_type,
            mode='eval'
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(gnn_loader, desc=f"Evaluating {split}"):
                batch = batch.to(self.device)
                batch_idx = batch.n_id[:batch.batch_size]
                
                # 获取LM特征
                texts = [self.user_seq[idx.item()] for idx in batch_idx]
                tokenized = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                    add_special_tokens=False
                )
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                lm_embeddings_batch, _ = self.qwen_model(tokenized)
                
                # 获取GNN特征
                gnn_out = self.gnn_model(batch.x, batch.edge_index, batch.edge_type)
                gnn_features = gnn_out[:batch.batch_size]
                
                # Fusion预测
                logits, _ = self.fusion_model(lm_embeddings_batch, gnn_features)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_labels.append(batch.labels[:batch.batch_size].cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return acc, f1
    
    def train(self):
        """训练主循环"""
        print(f"Starting training with mode: {self.train_mode}")
        
        for epoch in range(self.epochs):
            print(f"\n===== Epoch {epoch + 1}/{self.epochs} =====")
            
            # 训练
            if self.train_mode == 'joint':
                train_loss = self.train_epoch_joint(epoch)
            elif self.train_mode == 'sequential':
                train_loss = self.train_epoch_sequential(epoch)
            else:
                raise ValueError(f"Unknown train mode: {self.train_mode}")
            
            print(f"Train Loss: {train_loss:.4f}")
            
            # 验证
            valid_acc, valid_f1 = self.evaluate('valid')
            print(f"Valid Accuracy: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")
            
            # WandB日志
            if self.run is not None:
                self.run.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_accuracy': valid_acc,
                    'valid_f1': valid_f1
                })
            
            # 保存最佳模型
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best')
                print(f"New best model saved! Accuracy: {valid_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.eval_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # 测试
        print("\n===== Testing Best Model =====")
        self.load_checkpoint('best')
        test_acc, test_f1 = self.evaluate('test')
        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        if self.run is not None:
            self.run.log({
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'best_epoch': self.best_epoch
            })
        
        return test_acc, test_f1
    
    def save_checkpoint(self, name='best'):
        """保存检查点"""
        self.ckpt_filepath.mkdir(parents=True, exist_ok=True)
        torch.save({
            'qwen_model': self.qwen_model.state_dict(),
            'gnn_model': self.gnn_model.state_dict(),
            'fusion_model': self.fusion_model.state_dict(),
            'best_valid_acc': self.best_valid_acc,
            'best_epoch': self.best_epoch
        }, self.ckpt_filepath / f'{name}.pt')
    
    def load_checkpoint(self, name='best'):
        """加载检查点"""
        ckpt = torch.load(self.ckpt_filepath / f'{name}.pt')
        self.qwen_model.load_state_dict(ckpt['qwen_model'])
        self.gnn_model.load_state_dict(ckpt['gnn_model'])
        self.fusion_model.load_state_dict(ckpt['fusion_model'])
        self.best_valid_acc = ckpt['best_valid_acc']
        self.best_epoch = ckpt['best_epoch']
