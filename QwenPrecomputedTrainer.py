import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from utils import batch_linear_cka
from torch_geometric.utils import degree as calc_degree
from AttentionFusion import SupConLoss
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class QwenPrecomputedTrainer:
    def __init__(
        self,
        precomputed_embeddings,  # (num_users, 4096)
        gnn_model,
        fusion_model,
        pretrain,
        data_dict,
        device,
        epochs,
        lr,
        weight_decay,
        sample,
        ckpt_filepath='best_model.pt',
        supcon_temp=0.07,  # [Expert] 设为 0.07 以增强难样本挖掘
        lambda_supcon=0.1, 
        **kwargs,
    ):
        self.device = device
        self.pretrain = pretrain
        self.ckpt_filepath = Path(ckpt_filepath)
        self.epochs = epochs
        self.sample = sample
        
        # 1. Load Embeddings & Edges
        self.embeddings = precomputed_embeddings.to(device)
        self.edge_index = data_dict['edge_index'].to(device)
        num_nodes = self.embeddings.shape[0]
        
        # Handle Edge Types
        self.edge_type = data_dict.get('edge_type', None)
        if self.edge_type is not None: 
            self.edge_type = self.edge_type.to(device)
            
        # 2. Handle Labels   
        raw_labels = data_dict['labels'].to(device)

        # 处理 One-Hot 或 Class Index
        if raw_labels.dim() > 1 and raw_labels.shape[1] > 1:
            print(f"[Data] Detected One-Hot Labels {raw_labels.shape}. Converting to Class Indices.")
            self.labels = raw_labels.argmax(dim=1).long()
        else:
            self.labels = raw_labels.long()
            
        # 3. Initialize Masks 
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # 4. Load Indices & masks
        self.train_nodes = data_dict['train_idx'].to(device)
        self.test_nodes = data_dict['test_idx'].to(device)
        self.val_nodes = data_dict['valid_idx'].to(device)
           
        self.train_mask[self.train_nodes] = True
        self.val_mask[self.val_nodes] = True
        self.test_mask[self.test_nodes] = True
            
        # 5. Models & Optimizer
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_supcon = SupConLoss(temperature=supcon_temp, base_temperature=supcon_temp).to(device)
        self.lambda_supcon = lambda_supcon

        # 6. Node Degrees (for sampling if needed)
        num_nodes = self.embeddings.shape[0]
        row, col = self.edge_index
        deg = calc_degree(col, num_nodes=num_nodes, dtype=torch.float)
        deg = torch.log1p(deg)
        deg_max = deg.max()
        deg_min = deg.min()
        if deg_max > deg_min:
            deg = (deg - deg_min) / (deg_max - deg_min)
        else:
            deg = torch.zeros_like(deg)

        self.node_degrees = deg.unsqueeze(1).to(device)
        
        # Differential Learning Rate
        if self.pretrain:
            # Stage 1: pretraing GNN
            self.optimizer = torch.optim.AdamW(
                self.gnn.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # Stage 2: Fusion Training
            print(f"[Trainer] Applying Differential LR: Fusion={lr}, GNN={lr*0.01}")
            self.optimizer = torch.optim.AdamW([
                {'params': self.fusion.parameters(), 'lr': lr},         # Fusion 全速
                {'params': self.gnn.parameters(),    'lr': lr * 0.01}   # GNN 极低速微调
            ], weight_decay=weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_supcon = SupConLoss(temperature=supcon_temp)
        self.lambda_entropy = 0.01 
        self.lambda_supcon = lambda_supcon
        
        self.best_val_f1 = 0.0
        self.use_wandb = wandb.run is not None

        # Dataloader
        train_labels = self.labels[self.train_nodes].cpu().numpy()
        class_counts = np.bincount(train_labels)

        class_counts[class_counts == 0] = 1
        class_weights = 1. / class_counts
        sample_weights = class_weights[train_labels]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_dataset = torch.utils.data.TensorDataset(
            self.train_nodes, self.labels[self.train_nodes]
        )
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=256, 
            sampler=sampler, # 使用采样器
            drop_last=True   # 丢弃最后一个不完整的batch，避免SupCon计算NaN
        )

    def train(self):
        """
        Main training loop
        """
        for epoch in range(self.epochs):
            self.gnn.train()
            self.fusion.train()
            
            total_loss = 0
            gate_means = []
            
            for batch_nodes, batch_labels in self.dataloader:
                self.optimizer.zero_grad()
                
                # A. GNN Forward
                if self.edge_type is not None:
                    h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
                else:
                    h_gnn = self.gnn(self.embeddings, self.edge_index)
                
                batch_emb_lm = self.embeddings[batch_nodes]
                batch_emb_gnn = h_gnn[batch_nodes]
                
                
                batch_labels = batch_labels.to(self.device)
                batch_degree = self.node_degrees[batch_nodes]

                # B. Forward & Loss Calculation
                if self.pretrain:
                    # Stage 1: Only train GNN Classifier
                    logits = self.gnn.classifier(batch_emb_gnn)
                    loss = self.criterion_cls(logits, batch_labels)
                else:
                    # Stage 2: Fusion Training
                    logits, z_supcon, alpha = self.fusion(
                        batch_emb_lm, 
                        batch_emb_gnn, 
                        degree=batch_degree
                    )

                    gate_means.append(alpha.mean().item())

                    loss_cls = self.criterion_cls(logits, batch_labels) 
                    loss_supcon = self.criterion_supcon(z_supcon.unsqueeze(1), batch_labels)

                    # 计算双重 Loss
                    loss_entropy = -(alpha * torch.log(alpha + 1e-6) + 
                                   (1 - alpha) * torch.log(1 - alpha + 1e-6)).mean()
                    
                    loss = loss_cls + (self.lambda_supcon * loss_supcon) + (self.lambda_entropy * loss_entropy)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Step Scheduler
            self.scheduler.step()

            # C. Logging & Validation
            avg_loss = total_loss / len(self.dataloader)
            avg_gate = sum(gate_means)/len(gate_means) if len(gate_means) > 0 else 0.0
            
            # Evaluate (修复了 unpack error)
            val_acc, val_f1 = self.evaluate(self.val_nodes, split_name='val')
            
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Gate: {avg_gate:.4f}")
            
            if self.use_wandb:
                log_dict = {
                    "train/loss": avg_loss,
                    "val/f1": val_f1,
                    "val/acc": val_acc,
                    "train/lr": self.scheduler.get_last_lr()[0]
                }
                if not self.pretrain:
                    log_dict["train/gate_mean"] = avg_gate

                    if hasattr(self.fusion, 'gate_temperature'):
                         log_dict["train/gate_temp"] = self.fusion.gate_temperature.item()

                wandb.log(log_dict)

            # Save Best Model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint(epoch)
        
        return self.best_val_f1
    
    
    @torch.no_grad()
    def evaluate(self, nodes, split_name='val'):
        self.gnn.eval()
        self.fusion.eval()
        
        # 1. Forward Pass
        if self.edge_type is not None:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index, self.edge_type)
        else:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index)

        # --- CKA 诊断 (仅在非预训练且开启 WandB 时) ---
        if not self.pretrain and self.use_wandb and split_name == 'val':
            # 随机抽样计算 CKA
            sample_idx = torch.randperm(nodes.size(0))[:512]
            sample_nodes = nodes[sample_idx]
            feat_lm = self.embeddings[sample_nodes]
            feat_gnn = full_gnn_h[sample_nodes]
            cka_score = batch_linear_cka(feat_lm, feat_gnn)
            wandb.log({f"{split_name}/cka_score": cka_score.item()})

        # 2. Inference
        batch_emb_lm = self.embeddings[nodes]
        batch_emb_gnn = full_gnn_h[nodes]
        y_true = self.labels[nodes].cpu().numpy()

        if self.pretrain:
            logits = self.gnn.classifier(batch_emb_gnn)
        else:
            # Stage 2: Fusion
            logits, _, _ = self.fusion(batch_emb_lm, batch_emb_gnn)
            
        preds = logits.argmax(dim=1).cpu().numpy()
        
        # 3. 计算指标
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        
        # [FIX] 必须返回两个值，以匹配 train() 中的解包操作
        return acc, f1

    def save_checkpoint(self, epoch=0):
        self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            # 只在 Stage 2 保存 fusion
            'fusion_state_dict': self.fusion.state_dict() if not self.pretrain else None,
            'best_val_f1': self.best_val_f1
        }
        torch.save(state, self.ckpt_filepath)

    def load_checkpoint(self, path=None):
        p = path or self.ckpt_filepath
        print(f"Loading checkpoint from {p}")
        state = torch.load(p, map_location=self.device)
        
        self.gnn.load_state_dict(state['gnn_state_dict'])
        
        if not self.pretrain and state.get('fusion_state_dict') is not None:
            print("Loading Fusion state dict...")
            self.fusion.load_state_dict(state['fusion_state_dict'], strict=False)

        self.best_val_f1 = state.get('best_val_f1', 0.0)