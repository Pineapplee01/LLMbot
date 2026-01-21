import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from AttentionFusion import SupConLoss
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR


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
        supcon_temp=0.1,  # 温度系数，越小越关注 Hard Samples
        lambda_supcon=0.1, # 对比损失的权重
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
        num_nodes = self.embeddings.shape[0] # Total nodes
        
        # Handle Edge Types
        self.edge_type = data_dict.get('edge_type', None)
        if self.edge_type is not None: 
            self.edge_type = self.edge_type.to(device)
            
        # 2. Handle Labels
        raw_labels = data_dict['labels'].to(device)
        if raw_labels.dim() > 1 and raw_labels.shape[1] > 1:
            print(f"[Data] Detected One-Hot Labels {raw_labels.shape}. Converting to Class Indices.")
            self.labels = raw_labels.argmax(dim=1).long()
        else:
            self.labels = raw_labels.long()
            
        # 3. Initialize Masks 
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # 4. Fill Masks based on input type
        if 'train_mask' in data_dict:
            # Case A: Input is already Boolean Masks
            print("[Data] Using Boolean Masks from dataset.")
            self.train_nodes = data_dict['train_mask'].to(device)
            self.val_nodes = data_dict['val_mask'].to(device)
            self.test_nodes = data_dict['test_mask'].to(device)
            
            self.train_mask = self.train_nodes.bool()
            self.val_mask = self.val_nodes.bool()
            self.test_mask = self.test_nodes.bool()
            
        elif 'train_idx' in data_dict:
            # Case B: Input is Indices (YOUR CASE)
            print("[Data] Using Node Indices. Converting to Boolean Masks for Loss.")
            self.train_nodes = data_dict['train_idx'].to(device)
            self.test_nodes = data_dict['test_idx'].to(device)
            
            if 'val_idx' in data_dict:
                self.val_nodes = data_dict['val_idx'].to(device)
            elif 'valid_idx' in data_dict:
                self.val_nodes = data_dict['valid_idx'].to(device)
            else:
                # Create validation split if missing
                print("[Data] Warning: No validation split found. Creating 10% split.")
                perm = torch.randperm(len(self.train_nodes))
                split = int(len(self.train_nodes) * 0.9)
                self.val_nodes = self.train_nodes[perm[split:]]
                self.train_nodes = self.train_nodes[perm[:split]]

            # HERE IS THE FIX: Manually set True at indices
            self.train_mask[self.train_nodes] = True
            self.val_mask[self.val_nodes] = True
            self.test_mask[self.test_nodes] = True
            
        else:
            raise KeyError(f"Data dictionary keys: {data_dict.keys()} do not contain train splits.")

        # 5. Models & Optimizer
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_supcon = SupConLoss(temperature=supcon_temp, base_temperature=supcon_temp).to(device)
        self.lambda_supcon = lambda_supcon
        
        if self.pretrain:
            print(">>> [Trainer] Mode: GNN Pre-training (Frozen Fusion)")
            # 阶段一：只优化 GNN 的参数
            self.optimizer = torch.optim.AdamW(
                self.gnn.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            # 冻结 Fusion 以防万一
            for param in self.fusion.parameters():
                param.requires_grad = False

        else:
            print(">>> [Trainer] Mode: Joint Fusion Training")
            # 阶段二：优化 GNN + Fusion (你的原有逻辑)
            # 确保 Fusion 解冻
            for param in self.fusion.parameters():
                param.requires_grad = True

            fusion_params = list(map(id, self.fusion.contrastive_head.parameters()))
            base_params = filter(lambda p: id(p) not in fusion_params, 
                                 list(self.gnn.parameters()) + list(self.fusion.parameters()))

            self.optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': lr}, 
                {'params': self.fusion.contrastive_head.parameters(), 'lr': lr * 5}
            ], weight_decay=weight_decay)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.best_val_f1 = 0.0
        self._epoch = 0


    def train(self):

        # Wandb check
        if wandb.run is None:
            wandb.init(project="Qwenbot", name="SeGA_SupCon_Training")

        print(f"[{self.device}] Starting SeGA Training with SupCon...")
        
        patience = 20
        counter = 0

        for epoch in range(self.epochs):
            self._epoch = epoch
            self.gnn.train()

            if not self.pretrain:
                self.fusion.train()

            self.optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. GNN
            h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type) if self.edge_type is not None else self.gnn(self.embeddings, self.edge_index)

            # 2. Fusion
            if self.pretrain:
                # [阶段一] GNN 独立训练
                # 直接调用 GNN 内部的 classifier (GNNs.py 中定义的)
                logits = self.gnn.classifier(h_gnn)
                
                # Pretrain 阶段我们主要看 GNN 能不能分类，SupCon 也可以加在 h_gnn 上
                contrast_feat = h_gnn 
                alpha = torch.tensor([0.0]) # 占位符
            
            else:
                # [阶段二] Fusion 联合训练
                logits, contrast_feat, alpha = self.fusion(lm_features=self.embeddings, gnn_features=h_gnn)
            
            # --- Main Loss (Cross Entropy) ---
            loss_cls = self.criterion_cls(logits[self.train_nodes], self.labels[self.train_nodes])

            # --- Structural Loss ---
            train_feats = contrast_feat[self.train_nodes]
            train_labels = self.labels[self.train_nodes]

            train_feats_all = F.normalize(train_feats, dim=1)
            
            if self.sample == 'random':
            
                if train_feats_all.shape[0] > 2048:
                    perm = torch.randperm(train_feats_all.shape[0])[:2048]
                    batch_feats = train_feats_all[perm]
                    batch_labels = train_labels[perm]

                else:
                    batch_feats = train_feats_all
                    batch_labels = train_labels
                

            elif self.sample == 'hard':

                with torch.no_grad():

                    sim_matrix = torch.matmul(train_feats_all, train_feats_all.T)
                    sim_matrix.fill_diagonal_(-float('inf'))

                    weights = F.softmax(sim_matrix / 0.1, dim=1)
                    hard_indices = torch.multinomial(weights, num_samples=1).squeeze()
                
                batch_feats = train_feats_all[hard_indices]
                batch_labels = train_labels[hard_indices]

            loss_supcon = self.criterion_supcon(batch_feats, batch_labels)
            
            # 3. Total Loss (Multi-task Learning)
            current_lambda = self.lambda_supcon if epoch >= 3 else 0.0
            
            if not self.pretrain and epoch > 40 and loss_supcon > 7.0:
                current_lambda *= 0.5
            
            loss = loss_cls + (current_lambda * loss_supcon)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step() 
            
            # --- Validation ---
            val_acc, val_f1 = self.evaluate(self.val_nodes)
            
            # Logging
            log_dict = {
                'epoch': epoch, 
                'loss_total': loss.item(), 
                'loss_cls': loss_cls.item(),
                'loss_supcon': loss_supcon.item(), # 监控对比损失是否在下降
                'val_f1': val_f1, 
                'lr': self.scheduler.get_last_lr()[0],
                'gate_alpha_mean': alpha.mean().item() # 监控模型更偏向 Text 还是 Graph
            }
            wandb.log(log_dict)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | CE: {loss_cls.item():.4f} | SupCon: {loss_supcon.item():.4f} | Total: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val ACC: {val_acc:.4f} | Gate α: {alpha.mean().item():.4f}")
            
            # Checkpointing
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint(epoch)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.best_val_f1
    
    
    def evaluate(self, nodes):
        self.gnn.eval()
        if not self.pretrain:
            self.fusion.eval()

        with torch.no_grad():

            if self.edge_type is not None:
                h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
            else:
                h_gnn = self.gnn(self.embeddings, self.edge_index)
            
            if self.pretrain:
                # 评估 GNN 自己的分类头
                logits = self.gnn.classifier(h_gnn)
            else:
                # 评估 Fusion 的分类头
                logits, _, _ = self.fusion(self.embeddings, h_gnn)

            preds = logits.argmax(dim=1)
            y_true = self.labels[nodes].cpu().numpy()
            y_pred = preds[nodes].cpu().numpy()
            
            return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


    def save_checkpoint(self, epoch=0):
        self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }
        torch.save(state, self.ckpt_filepath)


    def load_checkpoint(self, path=None):
        p = path or self.ckpt_filepath
        state = torch.load(p, map_location=self.device)
        self.gnn.load_state_dict(state['gnn_state_dict'])

        if 'fusion_state_dict' in state:
            self.fusion.load_state_dict(state['fusion_state_dict'], strict=False)

        self.best_val_f1 = state['best_val_f1']
        print(f"Loaded checkpoint from epoch {state['epoch']}")