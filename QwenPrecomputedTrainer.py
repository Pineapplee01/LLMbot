import torch
import wandb
import numpy as np
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR


class QwenPrecomputedTrainer:
    def __init__(
        self,
        precomputed_embeddings,  # (num_users, 4096)
        gnn_model,
        fusion_model,
        data_dict,
        device,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        ckpt_filepath='best_model.pt',
        **kwargs,
    ):
        self.device = device
        self.ckpt_filepath = Path(ckpt_filepath)
        self.epochs = epochs
        
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
            
        # 3. CRITICAL FIX: Initialize Masks as False first
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
        
        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters()) + list(self.fusion.parameters()),
            lr=lr, 
            weight_decay=weight_decay
        )

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_f1 = 0.0
        self._epoch = 0

    def structural_loss(self, z, edge_index, labels, mask):
        """
        Calculates L_struct only on edges where BOTH nodes are in the training mask.
        Prevents Data Leakage from Test Set.
        """
        src, dst = edge_index
        
        # 1. Filter Edges: Both src and dst must be in the current mask (e.g., train_mask)
        # We create a boolean mask for edges
        edge_mask = mask[src] & mask[dst]
        
        if edge_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        # Keep only valid edges
        src_valid = src[edge_mask]
        dst_valid = dst[edge_mask]
        
        # 2. Calculate Label Agreement (Homophily)
        # 1 if same class, 0 if different
        # Note: labels can be [N] or [N, C] (one-hot). Handle both.
        if labels.dim() > 1:
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels
            
        same_class = (labels_idx[src_valid] == labels_idx[dst_valid]).float()
        
        # 3. Calculate Euclidean Distance (Squared)
        # Since z is normalized, ||u - v||^2 = 2 - 2*cos(u, v)
        # So this minimizes the angle between same-class neighbors.
        diff = (z[src_valid] - z[dst_valid]).pow(2).sum(dim=-1)
        
        # 4. Average Loss
        # We only penalize when (same_class == 1). 
        # We divide by the number of 'same' edges to keep scale consistent.
        loss = (same_class * diff).sum()
        
        # Avoid division by zero
        num_same_edges = same_class.sum() + 1e-6
        
        return loss / num_same_edges


    def train(self):
        # Wandb check
        if wandb.run is None:
            wandb.init(project="Qwenbot", name="SeGA_Training")

        print(f"[{self.device}] Starting SeGA Training (Full Batch)...")
        
        patience = 20
        counter = 0

        for epoch in range(self.epochs):
            self._epoch = epoch
            self.gnn.train()
            self.fusion.train()
            self.optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. GNN
            h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type) if self.edge_type is not None else self.gnn(self.embeddings, self.edge_index)

            # 2. Fusion
            logits, h_fused = self.fusion(lm_features=self.embeddings, gnn_features=h_gnn)
            
            # --- Main Loss (Cross Entropy) ---
            loss_cls = self.criterion(logits[self.train_nodes], self.labels[self.train_nodes])

            # --- Structural Loss ---
            loss_struct = self.structural_loss(
                z=h_fused, 
                edge_index=self.edge_index, 
                labels=self.labels, 
                mask=self.train_mask
            )

            current_lambda = 0.0
            if epoch >= 5:
                current_lambda = 0.1  # Tuning Target: 0.05 to 0.5

            loss = loss_cls + (current_lambda * loss_struct)

            loss.backward()
            self.optimizer.step()
            
            # --- Validation ---
            val_acc, val_f1 = self.evaluate(self.val_nodes)
            
            # Logging
            wandb.log({
                'epoch': epoch, 
                'train_loss': loss.item(), 
                'val_f1': val_f1, 
                'val_acc': val_acc,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | CE: {loss_cls.item():.4f} | Struct: {loss_struct.item():.4f} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

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
        self.fusion.eval()
        with torch.no_grad():
            if self.edge_type is not None:
                h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
            else:
                h_gnn = self.gnn(self.embeddings, self.edge_index)
            
            logits, _ = self.fusion(self.embeddings, h_gnn)
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
        self.fusion.load_state_dict(state['fusion_state_dict'])
        self.best_val_f1 = state['best_val_f1']
        print(f"Loaded checkpoint from epoch {state['epoch']}")