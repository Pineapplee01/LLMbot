import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from utils import batch_linear_cka, get_stats, calculate_structural_metrics
from torch_geometric.utils import degree as calc_degree, scatter
from AttentionFusion import SupConLoss, OrthogonalityLoss
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
        metadata,
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
        
        self.metadata = metadata.to(device) if metadata is not None else None

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
        self.ortho_loss_fn = OrthogonalityLoss()
        self.criterion_supcon = SupConLoss(temperature=supcon_temp).to(device)
        self.lambda_supcon = lambda_supcon

        # 6. Node Degrees (for sampling if needed)
        num_nodes = self.embeddings.shape[0]
        row, col = self.edge_index
        deg = calc_degree(col, num_nodes=num_nodes, dtype=torch.float)
        deg = torch.log1p(deg)
        deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)
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
            self.optimizer.zero_grad()
            
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
                    outputs = self.fusion(
                        batch_emb_lm, 
                        batch_emb_gnn, 
                    )

                    logits = outputs["logits"]
                    logits_gnn_aux = outputs["logits_gnn_aux"]
                    shared_gnn = outputs["shared_gnn"]
                    private_gnn = outputs["private_gnn"]
                    alpha = outputs["alpha"]

                    gate_means.append(alpha.mean().item())

                    loss_cls = self.criterion_cls(logits, batch_labels) 
                    loss_ortho = self.ortho_loss_fn(shared_gnn, private_gnn)
                    loss_aux = self.criterion_cls(logits_gnn_aux, batch_labels)

                    # 计算双重 Loss
                    loss_entropy = -(alpha * torch.log(alpha + 1e-6) + 
                                   (1 - alpha) * torch.log(1 - alpha + 1e-6)).mean()
                    
                    loss = loss_cls + 0.1 * loss_ortho + 0.3 * loss_aux
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Step Scheduler
            self.scheduler.step()

            # C. Logging & Validation
            avg_loss = total_loss / len(self.dataloader)
            avg_gate = sum(gate_means)/len(gate_means) if len(gate_means) > 0 else 0.0
            
            # Evaluate (修复了 unpack error)
            val_acc, val_f1, error_mask = self.evaluate(self.val_nodes, split_name='val')
            
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
        """
        Modified evaluate to optionally capture error indices.
        """
        self.gnn.eval()
        self.fusion.eval()
        
        # 1. Full Graph Inference (GNN)
        if self.edge_type is not None:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index, self.edge_type)
        else:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index, None)

        # 2. Batch Inference
        batch_emb_lm = self.embeddings[nodes]
        batch_emb_gnn = full_gnn_h[nodes]
         
        y_true = self.labels[nodes].cpu().numpy()

        if self.pretrain:
            logits = self.gnn.classifier(batch_emb_gnn)
        else:
            # Stage 2: Fusion (字典输出适配)
            outputs = self.fusion(batch_emb_lm, batch_emb_gnn)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            
            
        preds = logits.argmax(dim=1).cpu().numpy()
        
        
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')

        # 找出预测错误的 Local Indices (相对于 nodes 数组的索引)
        error_mask = (preds != y_true)
            
        # 获取这些错误的 Global Node Indices (原始图中的 ID)
        # nodes 是 Tensor, error_mask 是 numpy boolean array -> 需要转换一下
        error_mask_tensor = torch.from_numpy(error_mask).bool().to(nodes.device)
        bad_case_indices = nodes[error_mask_tensor]
            
        print(f"[{split_name}] Found {len(bad_case_indices)} bad cases out of {len(nodes)} samples.")
        return acc, f1, bad_case_indices

    def diagnose_errors(self, split=''):
        """
        [SeGA v5.0 Ultimate Diagnostic]
        包含: 三视图推理 + 结构特征分析 + Gate 行为分析
        """
        print("\n" + "="*60)
        print("🔬 [Diagnostic] Running Structure-Aware Analysis...")
        print("="*60)
        
        if split == 'train':
            eval_nodes = self.train_nodes
        elif split == 'val':
            eval_nodes = self.val_nodes
        elif split == 'test':
            eval_nodes = self.test_nodes
        else:
            raise ValueError(f"Unknown split name: {split}")

        self.gnn.eval()
        self.fusion.eval()

        with torch.no_grad():
            x_lm = self.embeddings
            edge_index = self.edge_index
            edge_type = self.edge_type

            # --- Step 0: 计算全图结构特征 ---
            log_deg, homophily = calculate_structural_metrics(x_lm, edge_index)

            # --- Step A: GNN 基础特征 ---
            if edge_type is not None:
                h_gnn_raw = self.gnn(x_lm, edge_index, edge_type)
            else:
                h_gnn_raw = self.gnn(x_lm, edge_index, None)

            # --- Step B: 三视图推理 ---

            # 1. Full
            out_full = self.fusion(x_lm, h_gnn_raw)
            logits_full = out_full['logits'] if isinstance(out_full, dict) else out_full[0]
            
            pred_full, conf_full, ent_full = get_stats(logits_full)
            
            # 2. Text Only (URGA 优先读取 logits_llm)
            if isinstance(out_full, dict) and 'logits_llm' in out_full:
                logits_text = out_full['logits_llm']
            else:
                h_gnn_zeros = torch.zeros_like(h_gnn_raw)
                out_text = self.fusion(x_lm, h_gnn_zeros)
                logits_text = out_text['logits'] if isinstance(out_text, dict) else out_text[0]
            
            pred_text, conf_text, ent_text = get_stats(logits_text)
                
            # 3. Graph Only
            x_lm_zeros = torch.zeros_like(x_lm)
            out_graph = self.fusion(x_lm_zeros, h_gnn_raw)
            logits_graph = out_graph['logits'] if isinstance(out_graph, dict) else out_graph[0]
            
            pred_graph, conf_graph, ent_graph = get_stats(logits_graph)
            
            # --- Step C: 数据收集 ---

            # 提取 Gate
            alpha_vals = torch.zeros(x_lm.shape[0])
            if isinstance(out_full, dict) and 'alpha' in out_full:
                alpha_vals = out_full['alpha'].squeeze()

            idx = eval_nodes.cpu().numpy()
            lbl = self.labels[eval_nodes].cpu().numpy()

            data = {
                'node_idx': idx,
                'label': lbl,
                
                # Full Model
                'pred_full': pred_full[eval_nodes].cpu().numpy(),
                'conf_full': conf_full[eval_nodes].cpu().numpy(),
                'is_correct_full': (pred_full[eval_nodes] == self.labels[eval_nodes]).cpu().numpy(),
                
                # Text Only
                'pred_text': pred_text[eval_nodes].cpu().numpy(),
                'conf_text': conf_text[eval_nodes].cpu().numpy(),
                'is_correct_text': (pred_text[eval_nodes] == self.labels[eval_nodes]).cpu().numpy(),
                
                # Graph Only
                'pred_graph': pred_graph[eval_nodes].cpu().numpy(),
                'conf_graph': conf_graph[eval_nodes].cpu().numpy(),
                'is_correct_graph': (pred_graph[eval_nodes] == self.labels[eval_nodes]).cpu().numpy(),
                
                # Analysis Metrics
                'gate': alpha_vals[eval_nodes].cpu().numpy(),
                'entropy': ent_full[eval_nodes].cpu().numpy(),
                'log_degree': log_deg[eval_nodes].cpu().numpy(),
                'homophily': homophily[eval_nodes].cpu().numpy()
            }

            df = pd.DataFrame(data)
            
            conditions = [
                (df['is_correct_text'] == False) & (df['is_correct_graph'] == True) ,
                (df['is_correct_text'] == True)  & (df['is_correct_graph'] == False),
                (df['is_correct_text'] == False) & (df['is_correct_graph'] == False),
                (df['is_correct_text'] == True)  & (df['is_correct_graph'] == True)
            ]
            choices = ['Text', 'Graph ', 'Both ', 'Clean']
            df['case_type'] = np.select(conditions, choices, default='Unknown')

            print("\n📊 [Diagnostic Summary]")
            print(df['case_type'].value_counts())
            
            return df

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