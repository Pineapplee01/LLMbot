import gc
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from utils import batch_linear_cka, get_stats, calculate_structural_metrics, analyze_uncertainty_dist
from torch_geometric.utils import degree as calc_degree, scatter
from AttentionFusion import SupConLoss, OrthogonalityLoss
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

class QwenPrecomputedTrainer:
    def __init__(
        self,
        precomputed_embeddings, 
        gnn_model,
        fusion_model,
        pretrain_gnn, 
        pretrain_llm, # 仅作为标记保留
        data_dict,
        device,
        epochs,
        lr,
        weight_decay,
        metadata,
        sample,
        ckpt_filepath='best_model.pt',
        supcon_temp=0.07,
        lambda_supcon=0.1, 
        **kwargs,
    ):
        self.device = device
        self.pretrain_gnn = pretrain_gnn
        self.pretrain_llm = pretrain_llm
        self.ckpt_filepath = Path(ckpt_filepath)
        self.epochs = epochs
        self.sample = sample
        
        # 1. Load Data
        self.embeddings = precomputed_embeddings.to(device)
        self.edge_index = data_dict['edge_index'].to(device)
        num_nodes = self.embeddings.shape[0]
        
        self.metadata = metadata.to(device) if metadata is not None else None
        self.edge_type = data_dict.get('edge_type', None)
        if self.edge_type is not None: 
            self.edge_type = self.edge_type.to(device)
            
        raw_labels = data_dict['labels'].to(device)
        if raw_labels.dim() > 1 and raw_labels.shape[1] > 1:
            self.labels = raw_labels.argmax(dim=1).long()
        else:
            self.labels = raw_labels.long()
            
        self.train_nodes = data_dict['train_idx'].to(device)
        self.val_nodes = data_dict['valid_idx'].to(device)
        self.test_nodes = data_dict['test_idx'].to(device)
            
        # 2. Models
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.ortho_loss_fn = OrthogonalityLoss()
        self.criterion_supcon = SupConLoss(temperature=supcon_temp).to(device)
        self.lambda_supcon = lambda_supcon

        # [Meta Features Containers]
        self.entropy_gnn = None
        self.jsd = None

        # 3. Precompute Structure
        print("[Trainer] Precomputing structural metrics...")
        self._precompute_structure_metrics(num_nodes)

        # 4. Optimizer Setup
        # [Fix] 这里的逻辑只决定 main loop (train) 的模式：GNN 或 Fusion
        # pretrain_llm 有自己的独立方法和优化器，不应影响主优化器的初始化
        if self.pretrain_gnn:
            print("[Trainer] Mode: GNN Pre-training (Stage 2)")
            self.optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            # Fusion Mode (Stage 3)
            print(f"[Trainer] Mode: Fusion Training (Stage 3) | Differential LR: Fusion={lr}, GNN={lr*0.01}")
            self.optimizer = torch.optim.AdamW([
                {'params': self.fusion.parameters(), 'lr': lr},
                {'params': self.gnn.parameters(),    'lr': lr * 0.01}
            ], weight_decay=weight_decay)

        # Scheduler
        if self.optimizer:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        
        self.best_val_f1 = 0.0
        self.use_wandb = wandb.run is not None
        self._init_dataloader()

    def _precompute_structure_metrics(self, num_nodes):
        with torch.no_grad():
            row, col = self.edge_index
            deg = calc_degree(col, num_nodes=num_nodes, dtype=torch.float).clamp(min=1)
            deg_log = torch.log1p(deg)
            self.node_degrees = ((deg_log - deg_log.min()) / (deg_log.max() - deg_log.min() + 1e-6)).unsqueeze(1)
            
            x_norm = F.normalize(self.embeddings, p=2, dim=1)
            indices = torch.stack([col, row]) 
            values = 1.0 / deg[col]
            
            adj_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
            neighbor_mean = torch.sparse.mm(adj_sparse, x_norm)
            
            mask_isolated = (deg == 0).unsqueeze(1)
            neighbor_mean = torch.where(mask_isolated, x_norm, neighbor_mean)
            neighbor_mean = F.normalize(neighbor_mean, p=2, dim=1)
            self.homophily = (x_norm * neighbor_mean).sum(dim=1, keepdim=True).clamp(0, 1)
        
        torch.cuda.empty_cache()

    def _init_dataloader(self):
        train_labels = self.labels[self.train_nodes].cpu().numpy()
        class_counts = np.bincount(train_labels)
        class_counts[class_counts == 0] = 1
        sample_weights = (1. / class_counts)[train_labels]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_dataset = torch.utils.data.TensorDataset(self.train_nodes, self.labels[self.train_nodes])
        self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=sampler, drop_last=True)

    def precompute_meta_features(self):
        """Precompute JSD and GNN Entropy before Stage 3"""
        print("[Trainer] Precomputing Meta-Features (GNN Entropy & JSD)...")
        self.gnn.eval()
        self.fusion.eval()
        
        with torch.no_grad():
            # 1. GNN Prob
            if self.edge_type is not None:
                h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
            else:
                h_gnn = self.gnn(self.embeddings, self.edge_index)
            
            logits_gnn = self.gnn.classifier(h_gnn)
            prob_gnn = F.softmax(logits_gnn, dim=1)
            self.entropy_gnn = -torch.sum(prob_gnn * torch.log(prob_gnn + 1e-9), dim=1, keepdim=True)
            
            # 2. Text VIB Prob
            logits_lm, _, _, _ = self.fusion.vib(self.embeddings)
            prob_lm = F.softmax(logits_lm, dim=1)
            
            # 3. JSD
            m = 0.5 * (prob_lm + prob_gnn)
            kl_lm = F.kl_div(torch.log(prob_lm + 1e-9), m, reduction='none').sum(dim=1, keepdim=True)
            kl_gnn = F.kl_div(torch.log(prob_gnn + 1e-9), m, reduction='none').sum(dim=1, keepdim=True)
            self.jsd = 0.5 * (kl_lm + kl_gnn)
            
        print("[Trainer] Meta-Features computed.")

    def pretrain_text_vib(self, epochs=15):
        """[Stage 1] VIB Pre-training"""
        print(f"\n🚀 [Stage 1] Starting Text Expert (VIB) Pre-training for {epochs} epochs...")
        # 定义局部优化器，不影响主流程
        optimizer = torch.optim.AdamW(self.fusion.vib.parameters(), lr=1e-3, weight_decay=1e-4)
        best_acc = 0.0
        lambda_vib = 1e-3 
        
        vib_ckpt_path = self.ckpt_filepath.parent / "best_text_vib.pt"

        for epoch in range(epochs):
            self.fusion.vib.train()
            total_loss = 0
            
            for batch_nodes, batch_labels in self.dataloader:
                optimizer.zero_grad()
                batch_nodes, batch_labels = batch_nodes.to(self.device), batch_labels.to(self.device)
                
                logits, _, _, kl_loss = self.fusion.vib(self.embeddings[batch_nodes])
                loss = self.criterion_cls(logits, batch_labels) + lambda_vib * kl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_acc = self._validate_text_only()
            print(f"   VIB Ep {epoch+1:02d} | Loss: {total_loss/len(self.dataloader):.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.fusion.vib.state_dict(), vib_ckpt_path)

        print(f"✅ Text Expert Finished. Best Acc: {best_acc:.4f}")
        # 加载最好的 VIB 权重，为后续 Fusion 做准备
        if vib_ckpt_path.exists():
            self.fusion.vib.load_state_dict(torch.load(vib_ckpt_path))  

    def _validate_text_only(self):
        self.fusion.vib.eval()
        preds, targets = [], []
        with torch.no_grad():
            emb = self.embeddings[self.val_nodes]
            lbl = self.labels[self.val_nodes]
            logits, _, _, _ = self.fusion.vib(emb)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(lbl.cpu())
        return accuracy_score(torch.cat(targets), torch.cat(preds))

    def train(self):
        """
        Handles Stage 2 (GNN Pretrain) OR Stage 3 (Fusion Train)
        Note: Stage 1 is handled explicitly by pretrain_text_vib()
        """
        WARMUP_EPOCHS = 5          
        lambda_oracle = 0.5        
        lambda_consist = 0.1       
        margin_val = 0.2
        lambda_vib = 1e-3

        # Fusion 阶段需要元特征
        if not self.pretrain_gnn and self.entropy_gnn is None:
            self.precompute_meta_features()

        for epoch in range(self.epochs):
            self.gnn.train()
            if not self.pretrain_gnn:
                self.fusion.train()
                
            self.optimizer.zero_grad()
            
            total_loss = 0
            gate_means = []
            
            curr_lambda_oracle = lambda_oracle if epoch >= WARMUP_EPOCHS else 0.0
            curr_lambda_consist = lambda_consist if epoch >= WARMUP_EPOCHS else 0.0
            
            for batch_nodes, batch_labels in self.dataloader:
                self.optimizer.zero_grad()
                batch_nodes, batch_labels = batch_nodes.to(self.device), batch_labels.to(self.device)

                # GNN Forward
                if self.edge_type is not None:
                    h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
                else:
                    h_gnn = self.gnn(self.embeddings, self.edge_index)
                
                batch_emb_gnn = h_gnn[batch_nodes]

                if self.pretrain_gnn:
                    # [Stage 2] GNN Only
                    logits = self.gnn.classifier(batch_emb_gnn)
                    loss = self.criterion_cls(logits, batch_labels)
                    loss_oracle, loss_consist, kl_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
                else:
                    # [Stage 3] Fusion
                    batch_emb_lm = self.embeddings[batch_nodes]
                    
                    outputs = self.fusion(
                        lm_emb=batch_emb_lm,      
                        gnn_emb=batch_emb_gnn,    
                        homophily=self.homophily[batch_nodes], 
                        degree=self.node_degrees[batch_nodes],
                        entropy_gnn=self.entropy_gnn[batch_nodes],
                        jsd=self.jsd[batch_nodes]
                    )

                    logits, alpha, kl_loss = outputs["logits"], outputs["alpha"], outputs["kl_loss"]
                    gate_means.append(alpha[:, 0].mean().item())

                    loss_cls = self.criterion_cls(logits, batch_labels) 
                    loss_aux = 0.5 * (self.criterion_cls(outputs["logits_lm"], batch_labels) + 
                                      self.criterion_cls(outputs["logits_gnn"], batch_labels))
                    
                    # Oracle Loss
                    loss_oracle = torch.tensor(0.0, device=self.device)
                    if curr_lambda_oracle > 0:
                        with torch.no_grad():
                            pred_lm = outputs["logits_lm"].argmax(dim=1)
                            pred_gnn = outputs["logits_gnn"].argmax(dim=1)
                            target_rank = torch.zeros_like(batch_labels, dtype=torch.float)
                            target_rank[(pred_lm == batch_labels) & (pred_gnn != batch_labels)] = 1.0
                            target_rank[(pred_lm != batch_labels) & (pred_gnn == batch_labels)] = -1.0
                            loss_mask = (target_rank != 0)

                        if loss_mask.sum() > 0:
                            loss_oracle = F.margin_ranking_loss(
                                outputs["r_lm_logits"].squeeze()[loss_mask], 
                                outputs["r_gnn_logits"].squeeze()[loss_mask], 
                                target_rank[loss_mask], 
                                margin=margin_val
                            )

                    loss_consist = torch.tensor(0.0, device=self.device)
                    if curr_lambda_consist > 0:
                        loss_consist = self.fusion.get_structure_consistency_loss(outputs["alpha"], self.homophily[batch_nodes])

                    loss = loss_cls + loss_aux + curr_lambda_oracle * loss_oracle + curr_lambda_consist * loss_consist + lambda_vib * kl_loss
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = total_loss / len(self.dataloader)
            avg_gate = sum(gate_means)/len(gate_means) if len(gate_means) > 0 else 0.0
            
            val_acc, val_f1, _ = self.evaluate(self.val_nodes, split_name='val')
            
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | Gate: {avg_gate:.4f}")
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint(epoch)
        
        return self.best_val_f1

    @torch.no_grad()
    def evaluate(self, nodes, split_name='val'):
        self.gnn.eval()
        self.fusion.eval()
        nodes = nodes.to(self.device)
        
        if self.edge_type is not None:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index, self.edge_type)
        else:
            full_gnn_h = self.gnn(self.embeddings, self.edge_index)

        batch_emb_gnn = full_gnn_h[nodes]
        y_true = self.labels[nodes].cpu().numpy()

        if self.pretrain_gnn:
            logits = self.gnn.classifier(batch_emb_gnn)
        else:
            if self.entropy_gnn is None: self.precompute_meta_features()
            batch_emb_lm = self.embeddings[nodes]
            outputs = self.fusion(
                lm_emb=batch_emb_lm, 
                gnn_emb=batch_emb_gnn, 
                homophily=self.homophily[nodes], 
                degree=self.node_degrees[nodes],
                entropy_gnn=self.entropy_gnn[nodes],
                jsd=self.jsd[nodes]
            )
            logits = outputs["logits"]
            
        preds = logits.argmax(dim=1).cpu().numpy()
        return accuracy_score(y_true, preds), f1_score(y_true, preds, average='macro'), nodes[torch.from_numpy(preds != y_true).to(self.device)]

    def diagnose_errors(self, split=''):
        print(f"🔬 [Diagnostic] Running Bad Case Study for {split}...")
        eval_nodes = getattr(self, f"{split}_nodes").to(self.device)
        # 转为 numpy 用于后续字典取值
        eval_nodes_np = eval_nodes.cpu().numpy()
        
        self.gnn.eval(); self.fusion.eval()
        if self.entropy_gnn is None: self.precompute_meta_features()

        with torch.no_grad():
            if self.edge_type is not None:
                h_gnn = self.gnn(self.embeddings, self.edge_index, self.edge_type)
            else:
                h_gnn = self.gnn(self.embeddings, self.edge_index)
            
            # 全量推理
            out = self.fusion(
                lm_emb=self.embeddings, gnn_emb=h_gnn,
                homophily=self.homophily, degree=self.node_degrees,
                entropy_gnn=self.entropy_gnn, jsd=self.jsd
            )
            prob_full = F.softmax(out['logits'], dim=1)
            
            # 单模态推理
            out_text, _, _, _ = self.fusion.vib(self.embeddings)
            prob_text = F.softmax(out_text, dim=1)
            prob_graph = F.softmax(self.gnn.classifier(h_gnn), dim=1)

            # [Fix] 关键修改：先从全量结果中切片，再比较
            # 1. 提取当前 Split 对应的预测结果 (N_eval,)
            pred_full_slice = prob_full.argmax(1)[eval_nodes].cpu().numpy()
            pred_text_slice = prob_text.argmax(1)[eval_nodes].cpu().numpy()
            pred_graph_slice = prob_graph.argmax(1)[eval_nodes].cpu().numpy()
            
            # 2. 提取当前 Split 对应的标签 (N_eval,)
            label_slice = self.labels[eval_nodes].cpu().numpy()

            def get_slice(t): 
                # 用于提取特征（如 homophily, gate 等）
                return t[eval_nodes].cpu().numpy()

            data = {
                'node_idx': eval_nodes_np,
                'label': label_slice,
                
                # Full Prediction
                'pred_full': pred_full_slice,
                'conf_full': get_slice(prob_full.max(1).values),
                'is_correct_full': (pred_full_slice == label_slice), # 此时维度一致 (8278,) == (8278,)
                
                # Text Prediction
                'pred_text': pred_text_slice,
                'conf_text': get_slice(prob_text.max(1).values),
                'is_correct_text': (pred_text_slice == label_slice),
                
                # Graph Prediction
                'pred_graph': pred_graph_slice,
                'conf_graph': get_slice(prob_graph.max(1).values),
                'is_correct_graph': (pred_graph_slice == label_slice),
                
                # Metrics
                'sigma_text': get_slice(out['sigma_text'].squeeze()),
                'gate': get_slice(out['alpha'][:, 0]),
                'homophily': get_slice(self.homophily.squeeze())
            }
            
            df = pd.DataFrame(data)
            
            # 标记 Case Type
            c = df
            cond = [
                (c['is_correct_text']) & (c['is_correct_graph']),
                (c['is_correct_text']) & (~c['is_correct_graph']),
                (~c['is_correct_text']) & (c['is_correct_graph']),
                (~c['is_correct_text']) & (~c['is_correct_graph'])
            ]
            df['case_type'] = np.select(cond, ['Easy', 'Text_Only', 'Graph_Only', 'Hard'], default='Unknown')
            
            print("\n[Diagnostic Summary]")
            print(df['case_type'].value_counts())
            print("\n[Metric Deep Dive]")
            print(df.groupby('case_type')[['sigma_text', 'gate', 'homophily']].mean())
            return df

    def save_checkpoint(self, epoch=0):
        self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {'epoch': epoch, 'best_val_f1': self.best_val_f1}
        # 如果是 GNN 预训练模式，只存 GNN
        if self.pretrain_gnn:
            state['gnn_state_dict'] = self.gnn.state_dict()
        # 否则 (Fusion模式)，存 Fusion 和 GNN (因为 GNN 可能被微调)
        else:
            state['gnn_state_dict'] = self.gnn.state_dict()
            state['fusion_state_dict'] = self.fusion.state_dict()
        torch.save(state, self.ckpt_filepath)

    def load_checkpoint(self, path=None):
        p = path or self.ckpt_filepath
        print(f"Loading checkpoint from {p}")
        state = torch.load(p, map_location=self.device)
        if 'gnn_state_dict' in state:
            self.gnn.load_state_dict(state['gnn_state_dict'])
        elif 'gnn' in state: # 兼容旧key
            self.gnn.load_state_dict(state['gnn'])
            
        if not self.pretrain_gnn:
            if 'fusion_state_dict' in state:
                print("Loading Fusion state dict...")
                self.fusion.load_state_dict(state['fusion_state_dict'], strict=False)
            elif 'fusion' in state:
                self.fusion.load_state_dict(state['fusion'], strict=False)
        self.best_val_f1 = state.get('best_val_f1', 0.0)