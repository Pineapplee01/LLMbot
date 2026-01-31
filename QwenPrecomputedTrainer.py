import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pandas as pd
import numpy as np

from utils import EdlLoss
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.utils import degree, scatter
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset

class MiniBatch:
    def __init__(self, n_id, edge_index, edge_type, batch_size):
        self.n_id = torch.tensor(n_id, dtype=torch.long)
        self.edge_index = edge_index
        self.batch_size = batch_size
        self.edge_type = edge_type
    
    def to(self, device):
        self.n_id = self.n_id.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_type is not None:
            self.edge_type = self.edge_type.to(device)
        return self

class NeighborSampler(DataLoader):
    """
    [Pure Python] 简易图邻居采样器
    替代 torch_geometric.loader.NeighborLoader，解决缺少 pyg-lib/torch-sparse 的报错问题。
    """
    def __init__(self, edge_index, edge_type, sizes, batch_size, input_nodes, shuffle=True, num_workers=0, **kwargs):
        self.edge_index = edge_index
        self.edge_type = edge_type # [NEW]
        self.sizes = sizes
        self.input_nodes = input_nodes
        self.num_nodes = edge_index.max().item() + 1
        
        print("Creating adjacency list (with types) for RobustNeighborSampler... (This runs once)")
        self.adj = [[] for _ in range(self.num_nodes)]
        self.adj_t = [[] for _ in range(self.num_nodes)] # [NEW] 存储边类型
        
        rows = edge_index[0].tolist()
        cols = edge_index[1].tolist()
        
        # 如果有 edge_type，转换为 list；否则为 None
        types = edge_type.tolist() if edge_type is not None else None
        
        # 构建反向图 (Target -> Source)
        if types is not None:
            for r, c, t in zip(rows, cols, types):
                self.adj[c].append(r)
                self.adj_t[c].append(t)
        else:
            for r, c in zip(rows, cols):
                self.adj[c].append(r)
            
        print("✅ Adjacency list created.")

        super().__init__(
            TensorDataset(input_nodes), 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.sample_subgraph,
            num_workers=num_workers
        )

    def sample_subgraph(self, batch_indices):
        seed_nodes = [item[0].item() for item in batch_indices]
        batch_size = len(seed_nodes)
        
        n_id = list(seed_nodes)
        n_id_set = set(seed_nodes)
        node_map = {n: i for i, n in enumerate(n_id)}
        
        edges_src = []
        edges_dst = []
        edges_type = [] # [NEW]
        
        current_layer_nodes = seed_nodes
        
        # BFS 采样
        for size in self.sizes:
            next_layer_nodes = []
            for target_node in current_layer_nodes:
                neighbors = self.adj[target_node]
                if len(neighbors) == 0: continue
                
                # 随机采样索引
                if len(neighbors) > size:
                    # 获取随机索引，以便同时取 neighbor 和 type
                    indices = np.random.choice(len(neighbors), size, replace=False)
                else:
                    indices = range(len(neighbors))
                
                for idx in indices:
                    source_node = neighbors[idx]
                    
                    if source_node not in n_id_set:
                        n_id_set.add(source_node)
                        n_id.append(source_node)
                        node_map[source_node] = len(n_id) - 1
                        next_layer_nodes.append(source_node)
                    
                    u = node_map[source_node]
                    v = node_map[target_node]
                    edges_src.append(u)
                    edges_dst.append(v)
                    
                    # 记录 Type
                    if self.edge_type is not None:
                        t = self.adj_t[target_node][idx]
                        edges_type.append(t)
            
            current_layer_nodes = next_layer_nodes

        if len(edges_src) > 0:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_type = torch.tensor(edges_type, dtype=torch.long) if self.edge_type is not None else None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long) if self.edge_type is not None else None
            
        return MiniBatch(n_id, edge_index, edge_type, batch_size)
    
class QwenPrecomputedTrainer:
    def __init__(
        self,
        precomputed_embeddings, 
        gnn_model,
        fusion_model,
        data_dict,
        dataloader,
        batch_size,
        device,
        epochs,
        lr,
        weight_decay,
        ckpt_filepath='best_model.pt',
        lambda_avuc=0.1,
        lambda_struct=0.1,
        lambda_supcon=0.1,
        supcon_temp=0.07,
        pretrain_gnn=False,
        pretrain_llm=False,
        pretrain_gate=False,
        sample='random',
        metadata=None,
        **kwargs,
    ):
        self.device = device
        self.ckpt_filepath = Path(ckpt_filepath)
        self.dataloader = dataloader
        self.batch_size =batch_size
        
        # Models
        self.embeddings = precomputed_embeddings.to('cpu')
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)
        
        # Configs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_avuc = lambda_avuc
        self.lambda_struct = lambda_struct
        self.lambda_supcon = lambda_supcon
        self.supcon_temp = supcon_temp
        self.use_wandb = wandb.run is not None
        
        # Flags
        self.pretrain_gnn = pretrain_gnn
        self.pretrain_llm = pretrain_llm
        self.pretrain_gate = pretrain_gate 
        
        # Data & Labels
        self.data_dict = data_dict
        self.labels = data_dict['labels'].to(device)
        if self.labels.dim() > 1 and self.labels.shape[1] > 1:
            self.labels = self.labels.argmax(dim=1)

        self.labels = self.labels.long()

        self.full_data = Data(
            x=self.embeddings, # CPU tensor
            y=self.labels,     # CPU tensor
            edge_index=data_dict['edge_index'], # CPU tensor
            edge_type=data_dict.get('edge_type', None)
        )

        # Graph Structure
        self.edge_index = data_dict['edge_index'].to(device)
        self.edge_type = data_dict['edge_type'].to(device) if 'edge_type' in data_dict else None
        
        # Meta Features
        self.homophily = self._precompute_homophily()
        self.node_degrees = self._precompute_degrees()
        
        # DataLoaders
        self.train_idx = data_dict['train_idx']
        self.val_idx = data_dict['valid_idx']
        self.test_idx = data_dict['test_idx']
        self.dataloader = self._create_dataloader(self.train_idx, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_idx, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_idx, shuffle=False)

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.edl_loss = EdlLoss()
        


        self._setup_optimizer()

    # =========================================================================
    #  🛠️ Helpers & Setup
    # =========================================================================

    def _create_dataloader(self, indices, shuffle):
        """
        [NEW] 根据配置选择 NeighborLoader 或 TensorDataset
        """
        if self.dataloader == 'neighbor':
            # Neighbor Sampling Loader
            # num_neighbors=[10, 10] 表示 2 层 GNN，每层采 10 个邻居
            return NeighborSampler(
                edge_index=self.data_dict['edge_index'],
                edge_type=self.edge_type,
                sizes=[10, 10], 
                batch_size=self.batch_size,
                input_nodes=indices, 
                shuffle=shuffle
            )
        else:
            dataset = torch.utils.data.TensorDataset(indices, self.labels[indices])
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _unpack_batch(self, batch):
        # 1. 如果是 MiniBatch 对象 (Neighbor Sampling)
        if hasattr(batch, 'n_id'): 
            n_id = batch.n_id
            batch_size = batch.batch_size
            edge_index = batch.edge_index.to(self.device)
            
            # 动态加载特征 (CPU -> GPU)
            x = self.embeddings[n_id].to(self.device)
            y = self.labels[n_id[:batch_size]].to(self.device)
            
            return x, y, edge_index, batch_size, n_id
            
        # 2. 如果是普通 Tuple (Full Graph Fallback)
        else:
            nodes, labels = batch
            nodes = nodes.to(self.device)
            
            # 全图 Forward
            h_all = self.gnn(self.embeddings.to(self.device), self.data_dict['edge_index'].to(self.device))
            if isinstance(h_all, tuple): h_all = h_all[0]
            
            x = h_all
            y = labels.to(self.device)
            edge_index = self.data_dict['edge_index'].to(self.device)
            batch_size = nodes.size(0)
            n_id = nodes
            
            return x, y, edge_index, batch_size, n_id

    def _setup_optimizer(self):
        """Initial optimizer setup"""
        params = list(filter(lambda p: p.requires_grad, self.gnn.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.fusion.parameters()))
        
        if len(params) > 0:
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _precompute_homophily(self):
        # Placeholder: Return zeros if not using homophily calculation
        # You can integrate your 'calculate_homophily' logic here if needed
        return torch.zeros(self.labels.size(0), 1).to(self.device)

    def _precompute_degrees(self):
        
        row, col = self.edge_index
        deg = degree(col, self.labels.size(0), dtype=torch.float)
        # Normalize
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        return deg.unsqueeze(1).to(self.device)

    def freeze_module(self, module, freeze=True):
        for param in module.parameters():
            param.requires_grad = not freeze

    def _compute_batch_consistency(self, x, edge_index, batch_size):
        """
        利用全图 Embedding 计算结构一致性，然后取 Batch 部分
        """
        row, col = self.edge_index
        # 计算全图边的相似度 (E, )
        edge_sim = F.cosine_similarity(x[row], x[col], dim=1)
        
        # 聚合到节点 (N, )
        # 归一化到 0~1: (sim + 1) / 2
        consistency_all = scatter(edge_sim, row, dim=0, dim_size=x.size(0), reduce='mean')
        consistency_all = (consistency_all + 1.0) / 2.0 
        
        # 提取 Batch 对应的分数 (B, 1)
        return consistency_all[:batch_size].unsqueeze(1)

    def train_epoch(self, epoch):
        if self.pretrain_gnn:
            return self._train_epoch_gnn(epoch)
        else:
            return self._train_epoch_fusion(epoch)

    def _train_epoch_gnn(self, epoch):

        self.gnn.train()
        self.fusion.gnn_proj.train()
        self.fusion.gnn_evidence_head.train()
        
        steps = 0
        total_loss = 0
        
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            x, labels, edge_index, batch_size, n_id = self._unpack_batch(batch)
            
            # --- 数据解包 ---
            if hasattr(batch, 'n_id'):
                h_sub = self.gnn(x, edge_index)
                h_gnn_target = h_sub[:batch_size]
                feat_for_consistency = x
            else:
                h_gnn_target = x[n_id]
                feat_for_consistency = x

            # --- 统一计算逻辑 ---
            consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
            
            # Evidence Head
            z_gnn = self.fusion.gnn_proj(h_gnn_target)
            logits, alpha, u, probs = self.fusion.gnn_evidence_head(z_gnn, consistency)
            
            loss_cls = self.criterion_cls(logits, labels)
            if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
            loss_edl = self.edl_loss(alpha, labels)
            loss = loss_cls + 0.2 * loss_edl
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item(); steps += 1
            if self.use_wandb and steps % 10 == 0: wandb.log({"loss/gnn_total": loss.item()})
                
        return total_loss / steps

    def _train_epoch_gnn_full_batch(self, epoch):
        """
        [OPTIMIZED] Full-Batch Training for GNN
        Avoids re-computing full graph convolution for every mini-batch.
        """
        self.gnn.train()
        self.fusion.gnn_proj.train()
        self.fusion.gnn_evidence_head.train()
        
        self.optimizer.zero_grad()
        
        # 1. Full Graph Forward (Once per Epoch)
        # h_all: [N, Dim]
        h_all = self._forward_gnn() 
        
        # 2. Calculate Full Graph Consistency (Once per Epoch)
        # consistency_all: [N, 1]
        row, col = self.edge_index
        edge_sim = F.cosine_similarity(h_all[row], h_all[col], dim=1)
        consistency_all = scatter(edge_sim, row, dim=0, dim_size=h_all.size(0), reduce='mean')
        consistency_all = (consistency_all + 1.0) / 2.0
        consistency_all = consistency_all.unsqueeze(1)
        
        # 3. Select Train Nodes Only
        # 只对训练集节点计算 Loss
        h_train = h_all[self.train_idx]
        c_train = consistency_all[self.train_idx]
        labels_train = self.labels[self.train_idx]
        
        # 4. Forward Evidence Head
        z_gnn = self.fusion.gnn_proj(h_train)
        logits, alpha, u, probs = self.fusion.gnn_evidence_head(z_gnn, c_train)
        
        # 5. Loss
        loss_cls = self.criterion_cls(logits, labels_train)
        if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
        loss_edl = self.edl_loss(alpha, labels_train)
        
        loss = loss_cls + 0.2 * loss_edl
        
        loss.backward()
        self.optimizer.step()
        
        if self.use_wandb:
             wandb.log({"loss/gnn_total": loss.item()})
             
        return loss.item()

    def _train_epoch_fusion(self, epoch):
        self.fusion.train(); self.gnn.train()
        total_loss = 0; steps = 0
        monitor = {'u_corr': [], 'u_wrong': [], 'gate': []}
        
        u_low, u_high = 0.30, 0.65
        kl_weight = self.lambda_struct * min(1.0, (epoch - 5) / 5) if epoch >= 5 else 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()
            
            # [FIXED] 使用 _unpack_batch
            x, labels, edge_index, batch_size, n_id = self._unpack_batch(batch)
            
            if hasattr(batch, 'n_id'):
                h_sub = self.gnn(x, edge_index)
                h_gnn_target = h_sub[:batch_size]
                h_lm_target = self.embeddings[n_id[:batch_size]].to(self.device)
                feat_for_consistency = x
            else:
                h_gnn_target = x[n_id]
                h_lm_target = self.embeddings[n_id].to(self.device)
                feat_for_consistency = x

            consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
            
            homophily_batch = self.homophily[n_id[:batch_size]].to(self.device)
            degrees_batch = self.node_degrees[n_id[:batch_size]].to(self.device)

            out = self.fusion(
                lm_emb=h_lm_target,
                gnn_emb=h_gnn_target,
                homophily=homophily_batch,
                degree=degrees_batch,
                consistency=consistency 
            )
            
            loss, loss_dict = self._compute_fusion_losses(
                out, labels, h_gnn_target, 
                kl_weight, 1.0, u_low, u_high, epoch
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item(); steps += 1
            self._update_monitor(monitor, out, labels)
            
        self._print_epoch_report(epoch, monitor)
        return total_loss / steps

    def _forward_gnn(self, batch_nodes):
        if self.edge_type is not None:
            h_all = self.gnn(self.embeddings, self.edge_index, self.edge_type)
        else:
            h_all = self.gnn(self.embeddings, self.edge_index)

        if isinstance(h_all, tuple):
            h_all = h_all[0]

        if batch_nodes is not None:
            return h_all, h_all[batch_nodes]
        
        return h_all

    def _compute_fusion_losses(self, out, labels, h_gnn, kl_weight, margin_weight, u_low, u_high, epoch):
        loss_cls = self.criterion_cls(out['logits'], labels)


        # Auxiliary Loss
        loss_text_aux = self.criterion_cls(out['logits_lm'], labels)
        
        if hasattr(self.fusion, 'gnn_classifier'):
             z_gnn = self.fusion.gnn_proj(h_gnn) # Sequential has activation
             logits_gnn = self.fusion.gnn_classifier(z_gnn)
        else:
             logits_gnn = self.gnn.classifier(h_gnn)

        loss_graph_aux = self.criterion_cls(logits_gnn, labels)
        
        # EDL Loss
        if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
        loss_edl_text = self.edl_loss(out['alpha_text'], labels)
        
        if 'alpha_gnn' in out:
            loss_edl_graph = self.edl_loss(out['alpha_gnn'], labels)
        else:
            loss_edl_graph = torch.tensor(0.0, device=self.device)

        # VIB KL Loss
        loss_vib = out['kl_loss'] * kl_weight
        
        # AvUC Loss 
        u_text = out['u_text']
        u_graph = out['u_graph'] 
        gate = out['gate_value']
        u_fused = gate * u_text + (1 - gate) * u_graph
        
        loss_avuc = self.calculate_avuc_loss(out['logits'], labels, u_fused) * self.lambda_avuc
        
        # Gate Consistency Loss
        loss_gate = self.fusion.get_structure_consistency_loss(
            out['gate_value'], self.homophily[labels.device.index], out['u_text']
        )
        
        # Margin Loss
        loss_margin = self.calculate_uncertainty_margin_loss(
            out['logits_lm'], labels, out['u_text'], u_low, u_high
        ) * margin_weight
        
        total_loss = loss_cls + \
                     (0.5 * loss_text_aux) + \
                     (0.5 * loss_graph_aux) + \
                     (0.2 * loss_edl_text) + \
                     (0.2 * loss_edl_graph) + \
                     loss_vib + \
                     loss_avuc + \
                     (0.1 * loss_gate) + \
                     loss_margin
        
        return total_loss, {
            "cls": loss_cls, 
            "margin": loss_margin, 
            "avuc": loss_avuc,
            "edl_g": loss_edl_graph,
            "aux_txt": loss_text_aux
        }

    # =========================================================================
    #  🧪 Evaluation & Diagnosis (FIXED)
    # =========================================================================

    def evaluate(self, split='val'):
        self.gnn.eval()
        self.fusion.eval()
        
        if split == 'val': loader = self.val_loader
        elif split == 'test': loader = self.test_loader
        else: loader = self.dataloader
        
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in loader:
                x, labels, edge_index, batch_size, n_id = self._unpack_batch(batch)
                if hasattr(batch, 'n_id'):
                    h_sub = self.gnn(x, edge_index); h_gnn_target = h_sub[:batch_size]
                    h_lm = self.embeddings[n_id[:batch_size]].to(self.device)
                    feat_for_consistency = x
                else:
                    h_gnn_target = x[n_id]; h_lm = self.embeddings[n_id].to(self.device); feat_for_consistency = x
                
                consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
                
                homophily_batch = self.homophily[n_id[:batch_size]].to(self.device)
                degrees_batch = self.node_degrees[n_id[:batch_size]].to(self.device)

                if self.pretrain_gnn:
                    z_gnn = self.fusion.gnn_proj(h_gnn_target)
                    logits, _, _, _ = self.fusion.gnn_evidence_head(z_gnn, consistency)
                else:
                    out = self.fusion(h_lm, h_gnn_target, homophily_batch, degrees_batch, consistency)
                    logits = out['logits']
                preds.append(logits.argmax(1).cpu()); targets.append(labels.cpu())
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        return f1_score(targets, preds, average='macro'), accuracy_score(targets, preds)

    def diagnose_errors(self, split='val'):
        self.gnn.eval(); self.fusion.eval()
        loader = self.val_loader if split=='val' else (self.test_loader if split=='test' else self.dataloader)
        results = []
        with torch.no_grad():
            for batch in loader:
                x, labels, edge_index, batch_size, n_id = self._unpack_batch(batch)
                if hasattr(batch, 'n_id'):
                    h_sub = self.gnn(x, edge_index); h_gnn_target = h_sub[:batch_size]
                    h_lm = self.embeddings[n_id[:batch_size]].to(self.device)
                    feat_for_consistency = x
                else:
                    h_gnn_target = x[n_id]; h_lm = self.embeddings[n_id].to(self.device); feat_for_consistency = x
                
                consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
                homophily_batch = self.homophily[n_id[:batch_size]].to(self.device)
                degrees_batch = self.node_degrees[n_id[:batch_size]].to(self.device)

                out = self.fusion(h_lm, h_gnn_target, homophily_batch, degrees_batch, consistency)
                
                probs_full = F.softmax(out['logits'], dim=1)
                conf_full, pred_full = probs_full.max(dim=1)
                probs_text = F.softmax(out['logits_lm'], dim=1)
                conf_text, pred_text = probs_text.max(dim=1)
                probs_graph = out['probs_gnn'] 
                conf_graph, pred_graph = probs_graph.max(dim=1)
                
                # Use CPU for iteration
                batch_nodes = n_id[:batch_size].cpu()
                labels_cpu = labels.cpu()
                
                for i in range(batch_size):
                    is_txt = (pred_text[i] == labels_cpu[i]).item()
                    is_gnn = (pred_graph[i] == labels_cpu[i]).item()
                    if is_txt and is_gnn: ctype = 'Easy'
                    elif not is_txt and not is_gnn: ctype = 'Hard'
                    elif is_txt: ctype = 'Text_Only'
                    else: ctype = 'Graph_Only'

                    results.append({
                        'node_idx': batch_nodes[i].item(),
                        'label': labels_cpu[i].item(),
                        'case_type': ctype,
                        'pred_full': pred_full[i].item(),
                        'conf_full': conf_full[i].item(),
                        'is_correct_full': (pred_full[i] == labels_cpu[i]).item(),
                        'gate': out['gate_value'][i].item(),
                        'u_text': out['u_text'][i].item(),
                        'u_graph': out['u_graph'][i].item(),
                        'conf_text': conf_text[i].item(),
                        'conf_graph': conf_graph[i].item(),
                        'homophily': homophily_batch[i].item()
                    })
        
        df = pd.DataFrame(results)
        if not df.empty:
            summary = df.groupby('case_type').agg({
                'node_idx': 'count', 'gate': 'mean', 'u_text': 'mean', 'u_graph': 'mean',
                'conf_graph': 'mean', 'is_correct_full': 'mean'
            }).rename(columns={'node_idx': 'Count', 'is_correct_full': 'Fusion_Acc'})
            print(f"\n[Diagnosis Summary] - {split.upper()}]\n{summary}")
        return df

    # =========================================================================
    #  🎛️ Stage Control Methods
    # =========================================================================

    def pretrain_text_vib(self, epochs=15):

        print(f"\n [PreTrainer] Text Expert (VIB) Pre-training ({epochs} epochs)")

        self.pretrain_gnn = False # We are not training GNN
        self.pretrain_llm = True  # Flag for logging (optional)
        
        # Freeze Everything EXCEPT VIB
        self.freeze_module(self.gnn, freeze=True)
        self.freeze_module(self.fusion.gnn_proj, freeze=True)
        self.freeze_module(self.fusion.gate_net, freeze=True)
        self.freeze_module(self.fusion.classifier, freeze=True)
        self.freeze_module(self.fusion.norm, freeze=True)
        self.freeze_module(self.fusion.gnn_evidence_head, freeze=True)
        self.freeze_module(self.fusion.vib, freeze=False)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.fusion.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Training Loop
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            self.fusion.vib.train()
            total_loss = 0
            correct = 0
            total = 0

            u_correct_list = []
            u_wrong_list = []
            
            # KL Warmup logic for VIB
            if epoch < 5: kl_weight = 0.0
            else: kl_weight = 0.01 * min(1.0, (epoch - 5) / 5) 
            
            for batch in self.dataloader:
                
                if isinstance(batch, (list, tuple)):
                    batch_nodes, batch_labels = batch
                else:
                    batch_nodes = batch
                    batch_labels = self.labels[batch_nodes]
                
                batch_nodes = batch_nodes.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Get Embeddings
                h_lm = self.embeddings[batch_nodes]
                
                # VIB Forward
                logits_lm, _, alpha_text, u_text, kl_loss = self.fusion.vib(h_lm)
                
                # Losses
                loss_cls = self.criterion_cls(logits_lm, batch_labels)
                if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
                loss_edl = self.edl_loss(alpha_text, batch_labels)
                loss_vib = kl_loss * kl_weight
                
                loss = loss_cls + loss_edl + loss_vib
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                with torch.no_grad():
                    preds = logits_lm.argmax(dim=1)
                    is_correct = (preds == batch_labels)
                    correct += is_correct.sum().item()
                    total += batch_labels.size(0)
                    
                    u_val = u_text.detach()
                    if u_val.dim() > 1: u_val = u_val.view(-1)
                    
                    u_correct_list.extend(u_val[is_correct].tolist())
                    u_wrong_list.extend(u_val[~is_correct].tolist())
                
            # Metrics
            avg_loss = total_loss / len(self.dataloader)

            train_acc = correct / total
            val_acc = self._validate_text_only()
            train_f1 = self.evaluate('train')[0]
            val_f1 = self.evaluate('val')[0]
            
            avg_u_c = np.mean(u_correct_list) if u_correct_list else 0
            avg_u_w = np.mean(u_wrong_list) if u_wrong_list else 0
            avg_u_gap = avg_u_w - avg_u_c

            u_c_std = np.std(u_correct_list) if len(u_correct_list) > 0 else 0.0
            u_w_std = np.std(u_wrong_list) if len(u_wrong_list) > 0 else 0.0
            scale_param =  F.softplus(self.fusion.vib.evidence_scale).item()

            
            print(f"   VIB Ep {epoch:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"   U_Corr: {avg_u_c:.4f} | U_Wrong: {avg_u_w:.4f} | Gap: {avg_u_gap:.4f}")
            
            # Validation (Text Only)
            if self.use_wandb:
                wandb.log({
                    "vib/loss": avg_loss,
                    "vib/train_acc": train_acc,
                    "vib/train_f1": train_f1,
                    "vib/val_acc": val_acc,
                    "vib/val_f1": val_f1,
                    "vib/u_correct": avg_u_c,
                    "vib/u_wrong": avg_u_w,
                    "vib/u_gap": avg_u_gap,
                    "vib/u_c_std": u_c_std,
                    "vib/u_w_std": u_w_std,
                    "vib/scale_param": scale_param
                })
                
            # Save if best
            if val_acc > best_acc:
                best_acc = val_acc
                # Save specifically the VIB state dict
                torch.save(self.fusion.vib.state_dict(), self.ckpt_filepath.parent / "best_text_vib.pt")
        
        print(f"Text Expert Finished.")
        self.freeze_module(self.fusion.vib, freeze=True)

    def _validate_text_only(self):
        """Helper for VIB validation"""
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
                h_lm = self.embeddings[nodes]
                
                logits, _, _, _, _ = self.fusion.vib(h_lm)
                preds.append(logits.argmax(dim=1).cpu())
                targets.append(labels.cpu())
                
        return accuracy_score(torch.cat(targets), torch.cat(preds))

    def _validate_gnn_stats(self):
        """
        Helper: Calculate GNN stats (Acc, U_Corr, U_Wrong, Gap)
        """
        self.gnn.eval()
        self.fusion.gnn_proj.eval()
        self.fusion.gnn_evidence_head.eval()
        
        preds_list, targets_list = [], []
        u_correct_list, u_wrong_list = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # [FIXED] 使用 _unpack_batch
                x, labels, edge_index, batch_size, n_id = self._unpack_batch(batch)
                
                if hasattr(batch, 'n_id'):
                    h_sub = self.gnn(x, edge_index)
                    h_gnn_target = h_sub[:batch_size]
                    feat_for_consistency = x
                else:
                    h_gnn_target = x[n_id]
                    feat_for_consistency = x

                consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
                z_gnn = self.fusion.gnn_proj(h_gnn_target)
                logits, _, u, _ = self.fusion.gnn_evidence_head(z_gnn, consistency)
                
                preds = logits.argmax(1)
                is_correct = (preds == labels)
                u_correct_list.extend(u[is_correct].cpu().tolist())
                u_wrong_list.extend(u[~is_correct].cpu().tolist())
                preds_list.append(preds.cpu())
                targets_list.append(labels.cpu())
                
        # Stats
        all_preds = torch.cat(preds_list)
        all_targets = torch.cat(targets_list)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        u_c_mean = np.mean(u_correct_list) if u_correct_list else 0.0
        u_w_mean = np.mean(u_wrong_list) if u_wrong_list else 0.0
        u_c_std = np.std(u_correct_list) if u_correct_list else 0.0
        u_w_std = np.std(u_wrong_list) if u_wrong_list else 0.0

        
        return f1, u_c_mean, u_w_mean, u_c_std, u_w_std

    def pretrain_gnn_stage(self, epochs):
        print(f"\nStage 2 | GNN Pre-training ({epochs} epochs)")
        
        self.pretrain_gnn = True
        self.freeze_module(self.fusion, True) 
        self.freeze_module(self.gnn, False)
        self.freeze_module(self.fusion.gnn_proj, False)
        self.freeze_module(self.fusion.gnn_evidence_head, False)

        self._setup_optimizer()
        
        best_f1 = 0
        for ep in range(1, epochs+1):
            
            loss = self._train_epoch_gnn(ep)
            f1, u_c_mean, u_w_mean, u_c_std, u_w_std = self._validate_gnn_stats()
            gap = u_w_mean - u_c_mean

            print(f"GNN Ep {ep:02d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | U_Gap: {gap:.4f}")
            print(f"U_Corr: {u_c_mean:.4f} | U_Wrong: {u_w_mean:.4f} | U_Corr_Std: {u_c_std:.4f} | U_Wrong_Std: {u_w_std:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "gnn/val_f1": f1,
                    "gnn/u_corr": u_c_mean, 
                    "gnn/u_wrong": u_w_mean, 
                    "gnn/u_corr_std": u_c_std,
                    "gnn/u_wrong_std": u_w_std,
                    "gnn/u_gap": gap,
                    "gnn/scale": F.softplus(self.fusion.gnn_evidence_head.evidence_scale).item()
                })

            if f1 > best_f1:
                best_f1 = f1
                self.save_checkpoint(ep)

    def pretrain_gate_stage(self, epochs):
        print(f"\nStage 3 | Gate Warmup ({epochs} epochs)")
        self.pretrain_gnn = False
        self.freeze_module(self.gnn, True)
        self.freeze_module(self.fusion.vib, True)
        self.freeze_module(self.fusion.gnn_proj, True)
        self.freeze_module(self.fusion.gate_net, False)
        self.freeze_module(self.fusion.classifier, False)
        self.freeze_module(self.fusion.norm, False)
        
        self._setup_optimizer()
        
        best_f1 = 0
        for ep in range(1, epochs+1):
            loss = self._train_epoch_fusion(ep)
            f1, acc = self.evaluate('val')
            gate_val = getattr(self, 'last_gate_mean', 0.5)
            print(f"Gate Ep {ep:02d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | Gate: {gate_val:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                self.save_checkpoint(ep)

    def train(self):
        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(epoch)
            val_f1, val_acc = self.evaluate('val')
            # append diagnostics if available
            extra = ""
            if hasattr(self, 'last_epoch_gate'):
                extra += f" | Gate: {self.last_epoch_gate:.4f}"
            if hasattr(self, 'last_epoch_u_gap'):
                extra += f" | U_Gap: {self.last_epoch_u_gap:.4f}"
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}{extra}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_checkpoint(epoch)
        return best_f1

    # =========================================================================
    #  📉 Loss Utils
    # =========================================================================

    def calculate_uncertainty_margin_loss(self, logits, labels, uncertainty, u_low, u_high):
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if uncertainty.dim() > 1: uncertainty = uncertainty.squeeze()
        mask_correct = (preds == labels)
        
        loss = torch.tensor(0.0, device=self.device)

        if mask_correct.sum() > 0:
            loss += F.relu(uncertainty[mask_correct] - u_low).mean()
        if (~mask_correct).sum() > 0:
            loss += F.relu(u_high - uncertainty[~mask_correct]).mean()
        return loss

    def calculate_avuc_loss(self, logits, labels, uncertainty):
        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        
        if uncertainty.dim() == 1: uncertainty = uncertainty.unsqueeze(1)
        if correct.dim() == 1: correct = correct.unsqueeze(1)

        uncertainty = torch.clamp(uncertainty, min=1e-6, max=1.0 - 1e-6)

        # [FIX] Ensure target_uncertainty matches input shape [B, 1]
        ac = correct * (1 - uncertainty)
        iu = (1 - correct) * uncertainty

        loss = -torch.log(torch.mean(ac + iu) + 1e-10)
            
        return loss

    # =========================================================================
    #  💾 IO Utils
    # =========================================================================

    def save_checkpoint(self, epoch):
        self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            'fusion_state_dict': self.fusion.state_dict()
        }
        torch.save(state, self.ckpt_filepath)

    def load_checkpoint(self, path=None):
        p = path or self.ckpt_filepath
        print(f"📥 Loading checkpoint from {p}")
        try:
            state = torch.load(p, map_location=self.device)
            # 兼容性加载：检查 keys
            if 'gnn_state_dict' in state:
                self.gnn.load_state_dict(state['gnn_state_dict'])
            if 'fusion_state_dict' in state:
                self.fusion.load_state_dict(state['fusion_state_dict'])
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    def _update_monitor(self, monitor, out, labels):
        with torch.no_grad():
            preds = out['logits_lm'].argmax(1)
            is_correct = (preds == labels)
            u = out['u_text'].detach()
            monitor['u_corr'].extend(u[is_correct].tolist())
            monitor['u_wrong'].extend(u[~is_correct].tolist())
            monitor['gate'].extend(out['gate_value'].flatten().tolist())

    def _log_wandb(self, loss, loss_dict, monitor):
        # [MODIFIED] 增加更详细的统计量
        if monitor['u_corr']:
            u_corr_arr = np.array(monitor['u_corr'])[-100:]
            u_c_mean = np.mean(u_corr_arr)
            u_c_std = np.std(u_corr_arr) # 正确样本的不确定性波动
        else:
            u_c_mean, u_c_std = 0, 0
            
        if monitor['u_wrong']:
            u_wrong_arr = np.array(monitor['u_wrong'])[-100:]
            u_w_mean = np.mean(u_wrong_arr)
            u_w_std = np.std(u_wrong_arr) # 错误样本的不确定性波动
        else:
            u_w_mean, u_w_std = 0, 0

        txt_scale = F.softplus(self.fusion.vib.evidence_scale).item()
        gnn_scale = F.softplus(self.fusion.gnn_evidence_head.evidence_scale).item()

        wandb.log({
            "loss/total": loss.item(),
            "loss/cls": loss_dict['cls'].item(),
            "loss/avuc": loss_dict['avuc'].item(),
            "loss/edl_g": loss_dict['edl_g'].item(),
            
            # 核心监控指标
            "diag/u_gap": u_w_mean - u_c_mean,  # 均值差 (原指标)
            "diag/u_corr_mean": u_c_mean,
            "diag/u_wrong_mean": u_w_mean,

            "diag/txt_scale": txt_scale, 
            "diag/gnn_scale": gnn_scale,
            
            # [NEW] 分布监控
            "diag/u_corr_std": u_c_std,    # 我们希望这个越小越好（正确样本都很确信）
            "diag/u_wrong_std": u_w_std   # 我们希望这个也收敛
        })

    def _print_epoch_report(self, epoch, monitor):
        u_c = np.mean(monitor['u_corr']) if monitor['u_corr'] else 0
        u_w = np.mean(monitor['u_wrong']) if monitor['u_wrong'] else 0
        gate = np.mean(monitor['gate']) if monitor['gate'] else 0
        print(f"📊 [Ep {epoch}] Gate: {gate:.4f} | U_Gap: {u_w - u_c:.4f} (C:{u_c:.4f} vs W:{u_w:.4f})")