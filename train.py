import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pandas as pd
import numpy as np

from utils import EdlLoss, AvUCLoss, kl_divergence_dirichlet
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

        self.data_loader_type = dataloader
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

        self.train_loader = self._create_dataloader(self.train_idx, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_idx, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_idx, shuffle=False)

        self.dataloader = self.train_loader  # 默认使用训练集的 dataloader 配置进行采样

        if 'homophily' in data_dict:
            print("[Trainer] Loading precomputed homophily from data_dict.")
            self.homophily = data_dict['homophily'].to(device)
        else:
            print("[Trainer] Computing homophily internally...")
            self.homophily = self._precompute_homophily().to(device)
            
        self.node_degrees = self._precompute_degrees()

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.edl_loss = EdlLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.avuc_criterion = AvUCLoss(beta=1.0)

        self._setup_optimizer()

    # =========================================================================
    #  🛠️ Helpers & Setup
    # =========================================================================

    def _create_dataloader(self, indices, shuffle):
        """
        [NEW] 根据配置选择 NeighborLoader 或 TensorDataset
        """
        if self.data_loader_type == 'neighbor':
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
        if hasattr(batch, 'n_id'): 
            n_id = batch.n_id
            batch_size = batch.batch_size
            edge_index = batch.edge_index.to(self.device)
            edge_type = batch.edge_type.to(self.device) if batch.edge_type is not None else None
            
            x = self.embeddings[n_id].to(self.device)
            y = self.labels[n_id[:batch_size]].to(self.device)
            
            return x, y, edge_index, edge_type, batch_size, n_id
            
        else:
            nodes, labels = batch
            nodes = nodes.to(self.device)
            
            # 全图 Forward Fallback
            h_all = self._call_gnn(
                self.embeddings.to(self.device), 
                self.data_dict['edge_index'].to(self.device),
                self.edge_type.to(self.device) if self.edge_type is not None else None
            )
            
            x = h_all
            y = labels.to(self.device)
            edge_index = self.data_dict['edge_index'].to(self.device)
            edge_type = self.edge_type.to(self.device) if self.edge_type is not None else None
            batch_size = nodes.size(0)
            n_id = nodes
            
            return x, y, edge_index, edge_type, batch_size, n_id

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

    def _call_gnn(self, x, edge_index, edge_type):
        if edge_type is not None:
            return self.gnn(x, edge_index, edge_type)
        else:
            return self.gnn(x, edge_index)

    def get_structure_consistency_loss(self, gate_value, homophily, u_text):
        """
        结构一致性损失 (Structure Consistency Loss)
        基于动态阈值 (Dynamic Relative Thresholding) 约束 Gate 的行为。
        
        Args:
            gate_value: [Batch, 1], Beta值, 越接近1表示越信Text
            homophily: [Batch, 1], 节点的结构同质性分数
            u_text: [Batch, 1], 文本模态的不确定性
        """
        # 1. 计算动态阈值 (基于当前Batch的统计)
        # 加上 detach() 防止梯度回传影响阈值计算，我们只希望优化 Gate
        u_mean = u_text.mean()
        u_std = u_text.std() + 1e-6
        thresh_low = (u_mean - 0.5 * u_std).detach()
        thresh_high = (u_mean + 0.5 * u_std).detach()

        # 2. 定义信任场景 (Inductive Biases)
        
        # 场景 A: 应该信任 Graph (Gate -> 0)
        # 条件: 结构很好 (Homophily > 0.7) 或者 文本非常不确定 (u_text > High)
        target_graph_trust = (homophily > 0.7) | (u_text > thresh_high)
        
        # 场景 B: 应该信任 Text (Gate -> 1)
        # 条件: 结构很差 (Homophily < 0.4) 并且 文本比较确信 (u_text < Low)
        target_text_trust = (homophily < 0.4) & (u_text < thresh_low)
        
        # 3. 计算损失
        loss = torch.tensor(0.0, device=gate_value.device)
        
        # 如果 Gate 应该小 (信Graph)，但实际很大，则惩罚 (gate^2)
        if target_graph_trust.any():
            loss += (gate_value[target_graph_trust] ** 2).mean()
            
        # 如果 Gate 应该大 (信Text)，但实际很小，则惩罚 ((1-gate)^2)
        if target_text_trust.any():
            loss += ((1 - gate_value[target_text_trust]) ** 2).mean()
            
        return loss

    def _compute_batch_consistency(self, x, edge_index, batch_size):
        """
        计算 batch 内节点的结构一致性
        """
        row, col = edge_index
        
        # 1. 特征归一化
        x_norm = F.normalize(x, p=2, dim=-1)
        
        # 2. 计算每条边的余弦相似度
        edge_sim = (x_norm[row] * x_norm[col]).sum(dim=-1)
        
        # 3. [FIX] 聚合到 col (Target Node)
        consistency_all = scatter(edge_sim, col, dim=0, dim_size=x.size(0), reduce='mean')
        
        # 4. 截取 batch 部分 (如果是 NeighborLoader，前 batch_size 个是 target nodes)
        if batch_size is not None:
            return consistency_all[:batch_size].unsqueeze(1)
            
        return consistency_all.unsqueeze(1)

    def train_epoch(self, epoch):
        if self.pretrain_gnn:
            return self._train_epoch_gnn(epoch)
        else:
            return self._train_epoch_fusion(epoch)

    def _train_epoch_gnn(self, epoch):
        self.gnn.train(); self.fusion.gnn_proj.train(); self.fusion.gnn_evidence_head.train()
        total_loss = 0; steps = 0

        for batch in self.dataloader:
            self.optimizer.zero_grad()
            
            x, labels, edge_index, edge_type, batch_size, n_id = self._unpack_batch(batch)
            
            # [FIXED] 使用 _call_gnn 传入 edge_type
            if hasattr(batch, 'n_id'):
                h_sub = self._call_gnn(x, edge_index, edge_type)
                h_gnn_target = h_sub[:batch_size]
                feat_for_consistency = x
            else:
                h_gnn_target = x[n_id]
                feat_for_consistency = x

            consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
            
            z_gnn = self.fusion.gnn_proj(h_gnn_target)
            logits, alpha, u, probs = self.fusion.gnn_evidence_head(z_gnn, consistency)
            
            loss_cls = self.criterion_cls(logits, labels)
            
            # ================= [修改开始] =================
            # 这里的 edl_loss 现在需要在 forward 中接收 epoch 参数
            # 来计算 KL 散度的退火系数
            loss_edl = self.edl_loss(alpha, labels, epoch)
            # ================= [修改结束] =================
            
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
        h_all = self._forward_gnn(batch_nodes=None) 
        
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
        consistency_train = consistency_all[self.train_idx]
        labels_train = self.labels[self.train_idx]
        
        # 4. Forward Evidence Head
        z_gnn = self.fusion.gnn_proj(h_train)
        logits, alpha, u, probs = self.fusion.gnn_evidence_head(z_gnn, consistency_train)
        
        # 5. Loss
        loss_cls = self.criterion_cls(logits, labels_train)
        loss_edl = self.edl_loss(alpha, labels_train, epoch)
        
        loss = loss_cls + 0.2 * loss_edl
        
        loss.backward()
        self.optimizer.step()
        
        if self.use_wandb:
             wandb.log({"loss/gnn_total": loss.item()})
             
        return loss.item()

    def _train_epoch_fusion(self, epoch):
        """
        训练融合阶段，冻结 GNN 和 Text Expert，只训练 Gate
        """
        self.fusion.train()

        # 冻结专家，只训练 Gate
        self.fusion.vib.eval()
        self.gnn.eval()
        self.fusion.gnn_proj.eval()
        self.fusion.gnn_evidence_head.eval()

        total_loss = 0
        all_gates = []
    
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            x, labels, edge_index, edge_type, batch_size, n_id = self._unpack_batch(batch)
            
            # data preparation
            if hasattr(batch, 'n_id'):
                h_sub = self._call_gnn(x, edge_index, edge_type)
                h_gnn = h_sub[:batch_size]
                h_lm = x[:batch_size]

                if self.homophily is not None and len(self.homophily) > 0:
                     # n_id 映射回原图索引
                     homophily_batch = self.homophily[n_id][:batch_size]
                else:
                     homophily_batch = torch.zeros(batch_size, 1, device=self.device)

                feat_for_consistency = x

            else:
                h_gnn = self._call_gnn(x, edge_index, edge_type)[n_id]
                h_lm = x[n_id]
                homophily_batch = self.homophily[n_id]
                feat_for_consistency = x

            # 计算一致性
            consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
            out = self.fusion(h_lm, h_gnn, homophily_batch, degree=None, consistency=consistency)
            
            with torch.no_grad():
                # 获取各自的预测结果
                pred_text = out['probs_lm'].argmax(dim=1)
                pred_gnn = out['probs_gnn'].argmax(dim=1)
                
                # 情况 A: Text 对，GNN 错 -> Gate 应该趋向 1.0 (信 Text)
                target_text = (pred_text == labels) & (pred_gnn != labels)
                target_gnn = (pred_text != labels) & (pred_gnn == labels)
                
                # 有效样本 Mask (只在有分歧的样本上训练 Gate)
                gate_mask = target_text | target_gnn
                
                # 构建标签: 默认 0.5 (无分歧时), Text赢填 1, GNN赢填 0
                gate_target = torch.zeros_like(out['gate_value'])
                gate_target[target_text] = 1.0
                gate_target[target_gnn] = 0.0
            
            # 3. 计算 Gate Loss (BCE)
            loss_gate = 0.0
            if gate_mask.sum() > 0:
                loss_gate = F.binary_cross_entropy(out['gate_value'][gate_mask], gate_target[gate_mask])

            # 4. Homophily 低的时候，Text 不太可信 (Gate 应该低)
            loss_struct = get_structure_consistency_loss(
                out['gate_value'], 
                homophily_batch, 
                out['u_text']
            )
            
            # 5. 总 Loss 组合
            # 移除了 loss_entropy，因为我们希望 gate 根据情况变化，而不是强行塌缩
            # 给 gate_supervision 较高的权重 (1.0) 强迫它学习区分
            loss = 1.0 * loss_gate + 0.1 * loss_struct
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_gates.append(out['gate_value'].detach().mean().item())
            
        self.last_gate_mean = np.mean(all_gates) if all_gates else 0.5
        return total_loss / len(self.dataloader)

    def _forward_gnn(self, batch_nodes=None):
       
        if batch_nodes is not None:
            # Mini-batch 模式 (通常 DataLoader 已经处理好子图，这里保留接口扩展性)
            raise NotImplementedError("Mini-batch forward logic is handled inside train_epoch loop.")
        else:
            # [Fix] Full-batch 模式正确获取全图数据
            x = self.embeddings.to(self.device)
            edge_index = self.edge_index
            edge_type = self.edge_type
            return self.gnn(x, edge_index, edge_type)

    def _compute_fusion_losses(self, out, labels, h_gnn, kl_weight, margin_weight, u_low, u_high, epoch):
        loss_cls = self.criterion_cls(out['logits'], labels)

        # [修改 D: 强力冻结逻辑]
        # 检测是否在 Stage 3 (Gate Warmup)
        # 只要 gnn 是 eval 模式 (training=False)，就说明我们在只练 Gate
        is_stage3 = not self.gnn.training 
        
        if is_stage3:
            # ❄️ Stage 3 绝对冻结: 屏蔽所有干扰项
            loss_text_aux = torch.tensor(0.0, device=self.device)
            loss_graph_aux = torch.tensor(0.0, device=self.device)
            loss_edl_text = torch.tensor(0.0, device=self.device)
            loss_edl_graph = torch.tensor(0.0, device=self.device)
            loss_vib = torch.tensor(0.0, device=self.device)
            loss_margin = torch.tensor(0.0, device=self.device)
            annealing_coef = 0.0
            
            # Stage 3 只保留: 1.主分类 Loss  2.Gate 引导 Loss
            loss_gate = self.fusion.get_structure_consistency_loss(
                out['gate_value'], self.homophily[labels.device.index], out['u_text']
            )
            # 提高权重，强迫 Gate 学习规则
            gate_weight = 5.0 
            
        else:
            # 🔥 Stage 4 / 正常模式 (保持原样)
            loss_text_aux = self.criterion_cls(out['logits_lm'], labels)
            
            # GNN Aux
            if hasattr(self.fusion, 'gnn_classifier'):
                 z_gnn = self.fusion.gnn_proj(h_gnn)
                 logits_gnn = self.fusion.gnn_classifier(z_gnn)
            else:
                 logits_gnn = self.gnn.classifier(h_gnn)
            loss_graph_aux = self.criterion_cls(logits_gnn, labels)
            
            # EDL Loss
            annealing_coef = min(0.1, epoch / 20.0)
            if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
            loss_edl_text = self.edl_loss(out['alpha_text'], labels)
            loss_edl_graph = self.edl_loss(out['alpha_gnn'], labels) if 'alpha_gnn' in out else 0.0
            
            loss_gate = self.fusion.get_structure_consistency_loss(
                out['gate_value'], self.homophily[labels.device.index], out['u_text']
            )
            gate_weight = 0.1
            loss_vib = out['kl_loss'] * kl_weight
            loss_margin = self.calculate_uncertainty_margin_loss(
                out['logits_lm'], labels, out['u_text'], u_low, u_high
            ) * margin_weight

        # AvUC Loss (保留用于监控)
        u_text = out['u_text']; u_graph = out['u_graph']; gate = out['gate_value']
        u_fused = gate * u_text + (1 - gate) * u_graph
        loss_avuc = self.calculate_avuc_loss(out['logits'], labels, u_fused) * self.lambda_avuc

        # 汇总
        total_loss = loss_cls + \
                     loss_text_aux + loss_graph_aux + \
                     (annealing_coef * loss_edl_text) + (annealing_coef * loss_edl_graph) + \
                     loss_vib + loss_avuc + \
                     (gate_weight * loss_gate) + \
                     loss_margin
                     
        return total_loss, { 
            "cls": loss_cls, 
            "gate": loss_gate,
            "edl_g": loss_edl_graph
        }

    # =========================================================================
    #  🧪 Evaluation & Diagnosis (FIXED)
    # =========================================================================

    def evaluate(self, split='val'):
        self.gnn.eval(); self.fusion.eval()
        loader = self.val_loader if split=='val' else (self.test_loader if split=='test' else self.dataloader)
        preds, targets = [], []
        
        # [新增] 用于统计 Gate 行为的容器
        gate_values = []
        u_gaps = []

        with torch.no_grad():
            for batch in loader:
                x, labels, edge_index, edge_type, batch_size, n_id = self._unpack_batch(batch)
                
                # ... (中间的数据准备逻辑保持不变, _call_gnn 等) ...
                if hasattr(batch, 'n_id'):
                    h_sub = self._call_gnn(x, edge_index, edge_type)
                    h_gnn_target = h_sub[:batch_size]
                    h_lm = self.embeddings[n_id[:batch_size]].to(self.device)
                    feat_for_consistency = x
                else:
                    h_gnn_target = x[n_id]
                    h_lm = self.embeddings[n_id.cpu()].to(self.device)
                    feat_for_consistency = x

                consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)
                homophily_batch = self.homophily[n_id[:batch_size]].to(self.device)
                degrees_batch = self.node_degrees[n_id[:batch_size]].to(self.device)
                
                # 前向传播
                out = self.fusion(
                    lm_emb=h_lm,
                    gnn_emb=h_gnn_target,
                    homophily=homophily_batch,
                    degree=degrees_batch,
                    consistency=consistency
                )
                
                # [新增] 收集统计数据
                gate_values.append(out['gate_value'].detach().cpu())
                # 计算 gap: u_text - u_graph (注意取 mean)
                gap = (out['u_text'] - out['u_graph']).mean()
                u_gaps.append(gap.item())

                preds.append(out['logits'].argmax(dim=-1).cpu())
                targets.append(labels.cpu())
        
        # [新增] 更新 Trainer 的状态，以便主循环打印
        if len(gate_values) > 0:
            all_gates = torch.cat(gate_values)
            self.last_gate_mean = all_gates.mean().item()
            self.last_u_gap = sum(u_gaps) / len(u_gaps)
        
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        
        return f1_score(targets, preds, average='macro'), accuracy_score(targets, preds)

    def diagnose_errors(self, split='val'):
        print(f"\n🔍 [Diagnosis] Analysis for {split}...")
        self.fusion.eval()
        self.gnn.eval()

        loader = self.val_loader if split == 'val' else (self.test_loader if split == 'test' else self.dataloader)
        all_records = []

        with torch.no_grad():
            for batch in loader:
                x, labels, edge_index, edge_type, batch_size, n_id = self._unpack_batch(batch)

                # --- 1. 构建输入 ---
                if hasattr(batch, 'n_id'):
                    h_sub = self._call_gnn(x, edge_index, edge_type)
                    h_gnn = h_sub[:batch_size]
                    h_lm = x[:batch_size]
                    homophily_batch = self.homophily[n_id][:batch_size]
                    feat_for_consistency = x
                else:
                    h_gnn = self._call_gnn(x, edge_index, edge_type)[n_id]
                    h_lm = x[n_id]
                    feat_for_consistency = x
                    homophily_batch = self.homophily[n_id]

                consistency = self._compute_batch_consistency(feat_for_consistency, edge_index, batch_size)

                # --- 2. 前向传播 ---
                out = self.fusion(h_lm, h_gnn, homophily_batch, degree=None, consistency=consistency)

                # --- 3. 提取结果 (强制拍平) ---
                # Fusion
                probs_full = F.softmax(out['logits'], dim=1)
                pred_full = probs_full.argmax(dim=1)
                is_correct = (pred_full == labels)
                
                # 2. Text 专家预测 (直接取 Logits)
                probs_text = F.softmax(out['logits_lm'], dim=1)
                pred_text = probs_text.argmax(dim=1)
                is_correct_text = (pred_text == labels)
                
                # 3. GNN 专家预测 (直接取 Logits)
                probs_graph = F.softmax(out['logits_gnn'], dim=1)
                pred_graph = probs_graph.argmax(dim=1)
                is_correct_graph = (pred_graph == labels)

                # --- 4. 样本分类 (Vectorized) ---
                mask_easy = is_correct_text & is_correct_graph
                mask_hard = (~is_correct_text) & (~is_correct_graph)
                mask_text_only = is_correct_text & (~is_correct_graph)
                mask_graph_only = (~is_correct_text) & is_correct_graph

                case_types = np.empty(batch_size, dtype=object)
                case_types[mask_easy.cpu().numpy()] = 'Easy'
                case_types[mask_hard.cpu().numpy()] = 'Hard'
                case_types[mask_text_only.cpu().numpy()] = 'Text_Only'
                case_types[mask_graph_only.cpu().numpy()] = 'Graph_Only'

                ent_graph = -torch.sum(probs_graph * torch.log(probs_graph + 1e-9), dim=1) / 0.693

                # --- 5. 数据打包 ---
                batch_data = {
                    'label': labels.cpu().numpy(),
                    'case_type': case_types,
                    'is_correct': is_correct.cpu().numpy(),
                    'gate': out['gate_value'].view(-1).cpu().numpy(),
                    'u_text': out['u_text'].view(-1).cpu().numpy(),
                    'u_graph': out['u_graph'].view(-1).cpu().numpy(),
                    'ent_graph': ent_graph.cpu().numpy(),
                    'gate_hit': np.zeros(len(labels))
                }
                all_records.append(pd.DataFrame(batch_data))


        # --- 6. 聚合与指标计算 ---
        if not all_records: return pd.DataFrame()
        df = pd.concat(all_records, ignore_index=True)

        # 计算 Gate Selectivity
        mask_text = df['case_type'] == 'Text_Only'
        df.loc[mask_text, 'gate_hit'] = (df.loc[mask_text, 'gate'] > 0.5).astype(int)
        
        # Graph Only Case: Gate 应 <= 0.5 (信Graph)
        mask_graph = df['case_type'] == 'Graph_Only'
        df.loc[mask_graph, 'gate_hit'] = (df.loc[mask_graph, 'gate'] <= 0.5).astype(int)
        
        return df

    def pretrain_text_vib(self, epochs=None):

        if epochs is None:
            epochs = self.epochs

        print(f"\n [PreTrainer] Text Expert (VIB) Pre-training ({epochs} epochs) [CE Warmup -> EDL]")

        # 1. 冻结策略
        self.pretrain_gnn = False 
        self.pretrain_llm = True

        # 冻结所有非 VIB 组件
        self.freeze_module(self.gnn, True)
        self.freeze_module(self.fusion.gnn_proj, True)
        self.freeze_module(self.fusion.gate_net, True)
        self.freeze_module(self.fusion.gnn_evidence_head, True)
        self.freeze_module(self.fusion.vib, False)
        
        # 2. 类别权重 (用于 CE Loss 和 EDL Loss)
        if not hasattr(self, 'class_weights') or self.class_weights is None:
            labels_np = self.labels.cpu().numpy()
            train_labels = labels_np[labels_np != -1]  
            class_counts = np.bincount(train_labels)
            total_samples = len(train_labels)
            num_classes = len(class_counts)
            
            # 使用 float32 避免半精度溢出
            self.class_weights = torch.tensor(
                total_samples / (num_classes * class_counts), 
                dtype=torch.float32
            ).to(self.device)
            print(f"   >>> Class Weights Initialized: {self.class_weights.cpu().numpy()}")

        # 3. 优化器
        # 注意：这里我们放宽 weight_decay，避免证据被压死
        base_params = []
        scale_params = []
        for n, p in self.fusion.vib.named_parameters():
            if not p.requires_grad: continue
            if 'evidence_scale' in n:
                scale_params.append(p)
            else:
                base_params.append(p)

        self.optimizer = torch.optim.AdamW([
            {'params': base_params, 'weight_decay': 1e-4}, 
            {'params': scale_params, 'weight_decay': 0.0} # Scale 参数不衰减，允许它增长
        ], lr=self.lr)

        # Training Loop
        best_f1 = 0.0
        WARMUP_EPOCHS = 5
        
        
        for epoch in range(1, epochs + 1):
            self.fusion.vib.train()
            total_loss = 0
            
            if epoch > WARMUP_EPOCHS:
                anneal_progress = (epoch - WARMUP_EPOCHS) / 10.0
                anneal_factor = min(1.0, max(0.0, anneal_progress))
            else:
                anneal_factor = 0.0
            
            u_correct_list = []
            u_wrong_list = []

            is_warmup = epoch <= WARMUP_EPOCHS
            
            for batch in self.dataloader:

                if hasattr(batch, 'n_id'): batch_nodes = batch.n_id
                elif isinstance(batch, (list, tuple)): batch_nodes, batch_labels = batch
                else: batch_nodes = batch

                batch_nodes = batch_nodes.to(self.device)
                batch_labels = self.labels[batch_nodes].to(self.device)
                h_lm = self.embeddings[batch_nodes.to(self.embeddings.device)].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward
                logits_lm, _, alpha_text, u_text, vib_kl = self.fusion.vib(h_lm)
                loss_vib_kl = vib_kl.mean() * 1e-4
                
                # CE Safety Net & Logit Norm
                loss_cls = self.edl_loss(alpha_text, batch_labels, epoch, epochs)

                if is_warmup:
                    # Warmup Phase: 只使用 CrossEntropy 快速收敛
                    loss_cls = F.cross_entropy(logits_lm, batch_labels, weight=self.class_weights)
                    
                    # [优化] Logit Norm 在这里正确计算
                    loss_logit_norm = 0.01 * torch.mean(logits_lm ** 2)
                    
                    loss = loss_cls + loss_logit_norm + loss_vib_kl
                else:
                    # EDL Phase
                    y_one_hot = F.one_hot(batch_labels, num_classes=2).float()
                    S = torch.sum(alpha_text, dim=1, keepdim=True)
                    sample_weights = self.class_weights[batch_labels]

                    # A. Fit Loss
                    edl_loss_per_sample = torch.sum(y_one_hot * (torch.log(S) - torch.log(alpha_text + 1e-7)), dim=1)
                    loss_fit = (edl_loss_per_sample * sample_weights).mean()

                    # B. KL Regularization
                    kl_dirichlet = kl_divergence_dirichlet(alpha_text)
                    loss_kl_dir = 0.05 * torch.mean(kl_dirichlet * sample_weights.unsqueeze(1))
                    
                    # C. Misleading Evidence Penalty
                    evidence = alpha_text - 1.0
                    non_target_mask = (1 - y_one_hot) 
                    loss_reg_misleading = anneal_factor * 0.5 * torch.mean(torch.sum(evidence * non_target_mask, dim=1))

                    # D. AvUC
                    pred_probs = alpha_text / S
                    logits_for_avuc = torch.log(pred_probs + 1e-7)
                    loss_avuc = anneal_factor * 1.0 * self.avuc_criterion(logits_for_avuc, batch_labels, u_text)
                    
                    # Total Loss
                    loss = loss_fit + loss_kl_dir + loss_reg_misleading + loss_avuc + loss_vib_kl
                    
                loss.backward()

                with torch.no_grad():
                     self.fusion.vib.evidence_scale.data.clamp_(max=3.0)

                torch.nn.utils.clip_grad_norm_(self.fusion.vib.parameters(), max_norm=2.0)
                self.optimizer.step()
                total_loss += loss.item()

                with torch.no_grad():
                    preds = logits_lm.argmax(dim=1)
                    is_correct = (preds == batch_labels)
                    u_val = u_text.detach().view(-1)
                    if is_correct.any(): u_correct_list.extend(u_val[is_correct].tolist())
                    if (~is_correct).any(): u_wrong_list.extend(u_val[~is_correct].tolist())
                
            # Metrics
            avg_loss = total_loss / len(self.dataloader)
            val_acc, val_f1 = self._validate_text_only()
            
            avg_u_c = np.mean(u_correct_list) if u_correct_list else 0
            avg_u_w = np.mean(u_wrong_list) if u_wrong_list else 0
            avg_u_gap = avg_u_w - avg_u_c

            avg_u_std = np.std(u_correct_list + u_wrong_list) if (u_correct_list + u_wrong_list) else 0
            avg_c_std = np.std(u_correct_list) if u_correct_list else 0
            avg_w_std = np.std(u_wrong_list) if u_wrong_list else 0

            scale_param = F.softplus(self.fusion.vib.evidence_scale).item()
        

            print(f"   VIB Ep {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            print(f"    Gap: {avg_u_gap:.4f} (W:{avg_u_w:.2f} - C:{avg_u_c:.2f}) | Scale: {scale_param:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "vib/loss": avg_loss,
                    "vib/val_f1": val_f1, 
                    "vib/u_correct": avg_u_c,
                    "vib/u_wrong": avg_u_w, 
                    "vib/u_gap": avg_u_gap,
                    "vib/u_std": avg_u_std,
                    "vib/u_correct_std": avg_c_std,
                    "vib/u_wrong_std": avg_w_std,
                    "vib/scale": scale_param
                })

            if val_f1 > best_f1:
                best_f1 = val_f1
                u_gap = avg_u_gap
                torch.save(self.fusion.vib.state_dict(), self.ckpt_filepath.parent / f"best_text_vib.pt")
        
        print(f"Text Expert Finished. best Val F1: {best_f1:.4f}, U_Gap: {u_gap:.4f}")
        self.freeze_module(self.fusion.vib, freeze=True)

    def _validate_text_only(self):
        """Helper for VIB validation"""
        self.fusion.vib.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                if hasattr(batch, 'n_id'):
                    # 如果是 MiniBatch 对象，提取 n_id
                    nodes = batch.n_id
                else:
                    # 如果已经是 Tensor
                    nodes = batch
                    
                nodes = nodes.to(self.device)
                
                # 1. 获取标签 (self.labels 通常在 GPU)
                batch_labels = self.labels[nodes]
                
                # 2. 获取 Embeddings (self.embeddings 通常在 CPU，需先转 CPU 索引再取)
                if self.embeddings.device == torch.device('cpu'):
                    batch_emb = self.embeddings[nodes.cpu()].to(self.device)
                else:
                    batch_emb = self.embeddings[nodes]
                
                # 3. 前向传播
                logits, _, _, _, _ = self.fusion.vib(batch_emb)
                
                # 4. 收集预测结果
                preds = logits.argmax(dim=1)
                val_preds.append(preds.cpu())
                val_labels.append(batch_labels.cpu())

        val_acc = accuracy_score(torch.cat(val_labels), torch.cat(val_preds))
        val_f1 = f1_score(torch.cat(val_labels), torch.cat(val_preds), average='macro')
        
        self.fusion.vib.train()  # 恢复训练模式
        return val_acc, val_f1
    
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
                # [FIXED] 解包 6 个值，包含 edge_type
                x, labels, edge_index, edge_type, batch_size, n_id = self._unpack_batch(batch)
                
                if hasattr(batch, 'n_id'):
                    h_sub = self._call_gnn(x, edge_index, edge_type)
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
                
                u_flat = u.view(-1) if u.dim() > 1 else u
                u_correct_list.extend(u_flat[is_correct].cpu().tolist())
                u_wrong_list.extend(u_flat[~is_correct].cpu().tolist())
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
        self.freeze_module(self.fusion.vib, True) 
        self.freeze_module(self.fusion.gate_net, True) 
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
        print(f"\n🚀 [Stage 3] Gate Mechanism Warmup ({epochs} epochs)")
        self.pretrain_gnn = False
        
        # 1. 严格冻结专家逻辑
        self.freeze_module(self.gnn, True)
        self.freeze_module(self.fusion.vib, True)
        self.freeze_module(self.fusion.gnn_proj, True)
        self.freeze_module(self.fusion.gnn_evidence_head, True)
        
        # 2. 激活待训练模块
        self.freeze_module(self.fusion.gate_net, False)
        
        gate_params = [p for p in self.fusion.parameters() if p.requires_grad]
        if len(gate_params) == 0:
            raise ValueError("⚠️ No parameters to train! Check freeze logic.")
        
        self.optimizer = torch.optim.AdamW(gate_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_f1 = 0
        for epoch in range(1, epochs + 1):

            self.current_epoch = epoch
            
            loss = self._train_epoch_fusion(epoch)
            
            # --- 验证循环 ---
            val_f1, val_acc = self.evaluate('val')
            
            gate_val = getattr(self, 'last_gate_mean', 0.5)
            u_gap = getattr(self, 'last_u_gap', 0.0)
            
            print(f"Gate Ep {epoch:02d} | Loss: {loss:8.4f} | Val F1: {val_f1:.4f} | Gate Avg: {gate_val:.3f} | U_Gap: {u_gap:.4f}")
            
            # 记录到 WandB
            wandb.log({
                "gate/loss": loss,
                "gate/val_f1": val_f1,
                "gate/gate_mean": gate_val,
                "gate/lr": self.optimizer.param_groups[0]['lr']
            })
            
            scheduler.step()
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_checkpoint(epoch)

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
        """
        Saves the model state.
        CRITICAL: Saves BOTH the GNN backbone AND the Fusion components (Proj+Head).
        This ensures Stage 3 loads exactly what Stage 2 trained.
        """
        # Ensure directory exists
        path = Path(self.ckpt_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.pretrain_gnn:
            # Save GNN + Fusion Partial State
            # This creates the 'gnn_state_dict' structure main.py expects
            save_dict = {
                'gnn_state_dict': {
                    **self.gnn.state_dict(),
                    # Prefix keys to match main.py loading logic
                    'classifier.0.weight': self.fusion.gnn_proj[0].weight,
                    'classifier.0.bias': self.fusion.gnn_proj[0].bias,
                    # Save head weights (assuming single linear layer)
                    'classifier.3.weight': self.fusion.gnn_evidence_head.evidence_layer.weight,
                    'classifier.3.bias': self.fusion.gnn_evidence_head.evidence_layer.bias,
                    'evidence_scale': self.fusion.gnn_evidence_head.evidence_scale
                },
                'epoch': epoch
            }
            torch.save(save_dict, path)
            # print(f"  💾 Saved GNN Expert to {path}")
            
        elif self.pretrain_llm:
            torch.save(self.fusion.vib.state_dict(), path)
            
        else:
            # Save full model for Gate/Fusion stages
            save_dict = {
                'gnn_state_dict': self.gnn.state_dict(),
                'fusion_state_dict': self.fusion.state_dict(),
                'epoch': epoch
            }
            torch.save(save_dict, path)

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

    def _audit_parameters(self, model_part, name="Module"):
        """
        [学术诊断] 审计模块的参数和梯度，定位 Loss 爆炸根源
        """
        total_norm = 0.0
        print(f"\n>>> [Audit {name}]")
        for p_name, p in model_part.named_parameters():
            if p.requires_grad:
                p_data = p.data.detach()
                # 统计权重大小
                w_max, w_std = p_data.max().item(), p_data.std().item()
                
                # 统计梯度大小
                g_stat = "None"
                if p.grad is not None:
                    g_data = p.grad.detach()
                    param_norm = g_data.norm(2).item()
                    total_norm += param_norm ** 2
                    g_stat = f"Norm:{param_norm:.2e} | MaxG:{g_data.max().item():.2e}"
                
                print(f"  {p_name:20s} | W_Max: {w_max:8.3f} | W_Std: {w_std:8.3f} | Grad: {g_stat}")
        
        total_norm = total_norm ** 0.5
        print(f"[*] Total Grad Norm: {total_norm:.4f}")
        return total_norm