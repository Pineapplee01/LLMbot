import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os
import numpy as np
import wandb
import json

from pathlib import Path
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter

def relation_aware_knn_pruning(data_dict, embeddings, k=5, z_score_threshold=0.0):
    """
    [SeGA v4.0 Core] Local Z-score/Top-K Pruning.
    修复了 edge_type 维度爆炸的 Bug，确保边索引和边类型严格对齐。
    """
    print(f"\n[Graph Refinement] Starting Relation-Aware KNN Pruning (K={k})...")
    
    edge_index = data_dict['edge_index']
    edge_type = data_dict['edge_type']
    device = edge_index.device
    num_nodes = embeddings.size(0)

    # 容器用于存放筛选后的边
    refined_edges_list = []
    refined_types_list = []
    
    # 移动 Embedding 到同一设备用于计算
    embeddings = embeddings.to(device)

    mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)

    # 获取图中包含的所有关系类型 (通常是 0 和 1)
    unique_relations = torch.unique(edge_type)

    total_kept = 0
    
    for r_type in unique_relations:
        # 1. 提取当前关系的所有边
        # mask shape: [E_total], sub_edges shape: [2, E_rel]
        r_mask = (edge_type == r_type)
        r_edges = edge_index[:, r_mask]

        global_indices = torch.where(r_mask)[0]
        
        if r_edges.shape[1] == 0:
            continue
            
        src_nodes = r_edges[0]
        dst_nodes = r_edges[1]
        
        # 2. 计算余弦相似度 (Element-wise)
        emb_src = F.normalize(embeddings[src_nodes], p=2, dim=1)
        emb_dst = F.normalize(embeddings[dst_nodes], p=2, dim=1)
        sims = torch.sum(emb_src * emb_dst, dim=1)

        
        # 3. 执行 Top-K 筛选
        unique_src = torch.unique(src_nodes)
        
        nodes_processed = 0
        edges_kept_local = 0
        
        # 但考虑到 TwiBot-20 规模尚可，此循环是安全的。
        # 如果追求极致速度，可使用 torch_scatter 或 argsort 优化。
        for u in unique_src:
            
            # 找到节点 u 发出的所有边在 sub_edges 中的索引
            loc_idx = torch.where(src_nodes == u)[0]
            u_sims = sims[loc_idx]

            k_actual = min(len(loc_idx), k)
            vals, topk_rel_idx = torch.topk(u_sims, k_actual)
            
            if len(loc_idx) > 2:
                mean = u_sims.mean()
                std = u_sims.std()
                
                # 动态阈值: 必须大于均值
                score_mask = vals > (mean + z_score_threshold * std)
                
                # 至少保留 1 个最好的，防止孤立
                if score_mask.sum() == 0:
                    score_mask[0] = True
                
                topk_rel_idx = topk_rel_idx[score_mask]

            indices = global_indices[loc_idx[topk_rel_idx]]
            mask[indices] = True
            
            nodes_processed += 1
            edges_kept_local += len(indices)
        
        print(f"  - Relation {r_type.item()}: Processed {nodes_processed} nodes. Kept {edges_kept_local}/{len(global_indices)} edges.")


    # 5. 合并所有关系的边
    new_edge_index = edge_index[:, mask]
    new_edge_type = edge_type[mask]

    print(f"[Graph Refinement] Total Pruned: {edge_index.shape[1]} -> {new_edge_index.shape[1]}")
    print(f"[Graph Refinement] Reduction Rate: {1 - new_edge_index.shape[1]/edge_index.shape[1]:.2%}")

    # 5. 添加自环 (关键！防止 RGT 崩溃)
    new_edge_index, new_edge_type = remove_self_loops(new_edge_index, new_edge_type)
    new_edge_index, _ = add_self_loops(new_edge_index, num_nodes=num_nodes)
    
    # 补充 edge_type
    num_self_loops = num_nodes
    # 假设自环是类型 0 (或者你可以定义为 max_type + 1)
    loop_types = torch.zeros(num_self_loops, dtype=torch.long, device=device)
    new_edge_type = torch.cat([new_edge_type, loop_types], dim=0)
    
    data_dict['edge_index'] = new_edge_index
    data_dict['edge_type'] = new_edge_type
    
    return data_dict

def _torch_load(path):
    # 尝试使用 weights_only（若当前 PyTorch 支持），否则回退到普通加载
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def seed_setting(seed_number):
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def setup_wandb(args, seed):
    run = wandb.init(
        project=args.project_name,
        name=args.experiment_name + f'_seed_{seed}',
        config=args 
    )
    return run


def load_raw_data(dataset_path, use_GNN=True):
    """
    🚀 **Enhanced TwiBot-20 Dataset Loader**
    
    Robust data loader for TwiBot-20 and similar datasets.
    Handles multiple path formats and ensures consistent file loading.
    
    Args:
        dataset_path (str): Path to dataset directory
            - Support formats: './datasets/TwiBot-20', 'TwiBot-20', './datasets/TwiBot20'
        use_GNN (bool): Whether to load graph structure data (edge_index, edge_type)
    
    Returns:
        dict: Dictionary containing loaded tensors:
            - 'train_idx': Training node indices [N_train] (TwiBot-20: [8278])
            - 'valid_idx': Validation node indices [N_valid] (TwiBot-20: [2365]) 
            - 'test_idx': Test node indices [N_test] (TwiBot-20: [1183])
            - 'labels': Node labels [N_nodes, N_classes] (TwiBot-20: [11826, 2] one-hot)
            - 'edge_index': Edge connectivity [2, N_edges] (TwiBot-20: [2, 16908])
            - 'edge_type': Edge types [N_edges] (TwiBot-20: [16908], values 0/1)
            - 'user_text': Text data from norm_user_text.json (if exists)
    
    File Name Mapping (Based on TwiBot-20 Structure Analysis):
        ✅ train_idx.pt    -> Training indices (consecutive: [0, 1, 2, ...])
        ✅ valid_idx.pt    -> Validation indices (consecutive: [8278, 8279, ...])  
        ✅ test_idx.pt     -> Test indices (consecutive: [10643, 10644, ...])
        ✅ labels.pt       -> One-hot labels [11826, 2] (Human: 5237, Bot: 6589)
        ✅ edge_index.pt   -> Graph edges [2, 16908] (COO format)
        ✅ edge_type.pt    -> Edge types [16908] (binary: 0 or 1)
        ✅ norm_user_text.json -> User text content
    
    Raises:
        FileNotFoundError: If dataset path cannot be resolved
        FileNotFoundError: If required data files are missing
    """
    # === 1. Path Resolution with Robust Error Handling ===
    data_dir = Path(dataset_path)
    print(f"[DataLoader] Resolving dataset path: '{dataset_path}'")

    # Try direct path first
    if not data_dir.exists():
        # Try ./datasets/<name> format
        cand = Path('./datasets') / data_dir.name
        if cand.exists():
            data_dir = cand
            print(f"[DataLoader] ✅ Found dataset at: {data_dir}")
        else:
            # Handle duplicated prefixes (e.g., './datasets/./datasets/TwiBot-20')
            s = str(data_dir).replace('./datasets/', '').replace('datasets/', '')
            cand2 = Path('./datasets') / s
            if cand2.exists():
                data_dir = cand2
                print(f"[DataLoader] ✅ Found dataset at: {data_dir} (cleaned path)")


    data_filepath = data_dir  # Use Path object for robust file operations

    print(f"[DataLoader] Loading data from: {data_filepath}")
    
    # === 2. Load Core Data Files ===
    print("[DataLoader] Loading data split indices...")
    
    # 🔹 Training/Validation/Test Splits (Index Format)
    # Based on analysis: TwiBot-20 uses consecutive indices
    train_idx = _torch_load(data_filepath / 'train_idx.pt')  # [8278] indices
    print(f"[DataLoader] ✅ train_idx: {train_idx.shape} (range: [{train_idx.min()}, {train_idx.max()}])")
    
    valid_idx = _torch_load(data_filepath / 'valid_idx.pt')  # [2365] indices  
    print(f"[DataLoader] ✅ valid_idx: {valid_idx.shape} (range: [{valid_idx.min()}, {valid_idx.max()}])")
    
    test_idx = _torch_load(data_filepath / 'test_idx.pt')    # [1183] indices
    print(f"[DataLoader] ✅ test_idx: {test_idx.shape} (range: [{test_idx.min()}, {test_idx.max()}])")
    
    print("[DataLoader] Loading node labels...")
    
    # 🔹 Labels (One-Hot Format Detection)
    # TwiBot-20: [11826, 2] one-hot -> Human: 5237, Bot: 6589
    labels = _torch_load(data_filepath / 'labels.pt')
    print(f"[DataLoader] ✅ labels: {labels.shape} dtype={labels.dtype}")


    # === 3. Load Graph Structure (if requested) ===
    if use_GNN:
        print("[DataLoader] Loading graph structure data...")
        
        # 🔹 Edge Index (Graph Connectivity)
        edge_index = _torch_load(data_filepath / 'edge_index.pt')
        
        # 🔹 Edge Type (Relation Types) - Optional

        edge_type = _torch_load(data_filepath / 'edge_type.pt')
    
        # 🔹 Text Data (User Metadata and Content) - Optional
        with open(data_filepath / 'norm_user_text.json', 'r', encoding='utf-8') as f:
            user_text = json.load(f)
            
        
        # Return complete dataset with graph structure
        data_dict = {
            'train_idx': train_idx,
            'valid_idx': valid_idx, 
            'test_idx': test_idx,
            'labels': labels,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'user_text': user_text
        }

    else:
        print("[DataLoader] 📝 Graph structure not requested (use_GNN=False)")
        # Return basic dataset without graph structure  
        data_dict = {
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx, 
            'labels': labels
        }
    
    # === 4. Final Validation ===
    total_nodes = len(train_idx) + len(valid_idx) + len(test_idx)
    expected_nodes = labels.shape[0]
    
    if total_nodes == expected_nodes:
        print(f"[DataLoader] ✅ Data consistency check passed: {total_nodes} total indices = {expected_nodes} labels")
    else:
        print(f"[DataLoader] ⚠️ Data consistency warning: {total_nodes} total indices ≠ {expected_nodes} labels")
    
    print(f"[DataLoader] 🎯 Dataset loaded successfully! Keys: {list(data_dict.keys())}")
    return data_dict


def load_distilled_knowledge(from_which_model, intermediate_data_filepath, iter):
    if from_which_model == 'LM':
        embeddings = torch.load(intermediate_data_filepath / f'embeddings_iter_{iter}.pt')
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return embeddings, soft_labels
    
    elif from_which_model == 'GNN':
       
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels

    elif from_which_model == 'MLP':
        soft_labels = torch.load(intermediate_data_filepath / f'soft_labels_iter_{iter}.pt')
        return soft_labels
    
    else:
        raise ValueError('"from_which_model" should be "LM", "GNN" or "MLP".')


def prepare_path(experiment_name):
    experiment_path = Path(experiment_name)
    ckpt_filepath = experiment_path / 'checkpoints'
    MLP_ckpt_filepath = ckpt_filepath / 'MLP'
    LM_ckpt_filepath = ckpt_filepath / 'LM'
    GNN_ckpt_filepath = ckpt_filepath / 'GNN'
    MLP_KD_ckpt_filepath = Path('MLP_KD')
    LM_prt_ckpt_filepath = ckpt_filepath / 'LM_pretrain'
    GNN_prt_ckpt_filepath = ckpt_filepath / 'GNN_pretrain'
    LM_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_prt_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    LM_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    GNN_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_KD_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    MLP_ckpt_filepath.mkdir(exist_ok=True, parents=True)
    
    LM_intermediate_data_filepath = experiment_path / 'intermediate' / 'LM'
    GNN_intermediate_data_filepath = experiment_path / 'intermediate' / 'GNN'
    MLP_intermediate_data_filepath = experiment_path / 'intermediate' / 'MLP'
    LM_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    GNN_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)
    MLP_intermediate_data_filepath.mkdir(exist_ok=True, parents=True)

    return LM_prt_ckpt_filepath, GNN_prt_ckpt_filepath, MLP_KD_ckpt_filepath, LM_ckpt_filepath, GNN_ckpt_filepath, MLP_ckpt_filepath, LM_intermediate_data_filepath, GNN_intermediate_data_filepath, MLP_intermediate_data_filepath
    
def reset_split(n_nodes, ratio):
    idx = torch.randperm(n_nodes)
    split = list(map(int, ratio.split(',')))
    train_ratio = split[0] / sum(split)
    valid_ratio = split[1] / sum(split)

    train_idx = idx[: int(train_ratio * n_nodes)]
    valid_idx = idx[int(train_ratio * n_nodes): int((train_ratio + valid_ratio) * n_nodes)]
    test_idx = idx[int((train_ratio + valid_ratio) * n_nodes):]
    return train_idx, valid_idx, test_idx

def batch_linear_cka(X, Y):
    """计算 Batch 内的线性 CKA 相似度，用于诊断模态鸿沟"""
    
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    gram_x = torch.matmul(X, X.t())
    gram_y = torch.matmul(Y, Y.t())
    
    f_x = torch.norm(gram_x, p='fro')
    f_y = torch.norm(gram_y, p='fro')
    
    return torch.sum(gram_x * gram_y) / (f_x * f_y + 1e-8)

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
    这会自动处理 Hard Sample Mining，因为温度系数会放大困难样本的梯度。
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size, dim] - 必须是归一化后的特征
        labels: [batch_size]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(1) # [batch, 1, dim]

        batch_size = features.shape[0]
        
        # 构造 Mask
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 Log-Prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算 Mean Log-Likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss