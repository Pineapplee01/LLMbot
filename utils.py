import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os
import numpy as np
import wandb
import json

from pathlib import Path
from torch_geometric.utils import add_self_loops

def semantic_structure_refinement(data_dict, embeddings, threshold=0.5):
    """
    [SeGA v3.0 Core] Semantic-Guided Graph Pruning (inspired by BotLGT).
    Refines graph structure by removing edges with low semantic similarity
    and ensuring self-loops for structural stability.
    
    Args:
        data_dict (dict): Dictionary containing 'edge_index' and 'edge_type'.
        embeddings (Tensor): Precomputed LLM embeddings [Num_Nodes, Dim].
        threshold (float): Cosine similarity threshold for pruning.
    
    Returns:
        dict: Updated data_dict with refined structure.
    """
    print(f"\n[Graph Refinement] Starting semantic pruning (Threshold={threshold})...")
    
    edge_index = data_dict['edge_index']
    device = embeddings.device
    
    # 1. Prepare Embeddings (Move to CPU to save GPU memory during big matrix ops if needed)
    # forcing computation on same device as embeddings
    src_idx, dst_idx = edge_index[0], edge_index[1]
    
    emb_src = embeddings[src_idx]
    emb_dst = embeddings[dst_idx]
    
    # 2. Compute Cosine Similarity for existing edges only
    # Sim(A, B) = (A . B) / (|A| * |B|)
    emb_src_norm = F.normalize(emb_src, p=2, dim=1)
    emb_dst_norm = F.normalize(emb_dst, p=2, dim=1)
    
    # Element-wise dot product
    similarity = (emb_src_norm * emb_dst_norm).sum(dim=1)
    
    # 3. Filter Edges
    mask = similarity > threshold
    pruned_edge_index = edge_index[:, mask]
    
    num_original = edge_index.shape[1]
    num_kept = pruned_edge_index.shape[1]
    num_dropped = num_original - num_kept
    
    print(f"[Graph Refinement] Edges processed: {num_original}")
    print(f"[Graph Refinement] Pruned edges:    {num_dropped} ({num_dropped/num_original:.2%})")
    
    # 4. Handle Edge Types (if exist)
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        data_dict['edge_type'] = data_dict['edge_type'][mask]
        # Note: If adding self-loops later, need to handle edge_type for self-loops
        # For simplicity in RGT, usually a special relation type is assigned or it's handled implicitly.
        # Here we preserve the alignment for RGT.
    
    # 5. [Crucial for Sparse Graphs] Add Self-Loops
    # Ensures isolated nodes (after pruning) can still aggregate their own features in GNN.
    final_edge_index, _ = add_self_loops(pruned_edge_index, num_nodes=embeddings.shape[0])
    
    print(f"[Graph Refinement] Final edges (with self-loops): {final_edge_index.shape[1]}")
    
    # Update data_dict
    data_dict['edge_index'] = final_edge_index
    
    # If edge_type exists, we need to pad it for self-loops. 
    # Usually, we assign a new relation id or 0 for self-loops.
    # Assuming RGT handles relations, we simply extend edge_type with a default relation (e.g., 0)
    if 'edge_type' in data_dict and data_dict['edge_type'] is not None:
        num_self_loops = final_edge_index.shape[1] - num_kept
        # Append relation 0 (or similar) for self-loops
        self_loop_types = torch.zeros(num_self_loops, dtype=torch.long, device=device)
        data_dict['edge_type'] = torch.cat([data_dict['edge_type'], self_loop_types], dim=0)

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