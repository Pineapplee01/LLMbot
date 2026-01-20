import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os
import numpy as np
import wandb
import json
from pathlib import Path


def _torch_load_robust(path):
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


def load_raw_data(dataset_path, use_GNN=False):
    """
    Robust loader: accept either:
      - './datasets/TwiBot-20'
      - 'TwiBot-20'
      - './datasets/TwiBot20'
    Resolve to an existing directory and avoid double './datasets/./datasets/...' bugs.
    Returns a dict with tensors.
    """
    data_dir = Path(dataset_path)

    # Try direct path, then ./datasets/<name>, then strip duplicated prefixes
    if not data_dir.exists():
        cand = Path('./datasets') / data_dir.name
        if cand.exists():
            data_dir = cand
        else:
            s = str(data_dir).replace('./datasets/', '').replace('datasets/', '')
            cand2 = Path('./datasets') / s
            if cand2.exists():
                data_dir = cand2

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found. Tried: '{dataset_path}' -> resolved to '{data_dir}'")

    data_filepath = data_dir  # Path object

    # Load files using Path / operator to avoid manual string concat errors
    train_idx = _torch_load_robust(data_filepath / 'train_idx.pt')
    valid_idx = _torch_load_robust(data_filepath / 'valid_idx.pt')
    test_idx  = _torch_load_robust(data_filepath / 'test_idx.pt')
    labels    = _torch_load_robust(data_filepath / 'labels.pt')
    edge_index = _torch_load_robust(data_filepath / 'edge_index.pt')
    try:
        edge_type = _torch_load_robust(data_filepath / 'edge_type.pt')
    except Exception:
        edge_type = None

    return {
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx,
        'labels': labels,
        'edge_index': edge_index,
        'edge_type': edge_type
    }


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