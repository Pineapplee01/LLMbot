import torch


def _as_long_cpu_tensor(idx):
    if idx is None:
        return torch.empty(0, dtype=torch.long)
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().reshape(-1)
    return torch.tensor(idx, dtype=torch.long).reshape(-1)


def _safe_pseudo_label_training_index(train_idx, valid_idx, test_idx, n_total, pl_ratio, pseudo_label_pool_idx=None):
    """Build train + explicit pseudo-label indices without ever using val/test by complement."""
    train_idx = _as_long_cpu_tensor(train_idx)
    valid_idx = _as_long_cpu_tensor(valid_idx)
    test_idx = _as_long_cpu_tensor(test_idx)
    pool_idx = _as_long_cpu_tensor(pseudo_label_pool_idx)
    n_train = int(train_idx.numel())
    if pool_idx.numel() == 0 or n_train == 0 or float(pl_ratio) <= 0.0:
        is_pl = torch.zeros_like(train_idx, dtype=torch.int64)
        return train_idx, is_pl, torch.empty(0, dtype=torch.long)

    val_test = set(valid_idx.tolist()) | set(test_idx.tolist())
    train_set = set(train_idx.tolist())
    pool_set = set(pool_idx.tolist())
    if pool_set & val_test:
        raise ValueError("pseudo_label_pool_idx must be disjoint from validation and test splits.")
    if pool_set & train_set:
        raise ValueError("pseudo_label_pool_idx must be disjoint from train_idx.")
    if pool_idx.min().item() < 0 or pool_idx.max().item() >= int(n_total):
        raise ValueError("pseudo_label_pool_idx contains node ids outside the graph.")

    n_pl = min(int(n_train * float(pl_ratio)), int(pool_idx.numel()))
    if n_pl <= 0:
        is_pl = torch.zeros_like(train_idx, dtype=torch.int64)
        return train_idx, is_pl, torch.empty(0, dtype=torch.long)
    chosen = pool_idx[torch.randperm(pool_idx.numel())[:n_pl]]
    train_idx_all = torch.cat((train_idx, chosen))
    is_pl = torch.ones_like(train_idx_all, dtype=torch.int64)
    is_pl[:n_train] = 0
    return train_idx_all, is_pl, chosen
