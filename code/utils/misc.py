"""
utils/misc.py
Shared utilities: seeding, belief fusion, structural features, wandb proxy.
"""
from __future__ import annotations

import importlib
import os
import random
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree as calc_degree, scatter


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_setting(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# ── Belief fusion ─────────────────────────────────────────────────────────────

def jsd_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence between two probability distributions."""
    p, q = p.clamp(min=eps), q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(dim=1, keepdim=True)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(dim=1, keepdim=True)
    return 0.5 * (kl_pm + kl_qm)


def weighted_belief_fusion(
    alpha_list: List[torch.Tensor],
    weight_list: List[torch.Tensor],
    eps: float = 1e-8,
    num_classes: int = 2,
) -> torch.Tensor:
    """Weighted Dempster-Shafer belief fusion over Dirichlet parameters."""
    evidences = [a - 1.0 for a in alpha_list]
    fused = torch.zeros_like(evidences[0])
    for w, e in zip(weight_list, evidences):
        fused = fused + w * e
    return fused + 1.0


# ── Graph structural features ─────────────────────────────────────────────────

def compute_directed_structural_features(
    edge_index: torch.Tensor, num_nodes: int, x: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute 5-dim per-node topology-only structural features.

    Returns [N, 5]:
        0: in_log_degree      (z-normed)
        1: out_log_degree     (z-normed)
        2: total_log_degree   (z-normed)
        3: neighbor_deg_var   (z-normed log variance of neighbor degrees)
        4: graph_missing      (1.0 if isolated node)

    The `x` parameter is accepted for API compatibility but not used.
    """
    row, col = edge_index
    in_deg = calc_degree(col, num_nodes=num_nodes, dtype=torch.float)
    out_deg = calc_degree(row, num_nodes=num_nodes, dtype=torch.float)
    total_deg = in_deg + out_deg

    def _znorm_log(deg: torch.Tensor) -> torch.Tensor:
        ld = torch.log1p(deg)
        return ((ld - ld.mean()) / (ld.std() + 1e-6)).unsqueeze(1)

    sym_edge = torch.unique(torch.cat([edge_index, edge_index.flip(0)], dim=1), dim=1)
    sym_row, sym_col = sym_edge
    neighbor_deg = total_deg[sym_row]
    neighbor_deg_mean = scatter(neighbor_deg, sym_col, dim=0, dim_size=num_nodes, reduce="mean")
    sq_diff = (neighbor_deg - neighbor_deg_mean[sym_col]) ** 2
    neighbor_deg_var = scatter(sq_diff, sym_col, dim=0, dim_size=num_nodes, reduce="mean")
    ndv = torch.log1p(neighbor_deg_var)
    ndv_normed = ((ndv - ndv.mean()) / (ndv.std() + 1e-6)).unsqueeze(1)
    graph_missing = (total_deg == 0).float().unsqueeze(1)

    return torch.cat([_znorm_log(in_deg), _znorm_log(out_deg), _znorm_log(total_deg), ndv_normed, graph_missing], dim=1)


# ── Wandb compatibility proxy ─────────────────────────────────────────────────

def _noop(*args, **kwargs):
    return None


class _NoOpBackend:
    run = None

    def init(self, *args, **kwargs): return None
    def log(self, *args, **kwargs): return None
    def finish(self, *args, **kwargs): return None


class WandbProxy:
    """Lazy wandb proxy — no-op until enable_wandb() is called."""

    def __init__(self) -> None:
        self._backend: Any = _NoOpBackend()

    @property
    def run(self):
        return getattr(self._backend, "run", None)

    def set_backend(self, backend: Optional[Any]) -> None:
        self._backend = backend if backend is not None else _NoOpBackend()

    def __getattr__(self, name: str):
        return getattr(self._backend, name, _noop)


wandb = WandbProxy()


def configure_disabled_env() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_CONSOLE", "off")


def enable_wandb(required: bool = True):
    if not isinstance(getattr(wandb, "_backend", None), _NoOpBackend):
        return wandb._backend
    try:
        backend = importlib.import_module("wandb")
    except ImportError:
        if required:
            raise
        return None
    wandb.set_backend(backend)
    return backend
