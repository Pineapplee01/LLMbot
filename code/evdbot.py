"""
LEGACY / INACTIVE MODULE

This file is preserved only as a historical reference. It is not part of the
active `code/` execution path, which now flows through:
`precompute.py` -> `main.py` -> `model.py` -> `train.py`.

The original source is wrapped below as an inert snapshot so it can still be
read during archaeology without remaining import-active.
"""

LEGACY_SOURCE = r'''
evdbot.py
EVD-Bot: Evidential Dual-View Bot Detection

Asynchronous Two-Stage Training Architecture:
  Phase 1: SemanticEvidentialExtractor trained alone → freeze outputs
  Phase 2: VarianceAwareGNN + DSFusion trained on frozen Phase 1 outputs

Modules:
  - EVDBotConfig              (centralized hyperparameters)
  - SemanticEvidentialExtractor (LLM embedding → Dirichlet params + projection)
  - VarianceAwareGNNLayer     (precision-weighted message passing)
  - VarianceAwareGNN          (multi-layer stack)
  - DSFusionModule            (structural evidence + Dempster-Shafer fusion)
  - EvidentialLoss            (Type II ML + KL with annealing)
  - Phase1_SemanticModel      (Phase 1 top-level)
  - Phase2_StructuralModel    (Phase 2 top-level)

References:
  - Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty", NeurIPS 2018
  - Han et al., "Trusted Multi-View Classification", ICLR 2021
  - Zügner & Günnemann, "Reliable Graph Neural Networks via Robust
    Aggregation", NeurIPS 2020
"""

import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class EVDBotConfig:
    """Centralized configuration for EVD-Bot.

    All hyperparameters are defined here. No values are hardcoded in modules.

    Attributes:
        d_llm: Dimensionality of input LLM embeddings.
        d_proj: Projection dimension (shared semantic→GNN interface).
        d_gnn_hidden: Hidden dimension of GNN layers.
        num_classes: Number of output classes.
        num_gnn_layers: Number of VarianceAwareGNN layers.
        num_relations: Number of edge relation types.
        dropout: Dropout rate.
        tau: Fallback routing threshold. At inference, if
            u_struct > u_sem * tau, fall back to semantic-only prediction.
        lambda_kl: Maximum KL divergence regularization weight.
        anneal_steps: Number of epochs to linearly anneal lambda_kl from 0.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        eps: Numerical stability epsilon.
        alpha_clamp_min: Minimum alpha value before digamma/lgamma calls.
    """

    d_llm: int = 4096
    d_proj: int = 256
    d_gnn_hidden: int = 256
    num_classes: int = 2
    num_gnn_layers: int = 2
    num_relations: int = 2
    dropout: float = 0.1
    tau: float = 1.5
    lambda_kl: float = 1.0
    anneal_steps: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eps: float = 1e-7
    alpha_clamp_min: float = 1.01


# ── Module 1: Semantic Evidential Extractor ───────────────────────────────────


class SemanticEvidentialExtractor(nn.Module):
    """Maps LLM embeddings to Dirichlet parameters and a projected feature.

    Two independent pathways:
      - Evidence path: h_llm → MLP → ReLU → e; alpha = e + 1
      - Projection path: h_llm → Linear → LayerNorm → ReLU → h_proj

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize SemanticEvidentialExtractor.

        Args:
            config: EVDBotConfig with d_llm, d_proj, num_classes, dropout.
        """
        super().__init__()
        self.config = config
        self.evidence_layer = nn.Sequential(
            nn.Linear(config.d_llm, config.d_proj),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_proj, config.num_classes),
        )
        self.projection_layer = nn.Sequential(
            nn.Linear(config.d_llm, config.d_proj),
            nn.LayerNorm(config.d_proj),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, h_llm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute semantic Dirichlet parameters and projected features.

        Args:
            h_llm: LLM embedding matrix of shape ``[N, D_llm]``.

        Returns:
            Dictionary with keys:
                - ``alpha_sem``: Dirichlet parameters of shape ``[N, K]``.
                - ``u_sem``: Semantic uncertainty of shape ``[N, 1]``.
                - ``h_proj``: Projected features of shape ``[N, d_proj]``.
        """
        e: torch.Tensor = F.relu(self.evidence_layer(h_llm))
        alpha_sem: torch.Tensor = e + 1.0
        S: torch.Tensor = alpha_sem.sum(dim=1, keepdim=True)
        u_sem: torch.Tensor = self.config.num_classes / S
        h_proj: torch.Tensor = self.projection_layer(h_llm)
        return {"alpha_sem": alpha_sem, "u_sem": u_sem, "h_proj": h_proj}


# ── Module 2: Variance-Aware GNN ─────────────────────────────────────────────


class VarianceAwareGNNLayer(MessagePassing):
    """Single GNN layer with precision-weighted message passing.

    Messages from neighbours are weighted by the inverse variance (precision)
    derived from each sender's semantic uncertainty. Relation-specific linear
    transforms are applied before weighting.

    Mathematical formulation:
        sigma_j^2 = exp(w_u * u_sem_j + b_u)
        msg_j     = (1 / (sigma_j^2 + eps)) * W_r @ h_j
        h_v'      = LeakyReLU(LN(sum(msg_j) + W_self @ h_v))

    Attributes:
        config: EVDBotConfig instance.
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        config: EVDBotConfig,
    ) -> None:
        """Initialize VarianceAwareGNNLayer.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            num_relations: Number of edge relation types.
            config: EVDBotConfig with eps.
        """
        super().__init__(aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config

        self.rel_linears = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
        )
        self.root_linear = nn.Linear(in_dim, out_dim, bias=True)
        self.ln = nn.LayerNorm(out_dim)

        self.w_u = nn.Parameter(torch.tensor(1.0))
        self.b_u = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        u_sem: torch.Tensor,
    ) -> torch.Tensor:
        """Run one layer of variance-aware message passing.

        Args:
            x: Node feature matrix of shape ``[N, in_dim]``.
            edge_index: Edge connectivity of shape ``[2, E]``.
            edge_type: Relation type per edge of shape ``[E]``.
            u_sem: Semantic uncertainty per node of shape ``[N, 1]``.

        Returns:
            Updated node features of shape ``[N, out_dim]``.
        """
        agg: torch.Tensor = self.propagate(
            edge_index, x=x, edge_type=edge_type, u_sem=u_sem
        )
        out: torch.Tensor = self.ln(agg + self.root_linear(x))
        return F.leaky_relu(out)

    def message(
        self,
        x_j: torch.Tensor,
        edge_type: torch.Tensor,
        u_sem_j: torch.Tensor,
    ) -> torch.Tensor:
        """Construct precision-weighted, relation-transformed messages.

        Args:
            x_j: Source node features of shape ``[E, in_dim]``.
            edge_type: Relation type per edge of shape ``[E]``.
            u_sem_j: Source node semantic uncertainty of shape ``[E, 1]``.

        Returns:
            Messages of shape ``[E, out_dim]``.
        """
        sigma_sq: torch.Tensor = torch.exp(self.w_u * u_sem_j + self.b_u)
        precision: torch.Tensor = 1.0 / (sigma_sq + self.config.eps)

        transformed = torch.zeros(
            x_j.size(0), self.out_dim, device=x_j.device, dtype=x_j.dtype
        )
        for r, lin in enumerate(self.rel_linears):
            mask: torch.Tensor = edge_type == r
            if mask.any():
                transformed[mask] = lin(x_j[mask])

        return precision * transformed


class VarianceAwareGNN(nn.Module):
    """Multi-layer variance-aware GNN stack.

    Stacks ``num_gnn_layers`` VarianceAwareGNNLayer instances with dropout.
    The first layer maps from ``d_proj`` to ``d_gnn_hidden``; subsequent
    layers are ``d_gnn_hidden`` to ``d_gnn_hidden``.

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize VarianceAwareGNN.

        Args:
            config: EVDBotConfig with d_proj, d_gnn_hidden, num_gnn_layers,
                num_relations, dropout.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for i in range(config.num_gnn_layers):
            in_dim = config.d_proj if i == 0 else config.d_gnn_hidden
            self.layers.append(
                VarianceAwareGNNLayer(
                    in_dim=in_dim,
                    out_dim=config.d_gnn_hidden,
                    num_relations=config.num_relations,
                    config=config,
                )
            )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        h_proj: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        u_sem: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full GNN stack.

        Args:
            h_proj: Projected node features of shape ``[N, d_proj]``.
            edge_index: Edge connectivity of shape ``[2, E]``.
            edge_type: Relation type per edge of shape ``[E]``.
            u_sem: Semantic uncertainty per node of shape ``[N, 1]``.
                Must be detached (no gradient to Phase 1).

        Returns:
            GNN output features of shape ``[N, d_gnn_hidden]``.
        """
        h: torch.Tensor = h_proj
        for layer in self.layers:
            h = layer(h, edge_index, edge_type, u_sem)
            h = self.dropout(h)
        return h


# ── Module 3: Dempster-Shafer Fusion ─────────────────────────────────────────


class DSFusionModule(nn.Module):
    """Structural evidence extraction and Dempster-Shafer fusion.

    Computes structural Dirichlet parameters from GNN output, then combines
    them with frozen semantic parameters via Dempster's rule:
        alpha_combined = alpha_sem + alpha_struct - 1

    During training, always uses the DS combination (gradients flow through
    structural branch). During inference, applies fallback routing: if
    structural uncertainty exceeds ``tau * semantic uncertainty``, the
    semantic-only prediction is used.

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize DSFusionModule.

        Args:
            config: EVDBotConfig with d_gnn_hidden, num_classes, tau.
        """
        super().__init__()
        self.config = config
        self.struct_evidence_layer = nn.Linear(
            config.d_gnn_hidden, config.num_classes
        )

    def forward(
        self,
        alpha_sem_frozen: torch.Tensor,
        u_sem_frozen: torch.Tensor,
        h_gnn: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse semantic and structural Dirichlet parameters.

        Args:
            alpha_sem_frozen: Frozen semantic Dirichlet parameters of shape
                ``[B, K]``. Must be detached.
            u_sem_frozen: Frozen semantic uncertainty of shape ``[B, 1]``.
                Must be detached.
            h_gnn: GNN output features of shape ``[B, d_gnn_hidden]``.

        Returns:
            Dictionary with keys:
                - ``alpha_struct``: Structural Dirichlet params ``[B, K]``.
                - ``u_struct``: Structural uncertainty ``[B, 1]``.
                - ``alpha_final``: Fused or fallback Dirichlet params ``[B, K]``.
                - ``u_final``: Final uncertainty ``[B, 1]``.
                - ``routing_mask``: ``[B, 1]``. 1.0 = DS used, 0.0 = fallback.
        """
        K: int = self.config.num_classes

        e_struct: torch.Tensor = F.relu(self.struct_evidence_layer(h_gnn))
        alpha_struct: torch.Tensor = e_struct + 1.0
        S_struct: torch.Tensor = alpha_struct.sum(dim=1, keepdim=True)
        u_struct: torch.Tensor = K / S_struct

        alpha_combined: torch.Tensor = alpha_sem_frozen + alpha_struct - 1.0

        if self.training:
            alpha_final = alpha_combined
            routing_mask = torch.ones_like(u_struct)
        else:
            routing_mask = (u_struct <= u_sem_frozen * self.config.tau).float()
            alpha_final = (
                routing_mask * alpha_combined
                + (1.0 - routing_mask) * alpha_sem_frozen
            )
            logger.debug(
                "DS routing: %.1f%% nodes use DS combination",
                routing_mask.mean().item() * 100,
            )

        u_final: torch.Tensor = K / alpha_final.sum(dim=1, keepdim=True)

        return {
            "alpha_struct": alpha_struct,
            "u_struct": u_struct,
            "alpha_final": alpha_final,
            "u_final": u_final,
            "routing_mask": routing_mask,
        }


# ── Module 4: Evidential Loss ────────────────────────────────────────────────


class EvidentialLoss(nn.Module):
    """Type II Maximum Likelihood loss with KL regularization.

    Single-view loss: called once per training phase (NOT a joint loss).

    Loss for a single view:
        L = digamma_CE(alpha, y) + lambda * KL[Dir(alpha_tilde) || Dir(1)]

    where alpha_tilde = y + (1 - y) * alpha removes evidence from the
    correct class before KL regularization, and lambda is annealed linearly
    from 0 to lambda_kl over ``anneal_steps`` epochs.

    References:
        Sensoy et al., NeurIPS 2018, Eq. 3-5.

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize EvidentialLoss.

        Args:
            config: EVDBotConfig with num_classes, lambda_kl, anneal_steps,
                eps, alpha_clamp_min.
        """
        super().__init__()
        self.config = config

    def forward(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute evidential loss for a single view.

        Args:
            alpha: Dirichlet parameters of shape ``[N, K]``.
            target: Ground-truth class indices of shape ``[N]``.
            epoch: Current epoch (1-indexed) for KL annealing.

        Returns:
            Dictionary with keys:
                - ``loss``: Scalar total loss.
                - ``loss_ce``: Scalar digamma cross-entropy (detached).
                - ``loss_kl``: Scalar KL term (detached).
                - ``lambda_kl``: Current annealing coefficient (float).
        """
        lambda_kl_current: float = self.config.lambda_kl * min(
            1.0, epoch / max(self.config.anneal_steps, 1)
        )

        alpha_safe: torch.Tensor = alpha.clamp(min=self.config.alpha_clamp_min)

        y: torch.Tensor = F.one_hot(
            target, num_classes=self.config.num_classes
        ).float()
        S: torch.Tensor = alpha_safe.sum(dim=1, keepdim=True)

        # Digamma-based cross-entropy
        loss_ce: torch.Tensor = torch.sum(
            y * (torch.digamma(S) - torch.digamma(alpha_safe)),
            dim=1,
            keepdim=True,
        )

        # KL regularization on modified alpha (remove correct-class evidence)
        alpha_tilde: torch.Tensor = y + (1.0 - y) * alpha_safe
        alpha_tilde = alpha_tilde.clamp(min=self.config.alpha_clamp_min)
        loss_kl: torch.Tensor = self._kl_divergence_dirichlet(
            alpha_tilde, self.config.num_classes
        )

        loss: torch.Tensor = torch.mean(loss_ce + lambda_kl_current * loss_kl)

        return {
            "loss": loss,
            "loss_ce": loss_ce.mean().detach(),
            "loss_kl": loss_kl.mean().detach(),
            "lambda_kl": lambda_kl_current,
        }

    @staticmethod
    def _kl_divergence_dirichlet(
        alpha: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """Compute KL[Dir(p|alpha) || Dir(p|1)].

        Args:
            alpha: Dirichlet parameters of shape ``[N, K]``.
            num_classes: Number of classes K.

        Returns:
            KL divergence of shape ``[N, 1]``.
        """
        ones = torch.ones(
            [1, num_classes], dtype=torch.float32, device=alpha.device
        )
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        return first_term + second_term


# ── Phase 1: Semantic Model ──────────────────────────────────────────────────


class Phase1_SemanticModel(nn.Module):
    """Phase 1 top-level model for semantic evidential training.

    Wraps SemanticEvidentialExtractor. Trained alone on LLM embeddings.
    After training, pre-compute and detach outputs (alpha_sem, u_sem, h_proj)
    for Phase 2 consumption.

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize Phase1_SemanticModel.

        Args:
            config: EVDBotConfig instance.
        """
        super().__init__()
        self.config = config
        self.extractor = SemanticEvidentialExtractor(config)

    def forward(self, h_llm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run semantic evidence extraction.

        Args:
            h_llm: LLM embedding matrix of shape ``[N, D_llm]``.

        Returns:
            Dictionary with keys:
                - ``alpha_sem``: Dirichlet parameters ``[N, K]``.
                - ``u_sem``: Semantic uncertainty ``[N, 1]``.
                - ``h_proj``: Projected features ``[N, d_proj]``.
        """
        return self.extractor(h_llm)

    @torch.no_grad()
    def precompute(self, h_llm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pre-compute and detach all outputs for Phase 2.

        Args:
            h_llm: LLM embedding matrix of shape ``[N, D_llm]``.

        Returns:
            Dictionary with detached CPU tensors:
                - ``alpha_sem``: ``[N, K]``
                - ``u_sem``: ``[N, 1]``
                - ``h_proj``: ``[N, d_proj]``
        """
        self.eval()
        out = self.extractor(h_llm)
        return {k: v.detach().cpu() for k, v in out.items()}


# ── Phase 2: Structural Model ────────────────────────────────────────────────


class Phase2_StructuralModel(nn.Module):
    """Phase 2 top-level model for structural evidential training.

    Consumes frozen Phase 1 outputs (h_proj, u_sem, alpha_sem) and graph
    structure. Trains the VarianceAwareGNN and DSFusionModule.

    All frozen inputs MUST be detached before being passed to forward().
    This class does NOT call .detach() internally to make the contract
    explicit to callers.

    Attributes:
        config: EVDBotConfig instance.
    """

    def __init__(self, config: EVDBotConfig) -> None:
        """Initialize Phase2_StructuralModel.

        Args:
            config: EVDBotConfig instance.
        """
        super().__init__()
        self.config = config
        self.gnn = VarianceAwareGNN(config)
        self.fusion = DSFusionModule(config)

    def forward(
        self,
        h_proj_frozen: torch.Tensor,
        u_sem_frozen: torch.Tensor,
        alpha_sem_frozen: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        seed_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run structural GNN and DS fusion.

        Args:
            h_proj_frozen: Frozen projected features of shape ``[N, d_proj]``.
                Must be detached.
            u_sem_frozen: Frozen semantic uncertainty of shape ``[N, 1]``.
                Must be detached.
            alpha_sem_frozen: Frozen semantic Dirichlet params of shape
                ``[N, K]``. Must be detached.
            edge_index: Edge connectivity of shape ``[2, E]``.
            edge_type: Relation type per edge of shape ``[E]``.
            seed_idx: Indices of target nodes of shape ``[B]``. If ``None``,
                all nodes are treated as targets.

        Returns:
            Dictionary with keys:
                - ``alpha_struct``: Structural Dirichlet params ``[B, K]``.
                - ``u_struct``: Structural uncertainty ``[B, 1]``.
                - ``alpha_final``: Fused Dirichlet params ``[B, K]``.
                - ``u_final``: Final uncertainty ``[B, 1]``.
                - ``routing_mask``: ``[B, 1]``.
                - ``alpha_sem``: Pass-through of frozen semantic params ``[B, K]``.
                - ``u_sem``: Pass-through of frozen semantic uncertainty ``[B, 1]``.
        """
        h_gnn: torch.Tensor = self.gnn(
            h_proj_frozen, edge_index, edge_type, u_sem_frozen
        )

        if seed_idx is not None:
            h_gnn_seed = h_gnn[seed_idx]
            alpha_sem_seed = alpha_sem_frozen[seed_idx]
            u_sem_seed = u_sem_frozen[seed_idx]
        else:
            h_gnn_seed = h_gnn
            alpha_sem_seed = alpha_sem_frozen
            u_sem_seed = u_sem_frozen

        fusion_out: Dict[str, torch.Tensor] = self.fusion(
            alpha_sem_seed, u_sem_seed, h_gnn_seed
        )

        fusion_out["alpha_sem"] = alpha_sem_seed
        fusion_out["u_sem"] = u_sem_seed
        return fusion_out


# ── Utility ───────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Seed all random generators for reproducibility.

    Sets seeds for ``random``, ``numpy``, ``torch``, and configures
    ``torch.backends.cudnn`` for deterministic behaviour.

    Args:
        seed: Integer seed value.
    """
    try:
        from utils import seed_setting  # type: ignore[import-untyped]

        seed_setting(seed)
        logger.info("Seeded via utils.seed_setting(%d)", seed)
    except ImportError:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        logger.info("Seeded locally (%d)", seed)
'''
