"""Model configuration objects for the NLPCC graph-detector line."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GraphDetectorConfig",
    "ModelConfig",
    "ReliabilityRoutingConfig",
]


@dataclass(frozen=True)
class ModelConfig:
    """Common neural model configuration used by the submission runner."""

    graph_backbone: str = "rgcn"
    text_encoder: str = "roberta-base"
    hidden_dim: int = 128
    dropout: float = 0.3


@dataclass(frozen=True)
class ReliabilityRoutingConfig:
    """Minimal declarative routing contract for the NLPCC submission package."""

    reliability_strategy: str = "weighted_reference_tail"
    routing_budget: float = 0.10
    support_k: int = 8


@dataclass(frozen=True)
class GraphDetectorConfig:
    """Configuration for graph-detector preparation and execution."""

    dataset: str = "TwiBot-20"
    embedding_path: str | None = None
    seeds: tuple[int, ...] = (1,)
    model: ModelConfig = ModelConfig()
    routing: ReliabilityRoutingConfig = ReliabilityRoutingConfig()
