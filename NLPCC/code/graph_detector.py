"""NLPCC graph-detector task boundary."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from models import GraphDetectorConfig, ModelConfig, ReliabilityRoutingConfig
from router import describe_reliability_strategy
from utils import parse_seed_list

__all__ = [
    "build_graph_detector_config",
    "describe_graph_detector_task",
    "run_graph_detector_task",
]


def build_graph_detector_config(args: Namespace) -> GraphDetectorConfig:
    """Build the graph-detector configuration from canonical CLI args."""

    model_config = ModelConfig(
        graph_backbone=args.graph_backbone,
        text_encoder=args.text_encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    routing_config = ReliabilityRoutingConfig(
        reliability_strategy=args.reliability_strategy,
        routing_budget=args.routing_budget,
        support_k=args.support_k,
    )
    return GraphDetectorConfig(
        dataset=args.dataset,
        embedding_path=args.embedding_path,
        seeds=tuple(parse_seed_list(args.seeds)),
        model=model_config,
        routing=routing_config,
    )


def describe_graph_detector_task(args: Namespace) -> dict[str, Any]:
    """Describe the selected graph-detector task without training side effects."""

    config = build_graph_detector_config(args)
    return {
        "experiment_task": "graph_detector_prepare",
        "dataset": config.dataset,
        "graph_backbone": config.model.graph_backbone,
        "text_encoder": config.model.text_encoder,
        "embedding_path": config.embedding_path,
        "reliability_strategy": config.routing.reliability_strategy,
        "routing_budget": config.routing.routing_budget,
        "support_k": config.routing.support_k,
        "routing_strategy_metadata": describe_reliability_strategy(
            config.routing.reliability_strategy
        ),
        "seeds": list(config.seeds),
    }


def run_graph_detector_task(args: Namespace) -> dict[str, Any]:
    """Dispatch supported NLPCC graph-detector tasks."""

    if args.experiment_task == "graph_detector_prepare":
        return describe_graph_detector_task(args)
    raise ValueError(f"Unsupported NLPCC experiment_task: {args.experiment_task}")
