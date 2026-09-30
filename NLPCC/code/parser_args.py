"""Canonical CLI contract for the NLPCC submission package."""

from __future__ import annotations

import argparse

from router import RELIABILITY_STRATEGY_CHOICES
from utils import parse_seed_list

__all__ = [
    "NLPCC_EXPERIMENT_TASKS",
    "build_parser",
    "parser_args",
]

NLPCC_EXPERIMENT_TASKS = (
    "graph_detector_prepare",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal NLPCC argument parser."""

    parser = argparse.ArgumentParser(
        description="NLPCC graph detector submission runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment_task",
        choices=NLPCC_EXPERIMENT_TASKS,
        default="graph_detector_prepare",
        help="Canonical NLPCC experiment task.",
    )
    parser.add_argument("--dataset", default="TwiBot-20", help="Dataset name.")
    parser.add_argument("--seeds", default="1", help="Comma-separated random seeds.")
    parser.add_argument("--graph_backbone", default="rgcn", help="Graph detector backbone.")
    parser.add_argument("--text_encoder", default="roberta-base", help="Text encoder name.")
    parser.add_argument("--embedding_path", default=None, help="Optional semantic embedding path.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument(
        "--reliability_strategy",
        choices=RELIABILITY_STRATEGY_CHOICES,
        default="weighted_reference_tail",
        help="Reliability strategy metadata recorded for this dry-run configuration.",
    )
    parser.add_argument(
        "--routing_budget",
        type=float,
        default=0.10,
        help="User-controlled routing intervention budget expressed as a fraction.",
    )
    parser.add_argument(
        "--support_k",
        type=int,
        default=8,
        help="Support-set size metadata recorded for the selected reliability strategy.",
    )
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    return parser


def parser_args(raw_args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args and normalize canonical fields."""

    args = build_parser().parse_args(raw_args)
    args.seeds = parse_seed_list(args.seeds)
    return args
