"""Reliability routing metadata and budget-selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

__all__ = [
    "RELIABILITY_STRATEGY_CHOICES",
    "ReliabilityStrategySpec",
    "describe_reliability_strategy",
    "rank_risk_scores",
    "resolve_reliability_strategy",
    "select_top_budget",
]


@dataclass(frozen=True)
class ReliabilityStrategySpec:
    """Declarative metadata for a named reliability-scoring strategy."""

    name: str
    role: str
    reviewer_facing: bool
    description: str
    general_case_strategy: str
    paper_method_strategy: str


_RELIABILITY_STRATEGY_REGISTRY: dict[str, ReliabilityStrategySpec] = {
    "weighted_reference_tail": ReliabilityStrategySpec(
        name="weighted_reference_tail",
        role="paper_method",
        reviewer_facing=True,
        description=(
            "Weighted reference-tail reliability strategy used as the reviewer-facing paper method."
        ),
        general_case_strategy="ncp_local",
        paper_method_strategy="weighted_reference_tail",
    ),
    "ncp_local": ReliabilityStrategySpec(
        name="ncp_local",
        role="general_case",
        reviewer_facing=False,
        description=(
            "Broader local-risk strategy family used as an internal generalization taxonomy."
        ),
        general_case_strategy="ncp_local",
        paper_method_strategy="weighted_reference_tail",
    ),
}

RELIABILITY_STRATEGY_CHOICES = tuple(_RELIABILITY_STRATEGY_REGISTRY)


def resolve_reliability_strategy(name: str) -> ReliabilityStrategySpec:
    """Return the normalized metadata for a named reliability strategy."""

    try:
        return _RELIABILITY_STRATEGY_REGISTRY[name]
    except KeyError as error:
        valid = ", ".join(RELIABILITY_STRATEGY_CHOICES)
        raise ValueError(
            f"Unsupported reliability strategy '{name}'. Expected one of: {valid}"
        ) from error


def describe_reliability_strategy(name: str) -> dict[str, object]:
    """Expose a JSON-serializable view of the normalized reliability strategy metadata."""

    spec = resolve_reliability_strategy(name)
    return asdict(spec)


def rank_risk_scores(risk_scores: Mapping[int, float]) -> list[tuple[int, float]]:
    """Sort node risk scores from highest to lowest risk."""

    return sorted(risk_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))


def select_top_budget(risk_scores: Mapping[int, float] | Sequence[float], budget: int) -> list[int]:
    """Return node ids selected by descending risk under a fixed budget."""

    if budget <= 0:
        return []
    if isinstance(risk_scores, Mapping):
        ranked = rank_risk_scores(risk_scores)
    else:
        ranked = rank_risk_scores({index: score for index, score in enumerate(risk_scores)})
    return [node_id for node_id, _score in ranked[:budget]]
