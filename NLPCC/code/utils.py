"""Shared parsing helpers for the NLPCC submission package."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "parse_seed_list",
]


def parse_seed_list(raw_value: str | int | Iterable[int] | None) -> list[int]:
    """Parse CLI seed input into a non-empty integer list."""

    if raw_value is None:
        return [1]
    if isinstance(raw_value, int):
        return [raw_value]
    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
        if not tokens:
            return [1]
        return [int(token) for token in tokens]
    return [int(seed) for seed in raw_value]
