"""Entrypoint for the trimmed NLPCC graph-detector package."""

from __future__ import annotations

import json
from collections.abc import Sequence

from graph_detector import run_graph_detector_task
from parser_args import parser_args

__all__ = [
    "main",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected NLPCC experiment task."""

    args = parser_args(list(argv) if argv is not None else None)
    result = run_graph_detector_task(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
