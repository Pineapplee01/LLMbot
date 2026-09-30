"""Data and artifact I/O for the NLPCC submission package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "read_json",
    "resolve_path",
    "write_json",
]


def resolve_path(path: str | Path) -> Path:
    """Expand a user-provided filesystem path."""

    return Path(path).expanduser().resolve()


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""

    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a UTF-8 JSON file and return the resolved output path."""

    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
