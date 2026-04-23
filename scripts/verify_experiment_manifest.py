#!/usr/bin/env python
"""Validate flat experiment manifests for research reproducibility.

Intended use:
  Run before interpreting metrics or using a run as evidence for a claim.

Assumptions:
  New claim-bearing manifests follow docs/research/experiment_manifest_schema.md.
  Legacy manifests may exist, but they are not valid for new claim completion
  unless converted to the flat schema.

How to extend safely:
  Add fields only after updating the schema doc and result checklist. Keep error
  messages actionable so experiment owners know exactly what to fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_STRING_FIELDS = [
    "experiment_id",
    "commit_hash",
    "task_name",
    "dataset_name",
    "dataset_version",
    "preprocessing_version",
    "model_name",
    "model_version_or_date",
    "prompt_or_template_version",
    "config_path",
    "command",
    "hardware_or_runtime_notes",
    "metrics_output_path",
    "artifact_output_path",
    "baseline_reference",
    "notes",
]

PATH_FIELDS = [
    "config_path",
    "metrics_output_path",
    "artifact_output_path",
]


def repo_root() -> Path:
    return Path.cwd().resolve()


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def resolve_path(raw: str, manifest_path: Path, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def error(field: str, problem: str, hint: str) -> str:
    return f"{field}: {problem} -> {hint}"


def validate_manifest(data: dict[str, Any], manifest_path: Path, allow_missing_paths: bool) -> list[str]:
    errors: list[str] = []
    root = repo_root()

    for field in REQUIRED_STRING_FIELDS:
        if not is_non_empty_string(data.get(field)):
            errors.append(error(field, "missing or empty", "add a non-empty string value"))

    split_id = data.get("split_id")
    split_description = data.get("split_description")
    if not is_non_empty_string(split_id) and not is_non_empty_string(split_description):
        errors.append(
            error(
                "split_id/split_description",
                "both missing or empty",
                "record a stable split_id or a clear split_description",
            )
        )

    seed_list = data.get("seed_list")
    if not isinstance(seed_list, list) or not seed_list:
        errors.append(error("seed_list", "missing, empty, or not an array", "use a non-empty array of integers"))
    elif not all(isinstance(seed, int) for seed in seed_list):
        errors.append(error("seed_list", "contains non-integer values", "use integers only, for example [1, 2, 3]"))

    if not allow_missing_paths:
        for field in PATH_FIELDS:
            raw = data.get(field)
            if is_non_empty_string(raw):
                resolved = resolve_path(raw, manifest_path, root)
                if not resolved.exists():
                    errors.append(error(field, f"path does not exist: {raw}", "create the file/dir or fix the path"))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--allow-missing-paths", action="store_true")
    args = parser.parse_args()

    all_errors: list[str] = []
    for raw_manifest in args.manifest:
        manifest_path = Path(raw_manifest)
        if not manifest_path.exists():
            all_errors.append(f"{raw_manifest}: manifest file does not exist -> pass an existing JSON manifest")
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            all_errors.append(f"{raw_manifest}: invalid JSON: {exc} -> fix JSON syntax")
            continue
        if not isinstance(data, dict):
            all_errors.append(f"{raw_manifest}: manifest must be a JSON object -> use the flat schema object")
            continue
        for item in validate_manifest(data, manifest_path.resolve(), args.allow_missing_paths):
            all_errors.append(f"{raw_manifest}: {item}")

    if all_errors:
        print("FAIL: experiment manifest validation failed")
        for item in all_errors:
            print(f"- {item}")
        return 1

    print(f"PASS: validated {len(args.manifest)} experiment manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
