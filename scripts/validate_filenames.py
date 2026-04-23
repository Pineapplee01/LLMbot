#!/usr/bin/env python
"""Validate filenames for deterministic, auditable research work.

Intended use:
  Run before closing a change to catch vague names that make research artifacts
  hard to audit later.

Assumptions:
  This script checks file names, not file contents. Existing external-interface
  names may be whitelisted explicitly.

How to extend safely:
  Add narrow allowlist entries for stable external names. Do not remove vague
  tokens globally just because one path needs an exception.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ALLOWED_BASENAMES = {
    "AGENTS.md",
    "README.md",
    "SKILL.md",
    "PLANS.md",
    "FINAL_PROPOSAL.md",
    "__init__.py",
    "manifest.json",
    "metrics.json",
    "summary.json",
    "config.json",
    "config.yaml",
    "config.yml",
    "run_request.json",
}

ALLOWED_STEMS_BY_SUFFIX = {
    ".py": {"conftest"},
    ".ts": set(),
    ".tsx": set(),
    ".js": set(),
    ".jsx": set(),
    ".json": {"manifest", "metrics", "summary", "config", "run_request"},
    ".yaml": {"config"},
    ".yml": {"config"},
    ".toml": {"pyproject"},
}

VAGUE_TOKENS = {
    "temp",
    "tmp",
    "new",
    "final",
    "copy",
    "misc",
    "helper",
    "helpers",
    "test2",
}

VAGUE_VERSION_RE = re.compile(r"^v\d+$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")

IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".cache",
    ".claude",
}


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out).resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def normalize(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root)
    except Exception:
        rel = path
    return str(rel).replace(os.sep, "/").lstrip("./")


def is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORED_PARTS:
        return True
    text = str(path).replace(os.sep, "/")
    return "/.claude/worktrees/" in text or "/wandb/" in text


def expand_paths(paths: list[str], root: Path) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not is_ignored(child):
                    expanded.append(child)
        elif path.is_file() and not is_ignored(path):
            expanded.append(path)
    return sorted(set(expanded))


def git_changed_files(root: Path, base_ref: str, staged: bool) -> list[Path]:
    cmd = ["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACMR"]
    if staged:
        cmd.append("--cached")
    else:
        cmd.append(base_ref)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        raise SystemExit(
            "FAIL: could not read changed files from git; pass explicit --paths instead."
        )
    return [root / line.strip() for line in out.splitlines() if line.strip()]


def tokens_for(path: Path) -> list[str]:
    stem = path.stem
    if path.suffix in {".ts", ".tsx", ".js", ".jsx"} and stem.endswith(".test"):
        stem = stem[: -len(".test")]
    return [tok for tok in TOKEN_RE.split(stem) if tok]


def is_allowlisted(path: Path) -> bool:
    if path.name in ALLOWED_BASENAMES:
        return True
    suffix = path.suffix.lower()
    return path.stem in ALLOWED_STEMS_BY_SUFFIX.get(suffix, set())


def bad_tokens(path: Path) -> list[str]:
    if is_allowlisted(path):
        return []
    bad = []
    for token in tokens_for(path):
        lowered = token.lower()
        if lowered in VAGUE_TOKENS or VAGUE_VERSION_RE.match(lowered):
            bad.append(token)
    return bad


def suggestion(path: Path, bad: list[str]) -> str:
    suffix = path.suffix or "<ext>"
    if path.name.startswith("test") or ".test" in path.name:
        return f"use a descriptive test name like test_manifest_validation{suffix}"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "use a purpose-specific config/artifact name such as experiment_manifest.json"
    return "rename with method, dataset, stage, or purpose instead of vague tokens"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--paths", nargs="*")
    args = parser.parse_args()

    root = repo_root()
    if args.paths:
        files = expand_paths(args.paths, root)
    else:
        files = [p for p in git_changed_files(root, args.base_ref, args.staged) if p.exists()]

    failures: list[str] = []
    for path in files:
        if is_ignored(path):
            continue
        bad = bad_tokens(path)
        if bad:
            failures.append(
                f"{normalize(path, root)}: ambiguous token(s) {', '.join(bad)} -> {suggestion(path, bad)}"
            )

    if failures:
        print("FAIL: ambiguous filenames detected")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"PASS: filename validation checked {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
