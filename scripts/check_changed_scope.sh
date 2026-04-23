#!/usr/bin/env bash
# Intended use:
#   Fail fast when changed files do not match the active task mode.
#
# Assumptions:
#   This script validates paths, not semantic intent. Evidence/generated zones
#   stay restricted even for experiment work unless a future policy explicitly
#   allows command-produced artifacts.
#
# How to extend safely:
#   Add narrow path globs to the relevant task mode. Do not widen restricted
#   evidence zones from inside a mode-specific allowlist.
set -euo pipefail

TASK_TYPE=""
BASE_REF="HEAD"
STAGED=0
PATH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-type)
      TASK_TYPE="${2:-}"
      shift 2
      ;;
    --base-ref)
      BASE_REF="${2:-HEAD}"
      shift 2
      ;;
    --staged)
      STAGED=1
      shift
      ;;
    --paths)
      shift
      while [[ $# -gt 0 ]]; do
        PATH_ARGS+=("$1")
        shift
      done
      ;;
    *)
      echo "FAIL: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_TYPE" ]]; then
  echo "FAIL: --task-type is required" >&2
  exit 2
fi

case "$TASK_TYPE" in
  governance|implementation|testing|experiment|analysis|review) ;;
  *)
    echo "FAIL: unsupported task type: $TASK_TYPE" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

normalize_path() {
  local p="$1"
  p="${p#./}"
  p="${p//\\//}"
  echo "$p"
}

collect_paths() {
  if [[ ${#PATH_ARGS[@]} -gt 0 ]]; then
    local raw
    for raw in "${PATH_ARGS[@]}"; do
      raw="$(normalize_path "$raw")"
      if [[ -d "$raw" ]]; then
        find "$raw" -type f 2>/dev/null || true
      elif [[ -f "$raw" ]]; then
        echo "$raw"
      else
        echo "$raw"
      fi
    done
  else
    if [[ "$STAGED" -eq 1 ]]; then
      git diff --name-only --diff-filter=ACMR --cached
    else
      git diff --name-only --diff-filter=ACMR "$BASE_REF"
    fi
  fi
}

is_restricted() {
  local p="$1"
  case "$p" in
    datasets/*|results/*|docs/published/*|.claude/worktrees/*) return 0 ;;
    LLMbot/saved_artifacts/*|LLMbot/checkpoints/*|LLMbot/code/wandb/*) return 0 ;;
    LMBot/results_*|LMBot/results_*/*|LMBot/datasets/*) return 0 ;;
    */__pycache__/*|__pycache__/*|*.pyc|*.pyo) return 0 ;;
    */.pytest_cache/*|.pytest_cache/*|*/.cache/*|.cache/*) return 0 ;;
    */tmp/*|tmp/*|*/wandb/*|wandb/*|logs/*|*.log) return 0 ;;
    LMBot/tmp*|LMBot/tmp*/*) return 0 ;;
  esac
  return 1
}

is_allowed() {
  local p="$1"
  case "$TASK_TYPE" in
    governance)
      case "$p" in
        .gitignore|AGENTS.md|README.md|.agents/*|docs/research/*|scripts/*|.github/*) return 0 ;;
      esac
      ;;
    implementation)
      case "$p" in
        .gitignore|AGENTS.md|README.md|RESEARCH_BRIEF.md|IDEA_CANDIDATES.md|IDEA_REPORT.md|lmbot.yaml) return 0 ;;
        LLMbot/*|LMBot/*|rewrite_pipeline/*|docs/*|scripts/*|.agents/*|tools/*) return 0 ;;
      esac
      ;;
    testing)
      case "$p" in
        scripts/*|.agents/*|docs/research/*|docs/guides/*|docs/protocols/*) return 0 ;;
        LLMbot/**/test/*|LLMbot/**/tests/*|LMBot/tests/*|LMBot/tests/**/*) return 0 ;;
        rewrite_pipeline/tests/*|rewrite_pipeline/tests/**/*) return 0 ;;
      esac
      ;;
    experiment)
      case "$p" in
        docs/research/*|docs/proposals/*|docs/wiki/experiments/*|refine-logs/*) return 0 ;;
        scripts/*|tools/*|.agents/*) return 0 ;;
        LLMbot/**/configs/*|LLMbot/**/experiments/*|LLMbot/**/scripts/*) return 0 ;;
        LLMbot/baseline/**/configs/*|LMBot/protocol/*|LMBot/docs/runbooks/*) return 0 ;;
        *manifest*.json|*config*.json|*config*.yaml|*config*.yml) return 0 ;;
      esac
      ;;
    analysis)
      case "$p" in
        AGENTS.md|README.md|RESEARCH_BRIEF.md|IDEA_CANDIDATES.md|IDEA_REPORT.md) return 0 ;;
        docs/*|scripts/*|.agents/*|tools/*|refine-logs/*.md|refine-logs/*.csv) return 0 ;;
        LLMbot/**/analysis/*|LMBot/lmbot/experimental/*) return 0 ;;
      esac
      ;;
    review)
      case "$p" in
        docs/reviews/*|docs/research/*|docs/wiki/*|docs/wiki/**/*|.agents/*|AGENTS.md|README.md) return 0 ;;
      esac
      ;;
  esac
  return 1
}

mapfile -t CHANGED < <(collect_paths | sed 's#^\./##' | sort -u)

FAILURES=()
for p in "${CHANGED[@]}"; do
  [[ -z "$p" ]] && continue
  p="$(normalize_path "$p")"
  if is_restricted "$p"; then
    FAILURES+=("$p: restricted evidence/generated zone")
  elif ! is_allowed "$p"; then
    FAILURES+=("$p: outside allowed zone for task type '$TASK_TYPE'")
  fi
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
  echo "FAIL: changed-scope validation failed"
  for failure in "${FAILURES[@]}"; do
    echo "- $failure"
  done
  exit 1
fi

echo "PASS: changed-scope validation checked ${#CHANGED[@]} path(s) for task type '$TASK_TYPE'"
