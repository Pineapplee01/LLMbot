#!/usr/bin/env bash
# Intended use:
#   One command for repo-local research governance checks.
#
# Assumptions:
#   The gate is deterministic and dependency-free. It validates changed paths and
#   manifests; it does not run experiments or judge scientific merit.
#
# How to extend safely:
#   Add new checks after filename and scope validation. Keep hard failures early
#   and print the final PASS summary only after every check succeeds.
set -euo pipefail

TASK_TYPE=""
BASE_REF="HEAD"
STAGED=0
ALLOW_MISSING_MANIFEST_PATHS=0
MANIFESTS=()
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
    --manifest)
      MANIFESTS+=("${2:-}")
      shift 2
      ;;
    --allow-missing-manifest-paths)
      ALLOW_MISSING_MANIFEST_PATHS=1
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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
else
  echo "FAIL: no Python interpreter found; set PYTHON=/path/to/python" >&2
  exit 2
fi

COMMON_ARGS=(--base-ref "$BASE_REF")
if [[ "$STAGED" -eq 1 ]]; then
  COMMON_ARGS+=(--staged)
fi

echo "CHECK: filename validation"
if [[ ${#PATH_ARGS[@]} -gt 0 ]]; then
  "$PYTHON_BIN" scripts/validate_filenames.py --paths "${PATH_ARGS[@]}"
else
  "$PYTHON_BIN" scripts/validate_filenames.py "${COMMON_ARGS[@]}"
fi

echo "CHECK: changed-scope validation"
if [[ ${#PATH_ARGS[@]} -gt 0 ]]; then
  bash scripts/check_changed_scope.sh --task-type "$TASK_TYPE" --paths "${PATH_ARGS[@]}"
else
  bash scripts/check_changed_scope.sh --task-type "$TASK_TYPE" "${COMMON_ARGS[@]}"
fi

collect_manifest_paths() {
  if [[ ${#PATH_ARGS[@]} -gt 0 ]]; then
    local raw
    for raw in "${PATH_ARGS[@]}"; do
      if [[ -d "$raw" ]]; then
        find "$raw" -type f -iname '*manifest*.json' 2>/dev/null || true
      elif [[ -f "$raw" && "$raw" == *manifest*.json ]]; then
        echo "$raw"
      fi
    done
  else
    if [[ "$STAGED" -eq 1 ]]; then
      git diff --name-only --diff-filter=ACMR --cached | grep -Ei 'manifest.*\.json$' || true
    else
      git diff --name-only --diff-filter=ACMR "$BASE_REF" | grep -Ei 'manifest.*\.json$' || true
    fi
  fi
}

if [[ ${#MANIFESTS[@]} -eq 0 ]]; then
  mapfile -t MANIFESTS < <(collect_manifest_paths | sort -u)
fi

echo "CHECK: experiment manifest validation"
if [[ ${#MANIFESTS[@]} -gt 0 ]]; then
  MANIFEST_ARGS=()
  if [[ "$ALLOW_MISSING_MANIFEST_PATHS" -eq 1 ]]; then
    MANIFEST_ARGS+=(--allow-missing-paths)
  fi
  "$PYTHON_BIN" scripts/verify_experiment_manifest.py "${MANIFESTS[@]}" "${MANIFEST_ARGS[@]}"
else
  echo "PASS: no experiment manifests to validate"
fi

echo "PASS: research gate completed for task type '$TASK_TYPE'"
