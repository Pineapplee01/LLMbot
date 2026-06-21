# LLMbot Experiment Runtime Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first slice of LLMbot experiment code cleaner and more compatible by centralizing local model-cache environment handling and documenting the experiment run boundary.

**Architecture:** Keep training behavior unchanged. Put shared process-environment helpers in `runtime_env.py` because it is already imported by the active mainline, but make it lightweight by delaying the PyTorch import until device helpers are called. Update only existing queue scripts that already set the same HuggingFace offline cache variables.

**Tech Stack:** Python standard library, existing LLMbot modules, PowerShell smoke checks, Git.

---

## File Structure

- Modify `runtime_env.py`: add cache/process environment helpers and delay `torch` import.
- Modify `launch_sampled_twibot22_base5_20260620.py`: reuse the shared offline model-cache helper.
- Modify `run_sampled_twibot22_base_5seed_20260620.py`: reuse the shared offline model-cache helper and keep manifest fields stable.
- Modify `run_twibot20_full_selective_residual_5seed_20260620.py`: reuse the shared helper while preserving CUDA/W&B overrides.
- Modify `README.md`: add a short experiment-runtime compatibility note.
- Modify `experiments.md`: add an operating note for local model-cache and queue-script boundaries.

Out of scope:

- No training algorithm changes.
- No experiment launch.
- No edits to generated `experiments/`, logs, checkpoints, models, or datasets.
- No new test files.

## Task 1: Runtime Environment Helper

- [x] Verify importing `runtime_env` is lightweight enough for queue scripts.
- [x] Add `clean_process_env`, `configure_model_cache_env`, and `build_offline_model_env`.
- [x] Keep device helpers compatible by importing `torch` only inside those helpers.
- [x] Validate with an inline Python check that imports the module and checks returned cache paths.

## Task 2: Queue Script Deduplication

- [x] Update sampled TwiBot-22 launcher to call `build_offline_model_env`.
- [x] Update sampled TwiBot-22 5-seed queue to call `build_offline_model_env`.
- [x] Update TwiBot-20 full selective residual queue to call `build_offline_model_env`, then reapply its CUDA/W&B overrides.
- [x] Validate script import/compile with `python -m py_compile`.

## Task 3: Experiment Documentation

- [x] Add README note that local model downloads/caches live under `G:\Research\BotDetection\models`.
- [x] Add experiments registry note that queue scripts should use stable experiment IDs, manifest paths, and `runtime_env.build_offline_model_env`.
- [x] Validate docs contain the cache path and helper name.

## Task 4: Final Checks

- [x] Run `python -m py_compile` for changed Python files.
- [x] Run inline helper behavior check.
- [x] Run `python main.py --help` if dependencies are available; otherwise capture the exact missing dependency.
- [x] Review `git diff --stat` and `git status --short --branch`.

## Task 5: Queue Logging Helper Extraction

- [x] Add lightweight timestamp, JSON-write, and logged subprocess helpers to `runtime_env.py`.
- [x] Update direct queue scripts with repeated manifest/logging wrappers to call those helpers without changing command arguments or manifest fields.
- [x] Keep scripts with custom process control out of scope unless a later slice can preserve their behavior exactly.
- [x] Tighten `experiments.md` status and current-queue rules so future queue entries stay indexable.
- [x] Validate with py_compile, an inline helper smoke check, `python main.py --help`, and `git diff --check`.

## Task 6: CSV Helper Extraction

- [x] Add `runtime_env.write_csv_rows_file(...)` for schema-driven CSV writes.
- [x] Update formal snapshot CSV writes to reuse the shared helper while preserving the old empty-without-fields behavior.
- [x] Update full residual by-seed summary CSV writes to reuse the shared helper while preserving the fixed schema and `\n` line terminator.
- [x] Extend `experiment_runbook.md` smoke checks to cover the shared helper and both script wrappers.

## Task 7: Consolidated Dry Validation

- [x] Add `check_experiment_helpers_20260620.py` as the single dry-validation entry point for root experiment helper cleanup.
- [x] Keep validation non-generating: compile outputs and smoke artifacts write only to temporary directories, with bytecode writes disabled for imported helpers.
- [x] Replace the long inline `experiment_runbook.md` validation block with the consolidated script, `python main.py --help`, and `git diff --check`.
- [x] Cover PowerShell resolution, launcher fail-fast behavior, import side effects, shared JSON readers, routed-mask helpers, CSV schema wrappers, and queue-manifest lifecycle helpers.

## Completion Notes

- Current nested `LLMbot` branch: `codex-llmbot-experiment-refactor`.
- Last verified nested commit at this checkpoint: `8ee4c6b`.
- Non-generating validation used only compile/import/helper smokes and `python main.py --help`.
- No queue, training, report-refresh, or evidence-generation command was run for this refactor plan.
- Parent BotDetection gitlink remains stale because the parent worktree Git metadata directory rejects writes for the current sandbox user.
