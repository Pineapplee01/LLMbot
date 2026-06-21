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

- [ ] Verify importing `runtime_env` is lightweight enough for queue scripts.
- [ ] Add `clean_process_env`, `configure_model_cache_env`, and `build_offline_model_env`.
- [ ] Keep device helpers compatible by importing `torch` only inside those helpers.
- [ ] Validate with an inline Python check that imports the module and checks returned cache paths.

## Task 2: Queue Script Deduplication

- [ ] Update sampled TwiBot-22 launcher to call `build_offline_model_env`.
- [ ] Update sampled TwiBot-22 5-seed queue to call `build_offline_model_env`.
- [ ] Update TwiBot-20 full selective residual queue to call `build_offline_model_env`, then reapply its CUDA/W&B overrides.
- [ ] Validate script import/compile with `python -m py_compile`.

## Task 3: Experiment Documentation

- [ ] Add README note that local model downloads/caches live under `G:\Research\BotDetection\models`.
- [ ] Add experiments registry note that queue scripts should use stable experiment IDs, manifest paths, and `runtime_env.build_offline_model_env`.
- [ ] Validate docs contain the cache path and helper name.

## Task 4: Final Checks

- [ ] Run `python -m py_compile` for changed Python files.
- [ ] Run inline helper behavior check.
- [ ] Run `python main.py --help` if dependencies are available; otherwise capture the exact missing dependency.
- [ ] Review `git diff --stat` and `git status --short --branch`.
