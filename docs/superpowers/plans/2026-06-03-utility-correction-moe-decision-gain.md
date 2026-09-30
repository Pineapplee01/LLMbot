# Utility Correction MoE Decision-Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the fixed-router `utility_correction_moe` path diagnosable and trainable against true correction benefit, not only loss-margin improvement.

**Architecture:** Extend the existing `LLMbot/trainer_glance.py` correction-MoE path in place. Add a CLI-controlled utility target that can preserve the old loss-advantage behavior or train utility heads on discrete decision gain (`base_wrong & expert_pred_correct`), and persist raw per-expert logits/loss/advantage tensors for offline threshold sweeps.

**Tech Stack:** Python, PyTorch, existing `LLMbot` strict `joint_router_refinement` pipeline.

---

### Task 1: Add CLI Surface

**Files:**
- Modify: `LLMbot/parser_args.py`
- Modify docs if CLI changes: `LLMbot/README.md`, `docs/code/parser.md`, `docs/code/research.md`, `docs/ARCHITECTURE.md`

- [ ] Add `--joint_correction_moe_utility_target {loss_advantage,decision_gain,hybrid}` with default `loss_advantage`.
- [ ] Document that `loss_advantage` preserves current behavior, `decision_gain` uses `base_wrong & expert_pred_correct`, and `hybrid` accepts either positive loss advantage or decision gain.

### Task 2: Train Utility Heads With Selectable Target

**Files:**
- Modify: `LLMbot/trainer_glance.py`

- [ ] Thread `joint_correction_moe_utility_target` into the strict joint refiner training call.
- [ ] In the `utility_correction_moe` loss block, keep computing `expert_loss_matrix`.
- [ ] Compute per-expert `loss_advantage_targets = base_loss - expert_loss - beta > 0`.
- [ ] Compute `decision_gain_targets = base_wrong & expert_pred_correct`.
- [ ] Select utility targets from the new CLI mode.
- [ ] Keep existing sample weighting and availability masks unchanged.

### Task 3: Save Per-Expert Diagnostics

**Files:**
- Modify: `LLMbot/trainer_glance.py`

- [ ] During split policy application, collect per-node `expert_logits`, `utility_logits`, `expert_pred`, `expert_loss`, and `expert_advantage` for eligible prompt-expert rows.
- [ ] Persist these tensors in `outputs.pt` under stable names for `utility_correction_moe`.
- [ ] Add compact scalar metadata to `metrics.json` / `manifest.json`: utility target mode and diagnostic tensor names.
- [ ] Add per-node test fields for selected expert target diagnostics without bloating rows with all logits.

### Task 4: Validate

**Files:**
- No new test files.

- [ ] Run `python -m py_compile LLMbot/trainer_glance.py LLMbot/parser_args.py LLMbot/main.py`.
- [ ] Run `python LLMbot/main.py --help` and verify the new flag appears.
- [ ] Do not run claim-grade experiments in the implementation step; hand off stable commands for server validation.

### Task 5: Documentation Sync

**Files:**
- Modify: `LLMbot/README.md`
- Modify: `docs/code/parser.md`
- Modify: `docs/code/research.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] Explain the new target modes and why `decision_gain` is distinct from loss advantage.
- [ ] Record that this is a fixed-router routed-node correction diagnostic, not a new backbone/router protocol.
- [ ] State that saved raw tensors enable offline beta/threshold sweeps without retraining.
