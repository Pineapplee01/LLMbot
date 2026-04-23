---
type: project
node_id: project:baseline_migration_comparison_context_2026-04-17
title: Baseline Migration and Comparison Context
created_at: 2026-04-17T00:00:00Z
updated_at: 2026-04-23T00:00:00Z
tags: [baseline, migration, comparison, context, lmbot]
---

# Baseline Migration and Comparison Context

## Purpose

This note is the short truth card for the 2026-04-17 baseline migration state. It exists so future sessions can recover the current layout without rereading the entire wiki history.

## Current Layout

- **Active method code**: `LLMbot/`
- **Migrated LMbot baseline**: `LLMbot/baseline/`
- **Authoritative docs root**: `docs/`

Within the migrated baseline:

- `LLMbot/baseline/core/` is the main LMbot WSDM 2024 training pipeline
- `LLMbot/baseline/analysis/` contains diagnostic and failure-regime analysis utilities
- `LLMbot/baseline/baselines/` contains additional comparison baselines

## Code Provenance

- `LLMbot/baseline/core` is mostly aligned with `docs/published/lmbot-wsdm2024/src`
- `GNNs.py` and `parser_args.py` differ from the published snapshot
- Treat `LLMbot/baseline/core` as a **runnable migrated baseline**, not as a fully untouched release mirror

## Comparison Guidance

Use these as the default comparison entrypoints:

- **Current mainline method**: `LLMbot/main.py`
- **LMbot baseline**: `LLMbot/baseline/core/main.py`
- **Comparison protocol**: `docs/protocols/baseline_comparability.md`

Important distinction:

- `LLMbot/code` rw1/rw4 results are the validated unified-protocol reference for the active method line
- `LLMbot/baseline/core` default behavior still needs explicit comparison controls in formal experiments

Known comparison risks in `LLMbot/baseline/core`:

- split reset behavior must be controlled explicitly
- Macro-F1 evaluation should be enforced explicitly rather than assumed from the default training code

## Baseline Status Card

### Reproduced

- LMbot RGCN GNN baseline: Macro-F1 `0.8723`
- LMbot LM co-trained baseline: Macro-F1 `0.8771`
- LMbot RGT single-pass GNN baseline: Macro-F1 `0.8619`

### Reported or reproduction-in-progress

- BotBR: about `86.8%`
- HyperScan: `87.2%`

## Use This Note When

Use this note when a future session needs to answer any of the following quickly:

- where the migrated baseline lives
- which code path is the active method line
- which docs location is authoritative
- whether the migrated baseline equals the published snapshot exactly
- which baseline numbers are already reproduced

## Historical Scope Update (2026-04-23)

- This 2026-04-17 card remains the historical record for the baseline-migration layout change.
- Current session routing now lives in `docs/wiki/project/mainline_switch_context_2026-04-23.md` and points new sessions to `LLMbot/baseline/`.
- Keep the `LLMbot/code` distinction here as historical comparison context, not as the current default route.
