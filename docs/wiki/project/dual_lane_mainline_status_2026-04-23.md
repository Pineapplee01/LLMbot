---
type: project
node_id: project:dual_lane_mainline_status_2026-04-23
title: Dual-Lane Mainline Status
created_at: 2026-04-23T00:00:00Z
updated_at: 2026-05-23T00:00:00Z
tags: [historical, baseline, mainline, dual-lane, status]
---

# Historical Dual-Lane Mainline Status

## Status

Historical note only.

This document described the older `LLMbot/baseline/core` dual-lane mainline
state on 2026-04-23. It is no longer the active routing target for current
implementation work.

## Current Interpretation Rule

- `LLMbot/` is now the only active mainline
- `LLMbot/baseline/core` is deprecated legacy code
- this note remains useful only as historical context for migration,
  comparison, or forensic review

## What Still Matters From This Note

The document remains relevant for:

- understanding the historical baseline harness shape
- interpreting older baseline-routed experiments
- tracking which comparison and stage concepts originated in the deprecated
  baseline line

## What No Longer Applies

These statements should no longer be treated as current truth:

- `LLMbot/baseline/core/main.py` is the formal experiment entrypoint
- `LLMbot/baseline/core` is the current mainline
- baseline/core tests define the active mainline validation surface

## Replacement Sources

For current truth, use:

- `LLMbot/AGENTS.md`
- `LLMbot/README.md`
- `docs/ARCHITECTURE.md`
- `docs/code/parser.md`
- `code.md`
