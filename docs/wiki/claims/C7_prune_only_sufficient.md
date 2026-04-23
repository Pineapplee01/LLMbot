---
type: claim
node_id: claim:C7
title: Prune-Only Single Pass is Sufficient
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [prune-only, simplicity, edge-addition]
---

# Claim: Prune-Only Single Pass is Sufficient

## Statement

Edge addition provides zero benefit beyond pruning. Prune-only single pass achieves identical performance to prune+add, simplifying the method.

## Evidence Status

✅ **SUPPORTED** — B4 validated from existing results

## Supporting Evidence

| Method | Macro F1 | ECE |
|--------|----------|-----|
| rw2_calib_prune (prune-only) | **0.7620** | **0.0235** |
| rw3_calib_prune_add (prune+add) | **0.7620** | **0.0235** |

- Identical F1 and ECE
- Edge addition adds complexity without benefit
- Simplifies paper story: "prune harmful edges" is cleaner than "prune and add"

## Tested By

- exp:kill_tests_b2_b3 (B4 comparison)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
