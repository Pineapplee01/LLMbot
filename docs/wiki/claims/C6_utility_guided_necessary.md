---
type: claim
node_id: claim:C6
title: Utility-Guided Pruning is Necessary, Not Generic Sparsification
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [ablation, utility, pruning, kill-test]
---

# Claim: Utility-Guided Pruning is Necessary, Not Generic Sparsification

## Statement

The q_graph trigger and soft gating components are both necessary for the method to work. Removing either degrades F1 and ECE, proving the gain comes from utility-guided pruning, not generic sparsification.

## Evidence Status

✅ **SUPPORTED** — B3 kill test PASS (2026-04-08)

## Supporting Evidence

| Ablation | Macro F1 | ECE | ΔF1 | ΔECE |
|----------|----------|-----|-----|------|
| Full (rw2) | **0.7620** | **0.0235** | — | — |
| No q_graph trigger | 0.7559 | 0.0402 | -0.0060 | +0.0166 |
| No soft gate | 0.7559 | 0.0661 | -0.0060 | +0.0426 |
| No explanation | 0.7557 | — | -0.0063 | — |

- F1 loss per ablation: -0.006 (required ≥0.005) ✅
- ECE loss: +0.017 to +0.043 (required ≥0.015) ✅
- Soft gating removal nearly triples ECE (0.0235 → 0.0661)

## Tested By

- exp:kill_tests_b2_b3 (B3 kill test)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
