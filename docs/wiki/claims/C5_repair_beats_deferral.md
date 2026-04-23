---
type: claim
node_id: claim:C5
title: Propagation-Level Repair Beats Output-Level Deferral
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [repair, deferral, propagation, kill-test]
---

# Claim: Propagation-Level Repair Beats Output-Level Deferral

## Statement

On the same triggered nodes, propagation-level repair (local edge pruning + re-propagation) dramatically outperforms output-level deferral (switching to semantic logits).

## Evidence Status

✅ **SUPPORTED** — B2 kill test STRONG PASS (2026-04-08)

## Supporting Evidence

### Head-to-Head Comparison (same trigger, seed 42)

| Method | Macro F1 | ECE | Disagree-Slice F1 |
|--------|----------|-----|-------------------|
| Deferral (tau=0.10) | 0.7444 | 0.0754 | 0.4324 |
| **Repair (rw4)** | **0.7603** | **0.0190** | **0.5881** |

- **Disagree-slice gain: +0.1557** (required ≥0.01) → **15x threshold**
- **Macro-F1 gain: +0.0159**
- **ECE improvement: -0.0564** (74% reduction)

### Why Repair Beats Deferral

- Deferral only changes the final label decision — contaminated messages have already propagated
- Repair removes harmful edges BEFORE re-propagation, cleaning the message-passing input
- This is the paper's dominant contribution: "repair the graph, don't just route around it"

## Tested By

- exp:kill_tests_b2_b3 (B2 kill test)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
