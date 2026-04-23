---
type: experiment
node_id: exp:confidence_fallback
title: Confidence Fallback Baseline (M2)
status: completed
priority: medium
seeds: [42]
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [baseline, confidence-fallback, slice-diagnostics]
---

# Confidence Fallback Baseline

## Result

Confidence fallback (tau=0.9) beats learned router by +1.1 F1 (0.798 vs 0.788).

## Slice Diagnostics

| Slice | Semantic F1 | Graph F1 | Fallback F1 | Notes |
|-------|------------|----------|-------------|-------|
| degree_gt20 | — | — | — | Graph hurts |
| disagreement | — | — | — | Graph hurts |
| degree_6_20 | — | — | — | Graph helps |
| Target slice (N=22) | 0.533 | — | 0.533 | Intervention needed |

## Key Finding

- Simple confidence fallback outperforms learned router
- Graph hurts on degree_gt20 and disagreement slices
- Graph only helps on degree_6_20 slice
- Target slice (N=22) has F1=0.533 for both semantic and fallback — needs propagation-level repair

## Implication

Validates claim:C2 — confidence-based fallback is indeed a strong baseline.
But also shows that fallback alone is insufficient for the hardest cases (target slice F1=0.533).
Propagation-level repair (rw4) achieves 0.5881 on disagreement slice, beating fallback.

## Artifacts

- `LLMbot/saved_artifacts/diagnostics/confidence_fallback.json`
- `LLMbot/saved_artifacts/diagnostics/slice_diagnostics.json`
