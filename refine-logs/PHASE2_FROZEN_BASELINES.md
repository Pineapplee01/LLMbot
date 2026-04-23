# Phase 2: Frozen Post-Hoc Baselines

**Date**: 2026-04-17
**Seeds**: 1, 2, 3
**GNN backbone**: RGCN (frozen checkpoint, no retraining)
**Protocol**: Post-hoc inference-time modifications only

## Results

| Baseline | Overall Acc | Δ vs noop | Notes |
|----------|------------|-----------|-------|
| noop | 0.8453±0.018 | — | Frozen RGCN, anchor emb |
| C&S (Correct & Smooth) | 0.8571±0.012 | +0.0118 | Label propagation post-processing |
| GNNGuard | BLOCKED | — | RGCN incompatible (monkey-patches TransformerConv) |

## Key Findings

1. **C&S is the strongest frozen baseline**: +1.18% acc over noop with no retraining
2. **GNNGuard blocked**: `rgt_forward_gnnguard()` monkey-patches `TransformerConv.message()` which is RGT-specific; requires full rewrite for RGCN compatibility
3. **C&S does not differentiate by regime**: applies uniform label smoothing, no regime-specific behavior

## Stop Condition Check

Can we answer: "Is the strongest frozen baseline sufficient to replace the rewrite/intervention line?"

**Answer: NO** — C&S (+0.012 acc) is weaker than:
- Phase 3 best rewrite (aux_50, F1=0.8703 vs original F1=0.8681, +0.003 F1 — marginal)
- Phase 4 semantic_a0.3 (target acc 0.6328 vs noop 0.5499 on risky nodes)
- Phase 5 M3 finetuned emb (F1=0.8593 retrained from scratch)

C&S improves overall accuracy but does not address the regime-specific failure pattern that motivates FRMI. The intervention line (Phase 4) provides action differentiation that C&S cannot.

**Phase 2: PASS** — strongest frozen baseline identified (C&S), insufficient to replace intervention.
