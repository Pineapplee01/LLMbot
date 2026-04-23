# Phase 5: 2×2 Embedding × Graph Matrix

**Date**: 2026-04-17
**Seeds**: 1, 2, 3
**GNN backbone**: RGCN (retrained from scratch, 200 epochs, AdamW + CosineAnnealing)
**Protocol**: Anchor emb = embeddings_iter_-1.pt; Finetuned emb = last available iter (seed 1,3: iter_2; seed 2: iter_1)

## Results

| Cell | Embedding | Graph | Acc (mean±std) | F1 (mean±std) |
|------|-----------|-------|---------------|--------------|
| M1 | anchor (iter_-1) | original | 0.8540±0.0101 | 0.8521±0.0094 |
| M2 | anchor (iter_-1) | aux_50 | 0.8560±0.0098 | 0.8541±0.0091 |
| M3 | finetuned (last iter) | original | **0.8611±0.0047** | **0.8593±0.0042** |
| M4 | finetuned (last iter) | aux_50 | 0.8583±0.0071 | 0.8562±0.0066 |

## Deltas

| Comparison | Δ Acc | Δ F1 | Interpretation |
|-----------|-------|------|----------------|
| M2 − M1 (graph effect, anchor emb) | +0.0020 | +0.0021 | aux_50 marginally helps with anchor emb |
| M4 − M3 (graph effect, finetuned emb) | −0.0028 | −0.0031 | aux_50 slightly hurts with finetuned emb |
| M3 − M1 (emb effect, original graph) | **+0.0070** | **+0.0072** | finetuned emb is the stronger signal |
| M4 − M2 (emb effect, aux_50 graph) | +0.0023 | +0.0021 | emb effect smaller on pruned graph |

## Key Findings

1. **Finetuned embedding is the dominant axis**: M3−M1 = +0.0072 F1, the largest delta in the matrix
2. **Graph rewrite effect is negligible and inconsistent**: +0.0021 with anchor emb, −0.0031 with finetuned emb — within noise (std ≈ 0.009)
3. **Best cell is M3** (finetuned emb + original graph): F1=0.8593 — graph pruning does not help the best embedding
4. **aux_50 graph does not compound with finetuned emb**: M4 (0.8562) < M3 (0.8593), suggesting the pruned graph removes edges that are informative for the finetuned representation
5. **Graph rewrite contribution is not separable from noise**: max |graph delta| = 0.0031, well within ±1 std

## Stop Condition Check

Required: write out M2−M1, M4−M3, M3−M1, M4−M2 deltas.

✅ All four deltas reported above.

## Implication for Research Direction

- The embedding quality (co-training iterations) matters more than graph structure modification
- Graph rewrite (aux_50) provides no reliable gain: +0.002 in one condition, −0.003 in another
- The FRMI action differentiation finding (Phase 4) remains valid as a mechanism claim, but the overall performance ceiling is set by embedding quality, not graph surgery
- **Phase 5: PASS** — matrix complete, deltas clear

## Combined Summary (Phases 3–5)

| Method | F1 | Δ vs M1 baseline |
|--------|-----|-----------------|
| M1: anchor + original (RGCN retrained) | 0.8521 | — |
| M2: anchor + aux_50 | 0.8541 | +0.0021 |
| M3: finetuned + original | **0.8593** | **+0.0072** |
| M4: finetuned + aux_50 | 0.8562 | +0.0041 |
| Phase 4 noop (frozen RGCN, anchor emb) | 0.8453 | — |
| Phase 4 semantic_a0.3 (frozen) | 0.8681 | — |

Note: Phase 4 semantic_a0.3 (0.8681) outperforms all retrained cells because it uses the frozen original RGCN checkpoint (which was trained with the full LMBot co-training pipeline including soft-label distillation), not a from-scratch retrain on anchor embeddings only.
