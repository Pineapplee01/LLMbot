# Phase 3: Rewrite/Retrain Baselines

**Date**: 2026-04-17
**Seeds**: 1, 2, 3
**GNN backbone**: RGCN (2-layer, 128d, 200 epochs, AdamW + CosineAnnealing)
**Embedding**: LMBot anchor (embeddings_iter_-1.pt, 768d RoBERTa)

## Results

| Config | Acc (mean±std) | F1 (mean±std) | ΔF1 | Edges |
|--------|---------------|--------------|-----|-------|
| original | 0.8532±0.011 | 0.8681±0.013 | — | 16908 |
| heur_90 | 0.8512±0.008 | 0.8669±0.010 | -0.002 | 15217 |
| heur_80 | 0.8507±0.009 | 0.8658±0.011 | -0.003 | 13526 |
| heur_70 | 0.8515±0.007 | 0.8670±0.010 | -0.002 | 11835 |
| learn_90 | 0.8538±0.009 | 0.8684±0.011 | +0.001 | 15217 |
| learn_80 | 0.8509±0.008 | 0.8670±0.010 | -0.002 | 13526 |
| learn_70 | 0.8546±0.005 | 0.8699±0.008 | +0.001 | 11835 |
| **aux_50** | **0.8560±0.009** | **0.8703±0.011** | **+0.003** | 8454 |

## Key Findings

1. **Strongest rewrite/retrain config**: `aux_50` (keep top 50% by learned reliability), F1=0.8703 (+0.003 over original)
2. **Heuristic pruning hurts**: All heur_* configs slightly degrade F1
3. **Learned pruning marginal**: learn_70 and learn_90 show tiny gains (+0.001)
4. **Gains within noise**: Best gain (+0.003) is well within 1 std (±0.011)
5. **Aggressive pruning doesn't help**: Removing 50% of edges (aux_50) gives the best result, suggesting the original graph has significant noise

## Stop Condition Check

1. Strongest rewrite/retrain config identified: **aux_50** (+0.003 F1)
2. Gain magnitude: **marginal** (within noise)
3. Worth entering Phase 5: **Questionable** — the gain is too small to be a reliable signal

## Implication

Graph preprocessing alone does not provide a strong signal for this dataset/backbone combination. The original RGCN on the original graph (F1=0.868) is already a strong baseline. Any method claiming graph modification as a contribution must show gains significantly larger than +0.003.
