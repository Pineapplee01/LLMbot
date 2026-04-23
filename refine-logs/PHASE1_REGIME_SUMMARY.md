# Phase 1: Regime Summary

**Date**: 2026-04-17
**Seeds**: 1, 2, 3 (3-seed mean)
**GNN backbone**: RGCN (2-layer, 128d, auto-detected from checkpoints)
**Diagnostics**: v2 (full-graph inference, prior-only regime definitions)

## RQ1: Error Concentration

Errors are **non-uniformly concentrated** and **cross-seed reproducible**.

### Expert Quadrant (test set, 3-seed mean ± std)
| Quadrant | % |
|----------|---|
| both_correct | 82.4% ± 1.0% |
| both_wrong | 12.1% ± 1.5% |
| lm_only_correct | 2.7% ± 1.0% |
| gnn_only_correct | 2.8% ± 1.0% |

### Error Rate by Degree
| Degree | Error Rate | n |
|--------|-----------|---|
| 0 (graph_missing) | 10.6% ± 1.5% | ~87 |
| 1-2 | 15.2% ± 1.2% | ~764 |
| 3-10 | **15.9%** ± 1.2% | ~274 |
| 11-50 | 9.8% ± 2.7% | ~57 |

### Regime Enrichment
| Regime | Error In | Error Out | Enrichment |
|--------|---------|----------|------------|
| prop_corruption_flag (top 25%) | **22.4%** | 12.2% | **1.84x** |
| sparse_evidence | 14.3% | 15.3% | 0.93x |
| graph_missing | 10.6% | 15.1% | 0.70x |

**Key finding**: prop_corruption is the only regime with strong error enrichment (1.84x). Sparse evidence and graph_missing do NOT enrich errors.

## RQ2: Trigger Quality
| Score | AUROC | Top-5% Precision |
|-------|-------|-----------------|
| entropy+disagree | **0.820** | **50.5%** |
| gnn_entropy | 0.819 | 48.1% |
| prop_corr_score | 0.649 | 34.2% |
| disagreement | 0.576 | 49.8% |

## both_wrong Analysis (12.1%, ~142/seed)
- 0% have disagreement (both experts confidently wrong)
- Enriched in low homophily (42.8% vs 32.7% overall)
- Enriched in prop_corruption (35.3% vs 25.0% overall)
- High entropy: GNN 0.497, LM 0.530

## Stop Condition Check
1. Error concentration non-uniform: **YES** (prop_corruption 1.84x, degree 3-10 highest)
2. Cross-seed reproducible: **YES** (stable across seeds 1-3)
3. v2 regime table based on full-graph diagnostics: **YES** (gnn_*_all.pt generated)

**Phase 1: PASS** — proceed to Phase 2.
