---
type: experiment
node_id: exp:failure_regime_diagnostics_v2
title: Failure-Regime Diagnostic Pipeline v2
status: completed
priority: critical
seeds: [1, 2, 3, 4, 5]
created_at: 2026-04-15T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [diagnostics, failure-regime, rq1, rq2, rq4, matched-budget, prior-only]
---

# Failure-Regime Diagnostic Pipeline v2

## Purpose

Per-node diagnostic analysis of RGT base detector + LM semantic expert to support RQ1-RQ4 of the frozen paper identity (failure-regime-conditioned minimal intervention).

## Methodology Fixes (v1 → v2)

1. **Oracle contamination eliminated**: Full-graph GNN inference on all 11826 nodes (v1 used oracle labels for train nodes in neighbor stats)
2. **Regime definitions are prior-only**: `prop_corruption_score` uses only deployable signals (pred_homophily, neighbor_entropy, own_entropy), never `gnn_correct`
3. **Sparse regime split**: `graph_missing` (degree=0) vs `weak_support` (0 < degree ≤ Q25)
4. **Matched-budget comparison added**: AUROC, AUPRC, top-k precision for risk scoring

## Scripts

- `LMBot/diagnose_gnn.py` v2 — full-graph GNN inference
- `LMBot/diagnose_lm.py` — LM pretrain checkpoint inference
- `LMBot/build_failure_regime_table.py` v2 — prior-only regime definitions
- `LMBot/analyze_regime.py` v2 — matched-budget + cross-tabs + breakdown

## Key Results (test set, 5-seed mean)

### RQ1: Error Concentration

| Regime (prior-defined) | Error Rate In | Error Rate Out | Enrichment |
|------------------------|---------------|----------------|------------|
| sparse_evidence (deg ≤ Q25) | 14.3% | 15.3% | 0.93x |
| graph_missing (deg=0) | 10.6% | 15.1% | 0.70x |
| weak_support (0<deg≤Q25) | 14.9% | 14.7% | 1.02x |
| **prop_corruption_flag (top 25%)** | **22.4%** | **12.2%** | **1.84x** |

Degree-stratified: deg 1-2 err=15.2%, deg 3-10 err=15.9%, deg 11-50 err=9.8%, deg 0 err=10.6%.
Oracle homophily (analysis-only): hom<0.3 err=20.1%, hom>0.7 err=13.4%.

**Caveat**: sparse_evidence does NOT enrich errors on this dataset. prop_corruption_flag (prior-defined composite score) is the strongest prior-only regime signal.

### RQ2: Matched-Budget Risk Capture

| Score | top-5% prec | top-10% | AUROC | AUPRC |
|-------|-------------|---------|-------|-------|
| gnn_entropy | 48.1% | 46.2% | 0.819 | 0.396 |
| disagreement | 49.8% | 32.8% | 0.576 | 0.213 |
| 1-gnn_max_prob | 48.1% | 46.2% | 0.819 | 0.396 |
| prop_corr_score | 34.2% | 30.1% | 0.649 | 0.241 |
| **entropy+disagree** | **50.5%** | **46.5%** | **0.820** | **0.405** |

**Interpretation**: Disagreement is high-precision at small budgets (top-5%) but low-coverage (AUROC 0.576). Entropy is the dominant ranking signal. Combining entropy+disagreement yields marginal but consistent improvement.

### Expert Quadrant Decomposition

| Quadrant | % of test nodes |
|----------|----------------|
| both_correct | 82.4% ± 1.0% |
| both_wrong | 12.1% ± 1.5% |
| lm_only_correct | 2.7% ± 1.0% |
| gnn_only_correct | 2.8% ± 1.0% |

### RQ4: Regime-Action Preference (on GNN-wrong nodes)

| Regime | LM rescue rate (in) | LM rescue rate (out) |
|--------|---------------------|----------------------|
| graph_missing | 30.0% | 17.8% |
| weak_support | 18.5% | 18.4% |
| prop_corruption_flag | 23.8% | 15.2% |

**Interpretation**: Semantic rescue is highest on graph_missing nodes (30% vs 18%), supporting regime-specific action. But absolute numbers are small (~9 nodes/seed for graph_missing). prop_corruption nodes also show elevated LM rescue (23.8% vs 15.2%).

### both_wrong Breakdown

- 12.1% of test nodes (142/seed)
- Enriched in low-homophily: 42.8% have hom<0.3 (vs 32.7% overall)
- Enriched in prop_corruption: 35.3% flagged (vs 25.0% overall)
- 0% have disagreement — both experts are confidently wrong
- High entropy: GNN 0.497 (vs 0.309 overall), LM 0.530 (vs 0.375 overall)

## Caveats (per rigorous self-critique)

1. **Semantic ceiling is narrow**: LM-only-correct = 2.7% → semantic enhance is a narrow operator, not a dominant gain source
2. **Oracle homophily is analysis-only**: Cannot be used as estimator input at deployment time
3. **Disagreement is high-precision but low-coverage**: Not yet proven superior to entropy-only as a full failure-mode estimator
4. **prop_corruption definition is heuristic**: Composite score weights (0.4/0.3/0.3) are not optimized
5. **both_wrong is not yet explained**: May include label noise, adversarial semantics, or model capacity limits

## Artifacts

- Regime tables: `TwiBot-20_RGT_seed_{1-5}/diagnostics/regime_table_{val,test}.csv`
- Per-node diagnostics: `gnn_*_{all,val,test}.pt`, `lm_*_{val,test}.pt`
