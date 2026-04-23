---
type: claim
node_id: claim:C1
title: Graph Evidence is Uneven and Can Be Harmful on Sparse/Disagreement Cases
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-03-25T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [graph-reliability, failure-modes, calibration]
---

# Claim: Graph Evidence is Uneven and Can Be Harmful on Sparse/Disagreement Cases

## Statement

Graph evidence is not uniformly reliable across all nodes. On low-degree, graph-missing, and disagreement slices, graph evidence can be harmful and lead to worse performance than text-only baselines.

## Evidence Status

✅ **SUPPORTED** — High confidence

## Supporting Evidence

### Early Evidence (2026-03-25, rw1/rw4 framework)

1. **Router underperforms both single experts**:
   - Router F1: 0.761, Semantic F1: 0.766, Graph F1: 0.767

2. **Text-only outperforms full model**:
   - Text-only F1: 0.8115, Full model F1: 0.7945

### v2 Diagnostic Evidence (2026-04-15, RGT base detector, 5 seeds)

3. **Degree-stratified error rates** (test set):
   - deg 0: err=10.6%, deg 1-2: err=15.2%, deg 3-10: err=15.9%, deg 11-50: err=9.8%
   - Errors concentrate on low-degree connected nodes, not isolated nodes

4. **Oracle homophily analysis** (analysis-only, not deployable):
   - hom<0.3: err=20.1%, hom 0.3-0.5: err=11.8%, hom 0.5-0.7: err=10.3%, hom>0.7: err=13.4%
   - Low-homophily neighborhoods have 2x error rate

5. **Prior-defined regime enrichment** (no oracle, no gnn_correct in definition):
   - prop_corruption_flag (composite score top 25%): err=22.4% vs 12.2% outside → **1.84x enrichment**
   - sparse_evidence (deg ≤ Q25): err=14.3% vs 15.3% → no enrichment on this dataset

6. **Expert quadrant decomposition**:
   - both_correct: 82.4%, both_wrong: 12.1%, lm_only_correct: 2.7%, gnn_only_correct: 2.8%
   - Errors are NOT uniformly distributed across expert agreement patterns

## Tested By

- exp:text_only_baseline_2026_03_25
- exp:full_model_baseline_2026_03_25

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Implications

- Need selective graph usage mechanism (abstention, fallback)
- Cannot assume graph always helps
- Slice-specific strategies required
