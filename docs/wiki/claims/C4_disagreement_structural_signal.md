---
type: claim
node_id: claim:C4
title: Expert Disagreement is a High-Precision Risk Cue with Limited Coverage
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [disagreement, trigger, structural-signal, matched-budget]
---

# Claim: Expert Disagreement is a High-Precision Risk Cue with Limited Coverage

## Statement

Prediction disagreement between semantic (LM) and graph (GNN) experts identifies a small, highly risky subset of nodes. When experts disagree, GNN accuracy drops to ~50.9% (vs 87.2% when they agree). However, disagreement alone has limited coverage (5.5% of nodes) and low AUROC (0.576), making it a precision-oriented cue rather than a complete failure-mode estimator.

## Evidence Status

✅ **SUPPORTED** — Validated in B1 trigger characterization + v2 matched-budget comparison

## Supporting Evidence

### B1 Trigger Characterization (2026-04-08)
- Trigger coverage: 14.2% of nodes (selective, not blanket)
- On triggered nodes: semantic expert beats graph expert

### v2 Matched-Budget Comparison (2026-04-15, RGT base detector, 5 seeds)

| Score | top-5% precision | AUROC | AUPRC |
|-------|-----------------|-------|-------|
| gnn_entropy | 48.1% | 0.819 | 0.396 |
| **disagreement** | **49.8%** | 0.576 | 0.213 |
| entropy+disagree | 50.5% | **0.820** | **0.405** |

- Disagreement: highest top-5% precision (49.8%) but lowest AUROC (0.576)
- Entropy: dominant ranking signal (AUROC 0.819)
- Combined: marginal but consistent improvement over entropy alone
- When experts agree: GNN acc=0.872; when disagree: GNN acc=0.509

### Interpretation

Disagreement should be treated as a **high-precision, low-coverage risk feature** rather than a standalone failure-mode estimator. It is most valuable when combined with entropy-based signals. The claim is softened from "identifies unreliable propagation" to "identifies a high-risk subset" — the causal mechanism (propagation corruption vs other failure modes) is not yet proven.

## Tested By

- exp:kill_tests_b2_b3 (B1 characterization)
- exp:failure_regime_diagnostics_v2 (matched-budget comparison)
