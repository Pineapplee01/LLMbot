# Protocol Freeze: FRMI Full Paper

**Date**: 2026-04-20
**Status**: FROZEN — no modifications after this date without explicit justification

---

## 1. Dataset & Split Protocol

**Primary**: TwiBot-22 official train/val/test split (from `split.csv`)
- Train: official train set
- Validation: official val set (all threshold/hyperparameter selection)
- Test: forward-only evaluation (no tuning, no threshold adjustment)

**Secondary** (appendix): TwiBot-20 official split

**OOF protocol**: k-fold cross-validation WITHIN the official train split only.
- Purpose: generate failure labels for probe training (Block C)
- k=5 folds, stratified by label
- OOF predictions NEVER enter main result tables
- OOF failure labels = binary indicator of whether base GATv2 misclassified the node

---

## 2. Pre-Registered Subgroup Definitions

All thresholds computed on VALIDATION set, frozen, applied to test without modification.

| Subgroup | Definition | Threshold Rule |
|----------|-----------|----------------|
| Sparse-Evidence | Bottom-20% by in-degree | degree < 20th percentile of val set |
| Propagation-Corruption | Bottom-20% by adjusted neighborhood informativeness | NI < 20th percentile of val set |
| Semantic-Structural Conflict | Top-20% by \|p_LM - p_GNN\| | divergence > 80th percentile of val set |
| Camouflage-Heavy | ≥50% 1-hop neighbors have opposite predicted label | binary rule, no threshold to tune |
| High-Confidence-Wrong (HCW) | MSP > 0.9 AND incorrect prediction | MSP threshold fixed at 0.9; correctness from OOF |

Notes:
- "Adjusted neighborhood informativeness" follows the NeurIPS 2023 homophily metrics paper — not naive edge homophily
- Subgroups may overlap (a node can be both sparse-evidence and camouflage-heavy)
- Subgroup membership is computed per-node, not per-edge

---

## 3. Probe & Policy Hyperparameters

All selected on validation set only:
- Probe: L2-regularized logistic regression, regularization C selected via val AUROC
- Intervention budget: 15% of test nodes (fixed; sensitivity analysis in appendix)
- Semantic residual alpha: selected on val set, fixed for all test seeds
- Edge reweight threshold: selected on val set, fixed for all test seeds
- Regime assignment rules: frozen after val analysis

---

## 4. Statistical Protocol

- 3 random seeds for all stochastic experiments
- Report: mean ± std across seeds in all tables
- Core comparisons: paired bootstrap over nodes on same test set (1000 resamples, 95% CI)
  - D7 vs D2, D3, D5, D6 (policy comparison)
  - Matched vs swapped actions within each regime (causal test)
- Significance threshold: p < 0.05 (bootstrap)

---

## 5. Metrics

**Primary** (main tables):
- Overall Macro-F1, Bot-F1
- Hard-subgroup ΔF1 (per pre-registered subgroup)
- Easy-node preservation (bottom-50% safest nodes, no degradation)
- Budgeted intervention gain (F1 improvement per % nodes intervened)

**Probe-specific** (Table 2):
- Utility@15% (precision of intervention at 15% budget)
- HCW capture rate (fraction of high-confidence-wrong in top-k)
- Risk-capture curve (probe vs uncertainty-only)
- AUROC, AUPRC (secondary)

**Appendix**:
- ECE, Brier score
- Conformal coverage / set size (optional)

---

## 6. Backbone Configuration

**Primary**: GATv2 (2-layer GATv2Conv)
- Hidden dim: 128
- Heads: 8
- Dropout: 0.4
- Activation: LeakyReLU
- Input: frozen LM embeddings (768-dim from co-trained RoBERTa)

**Confirmatory** (appendix): BotRGCN (existing implementation)
- Same hyperparameters where applicable
- Portability criterion: P1/P2/S1 hold directionally (same sign)

---

## 7. What Is NOT Allowed After Freeze

- Changing TwiBot-22 split
- Tuning any hyperparameter on test set
- Adding/removing subgroup definitions based on test results
- Changing probe training target based on test performance
- Adjusting intervention budget based on test outcomes
- Post-hoc selection of which conditions to report
