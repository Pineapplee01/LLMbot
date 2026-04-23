---
type: claim
node_id: claim:C8
title: Different Failure Regimes Show Different Corrective Action Preferences
status: preliminary
confidence: moderate
evidence_type: empirical
created_at: 2026-04-15T00:00:00Z
updated_at: 2026-04-15T00:00:00Z
tags: [failure-regime, action-preference, semantic-enhance, propagation-corruption, rq4]
---

# Claim: Different Failure Regimes Show Different Corrective Action Preferences

## Statement

When the base GNN detector fails, the effectiveness of semantic rescue (LM-only correction) varies systematically across prior-defined failure regimes. Graph-missing nodes show the highest semantic recoverability (30%), while propagation-corruption nodes show elevated but lower recoverability (23.8%). This supports the conjecture that different regimes require different corrective actions.

## Evidence Status

⚠️ **PRELIMINARY** — Upper-bound style evidence only. Semantic usefulness shown via LM rescue rate, but actual intervention mechanisms not yet tested.

## Supporting Evidence (exp:failure_regime_diagnostics_v2)

### LM Rescue Rate on GNN-Wrong Nodes

| Regime | LM rescue (in) | LM rescue (out) | Δ |
|--------|----------------|-----------------|---|
| graph_missing (deg=0) | 30.0% | 17.8% | +12.2pp |
| weak_support (0<deg≤Q25) | 18.5% | 18.4% | +0.1pp |
| prop_corruption_flag | 23.8% | 15.2% | +8.6pp |

### Quadrant × Degree Cross-Tab

| Quadrant | deg=0 | deg 1-2 | deg 3-10 | deg 11-50 |
|----------|-------|---------|----------|-----------|
| both_correct | 86.7% | 81.7% | 81.8% | 88.4% |
| both_wrong | 7.4% | 12.6% | 13.0% | 8.4% |
| lm_only_correct | 3.2% | 2.7% | 2.9% | 1.4% |
| gnn_only_correct | 2.8% | 3.0% | 2.3% | 1.8% |

### Interpretation

- **graph_missing**: GNN has no graph signal → semantic rescue is most effective here. Supports Semantic Enhance as the preferred action.
- **weak_support**: Marginal graph signal → semantic rescue offers no advantage. Neither semantic nor graph-side action is clearly preferred.
- **prop_corruption**: Corrupted propagation → LM rescue is elevated (23.8%) but not dominant. Suggests propagation-side correction may be needed in addition to or instead of semantic rescue.

## Caveats

1. **Upper-bound evidence**: LM rescue rate measures "could semantic help?" not "does semantic intervention actually help?"
2. **Small absolute numbers**: graph_missing has ~9 GNN-wrong nodes per seed — high variance
3. **Semantic ceiling is narrow**: Total LM-only-correct = 2.7% of test set
4. **No actual intervention tested**: Need to implement and evaluate regime-conditioned actions to confirm

## Tested By

- exp:failure_regime_diagnostics_v2

## Implications

- Paper should NOT be written as semantic-heavy story (ceiling too low)
- Semantic Enhance is a narrow, regime-specific operator for graph_missing/sparse nodes
- Propagation-corruption regime likely needs graph-structure-level intervention
- both_wrong (12.1%) remains outside current action space — future work
