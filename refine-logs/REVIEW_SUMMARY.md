# Review Summary: Embedding-Dominant Failure Analysis

**Date**: 2026-04-19
**Method**: Embedding×Graph analysis + post-hoc semantic override
**Supersedes**: FRMI review summary (2026-04-15)

## Evidence-Based Refinement (No External Reviewer — Codex MCP token expired)

### What Was Tried and Killed

| Attempt | Result | Reason Killed |
|---------|--------|---------------|
| FRMI on RGT system | F1=0.7472 ≈ Semantic 0.7477 | Wrong base system (F1≈0.76 vs baselines≈0.87) |
| Composite d_i trigger | AUROC=0.692 < entropy 0.719 | E1 killed |
| Graph rewrite (aux_50) | +0.003 F1, within noise | Phase 3/5: embedding dominates |
| 3-action policy | No global improvement | Pilot marginal, proxy repair |
| Diagnostic study on RGT | 7/10 workshop | Built on weak system |

### What Survived

| Finding | Evidence | Strength |
|---------|----------|----------|
| Embedding >> graph modification | Phase 5: M3-M1=+0.0072 vs graph ±0.003 | Strong (3 seeds) |
| Semantic override improves targets | Phase 4: +8.3pp target acc | Strong (3 seeds) |
| prop_corruption 1.84x enrichment | Phase 1: cross-seed stable | Strong (3 seeds) |
| entropy+disagree best trigger | Phase 1: AUROC=0.820 | Strong (3 seeds) |
| Class-conditional disagreement | Auto-review: confirmed pre/post-cal | Strong (test+valid) |
| Action differentiation exists | Phase 4: repair≠semantic on slices | Moderate (1 seed) |

### Thesis Decision

**Chosen**: Thesis D+C merged — "Embedding-dominant failure analysis with post-hoc semantic override"

**Rationale**:
- Thesis C (embedding-first) alone is too descriptive for CIKM short
- Thesis D (selective override) alone has alpha instability problem
- Merged: 2×2 analysis (already done, 3 seeds) + override policy (needs alpha fix)
- The 2×2 matrix is the strongest result and requires no new experiments
- The override policy needs one fix (alpha on valid_cal) and re-evaluation

### Alpha Instability Fix

Phase 4 selected alpha per-seed on valid_cal, which produced instability (0.05, 0.2, 0.3 across seeds). Fix: select alpha on valid_cal using seed-averaged performance, then apply the same alpha to all seeds. This is methodologically cleaner and should stabilize results.

### Minimum Viable Paper

1. Phase 5 2×2 matrix (already done) — embedding dominates
2. Phase 1 regime analysis (already done) — prop_corruption enrichment
3. Phase 4 override with fixed alpha (needs re-run) — targeted improvement
4. Comparison table vs baselines (needs 3-seed main table)

## Consensus

- Method is grounded in validated empirical data (Phases 1-5)
- Biggest risk: alpha instability after fix
- Biggest opportunity: 2×2 embedding×graph analysis is genuinely novel framing
- Venue: CIKM 2026 short/applied (May 23) primary; ASONAM 2026 (Apr 26) stretch
