# Idea Candidates (Compact)

| # | Idea | d_i | Policy | Repair | Novelty | Risk | Status |
|---|------|-----|--------|--------|---------|------|--------|
| 1 | FRMI | composite (prop_corruption + disagreement + rarity) | 3-way MLP | trust-weighted attenuation | 7/10 | LOW-MED | RECOMMENDED |
| 2 | RASS | regime classifier | regime-to-action map | regime-specific attenuation | 6/10 | LOW | BACKUP |
| 3 | UGBP-P | BP entropy + disagreement | convergence-based | trust-weighted BP messages | 8/10 | MEDIUM | HIGH NOVELTY |
| 4 | EDP-AR | evidential vacuity * conflict | threshold-based | trust-weighted | 6/10 | MEDIUM | ALTERNATIVE |
| 5 | SSCR | embedding cosine distance | learned 3-way | embedding-guided reweight | 5/10 | LOW | SIMPLER |
| 6 | CEU-S | rw4 trigger + prop_corruption | rule-based | soft keep-utility | 4/10 | LOW | MINIMAL EXTENSION |

## Active Idea: #1 — FRMI (Failure-Regime-Conditioned Minimal Intervention)
- **Hypothesis**: Composite d_i + 3-action policy + directional attenuation improves over binary trigger + pruning
- **Key evidence**: prop_corruption enriches errors 1.84x; rw4 repair beats deferral +0.16 F1; prune-only works
- **Next step**: Implement in LLMbot/, pilot on seed 42, compare against rw4 + ablations
- **Kill criterion**: If FRMI doesn't beat rw4 on disagree slice AND global F1 < LM-only (0.8756)
