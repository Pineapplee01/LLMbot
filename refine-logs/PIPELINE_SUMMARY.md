# Pipeline Summary

**Problem**: Social bot detection errors concentrate in recoverable failure regimes; graph modification provides marginal gains while embedding quality dominates
**Final Method Thesis**: Embedding quality (co-training iterations) dominates graph structure modification on TwiBot-20. A post-hoc semantic override policy on high-entropy nodes improves targeted failure slices without retraining.
**Final Verdict**: READY (for E0-E3 experiments)
**Date**: 2026-04-19

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution: 2×2 embedding×graph systematic analysis (Phase 5, already done, 3 seeds)
- Supporting contribution: Post-hoc semantic override policy on failure-regime-triggered nodes
- Explicitly rejected complexity: Graph modification as primary contribution; composite d_i trigger; FRMI 3-action policy; per-seed alpha tuning; end-to-end retraining

## Evidence Already Collected (No New Experiments Needed)

| Finding | Evidence | Strength |
|---------|----------|----------|
| Embedding >> graph modification | Phase 5: M3-M1=+0.0072 vs graph ±0.003 | Strong (3 seeds) |
| prop_corruption 1.84x enrichment | Phase 1: cross-seed stable | Strong (3 seeds) |
| entropy+disagree best trigger | Phase 1: AUROC=0.820 | Strong (3 seeds) |
| C&S is best frozen baseline | Phase 2: +1.18% acc | Strong (3 seeds) |
| Graph rewrite within noise | Phase 3: ±0.003 F1 | Strong (3 seeds) |
| Semantic override improves targets | Phase 4: +8.3pp target acc | Moderate (alpha unstable) |
| Class-conditional disagreement | Auto-review: confirmed pre/post-cal | Strong (test+valid) |

## Must-Prove Claims

- C1: Embedding quality (co-training) dominates graph modification — M3-M1 = +0.0072 F1 vs graph ±0.003 (DONE)
- C2: Post-hoc semantic override improves targeted failure slices ≥5pp without retraining (NEEDS E0+E1)

## First Runs to Launch

1. **E0**: Fix alpha instability — select alpha on valid_cal (seed-averaged), apply fixed alpha to test
2. **E1**: Override evaluation — 3-seed mean±std on test, regime breakdown
3. **E2**: Main comparison table vs LMbot/BotBR/HyperScan baselines

## Main Risks

- Alpha instability after fix: If alpha still varies >0.15 across seeds → fall back to diagnostic-only paper
- Override < 3pp improvement: If target acc improvement < 3pp → fall back to diagnostic-only paper
- Fallback paper: "Embedding Quality Dominates Graph Structure Modification" — requires 0 additional GPU-hours

## Next Action

- Proceed to `/run-experiment` for E0 (alpha fix, 0.5h) → E1 (override eval, 1h) → E2 (main table, 2-3h)
- Script: `LLMbot/baseline/analysis/frmi_experiment.py` (adapt for fixed-alpha protocol)
- Decision gate after E0: if kill condition triggers, write fallback paper immediately
