---
type: experiment
node_id: exp:kill_tests_b2_b3
title: Kill Tests B2 (Repair vs Deferral) + B3 (Utility Ablation)
status: completed
priority: critical
seeds: [42]
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [kill-test, repair-vs-deferral, ablation, calibration]
---

# Kill Tests B2 + B3 — BOTH PASSED

## B2: Repair vs Deferral — ✅ STRONG PASS

**Thesis**: Propagation-level repair (local edge pruning + re-propagation) beats output-level deferral.

| System | Macro F1 | ECE | Disagree-Slice F1 | Coverage |
|--------|----------|-----|-------------------|----------|
| rw1_calibration_only (baseline) | 0.7391 | 0.0738 | 0.4079 | 29.1% |
| same-trigger-defer (tau=0.10) | 0.7444 | 0.0754 | 0.4324 | 28.83% |
| **rw4_disagreement_local (repair)** | **0.7603** | **0.0190** | **0.5881** | 22.3% |

**Key metric**: Disagree-slice F1 gain = **+0.1557** (required ≥0.01) → **15x threshold**

**ECE improvement**: -0.0548 (74% reduction)

## B3: Utility Ablation — ✅ PASS

**Thesis**: Utility-guided pruning (q_graph trigger + soft gating) drives the gain, not generic sparsification.

| System | Macro F1 | ECE | ΔF1 vs Full | ΔECE vs Full |
|--------|----------|-----|-------------|-------------|
| rw2_calib_prune (full) | **0.7620** | **0.0235** | — | — |
| no_qgraph_trigger | 0.7559 | 0.0402 | -0.0060 | +0.0166 |
| no_soft_gate | 0.7559 | 0.0661 | -0.0060 | +0.0426 |
| no_expl | 0.7557 | — | -0.0063 | — |

**F1 loss**: -0.006 per ablation (required ≥0.005) ✅
**ECE loss**: +0.017 to +0.043 (required ≥0.015) ✅

## B4: Prune-Only Sufficiency — ✅ ALREADY VALIDATED

- rw2_calib_prune (prune-only): 0.7620 F1
- rw3_calib_prune_add (prune+add): 0.7620 F1 (identical)
- **Conclusion**: Edge addition provides zero benefit. Prune-only is sufficient.

## Verdict

**PROCEED TO PUBLICATION** — All kill tests passed decisively.

## Artifacts

All in `LLMbot/saved_artifacts/`:
- `rw1_calibration_only/seed_42/` — Baseline
- `rw4_disagreement_local/seed_42/` — Main method
- `rw2_calib_prune/seed_42/` — Upper-bound competitor
- `rw2_calib_prune_no_qgraph_trigger/seed_42/` — Ablation
- `rw2_calib_prune_no_soft_gate/seed_42/` — Ablation
- `rw2_calib_prune_no_expl/seed_42/` — Ablation
- `rw3_calib_prune_add/seed_42/` — Add-budget control
