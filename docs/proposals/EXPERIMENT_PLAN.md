# Experiment Plan: Reliability-Guided Test-Time Graph Surgery

**Date**: 2026-04-08  
**Method**: Reliability-Guided Test-Time Graph Surgery (prune-only)  
**Target Venue**: WWW 2026 / AAAI 2026  
**Codex Thread**: 019d6b99-a9dc-7522-9d1e-579db0df883c

---

## Claims Being Validated

| ID | Claim |
|----|-------|
| C1 | Prediction disagreement + low q_graph identifies nodes with unreliable graph propagation |
| C2 | Propagation-level repair beats output-level deferral under the same trigger |
| C3 | Utility-based pruning, not generic sparsification, drives the gain |
| C4 | Prune-only single pass is sufficient; extra graph editing is unnecessary |

---

## Research Phase Alignment

Use `docs/research/project_phase_taxonomy.md` for A-G phase naming. These labels are flexible research-planning labels and do not rename experiment block IDs such as B0, B1, or B2.

| Experiment Block | A-G Phase Alignment | Notes |
| --- | --- | --- |
| B0 | 阶段 B：GNN 输出 → post-hoc estimator | Preflight and reproducibility from existing outputs. |
| B1 | 阶段 B：GNN 输出 → post-hoc estimator | Trigger characterization from saved tensors and post-hoc signals. |
| B2 | 阶段 D：risk/regime → local propagation repair; 阶段 G：positioning / stress test | Repair vs same-trigger deferral. |
| B3 | 阶段 B：GNN 输出 → post-hoc estimator; 阶段 D：risk/regime → local propagation repair | Utility and gating ablations. |
| B4 | 阶段 D：risk/regime → local propagation repair | Prune-only mechanism scope. |
| B5 | 阶段 G：positioning / stress test | Threshold robustness. |
| B6 | 阶段 G：positioning / stress test | Multi-seed stability. |
| B7 | 阶段 G：positioning / stress test | Mechanism figures and paper-facing analysis. |

---

## Experiment Blocks

### B0 — Preflight & Reproducibility
- **Claim**: none (sanity)
- **Systems**: m5_sem_ibedl_ats, m7_graph_rgt_gats, m9_router_full, rw0_raw_dual_router, rw1_calibration_only
- **Metrics**: macro-F1, ECE, slice F1, trigger coverage
- **Success**: unit tests pass; anchor numbers reproduce within ±0.002 of m5=0.766, m7=0.767, m9=0.761
- **Kill**: broken tests or summary drift >0.01 — stop until code/artifact mismatch fixed
- **GPU hours**: 0.0 (uses existing artifacts)
- **Depends on**: nothing (root block)

### B1 — Trigger Characterization (C1)
- **Claim**: C1
- **Systems**: rw1_calibration_only; post-hoc trigger analysis from saved tensors
- **Trigger definition**: `pred_sem != pred_graph_raw` AND `q_sem*(1-u_sem)*(1-q_graph_raw) > 0.20`
- **Metrics**: trigger coverage, graph F1 on trigger slice, semantic F1 on trigger slice
- **Success**: coverage 8–20%; graph F1 on trigger slice ≥0.10 below all-node F1; semantic F1 ≥0.05 above graph on trigger slice
- **Kill**: coverage <5% or >30%, or semantic does not beat graph on trigger slice by ≥0.05
- **GPU hours**: 0.0 (post-hoc analysis from rw1 tensors)
- **Depends on**: B0; unlocks B2 and B3

### B2 — Repair vs Deferral (C2) ← KILL TEST
- **Claim**: C2
- **Systems**: rw1_calibration_only, same-trigger semantic deferral (post-hoc from rw1), rw4_disagreement_local, rw2_calib_prune (upper bound)
- **Metrics**: macro-F1, ECE, trigger-slice F1, matched coverage
- **Success**: repair beats same-trigger deferral by ≥0.01 macro-F1 at matched coverage; ECE not worse by >0.01
- **Kill**: same-trigger deferral matches or beats repair; or repair only wins by using materially higher coverage (>+3 points)
- **GPU hours**: 0.5 (one full rw4 seed-42 confirmation; deferral is post-hoc)
- **Depends on**: B1; can run parallel with B3

### B3 — Utility Ablation (C3) ← KILL TEST
- **Claim**: C3
- **Systems**: rw2_calib_prune, rw2_calib_prune_no_qgraph_trigger, rw2_calib_prune_no_soft_gate, rw2_calib_prune_no_expl, rw1_calibration_only
- **Metrics**: macro-F1, ECE, disagreement-slice F1, trigger coverage, edit coverage
- **Success**: rw2_calib_prune beats both no_qgraph_trigger and no_soft_gate by ≥0.005 F1 and ≥0.015 lower ECE; no_expl does not exceed rw2_calib_prune
- **Kill**: ungated pruning or no-soft-gate matches rw2_calib_prune within 0.002 F1
- **GPU hours**: 0.0 (reuses existing scouts)
- **Depends on**: B1; can run parallel with B2

### B4 — Prune-Only Sufficiency (C4)
- **Claim**: C4
- **Systems**: rw2_calib_prune, rw3_calib_prune_add, rw4_disagreement_local, rw4_no_add (manual clone with add_budget_in=0, add_budget_out=0)
- **Metrics**: macro-F1, ECE, disagreement-slice F1, trigger coverage
- **Success**: rw4_no_add within 0.002 F1 of rw4_disagreement_local; rw3_calib_prune_add does not beat best prune-only by >0.003
- **Kill**: add-enabled systems beat prune-only by >0.005 F1 or >0.02 on disagreement slice
- **GPU hours**: 0.5
- **Depends on**: B2 and B3

### B5 — Threshold Robustness
- **Claim**: C1–C4 (robustness)
- **Systems**: B4 winner; sweep graph_rewrite_threshold ∈ {0.15, 0.20, 0.25}, graph_rewrite_keep_threshold ∈ {0.50, 0.55, 0.60}, drop budgets {1, 2}
- **Metrics**: macro-F1, ECE, disagreement-slice F1, trigger coverage
- **Success**: ≥3 neighboring settings within 0.003 F1 of best; ECE varies <0.015; coverage stays in 8–20% band
- **Kill**: best point isolated by >0.006 from all neighbors, or coverage doubles under tiny threshold change
- **GPU hours**: 2.0
- **Depends on**: B4

### B6 — Multi-Seed Confirmation (Publication Grade)
- **Claim**: C1–C4 (publication-grade)
- **Systems**: m5_sem_ibedl_ats, m7_graph_rgt_gats, m9_router_full, rw1_calibration_only, B5 finalist; same-trigger deferral post-hoc from each rw1 seed
- **Seeds**: 42, 123, 456 (minimum 3; 5 preferred)
- **Metrics**: mean±std macro-F1, mean±std ECE, disagreement-slice F1, trigger coverage
- **Success**: finalist mean F1 ≥ rw1+0.01, mean ECE ≤0.035, disagreement-slice F1 ≥0.55, finalist beats same-trigger deferral in ≥4/5 seeds
- **Kill**: multi-seed gain <0.005 — stop paperization
- **GPU hours**: 6.0 (3 seeds × ~2h each)
- **Depends on**: B5

### B7 — Mechanism Figures & Analysis
- **Claim**: C1, C3 (mechanism evidence)
- **Systems**: B6 finalist
- **Analysis**: triggered-node heterophily distribution, pruned vs retained edge utility histogram, trigger coverage vs threshold curve, disagreement-slice F1 vs threshold
- **Metrics**: local heterophily ratio, edge utility distributions, coverage curves
- **Success**: triggered nodes show measurably higher local heterophily; pruned edges have lower utility than retained; figures are reviewer-clean
- **Kill**: mechanism figures do not support the story — narrow claim to empirical performance only
- **GPU hours**: 0.5 (analysis only)
- **Depends on**: B3 and B6

---

## Run Order (DAG)

```
B0 → B1
B1 → B2 (parallel with B3)
B1 → B3 (parallel with B2)
B2 → B4
B3 → B4
B4 → B5
B5 → B6
B3 → B7
B6 → B7
```

---

## Decision Gates

| After | Decision |
|-------|----------|
| B1 | If trigger coverage <5% or semantic doesn't beat graph on slice → rethink trigger definition |
| B2 | If deferral ≥ repair → drop propagation-repair claim; pivot to analysis paper |
| B3 | If controls match → utility story too weak; simplify claim to sparsification |
| B4 | If add beats prune → revise mechanism to include add |
| B5 | If method is brittle → workshop-only result |
| B6 | If multi-seed gain <0.005 → stop paperization |
| B7 | If mechanism figures unclear → narrow to empirical claim only |

---

## Budget Summary

| Block | GPU Hours | Notes |
|-------|-----------|-------|
| B0 | 0.0 | Existing artifacts |
| B1 | 0.0 | Post-hoc analysis |
| B2 | 0.5 | One rw4 full run |
| B3 | 0.0 | Existing scouts |
| B4 | 0.5 | One rw4_no_add run |
| B5 | 2.0 | Threshold sweep |
| B6 | 6.0 | 3–5 seeds |
| B7 | 0.5 | Analysis only |
| **Total** | **~9.5 GPU hours** | Within 1-week budget |

---

## First 3 Commands to Run

```powershell
# 1. rw1_calibration_only (seed 42) — anchor + trigger analysis source
python code/main.py \
  --dataset_path ..\datasets\TwiBot-20 \
  --q_final_path ..\datasets\TwiBot-20\stage1_artifacts\pool_last_ins_on_deb_0_pad_right_dt_auto_len_2048_sd_42_L_sweep\qwen3_emb_L-1.pt \
  --system_mode dual_router \
  --report_group_id rw1_calibration_only \
  --seed 42 \
  --epochs 30 \
  --batch_size 256 \
  --disable_wandb

# 2. Precompute explanation cache from rw1 artifacts
python code/explanation_cache_precompute.py \
  --dataset_path ..\datasets\TwiBot-20 \
  --artifact_dir saved_artifacts\rw1_calibration_only\seed_42 \
  --output_dir saved_artifacts\rw1_calibration_only\seed_42\explanation_cache

# 3. rw4_disagreement_local (seed 42) — main thesis method
python code/main.py \
  --dataset_path ..\datasets\TwiBot-20 \
  --q_final_path ..\datasets\TwiBot-20\stage1_artifacts\pool_last_ins_on_deb_0_pad_right_dt_auto_len_2048_sd_42_L_sweep\qwen3_emb_L-1.pt \
  --system_mode dual_router \
  --report_group_id rw4_disagreement_local \
  --explanation_cache_path saved_artifacts\rw1_calibration_only\seed_42\explanation_cache \
  --seed 42 \
  --epochs 30 \
  --batch_size 256 \
  --disable_wandb
```

**4th command** (after above): same full run for `rw2_calib_prune` — strongest screened competitor to rw4.
