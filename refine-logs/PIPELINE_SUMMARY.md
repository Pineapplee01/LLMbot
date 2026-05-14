# Pipeline Summary (v8) — Evidence-Graph Construction for LLM-Consumable Ego Context

**Method**: EQC v8 reframed as **Stage-5 4-tuple protocol** (schema + calibrated-quantile bucketization + canonical ordering + fixed token budget) for LLM-consumable graph-evidence artifacts. User rescoped LLM refiner from "optional" (v7) back to "mandatory main stage" (v8); 7-stage pipeline hard-locked.
**Problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage; scalar post-hoc calibration yields no operational ego-uncertainty signal for selective, budgeted, reversible refinement on a frozen RoBERTa + BotRGCN backbone.
**Final Method Thesis**: A 4-tuple protocol — (i) fixed 5-section schema, (ii) deterministic Q_66/Q_33 bucketization on calibrated similarity, (iii) canonical intra-bucket rank order `(PPR_rank, −sim)`, (iv) pre-registered 512-token budget with deterministic truncation — that converts frozen-backbone uncertain-ego contexts into LLM-consumable graph-evidence artifacts. The protocol's value is isolated by **9-variant Block-E + Block-CT evaluation** (scramble controls, 2-LLM robustness, cross-dataset calibration transfer), not by best-prompt search.
**Final Verdict**: **REVISE at MAX_ROUNDS=5** — structurally READY-adjacent (overall 8.63/10 at Round 5, NONE drift, zero structural blockers). Per Round-5 reviewer: "**remaining gap is primarily empirical, not structural**... CQ/MS ≥ 9 depends on Block-E + Block-CT results confirming the acceptance criteria. No further proposal polishing will move the needle."
**Date**: 2026-05-09
**Supersedes**: PIPELINE_SUMMARY.md v7 (2026-05-09 earlier this session; user rescoped LLM stage back to mandatory main). v7 artifacts remain in git history.

---

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (v5)
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Research audit: `refine-logs/RESEARCH_AUDIT.md` (research.md ↔ deliverables conformance)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (v5)
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md` (v5)
- Per-round artifacts: `round-0-initial-proposal.md`, `round-{1,2,3,4,5,6}-review.md`, `round-{1,2,3,4,5}-refinement.md`
- Score evolution: `refine-logs/score-history.md`
- Checkpoint: `refine-logs/REFINE_STATE.json` (`status: "completed"`)

## Contribution Snapshot (v8)

- **Dominant contribution**: **Stage-5 4-tuple protocol** — schema + deterministic Q_66/Q_33 calibrated-quantile bucketization + canonical `(PPR_rank, −sim)` intra-bucket ordering + pre-registered 512-token budget with deterministic truncation. Defined abstractly (independent of upstream plumbing); value isolated by scramble controls (role, order, collapse), encoder controls (E-null, E-RoBERTa, E-LLM), 2-LLM robustness (Qwen-2.5-7B, Mistral-7B-Instruct), and cross-dataset calibration transfer.
- **Supporting plumbing (explicitly non-novel)**: Stage-1 2-term composite `s_lbl + w_tg·JSD(p_LM ‖ p_GNN)`; Stage-3 2-iteration retrieval with top-50% quantile gate; Stage-4 calibrated-expected-error `ê_φ` 2-criterion accept (5-fold cross-fit on valid_cal); Stage-6 9218-param linear probe over concat(h_GNN_post, X_R, z_LLM).
- **Explicitly rejected complexity**: LLM-as-classifier; binary edge-reliability headline; global structure learning; CP coverage theorem; causal identification; s_reliability binary head; p_role 4-way training head; prompt-optimality claims; ordering-as-bonus claims; argmax-confidence accept proxy; hard-gate thresholds; retrieval_trail in canonical schema.

## Must-Prove Claims (6)

- **C (dominant, 4-part acceptance)**: (i) E-LLM-Qwen > E-RoBERTa > E-null with consistent direction on BOTH datasets; (ii) ≥ 1 seed-std on ≥ 1 dataset; (iii) E-LLM-Qwen > {shuffle-role, shuffle-order, collapse} consistent direction on BOTH; (iv) E-LLM-Qwen ≈ E-LLM-Mistral within 1 seed-std on EACH dataset.
- **S1** Stage necessity audit (Block-P) — transparency, not claim.
- **S2** Iteration budget-matched (Block-R) — R_2 > R_1-k* + R_2-seeds ≥ 1 seed-std on ≥ 1 dataset; else T_ret=1 in final.
- **S3** Linear probe sufficient (Block-L) — L-linear ≈ L-MLP within 1 seed-std; else escalate.
- **S4** Cross-dataset robustness on MGTAB.
- **S5** Cross-dataset calibration transfer (Block-CT) — CT-xfer drops ≤ 2 seed-std from CT-same.

Each claim has a pre-committed falsification path. Under any Claim-C partial failure, the corresponding Stage-5 schema element is dropped (role/order/collapse → contract to 3-tuple).

## First Runs to Launch

1. **Core machinery (W1–W3, ~ 16 GPU-hr)**: extend `GraphConformalSetEstimator` with 2-term Stage-1 composite; Stage-A calibration script with 5-fold `ê_φ` cross-fit + `T_valid_cal` + `τ_bucket` + `q_hat` + `τ*`; extend `EgoRefinementRepairOperator` with top-50% quantile retrieval + canonical ordering + ê_φ-gated accept + Stage-5 v1.0 serializer; pre-compute LLM embeddings (Qwen + Mistral) on hard nodes in train_cal ∪ valid_cal ∪ test.
2. **Block-E main matrix (W4, ~ 36 GPU-hr)**: 7 variants × 3 seeds × 2 datasets with cached LLM embeddings. Evaluate 4-part Claim C acceptance. If any sub-condition fails, apply pre-committed contraction immediately.
3. **Block-P / Block-R / Block-L transparency audit (W4–W5, ~ 44 GPU-hr)**: stage necessity + retrieval budget-match + probe sufficiency.
4. **Block-CT + MGTAB cross-dataset (W5, ~ 14 GPU-hr)**.

## Main Risks

- **Block-E (i) fails** (E-LLM-Qwen ≤ E-RoBERTa): evidence graph adds no signal; return to v7 EQC framing.
- **Block-E (iii) fails specific scramble**: that structural element dropped from Stage-5 schema; protocol shrinks (4-tuple → 3-tuple).
- **Block-E (iv) fails** (Qwen ≫ Mistral): protocol is model-specific; honest report.
- **Block-CT fails**: calibration partially dataset-specific.
- **ê_φ cross-fit disagreement**: simpler entropy fallback.

## Next Action

- **`/experiment-plan`** on `refine-logs/FINAL_PROPOSAL.md` (v8). Round-5 reviewer offered to generate a "one-page Block-E + CT result interpretation rubric" for the paper-writing phase.
- **Do not** attempt further `/research-refine` rounds — MAX_ROUNDS=5 reached and Round-5 reviewer explicitly stated "no further proposal polishing will move the needle". Remaining gap is empirical.

---

## Round-by-Round Score Summary

| Round | Difficulty | Overall | Verdict |
|---|---|---|---|
| v7-R1 through v7-R8 | standard + nightmare | 6.5 → 9.2 READY (nightmare) | v7 cycle: EQC as single-primitive controller, LLM optional |
| v8-R1 | standard | **6.675** | RETHINK (3 parallel novelties sprawl) |
| v8-R2/R3 | standard | **7.585** | REVISE (LOW drift; Stage-5 intellectual center designated) |
| v8-R4 | standard | **8.09** | REVISE (NONE drift; CQ ceiling — "schema engineering" risk) |
| v8-R5 | standard | **8.63** | **REVISE at MAX_ROUNDS; NONE drift; structurally complete; empirical gap only** |

v8 cycle: 5 rounds, RETHINK → REVISE monotone improvement (+1.955). READY requires empirical confirmation; `/research-refine` cannot produce it.
