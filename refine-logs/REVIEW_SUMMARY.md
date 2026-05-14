# Review Summary — Evidence-Graph Construction (v8 cycle; supersedes v7 EQC READY)

**Method**: Stage-5 4-tuple protocol for LLM-consumable graph-evidence artifacts. User rescoped LLM refiner from "optional" (v7) to "mandatory main stage" (v8); 7-stage pipeline hard-locked.
**Final score**: 8.63/10 REVISE at MAX_ROUNDS=5, NONE drift, structurally complete, empirical gap only.
**Supersedes**: v7 cycle artifacts (R1-R8 READY 9.2 nightmare for a different scope — LLM optional).

## v8 Cycle Rounds

| Round | Main concern | Resolution | Verdict |
|---|---|---|---|
| v8-R1 | 3 parallel novelties (retrieval loop + rewriter + LLM refiner) sprawl → dominant-contribution story diffuse | RETHINK verdict; 6.675/10; reviewer explicitly asks "demote two of three to plumbing" | RETHINK |
| v8-R2 | Apply RETHINK: designated Stage-5 evidence-graph construction as intellectual center; demoted Stages 1/3/4 to pre-committed plumbing; all 6 reviewer simplifications + 3 modernizations | Jump to LOW drift; CQ self-confirmation residual (Stage-4 argmax proxy) | REVISE 7.585 |
| v8-R3 | 2 CRITICAL + 2 IMPORTANT + 1 MINOR: scrambled-structure controls in Block-E; calibrated expected-error ê_φ; tightened 4-tuple definition of Stage-5; pre-registered token budget; param-count clarification | Applied all 5 | — |
| v8-R4 | CQ ceiling — protocol reads as "schema engineering" without ordering-invariance + LLM-family robustness; Claim C bar permissive | Promoted edge-order invariance to Block-E main; added 2-LLM (Qwen + Mistral) robustness; tightened Claim C to 4-part criterion; Block-CT to main secondary pillar; ê_φ 5-fold cross-fit; `retrieval_trail` demoted to appendix | REVISE 8.09 |
| v8-R5 | Final check at MAX_ROUNDS=5: structural vs empirical gap? | All structural blockers resolved; reviewer: "remaining gap is empirical, not structural... no further proposal polishing will move the needle" | REVISE 8.63 (terminal) |

## Final Status

- **Anchor**: user-locked 7-stage pipeline preserved across all 5 rounds. NONE drift.
- **Focus**: tight. One dominant contribution (Stage-5 4-tuple protocol), explicitly-non-novel plumbing.
- **Modernity**: appropriately frontier-aware. LLM as enhancer not predictor; 2-LLM robustness check; cross-dataset calibration transfer; probability-space disagreement.
- **Strongest parts of final method**:
  - Stage-5 protocol definable abstractly (schema + bucketization + canonical ordering + budget), independent of plumbing.
  - 4-part Claim C acceptance criterion makes the dominant claim cross-dataset + scramble + LLM-family robust.
  - `ê_φ` 5-fold cross-fit actually externalizes Stage-4 accept rule via held-out label supervision.
  - Pre-committed Block-E failure handling: any structural element failing scramble test is dropped from schema (protocol shrinks, not post-hoc defense).
- **Remaining weaknesses**:
  - CQ/MS ≥ 9 requires Block-E + Block-CT results to confirm the acceptance criteria. Proposal-polish exhausted.
  - Reviewer notes "a protocol whose value is in invariance/sensitivity tests" has an inherent CQ ceiling in top-venue perception unless empirical dominance is overwhelming.
  - `ê_φ` is still a meta-model over backbone-derived features; cannot detect confident-wrong regimes not represented in valid_cal.

**Problem**: Social bot detection residual errors concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage; scalar post-hoc calibration provides no operational ego-uncertainty signal for selective, budgeted, reversible ego refinement on a frozen Stage-1 RoBERTa + BotRGCN backbone.
**Initial Approach**: conformal ego-graph quality estimator → hard-node router → SKETCH/GAugLLM text-graph coupled modifier → soft ego reweight, per `research.md` (2026-05-09).
**Date**: 2026-05-09
**Rounds**: 6 / 5 (MAX_ROUNDS originally; user-directed rescope triggered Rounds 5–6 at nightmare difficulty).
**Final Score**: 9.2 / 10
**Final Verdict**: **READY (nightmare difficulty)**
**Supersedes**: `REVIEW_SUMMARY.md` v4 (2026-05-09 earlier this session) and v1 (2026-04-19 Embedding-Dominant charter); `PAPER_CHARTER.md` (2026-04-22 FRMI v2).

---

## Problem Anchor (verbatim, used across all 6 rounds)

- **Bottom-line problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage — not uniformly.
- **Must-solve bottleneck**: scalar post-hoc calibration yields no operational ego-uncertainty signal gating selective, budgeted, reversible ego refinement; binary edge-reliability methods collapse camouflage; global structure learning is non-local and non-rollbackable.
- **Non-goals**: global structure learning; LLM as final classifier; re-proving binary edge reliability; full causal-graph repair; new GNN architecture.
- **Constraints**: frozen Stage-1 RoBERTa + BotRGCN/RGCN backbone; no base-GNN retraining; soft-reweight existing ego edges only; existing project GPU budget; `max_iter = 1`.
- **Success condition**: on TwiBot-20 + TwiBot-22, triggered local ego refinement raises macro-F1 / reduces AURC on high-uncertainty ego slice without hurting global macro-F1 under matched budget; ablations isolate trigger / rewrite+gate mechanism / refiner readout.

---

## Round-by-Round Resolution Log

| Round | Difficulty | Overall | Main Reviewer Concerns | What This Round Changed | Solved? |
|-------|------------|---------|-------------------------|--------------------------|---------|
| 1 | standard | 6.5 | Binary-CP thinness; contribution sprawl; hand-mixed abstain-risk; inductive-CP overclaim; ablation bloat | Composite ego-quality score; router deleted; deterministic modifier in v1; CF → sensitivity (appendix); 3 claim blocks; "calibrated selection rule" framing | Partial |
| 2 | standard | 8.1 | `s_composite` under-defined; `s_tg` space-mismatch; 7-scalar "hidden training"; no non-conformal baseline; inductive-CP language | `s_composite = min_y s(v,y)` pinned; `s_tg` = JSD(p_LM ‖ p_GNN); `w_lbl ≡ 1` (7→6); T4-NS baseline; tuning protocol locked; edge cases specified; CP language dropped | Yes |
| 3 | standard | 8.8 | Locality not pinned; `virtual_context_edges` unused; "zero inference params" misleads | Stage-6 = L-hop-local + fail-loud test; virtual context dropped (Round 3); wording fixed; Pilot Gates A/B with pre-committed contraction | Yes (spec-level) |
| 4 | standard | 9.2 | None material | No changes | READY spec-level |
| **5** | **nightmare** | **8.6** | **USER RESCOPE** (4 locked decisions + 9 uncertainties) cost 0.6 pts; Stage-7/8 smuggles Stage-8 as second mechanism; BotBR/BECE differentiation weak; C4 ratio-only statistically awkward; 8-scalar search re-introduces hidden DOF | Applied user rescope (modifier locked to continuous utility; LLM branch restored as optional v1 ablation; artifact + virtual context edges restored; estimator claim tightened) | REVISE (rescope cost) |
| **6** | **nightmare** | **9.2** | None IMPORTANT | Round-5 feedback applied: Block-M + C5 hard kill gate; BotBR-public faithful reproduction + BR-sensitivity; C4 → non-inferiority δ=0.015 + 95% CI + dual criterion; τ+/τ- frozen as valid_cal quantiles (8→6 scalars); R123b CF-signal-permutation falsifier; strict kill rule on A-Modifier; Stage-8 main = L0/L2 only; "valid_cal-quantile-calibrated" language discipline | **READY nightmare** |

## Overall Evolution

- **Rounds 1–4 (standard)**: method went from "conformal + coupled modifier + retrieval + counterfactual supervision + LLM evidence prompt + iterative loop" (research.md full spec) to EQC² single-primitive controller with one deterministic modifier and zero base-backbone retraining. Primary moves: reframe non-conformity score as ego-quality composite (R1); probability-space JSD for `s_tg` (R2); L-hop-local Stage-6 with fail-loud test (R3); pilot-gate empirical handoffs (R3). Ended at 9.2 READY spec-level.
- **Rounds 5–6 (nightmare, user-directed rescope)**: user locked 4 scope decisions (modifier = continuous utility only; v1 main = artifact-only refiner; rewrite = soft on ego only; estimator claim = prediction-set uncertainty under ego context) and flagged 9 uncertainties. Round 5 at nightmare difficulty scored 8.6 REVISE — the restored Stage-7/8 artifact+refiner re-introduced sprawl risk. Round 6 added mechanism-isolation Block-M with the **C5 hard kill gate** (M-ro carries ≥ 0.5 of total lift), BR-public faithful reproduction, C4 non-inferiority CI framing, τ+/τ- quantile freeze, and strict kill rule on A-Modifier. Closed at 9.2 READY nightmare.

## Final Status

- **Anchor status**: **preserved** — reviewer confirmed NONE drift in every round, including after the user-directed rescope.
- **Focus status**: **tight by construction** — EQC² is dominant *iff* C5 holds; if C5 fails (Pilot C or Block-M), the paper pre-commits to recentering on artifact+refiner. Kill-gated bifurcation removes "one-of-four" ambiguity.
- **Modernity status**: **appropriately frontier-aware** — frozen RoBERTa + frozen LM/GNN heads inside probability-space JSD; no forced LLM components; LLM-enhanced branch is an optional ablation bounded by C4.
- **Strongest parts of final method**:
  - Same primitive (`s_composite`) used three ways — maximum economy of mechanism.
  - Non-conformal baseline (T4-NS) falsifies the set-based contribution in Block-T.
  - Mechanism-isolation block (Block-M) with C5 hard kill gate empirically bounds Stage-8 as commodity readout.
  - BotBR-public faithful reproduction + BR sensitivity directly addresses prior-art differentiation.
  - L-hop-local Stage-6 with fail-loud locality + virtual-context-isolation unit tests.
  - Strict kill rule on A-Modifier (R121/R122/R123/R123b) makes the learned-modifier appendix a credibility instrument.
  - No base-backbone retraining; all learning is calibration-time.
- **Remaining weaknesses (all empirical, not spec-level)**:
  - 5 empirical claims (C1–C5) and 3 pilot gates must pass. All have pre-committed falsification paths and fallback contractions.
  - Stage-8a artifact refiner (2-layer MLP trained on `train_cal` labels) has *perceptual* sprawl risk — mitigated by Block-M/C5 empirical bound and by presentational framing.
  - `s_rec` auto-encoder may not generalize across datasets — auto-drops via `w_rec → 0`.
  - Budget ~ 140–220 GPU-hr (up from Round-4 ~ 85–125 due to Block-M + Block-R promotion + BR-public + BR-sensitivity + Pilot C + D-EstSem).
