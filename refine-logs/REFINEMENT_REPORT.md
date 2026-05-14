# Refinement Report — Evidence-Graph Construction (v8 cycle supersedes v7 READY)

**Method**: v8 — Stage-5 4-tuple protocol for LLM-consumable graph-evidence artifacts. User rescoped LLM refiner from "optional" (v7) to "mandatory main stage" (v8); 7-stage pipeline hard-locked.
**Final score**: 8.63/10 REVISE at MAX_ROUNDS=5; NONE drift; structurally complete.
**Supersedes**: v7 EQC cycle (Rounds 1-8, READY nightmare 9.2 for "LLM-optional" scope).

## v8 Score Evolution

| Round | PF | MS | CQ | FL | FS | VF | VR | Overall | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| v8-1 | 8.0 | 6.5 | 6.0 | 7.0 | 7.0 | 6.5 | 6.5 | 6.675 | RETHINK |
| v8-3 | 8.5 | 7.3 | 7.1 | 7.8 | 8.1 | 7.4 | 7.2 | 7.585 | REVISE |
| v8-4 | 8.8 | 8.0 | 7.6 | 8.0 | 8.8 | 8.1 | 7.7 | 8.09 | REVISE |
| v8-5 | 9.0 | 8.5 | 8.6 | 8.3 | 9.2 | 8.6 | 8.2 | **8.63** | **REVISE (terminal)** |

Monotone +1.955 over 5 rounds.

## v8 Method Evolution Highlights

1. **R1 RETHINK → R2 intellectual-center designation**: picked Stage-5 evidence-graph construction as sole intellectual center; demoted Stages 1/3/4 to "pre-committed plumbing with explicit non-claims"; Stage-6 linear probe default (MLP only if ablation justifies).
2. **R3 CRITICAL fixes**: calibrated expected-error proxy `ê_φ` replaces argmax-confidence in Stage-4 accept (actually external via valid_cal labels); scrambled-structure controls (role/trail/collapse) promoted to Block-E main.
3. **R4 READY-shot lever (CQ ceiling move)**: edge-order invariance promoted to Block-E main; 2-LLM robustness (Qwen + Mistral); Claim C upgraded to 4-part cross-dataset-consistent criterion; `ê_φ` 5-fold cross-fit; Block-CT cross-dataset calibration transfer promoted to main secondary pillar.
4. **R5 terminal**: reviewer confirms "remaining gap is primarily empirical, not structural. No further proposal polishing will fix it."

## Pushback/Drift Log (v8)

| Round | Reviewer said | Response | Outcome |
|---|---|---|---|
| R1 | "3 parallel novelties dilute single-contribution story; demote 2 of 3 to plumbing" | Accepted fully. Designated Stage-5 as center. | Jumped to LOW drift |
| R3 | "Stage-4 argmax-confidence is still self-referential" | Accepted. Replaced with ê_φ cross-fit on valid_cal labels. | CRITICAL resolved |
| R4 | "Protocol-without-invariance tests reads as schema engineering" | Accepted. Added edge-order invariance + 2-LLM robustness + 4-part Claim C. | CQ +0.5 |
| R5 | "Empirical gap only" | Accepted terminal. | MAX_ROUNDS |

Zero drift; no pushback required (every reviewer suggestion directly addressed the anchor, not adjacent concerns).

## Remaining Weaknesses

- **CQ/MS ≥ 9 depends on empirical results**: Block-E + Block-CT must confirm acceptance criteria on both datasets and both LLMs.
- **Protocol-as-contribution ceiling**: top-venue reviewers may label any "schema + bucketization + ordering + budget" as schema engineering absent overwhelming empirical dominance.
- **`ê_φ` expressivity limit**: cannot detect backbone's confident-wrong regimes not represented in valid_cal. Mitigated by Block-CT + nested-split robustness but inherent.

## Next Action

- `/experiment-plan` on `FINAL_PROPOSAL.md` v8. Round-5 reviewer offered to generate "one-page Block-E + CT result interpretation rubric" for paper-writing phase.
- Do NOT rerun `/research-refine` — MAX_ROUNDS reached and reviewer explicitly stated no further proposal polishing will help.

---

**Problem**: Social bot detection residual errors concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage; scalar post-hoc calibration provides no operational ego-uncertainty signal for selective, budgeted, reversible ego refinement on a frozen Stage-1 RoBERTa + BotRGCN backbone.
**Initial Approach**: conformal ego-graph quality estimator → hard-node router → SKETCH/GAugLLM coupled modifier → soft ego reweight (`research.md` 2026-05-09).
**Date**: 2026-05-09
**Rounds**: 6 (4 at standard difficulty, 2 at **nightmare** difficulty)
**Final Score**: 9.2 / 10 at nightmare difficulty
**Final Verdict**: **READY**
**Supersedes**: v4 (2026-05-09 earlier in session); v1 (2026-04-19 Embedding-Dominant).

---

## Problem Anchor

[verbatim carried through all 6 rounds — see `REVIEW_SUMMARY.md` and `round-0-initial-proposal.md`]

## Output Files

- `refine-logs/REVIEW_SUMMARY.md` — round-by-round resolution log.
- `refine-logs/FINAL_PROPOSAL.md` — v5 clean final proposal.
- `refine-logs/REFINEMENT_REPORT.md` — this file.
- `refine-logs/RESEARCH_AUDIT.md` — research.md ↔ deliverables conformance audit.
- `refine-logs/EXPERIMENT_PLAN.md` — v5 experiment plan.
- `refine-logs/EXPERIMENT_TRACKER.md` — v5 tracker.
- `refine-logs/PIPELINE_SUMMARY.md` — v5 pipeline summary.
- `refine-logs/score-history.md` — full score evolution.
- `refine-logs/round-0-initial-proposal.md`, `round-{1..6}-review.md`, `round-{1..5}-refinement.md` — per-round artifacts.
- `refine-logs/REFINE_STATE.json` — `status: "completed"`.

## Score Evolution

| Round | Difficulty | PF | MS | CQ | FL | FS | VF | VR | Overall | Verdict |
|-------|------------|----|----|----|----|----|----|----|---------|---------|
| 1 | standard | 8 | 6 | 6 | 7 | 6 | 6 | 6 | 6.5 | REVISE |
| 2 | standard | 9 | 8 | 8 | 8 | 8 | 8 | 6 | 8.1 | REVISE |
| 3 | standard | 9 | 9 | 9 | 8 | 9 | 9 | 8 | 8.8 | REVISE |
| 4 | standard | 10 | 10 | 9 | 8 | 9 | 9 | 8 | 9.2 | READY (spec-level, conditional on Pilot Gates A/B) |
| 5 | **nightmare** | 9.4 | 9.2 | 8.1 | 8.4 | 8.0 | 8.5 | 6.8 | 8.6 | REVISE (rescope cost) |
| 6 | **nightmare** | 9.6 | 9.5 | 9.1 | 8.9 | 8.7 | 9.4 | 9.1 | **9.2** | **READY** |

PF = Problem Fidelity · MS = Method Specificity · CQ = Contribution Quality · FL = Frontier Leverage · FS = Feasibility · VF = Validation Focus · VR = Venue Readiness.

## Round-by-Round Review Record

| Round | Main Concerns | What Was Changed | Result |
|-------|----------------|------------------|--------|
| 1 | Binary-CP thinness; contribution sprawl; hand-mixed abstain risk; inductive-CP overclaim; ablation bloat | Composite ego-quality score; router deleted; deterministic modifier in v1; CF → sensitivity (appendix); 3 claim blocks; "calibrated selection rule" framing | 6.5 → 8.1 (Venue Readiness stuck at 6) |
| 2 | `s_composite` under-defined; `s_tg` space-mismatch; 7-scalar hidden-training; no non-conformal baseline; inductive-CP language | `s_composite = min_y s(v,y)` pinned; JSD(p_LM ‖ p_GNN); `w_lbl ≡ 1` (7→6); T4-NS baseline; tuning protocol locked; edge cases specified; CP language dropped | 8.1 → 8.8 (VR 6 → 8) |
| 3 | Locality not pinned; `virtual_context_edges` unused; "zero inference params" misleads | Stage-6 = L-hop-local + fail-loud test; virtual context dropped (Round 3); wording fixed; Pilot Gates A/B with contraction | 8.8 → 9.2 (READY spec-level) |
| 4 | None material (reviewer signoff) | — | READY spec-level |
| **5** | **USER RESCOPE** (4 locked decisions + 9 uncertainties); Stage-7/8 smuggles Stage-8; BR differentiation weak; C4 ratio-only; 8-scalar hidden DOF returning | Applied user rescope (modifier=continuous utility only; v1 main=artifact-only refiner; rewrite=soft on ego only; estimator claim=prediction-set uncertainty under ego context) | 9.2 → 8.6 (nightmare REVISE; rescope cost) |
| **6** | None IMPORTANT (reviewer signoff at nightmare) | Block-M + C5 hard kill gate; BR-public + BR-sensitivity; C4 non-inferiority (δ=0.015, 95% CI, dual criterion); τ+/τ- quantile freeze (8→6 scalars); R123b CF-signal-permutation; strict kill rule on A-Modifier; Stage-8 main = L0/L2; "valid_cal-quantile-calibrated" language | **8.6 → 9.2 READY nightmare** |

## Final Proposal Snapshot (v5)

Canonical clean version: `refine-logs/FINAL_PROPOSAL.md` (v5). Final thesis in six bullets:

- **EQC²** — one composite non-conformity score `s(v,y) = s_lbl(v,y) + w_tg·JSD(p_LM ‖ p_GNN) + w_het·s_het + w_rec·s_rec` (`w_lbl ≡ 1`, τ+/τ- frozen as valid_cal quantiles, `q_hat` + `τ*` on `valid_cal`).
- The **same primitive** `s_composite(v) = min_y s(v,y)` is used three ways: lexicographic trigger, rewrite-amplitude factor, and L-hop-local accept/rollback gate.
- The rewrite is a **deterministic continuous-utility** `u_edge` modifier + single-line clip-based soft reweight over existing ego edges only, followed by **L-hop-local Stage-6** with a fail-loud locality unit test.
- The paper's **set-based contribution** is explicitly falsifiable via T4-NS in Block-T, and the **rewrite+gate contribution** is explicitly falsifiable via C5 hard kill gate in Block-M (`M-ro − M-base ≥ 0.5 · (M-rr − M-base)` on both datasets).
- **No base-backbone retraining.** All learning is calibration-time: one AE on `train_cal`, one 6-scalar coordinate-descent on `valid_cal`, one `q_hat`, one `τ*`, two frozen `τ+/τ-` quantiles, Stage-7 artifact serialization (zero params), Stage-8a artifact-only refiner MLP on `train_cal` labels.
- **Optional v1 LLM branch** (Stage-8b) tested via C4 non-inferiority: `slice_F1(L2) − slice_F1(L0) ≤ 0.015` with 95% CI, AND `slice_F1(L0) ≥ 0.90 · slice_F1(L2)`.

## Method Evolution Highlights

1. **Most important simplification / focusing move** — Round 1 contribution consolidation: deleted trainable router, demoted learned modifier to appendix, collapsed ablation matrix to three claim blocks. Turned "a system of reasonable parts" into a single-primitive controller thesis.
2. **Most important mechanism upgrade** — Round 1 reframe of the non-conformity score as an ego-quality composite (Round 2 formalized via JSD). Structural answer to the reviewer's binary-CP-thinness concern; without it, the controller degrades to a label-only CP wrapper.
3. **Most important modernization / simplicity justification** — Round 3 locality pinning (L-hop-local Stage-6 with fail-loud test) + Round 3 pilot-gate handoff, plus Round 6 Block-M + C5 hard kill gate (binds Stage-8 as commodity readout, empirically, not rhetorically).
4. **Most important rescope absorption** — Round 5 user-directed rescope (restoring LLM branch + artifact + virtual context edges + locking modifier to continuous utility) cost 0.6 points at nightmare difficulty. Round 6 recovered it fully (9.2) via Block-M + C5 + BR-public + C4 non-inferiority + τ freeze + strict A-Modifier kill rule.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | Implicit: "You reintroduced scalar confidence in a new wrapper." | Accepted as legitimate signal; fixed mechanically via composite score (not argued). | Accepted. |
| 2 | "Rename 'conformal'; quantile-calibrated set-based controller is safer." | Partial: kept "EQC²" as short name, long form "Ego-Quality Quantile-Calibrated Controller"; no coverage claim; inductive-CP caveat. | Partial pushback; reviewer did not re-object. |
| 3 | "Show T4 > T4-NS / G1 > G2 (empirical)." | Cannot land inside `/research-refine`. Packaged as Pilot Gates A/B with pre-committed contraction. | Accepted Round 4: "Yes for /research-refine stage; falsifiable pilot gates resolve spec-level blocker." |
| 5 | Nightmare reviewer: "Stage-8 is a smuggled contribution; EQC² risks being one-of-four." | Accepted: added Block-M + C5 hard kill gate; demoted Stage-8 to commodity readout bounded by C5. | Reviewer Round 6: "Yes, under your own governance: EQC² is dominant by construction." |
| 5 | "C4 ratio-only is statistically awkward." | Accepted: non-inferiority CI framing + dual criterion. | Round 6: "Meets the 2026 bar in a way that is review-proof." |
| 5 | "BR / BR-soft may not be sufficient BotBR differentiation." | Accepted: BR-public faithful reproduction attempt + documented incompatibility report + BR/BR-soft sensitivity. | Round 6: "Unusually strong and should disarm novelty attacks." |
| 5 | "8-scalar search re-introduces hidden-DOF." | Accepted: τ+/τ- frozen as valid_cal quantiles (75th/25th); search back to 6 scalars. | Round 6: "Package that closes quiet hyperparameter fishing." |
| — | Cumulative: reviewer never requested LLM-as-classifier, binary-reliability headline, global-graph-rewrite. | No pushback needed. | Anchor preserved across all 6 rounds (NONE drift in every round). |

**Drift Warning accumulation across rounds**: Round 1 partial (binary-CP thinness, resolved); Rounds 2–6: **NONE**.

## Remaining Weaknesses

- **All empirical, zero spec-level**. Five claims (C1–C5) and three pilot gates (A, B, C) must pass. Every one has pre-committed falsification paths and contraction plans. If any gate fails, the paper pre-commits to the narrower framing — no patching with extra modules.
- **Compute budget grew**: Round-4 ~ 85–125 GPU-hr → Round-6 ~ 140–220 GPU-hr. Increase attributed to Block-M (NEW, ~ 20 GPU-hr), Block-R promoted from appendix (~ 25 GPU-hr), Block-BR-public + Block-BR-sensitivity (~ 15–30 GPU-hr), Pilot C (~ 2 GPU-hr), D-EstSem diagnostic (~ 3 GPU-hr), and paired-bootstrap CI analysis.
- **Stage-8a perceptual risk**: the artifact refiner MLP is a new supervised component. Mitigated empirically by Block-M (M-re variant shows the refiner alone does not carry the gain) and bounded by C5 and C4.

---

## Raw Reviewer Responses

<details>
<summary>Round 1 Review — standard (overall 6.5, REVISE)</summary>

High-Level Take: post-hoc, local, budgeted, reversible refinement on frozen backbone. Main risk: conformal ego-quality is conformalized label uncertainty; in binary bot detection set geometry collapses. Secondary risk: contribution sprawl by gravity — CF-supervised coupled modifier becomes the real "thing".

Scores: 8 / 6 / 6 / 7 / 6 / 6 / 6. Overall 6.5. Verdict REVISE.

Key fixes demanded: (1) define ONE deterministic `g(v)` and remove trainable router; (2) make main contribution = conformal-triggered + conformal-rollbacked local refinement; demote learned modifier; add deterministic modifier baseline; (3) cap CF supervision to top-K edges or use gradient attribution; (4) collapse ablation grid to (trigger swap, gate on/off, cross-dataset once); (5) reframe as "calibrated selection rule" rather than CP coverage; make non-conformity score quality-aware.

Simplifications: delete trainable router; deterministic modifier baseline; single monotone scalar abstain risk. Modernization: enrich non-conformity score with quality-specific signals; gradient attribution instead of CF reruns. Drift partial.

Answers: E4 ablation-only; CF = "model-based sensitivity supervision" (not causal); abstain risk hand-mix unnecessary; MGTAB preferred over Cresci-17.

</details>

<details>
<summary>Round 2 Review — standard (overall 8.1, REVISE)</summary>

Anchor preserved. Binary-CP thinness materially addressed via quality-aware composite score. Focus sharper after router deletion and appendix demotion.

Scores: 9 / 8 / 8 / 8 / 8 / 8 / 6. Overall 8.1. Verdict REVISE.

Venue Readiness (6, CRITICAL): paper reads as "well-tuned composite heuristic + quantile threshold" unless (1) non-conformal baseline added (T4-NS), (2) tuning protocol airtight, (3) language shifts from "conformal" to "quantile-calibrated".

Action items: formal `s_composite` and `margin` definitions; `s_tg` representation-mismatch fix (RoBERTa vs GNN embedding different spaces — switch to probability-space divergence); add no-set baseline; lock tuning protocol; specify binary-set edge cases.

Simplifications: drop AE unless additive; `w_lbl = 1` (7→6 scalars); "Quantile-Calibrated Ego-Quality Controller". Modernization: NONE (already modern). Drift NONE.

Signoff: deterministic modifier as supporting contribution ACCEPTABLE; `max_iter=1` v1 main ACCEPTABLE.

</details>

<details>
<summary>Round 3 Review — standard (overall 8.8, REVISE)</summary>

Anchor preserved. Prior VR blocker substantially addressed by (i) formal definitions, (ii) JSD fix, (iii) locked tuning protocol + leakage asserts, (iv) T4-NS falsifier.

Scores: 9 / 9 / 9 / 8 / 9 / 9 / 8. Overall 8.8. Verdict REVISE.

Remaining items: (1) show T4 > T4-NS (empirical pilot), (2) show G1 > G2 (empirical pilot), (3) specify locality of recomputation after reweighting.

Simplifications: drop `virtual_context_edges` from v1 main; tighten "zero trainable inference params" wording. Modernization NONE. Drift NONE.

"If those three land, this moves from excellent, tight proposal to plausibly READY."

</details>

<details>
<summary>Round 4 Review — standard (overall 9.2, READY spec-level)</summary>

Anchor preserved. Locality of recomputation resolved and auditable (L-hop-for-L-layer consistent with standard message-passing dependency structure, DGL neighbor-sampling docs cited). Set-based necessity + gate necessity both falsifiable via Pilot Gates A/B with pre-committed contraction.

Scores: 10 / 10 / 9 / 8 / 9 / 9 / 8. Overall 9.2.

Verdict: **READY (spec-level; conditional on Pilot Gates A/B)**. Paper-level READY conditional on pilots landing.

Drift NONE. Simplifications NONE (material). Modernization NONE.

Remaining action items are execution-phase: run Pilot A and Pilot B; log locality / audit metrics; freeze tuning protocol if pilots pass; follow pre-committed contraction if pilots fail.

</details>

<details>
<summary>Round 5 Review — NIGHTMARE (overall 8.6, REVISE)</summary>

Anchor preserved; no drift. Rescope cost 0.6 points relative to Round 4. Primary damage: Venue Readiness 8 → 6.8.

Scores: 9.4 / 9.2 / 8.1 / 8.4 / 8.0 / 8.5 / 6.8. Overall 8.6. Verdict REVISE.

Venue Readiness (6.8, IMPORTANT): Stage-7/8 risks reading as second mechanism competing with EQC²; reviewers will attribute gains to refiner rather than controller.

Three IMPORTANT items: (1) mechanism-isolation block (NoRewrite+Refiner vs RewriteOnly vs Rewrite+Refiner); (2) BotBR/BECE faithful reproduction OR explicit incompatibility + BR/BR-soft sensitivity; (3) upgrade C4 to non-inferiority (absolute margin + CIs). One MINOR: freeze τ+/τ- as valid_cal quantiles.

Simplifications: Stage-8 main = L0 / L2 only; A-Modifier fully kill-gated (any R121/R122/R123 fire ⇒ drop entire section); freeze τ. Modernization: non-inferiority framing; AURC curves; 3-seed floor.

Direct answers: Block-M addresses Stage-8-smuggled-contribution; C5 threshold 0.5 just right; C4 ratio-only somewhat lenient; R121/R122/R123 close to sufficient, add CF-permutation (R123b); 8-scalar tractable but perception risk returns; BR/BR-soft sufficient with BR-public + sensitivity; Block-R promotion justified.

</details>

<details>
<summary>Round 6 Review — NIGHTMARE (overall 9.2, READY)</summary>

Anchor preserved; no drift. Every load-bearing claim (C1–C5, S1–S3) has an explicit falsification path (or hard kill gate).

Scores: 9.6 / 9.5 / 9.1 / 8.9 / 8.7 / 9.4 / 9.1. Overall 9.2.

Verdict: **READY (nightmare difficulty)**. Zero dimensions < 7. Zero CRITICAL/IMPORTANT action items.

Simplifications NONE (v5 already removed main sprawl sources). Modernizations MINOR (×2): report C5 with paired-bootstrap CI on difference-of-differences; pre-register one fixed intervention ratio (or small set like {5%, 10%}) for AURC. Drift NONE.

Direct answers to audit (a)-(g):
(a) Block-M + C5 hard kill gate cleanly separates rewrite/controller from supervised refiner. EQC² dominant by construction.
(b) C5's 0.5 threshold just right (not 0.67+).
(c) C4 non-inferiority with δ=0.015 + 95% CI meets 2026 bar.
(d) Strict kill rule on learned-modifier correctly honest; converts R114 into credibility instrument.
(e) τ+/τ- freeze + 6-scalar + stability plots + disclosed coordinate-descent is the package that closes hidden-DOF concern.
(f) BR-public + incompatibility report + BR-sensitivity at venue bar; "unusually strong decision rule" disarms novelty attacks.
(g) EQC² unambiguously dominant by construction. Only residual risk is presentational.

Remaining action items NONE (spec-level). Execution pre-committed by Pilot Gates A/B/C and decisive blocks.

</details>

---

## Next Steps

- **Immediate (execution phase)**: run `/experiment-plan` output (`refine-logs/EXPERIMENT_PLAN.md`) via `/run-experiment`.
- **Week 0 (pilot gates)**: execute **R105** (Pilot A: T4 vs T4-NS), **R106** (Pilot B: G1 vs G2), **R132** (Pilot C: M-base vs M-ro) on TwiBot-20 dev, single seed, ~ 6 GPU-hr total. Each gate pre-commits to a fallback contraction plan.
- **Weeks 1–5 (conditional on all 3 pilots passing)**: core machinery → Block-T + Block-G → Block-M + Block-R → Block-X on MGTAB → Block-BR-public + Block-BR-sensitivity → appendices (A-Modifier strict-kill-gated by R121/R122/R123/R123b; A-Prompt conditional on C4) → analysis + paper draft via `/paper-plan` → `/paper-write`.
- **Post-execution (conditional on all 5 claims holding)**: paper submission to NeurIPS / ICML / ICLR 2026 venue per the READY nightmare-difficulty verdict.
