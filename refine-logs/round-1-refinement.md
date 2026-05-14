# Round 1 Refinement

**Round**: 1
**Review source**: `refine-logs/round-1-review.md` (overall 6.5, REVISE)
**Headline revision**: mechanism-level reframe — the non-conformity score itself becomes **ego-quality-aware**, solving the binary-CP thinness critique and sharpening the dominant contribution into a single composite controller (EQC²). Learned modifier + counterfactual supervision demoted to ablation. Trainable router deleted. Ablation matrix collapsed to three claim blocks.

---

## Problem Anchor (verbatim from Round 0)

- **Bottom-line problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage — not uniformly.
- **Must-solve bottleneck**: Scalar post-hoc calibration (MSP / entropy / margin / temperature / GATS / CaGCN) produces a confidence number but no **operational ego-graph-quality signal** that can gate selective, budgeted, reversible ego refinement. Hard edge-reliability methods (BotBR SIGIR'25, BECE TNNLS'25) collapse bot-human heterophily into binary reliable/unreliable, destroying camouflage evidence. Global graph structure learning (LDS, Pro-GNN, IDGL) is too expensive, non-local, non-rollbackable on a frozen backbone.
- **Non-goals**: global graph structure learning; LLM as final classifier; re-proving binary edge reliability; full causal-graph repair; a new GNN architecture.
- **Constraints**: frozen Stage-1 RoBERTa + BotRGCN/RGCN backbone; no base-GNN retraining; v1 no LLM calls; soft-reweight existing ego edges only; existing project GPU budget; main method `max_iter = 1`.
- **Success condition**: on TwiBot-20 + TwiBot-22, ego-quality-triggered local ego refinement raises macro-F1 and/or reduces AURC / average set size on the high-quality-risk slice without hurting global macro-F1, under matched intervention budget. Ablations must separately isolate (a) the quality trigger, (b) the refinement content, (c) the accept/rollback gate.

---

## Anchor Check

- **Original bottleneck still targeted**: yes. The revised non-conformity score makes the trigger *more* local-ego-quality-focused, not less. The dominant mechanism is still a post-hoc, reversible, budgeted local refiner on a frozen backbone.
- **Why the revised method still addresses it**: the composite non-conformity score explicitly encodes local heterophily + text-graph disagreement + ego reconstruction error alongside label non-conformity. This directly responds to the reviewer's concern that label-only CP collapses to near-scalar on binary tasks — the controller is now reading ego-quality, not just posterior uncertainty.
- **Reviewer suggestions rejected as drift**:
  - None. All of the reviewer's CRITICAL / IMPORTANT fixes are accepted in spirit or integrated below.
  - The reviewer-offered "reframe the controller as a calibrated selection rule rather than making full CP-validity claims" is **accepted explicitly**: we drop marginal-coverage claims and reframe the controller as a *calibrated selection rule* with inductive-CP caveats cited.

## Simplicity Check

- **Dominant contribution after revision**: EQC² — an **Ego-Quality Conformal Controller** that (a) emits a per-node selection score from a conformalized composite non-conformity score combining label non-conformity + text-graph disagreement + local heterophily + ego-embedding reconstruction error, and (b) uses the SAME selection score as an accept/rollback gate for a single-pass soft ego rewrite. One primitive, used as both trigger and gate.
- **Components removed or merged**:
  - **Trainable router** → deleted. Replaced by deterministic lexicographic rule `g(v)` on `(set_size, margin)` plus quality-composite threshold fit on `valid_cal`.
  - **Hand-mixed `0.70·size_risk + 0.30·margin_risk`** → deleted. Replaced with a single monotone scalar + one threshold (lexicographic fallback on ties).
  - **Counterfactual edge supervision via GNN reruns** → renamed to **model-based sensitivity attribution**, implemented as single-backward-pass edge-gradient proxy. No per-edge GNN rerun loop in v1 main.
  - **3-head learned modifier (s_reliability / u_edge / p_role)** → demoted to ablation; v1 main uses a **deterministic modifier** over frozen signals (cosine text similarity + relation-type weight + text-graph disagreement indicator). Learned modifier is an optional upgrade, gated on whether the deterministic baseline already moves the needle.
- **Reviewer suggestions rejected as unnecessary complexity**: none; we applied all reviewer simplifications.
- **Why the remaining mechanism is still the smallest adequate route**: only ONE primitive is trainable at all (the optional modifier, ablation-only); the v1 main method has **zero trainable components** when the deterministic modifier is used. The controller is a pure post-hoc calibration + selection rule on top of a frozen backbone.

---

## Changes Made

### 1. Binary-CP thinness → **quality-aware composite non-conformity score**

- **Reviewer said**: "In binary bot detection, CP sets are usually just {bot}, {human}, or {bot,human}, so set geometry can collapse to a near-scalar threshold signal unless the score is tied more directly to ego-graph quality / text-graph conflict." (CRITICAL, Venue Readiness).
- **Action**: Redefine the non-conformity score itself as an ego-quality composite.
  - `s_lbl(v, y) = 1 - p_base(y | v)` — standard label non-conformity (retained).
  - `s_tg(v)    = 1 - cos(X_R[v], h_GNN[v])` — text-graph disagreement, using frozen RoBERTa embedding vs frozen GNN node embedding.
  - `s_het(v)   = 1 - (#{u in N(v): y_hat(u) = y_hat(v)}) / max(deg(v), 1)` — local heterophily of *predicted* labels inside the ego (no test label access).
  - `s_rec(v)   = || x_R[v] - Decode(Encode(x_R[v], Ego_v)) ||` — ego-embedding reconstruction error from a small frozen auto-encoder trained on train_cal only (one-time 5-minute MLP fit; counts as zero trainable components for the backbone pipeline).
  - **Composite**: `s(v, y) = w_lbl · s_lbl(v, y) + w_tg · s_tg(v) + w_het · s_het(v) + w_rec · s_rec(v)`, with weights `(w_lbl, w_tg, w_het, w_rec)` fit by ONE constrained scalar search on `valid_cal` to maximize hard-node-slice recall at target coverage — no per-dimension tuning. Weights are reported in `calibration_metadata`.
- **Reasoning**: makes the CP primitive read ego-quality directly; escapes binary-label-geometry collapse; retains calibration discipline (test labels never enter the score or the weight search).
- **Impact on core method**: the controller is now substantively different from any scalar uncertainty estimator. This is the mechanism-level upgrade the reviewer demanded.

### 2. Contribution sprawl → **single-primitive controller + deterministic modifier default**

- **Reviewer said**: "Make the main paper contribution conformal-triggered + conformal-rollbacked local refinement. Demote the learned modifier. Provide a deterministic modifier baseline." (CRITICAL, Contribution Quality).
- **Action**:
  - **v1 main method modifier** is now **deterministic**: `modifier_score(e) = α_text · cos(X_R[v], X_R[u]) + α_rel · 𝟙[rel(e) ∈ R_trusted] + α_tgd · disagreement(v, u)`, where `R_trusted` is a dataset-declared relation subset (e.g. TwiBot `follow` / `mention`) and `disagreement` is a 3-valued indicator on sign of `s_tg`. Coefficients `(α_text, α_rel, α_tgd)` are fit by the same one-shot scalar search on `valid_cal` — ONE joint tuning pass for the controller + modifier, not two.
  - Learned modifier moves to a **single-head continuous scorer ablation**; removed from the v1 paper's headline contribution list.
  - Dominant contribution is now named and scoped: **EQC² — Ego-Quality Conformal Controller**, which is a *selection rule* and an *accept/rollback gate*, both driven by the same conformalized composite score. The refinement content (what edges get reweighted) is a commodity plugin the controller drives.
- **Reasoning**: the paper now reads as one sharp mechanism; the refinement module is a replaceable downstream consumer rather than a parallel contribution.
- **Impact on core method**: zero trainable components in the v1 main pipeline. All "learning" is reduced to (i) the one-shot auto-encoder for `s_rec` (5 min MLP fit on train_cal) and (ii) the scalar weight search.

### 3. Counterfactual GNN reruns → **gradient-attribution sensitivity (ablation only)**

- **Reviewer said**: "Cap supervision to top-K candidate edges per hard node OR replace counterfactual reruns with a single backward pass attribution proxy." (IMPORTANT, Feasibility). Also: "Rename counterfactual supervision as model-based sensitivity supervision; not causal identification." (Open Q2).
- **Action**: In ablation only, supervise the learned modifier via edge-weight gradient attribution on the frozen backbone: `d L_v / d w_e` computed in a single backward pass at the frozen checkpoint. This is called **model-based sensitivity supervision** in the paper (never "counterfactual"). Candidate edges per hard node capped at `K = 8` (existing ego edges, ranked by text similarity). No repeated GNN forward passes.
- **Reasoning**: compute is linear in `|hard_nodes| · K` backward hops; fits inside the project GPU budget; the framing is model-internal sensitivity, not causal.
- **Impact on core method**: since the learned modifier is ablation-only, this whole training recipe is optional for the headline claim.

### 4. Trainable router → **deterministic lexicographic selection rule**

- **Reviewer said**: "Define ONE trigger score g(v) deterministically; remove the trainable router in v1." (CRITICAL, Method Specificity).
- **Action**: `g(v) = (set_size(v), -coverage_margin(v), -s_composite(v))` evaluated lexicographically against a `valid_cal`-tuned threshold `τ*`. Hard nodes = nodes with `g(v) ≥ τ*`. `residual_risk_manifest` is kept only as a provenance log (what selection signal fired), not as a second learned head.
- **Reasoning**: one primitive, one threshold, one audit trail.
- **Impact on core method**: eliminates an entire trainable head from the pipeline.

### 5. Ablation bloat → **three claim blocks**

- **Reviewer said**: "Keep only what proves the controller thesis: (1) trigger swap, (2) rollback gate on/off, (3) cross-dataset once." (IMPORTANT, Validation Focus).
- **Action**: the Claim-Driven Validation Sketch in the revised proposal has exactly three blocks (see below). Everything else (modifier output form B0/B1/B2, learned-vs-deterministic modifier, E4/E5 diffusion, Cresci-17) moves to an **appendix-only** bucket. The appendix is not advertised as ablation mandatory for the main claim.
- **Reasoning**: reviewer fatigue is a real risk at NeurIPS/ICML/ICLR; the controller thesis is provable in three blocks.
- **Impact on core method**: none; scope only.

### 6. CP-validity overclaim → **calibrated selection rule framing + inductive-CP caveat**

- **Reviewer said**: "Conformal validity language can draw scrutiny on graph exchangeability / inductive issues; standard CP guarantees can break in inductive GNN regimes due to score shifts from message passing." (IMPORTANT, Venue Readiness).
- **Action**: We now frame EQC² as a **calibrated selection rule**, not a coverage-guaranteed conformal predictor. We explicitly acknowledge that under inductive message-passing regimes, marginal coverage is not guaranteed; we report empirical coverage on a held-out `test_cal` split as diagnostic rather than as a theorem.
- **Reasoning**: sidesteps the inductive-CP trap entirely without surrendering the controller mechanism.
- **Impact on core method**: none; framing only, though it affects how Section 2 and Section 10 of the final paper are written.

### 7. Dataset choice → **MGTAB replaces Cresci-17 as secondary robustness**

- **Reviewer said**: "Prefer MGTAB — explicitly graph-based, multi-relational (7 relation types); better stress-tests edge reweighting / heterophily-camouflage." (Open Q4).
- **Action**: secondary robustness target = MGTAB (7 relations, multi-relational). TwiBot-22 remains primary robustness; TwiBot-20 remains primary development.
- **Reasoning**: relation-type diversity directly stresses the `R_trusted` relation gating and the heterophily-camouflage story.

---

## Revised Proposal

# Research Proposal: EQC² — Ego-Quality Conformal Controller for Post-Hoc Local Ego Refinement in Social Bot Detection

**Version**: Round 1 refinement
**Date**: 2026-05-09

## Problem Anchor

> [as above, verbatim]

## Technical Gap

The frozen RoBERTa → BotRGCN pipeline leaves two classes of residual error: **ego-quality errors** (noisy / camouflaged ego) and **text-graph conflict errors** (RoBERTa and GNN evidence disagree). Naive fixes (scalar calibration, binary edge reliability, global structure learning, LLM-as-classifier, selective LLM consult) either lack an ego-quality handle, duplicate BotBR/BECE, demand retraining, or violate anchor constraints. The missing primitive is a **post-hoc ego-quality estimator** that provides a single calibrated selection score usable both as a hard-node trigger *and* as an accept/rollback gate for local refinement.

**Critically**, in binary bot detection the naive label-only conformal geometry collapses to a near-scalar signal. The primitive only becomes meaningful when its non-conformity score itself encodes ego-quality. That is the core methodological move.

## Method Thesis

**One-sentence thesis**: An ego-quality-aware conformal non-conformity score, reused as both selection trigger and accept/rollback gate, enables budgeted reversible local ego refinement on a frozen backbone — dominating scalar uncertainty triggers without duplicating binary edge-reliability literature and without requiring LLM calls or global structure learning.

- **Smallest adequate intervention**: zero new trainable parameters in the v1 main pipeline (one optional 5-minute auto-encoder for `s_rec`); one monotone selection rule; one soft-reweight rewrite arithmetic; one accept/rollback rule.
- **Frontier-native posture**: foundation-model-era RoBERTa embeddings used as a frozen semantic prior inside the composite score (for `s_tg` and `s_rec`) and inside the deterministic modifier (cosine text similarity). No LLM classifier, no graph-prompted LLM reasoning.

## Contribution Focus

- **Dominant contribution (one sentence)**: EQC² — a calibrated ego-quality composite non-conformity score that serves simultaneously as a hard-node selection rule and as an accept/rollback gate for post-hoc local ego refinement, sidestepping binary-CP thinness by construction.
- **Supporting contribution (optional)**: a minimal, deterministic text-graph modifier over existing ego edges that converts the EQC² selection into a budgeted soft reweight without adding trainable parameters.
- **Explicit non-contributions**: no binary edge-reliability headline (BotBR/BECE territory); no LLM-understands-graphs claim (TMLR 2024 counter-evidence); no global structure learning; no causal identification; no new SOTA detector.

## Proposed Method

### Complexity Budget

| Slot | v1 main | v2 / ablation |
|------|---------|---------------|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN/RGCN base GNN; existing ego adjacency; kNN semantic index | — |
| One-time fit | (i) `s_rec` auto-encoder MLP (2 layers, ~5 min CPU); (ii) scalar weight-vector `w = (w_lbl, w_tg, w_het, w_rec, α_text, α_rel, α_tgd)` by one constrained search on `valid_cal` | Learned modifier (single-head continuous scorer) — ablation only, gradient-attribution supervised |
| Not trainable at all | Composite non-conformity score; lexicographic selection rule; soft-reweight arithmetic; accept/rollback comparator | DAPS/NAPS diffusion, SNAPS semantic-kNN aggregation — ablation only |
| Excluded | LLM evidence prompt (v2); global structure learning; hard delete; causal identification | — |

Total trainable parameters in v1 main: **effectively zero** (the `s_rec` auto-encoder is a one-time frozen artifact; the weight vector is a 7-scalar search).

### System Overview

```
Stage-0 (frozen)  RoBERTa X_R → BotRGCN → base_probs(v), h_GNN(v)
Stage-A (one-time, CPU)  Fit s_rec auto-encoder on train_cal;
                         grid-/coordinate-search weights w on valid_cal
                         → τ*, R_trusted, w_*

Stage-1 (per node)  s(v, y) = w_lbl · s_lbl(v,y) + w_tg · s_tg(v)
                            + w_het · s_het(v) + w_rec · s_rec(v)
                    prediction_set(v) = {y : s(v, y) ≤ q_hat}   (q_hat on cal split)
                    set_size(v), margin(v), s_composite(v)

Stage-2 (deterministic trigger)  g(v) = (set_size(v), -margin(v), -s_composite(v))
                                 hard(v) = g(v) ≥_lex τ*
                                 provenance: residual_risk_manifest(v)

Stage-3 (retrieval, hard v only)  propagation_edges(v) = existing ego edges
                                  virtual_context_edges(v) = kNN_RoBERTa \ ego   (evidence-only)

Stage-4 (deterministic modifier)  modifier_score(e) = α_text · cos(X_R[v], X_R[u])
                                                    + α_rel  · 𝟙[rel(e) ∈ R_trusted]
                                                    + α_tgd  · disagreement(v, u)

Stage-5 (soft rewrite)  w_e' = w_e · clip(1 + λ · budget(v) · modifier_score(e)
                                              · (1 - s_composite(v)), 1-λ, 1+λ)
                        budget(v) = min(B_max, κ · s_composite(v))

Stage-6 (accept / rollback)  re-evaluate Stage-1 on Ego_v';
                             accept iff s_composite(v) strictly decreases
                                        AND set_size(v) does not increase;
                             else rollback.
```

All of this is single-pass (`max_iter = 1`). No dropout of components, no second training loop.

### Core Mechanism — EQC² Score

Define four normalized-to-`[0,1]` components:

- `s_lbl(v, y) = 1 - p_base(y | v)` — standard label non-conformity from the frozen classifier.
- `s_tg(v)    = 0.5 · (1 - cos(X_R[v], h_GNN[v]))` — text-graph disagreement on the same node, via frozen embeddings.
- `s_het(v)   = 1 - (#{u ∈ N(v) : y_hat(u) = y_hat(v)}) / max(deg(v), 1)` — ego-local heterophily of predicted labels (no test label access).
- `s_rec(v)   = min(1, || X_R[v] - Decode(Encode(X_R[v], Ego_v)) ||_2 / δ_ref)` — ego-embedding reconstruction error against a one-time MLP auto-encoder fit on train_cal.

Composite: `s(v, y) = w_lbl · s_lbl(v, y) + (w_tg · s_tg(v) + w_het · s_het(v) + w_rec · s_rec(v))`.

Calibration-split quantile: `q_hat = Quantile_{(n_cal+1)(1-α)/n_cal}({s(v_i, y_i) : v_i ∈ cal_split})`.

**Calibrated selection rule (v1 primary)**: `hard(v) = 1[g(v) ≥_lex τ*]` where `g(v) = (set_size(v), -margin(v), -s_composite(v))` and `τ*` is the single lexicographic cut chosen on `valid_cal` to maximize hard-node-slice recall at a target intervention ratio. Framed as a calibrated selection rule, not a coverage-guaranteed predictor; inductive-CP caveats cited explicitly in the paper's method section.

**Accept/rollback gate**: uses the SAME `s_composite`. `accept ⇔ s_composite(v; Ego_v') < s_composite(v; Ego_v)` AND `set_size(v; Ego_v') ≤ set_size(v; Ego_v)`.

**Why the mechanism is sharp**: trigger and gate are the same primitive; the primitive is substantively quality-aware (not just label-uncertainty-aware); the coefficients are fit by a single joint search; the inference path is single-pass.

### Deterministic Modifier (v1 main)

`modifier_score(e) = α_text · cos(X_R[v], X_R[u]) + α_rel · 𝟙[rel(e) ∈ R_trusted] + α_tgd · disagreement(v, u)`.

`R_trusted` is dataset-declared: for TwiBot-20/22, `{follow, reply}`; for MGTAB, the three graph-theoretically densest relations. The coefficients `(α_text, α_rel, α_tgd)` join the EQC² joint search on `valid_cal` (total 7 scalars).

### Soft Rewrite + Accept/Rollback

`w_e' = w_e · clip(1 + λ · budget(v) · modifier_score(e) · (1 - s_composite(v)), 1-λ, 1+λ)`, with `λ ∈ [0, 0.5]` and `budget(v) = min(B_max, κ · s_composite(v))`. Accept/rollback per the EQC² gate. One pass only.

### Modern Primitive Usage

- RoBERTa frozen embeddings serve inside `s_tg`, `s_rec`, the deterministic modifier, and the semantic-kNN retrieval. No LLM classifier. No graph-prompted LLM reasoning.
- Foundation-model-era primitives deliberately NOT used in v1: LLM evidence prompt, GLEM-style LM-GNN EM loop, LLM-as-final-classifier. Justified by anchor + TMLR-2024.

### Integration Into Existing Pipeline

- Extend `GraphConformalSetEstimator` at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) with the composite non-conformity score, the frozen auto-encoder, and the lexicographic selection rule. Add `conformal_filter_mode ∈ {off, composite_v1, label_only}` for ablation.
- Add `s_rec` auto-encoder fit script; persist as a Stage-0 artifact alongside the base GNN checkpoint.
- Extend `EgoRefinementRepairOperator` at [operators.py:194](LLMbot/baseline/core/operators.py#L194) with the deterministic modifier (no training) and the EQC² accept/rollback gate. Learned modifier stays as a separate class, flagged off in v1 main.

### Failure Modes

| Mode | Detection | Fallback |
|---|---|---|
| Auto-encoder overfits train_cal | Gap between `s_rec` on valid_cal vs test | Shrink `δ_ref`; if unresolved, drop `s_rec` from composite (set `w_rec=0` in weight search) |
| Weight search collapses to `w_lbl=1` | Inspect `valid_cal` weight vector | Signals EQC² offers no gain over label-only CP; paper claim contracts to "EQC² composite is necessary to beat scalar triggers; when reduced to label-only CP, gains vanish" |
| Set-size always = 1 on binary task | Diagnostic log | Expected failure mode; lexicographic rule falls back on `-s_composite` and still routes hard nodes |
| Accept/rollback thrashing > r% | Per-node rollback rate | Lower `λ`, lower `B_max`, tighten `R_trusted` |
| Inductive CP violation | Empirical coverage gap on `test_cal` vs nominal | Already expected; paper explicitly does not claim coverage, only selection |

### Novelty and Elegance

Closest prior: CF-GNN (conformal reporting on graphs), DAPS/NAPS/SNAPS (graph-diffused non-conformity), GATS/CaGCN (scalar GNN calibration), BotBR/BECE (binary edge reliability), GAugLLM (coupled text-graph modifier in contrastive learning), GLANCE (selective LLM consult), LOGIN (LLM consultant loop).

Exact differences:
- We do **not** claim CP coverage; we claim a **calibrated selection rule** — the first use of an ego-quality-aware composite non-conformity score as a simultaneous trigger and accept/rollback gate for local refinement in bot detection.
- We reuse the same primitive for both roles, eliminating the need for a second learned router.
- The rewrite is soft, budgeted, rollbackable, and preserves camouflage edges as downgraded-but-retained evidence (not deleted), orthogonal to BotBR/BECE binary reliability.

Parsimony: v1 main has **zero trainable pipeline parameters**; 7 scalars fit by a single search; one deterministic rewrite rule; one rollback comparator.

## Claim-Driven Validation Sketch

> Exactly three claim blocks. Everything else is appendix-only.

### Claim 1 — EQC² composite trigger dominates scalar and label-only CP triggers

- Keep refinement identical (deterministic modifier + soft reweight + accept/rollback). Swap only the trigger.
- Trigger variants: **T1** entropy (scalar baseline); **T2** temperature-scaled entropy; **T3** label-only CP (`s = s_lbl` only, equivalent to standard binary CP + lexicographic rule); **T4** EQC² composite (v1 main).
- Datasets: TwiBot-20 dev; TwiBot-22 primary robustness.
- Seeds: 3.
- Primary metric: macro-F1 on the high-quality-risk slice (top-20% `s_composite`).
- Secondary: global macro-F1 (non-degradation gate), AURC, ECE diagnostic, average rollback rate.
- Expected evidence: T4 > T3 > T2 ≥ T1 on slice macro-F1; T4 does not hurt global macro-F1; rollback rate stays below 15%. If T4 ≈ T3 the dominant-contribution claim contracts as described in Failure Modes.

### Claim 2 — EQC² accept/rollback gate is load-bearing

- Keep trigger = T4 and modifier identical. Toggle the gate.
- Gate variants: **G1** gate ON (v1 main); **G2** gate OFF (apply every rewrite unconditionally); **G3** gate uses a different signal (delta-loss on valid labels — a "supervised" rollback cheat).
- Datasets: TwiBot-20 dev + TwiBot-22.
- Primary metric: global macro-F1 (G2 should degrade; G1 should not).
- Secondary: slice macro-F1 (G1 vs G3 should be within noise, proving the unsupervised gate suffices).
- Expected evidence: G1 ≥ G2 on global F1 with matched budget; G1 ≈ G3 on slice F1 (same-primitive-as-gate suffices).

### Claim 3 — Cross-dataset robustness on multi-relational graph

- Best of Claim 1 + Claim 2 (T4 + G1 + deterministic modifier) on **MGTAB** (7 relations, multi-relational).
- Seeds: 3. Matched budget from TwiBot-20.
- Primary metric: global macro-F1 no-degradation + slice macro-F1 improvement.
- Expected evidence: directionally positive slice gain persists on MGTAB; the `R_trusted` relation selection transfers.

### Appendix-only (no headline-claim weight)

- Learned modifier (gradient-attribution-supervised) vs deterministic modifier.
- Modifier output form: binary vs continuous vs role-aware.
- Diffusion ingredients (DAPS/NAPS, SNAPS) in the non-conformity score.
- Ingredient ablation inside `s_composite` (drop `s_tg`, `s_het`, `s_rec` individually).
- Iteration ablation (`max_iter = 2` with rollback).

## Experiment Handoff Inputs

- Must-prove: C1 (EQC² trigger), C2 (EQC² gate), C3 (MGTAB robustness).
- Must-run: T1/T2/T3/T4, G1/G2/G3, TwiBot-20 + TwiBot-22 + MGTAB.
- Datasets: TwiBot-20 (dev), TwiBot-22 (primary robustness), MGTAB (secondary/multi-relational).
- Critical metrics: macro-F1 global + high-quality-risk slice; AURC diagnostic; average rollback rate; intervention ratio; LLM call ratio = 0.
- Highest-risk assumptions:
  - 7-scalar joint search on `valid_cal` transfers to test without per-seed retuning.
  - `s_rec` auto-encoder generalizes across TwiBot-20 → TwiBot-22 without refit.
  - Accept/rollback based on `s_composite` does not starve hard nodes of any useful rewrite.

## Compute & Timeline

- Total: ~ 40–80 GPU-hr for v1 main (3 datasets × 3 seeds × deterministic pipeline); +30–50 GPU-hr for appendix.
- Week 1: extend estimator with composite score, fit `s_rec` auto-encoder, scalar search.
- Week 2: extend operator with deterministic modifier + EQC² gate.
- Week 3: Claim 1 + Claim 2 on TwiBot-20 + TwiBot-22.
- Week 4: Claim 3 on MGTAB + appendix ablations.
- Week 5: analysis + paper draft.

## Remaining Risks and Planned Pushbacks

- If Round 2 reviewer asks to add a learned modifier to v1 main → reject, cite this round's contribution-quality finding and the one-shot 7-scalar search parsimony.
- If reviewer asks to promote `max_iter=2` to main → reject, cite research.md §13 Risk 5 (iterative confirmation bias).
- If reviewer asks for CP coverage theorem → reject, cite inductive-CP caveat, note we explicitly claim only calibrated selection.
