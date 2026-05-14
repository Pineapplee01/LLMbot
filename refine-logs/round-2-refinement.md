# Round 2 Refinement

**Round**: 2
**Review source**: `refine-logs/round-2-review.md` (overall 8.1, REVISE; one CRITICAL blocker on Venue Readiness)
**Headline revision**: five specificity/tuning-protocol fixes + two simplifications. No new mechanism, no new trainable component. Goal: close Venue Readiness 6 → 8+ by making every load-bearing choice airtight.

---

## Problem Anchor (verbatim from Round 0)

- Bottom-line problem, must-solve bottleneck, non-goals, constraints, success condition: unchanged from Round 0/1. Carried verbatim.

## Anchor Check

- Original bottleneck still targeted: yes. Every change is a specificity / tuning-protocol tightening; the mechanism (composite ego-quality non-conformity score reused as trigger and accept/rollback gate) is unchanged.
- Reviewer suggestions rejected as drift: none. All five action items + one simplification applied. One simplification (rename "conformal") partially applied — kept for the mechanism description, dropped from any coverage claim.

## Simplicity Check

- Dominant contribution remains EQC² (Ego-Quality Conformal Controller). No new parallel contribution sprouted.
- Components removed / merged:
  - 7-scalar joint search → **6-scalar search** (fixed `w_lbl = 1` per reviewer simplification option 2; search `{w_tg, w_het, w_rec, α_text, α_rel, α_tgd}`).
  - `s_rec` auto-encoder kept, but the joint search is now constrained so that `w_rec = 0` is a legitimate solution; if selected on `valid_cal`, the auto-encoder is dropped from the published v1 main pipeline automatically.
- Simplicity Check passes: v1 main still has zero trainable pipeline parameters (AE is a one-time fit; 6 scalars + one threshold are a calibration search).

---

## Changes Made (Round 2)

### 1. Formal definition of `s_composite(v)` and `margin(v)` (Action 1)

- **Reviewer said**: "Define `s_composite(v)` and `margin(v)` formally — is `s_composite = min_y s(v,y)`? `s(v, ŷ)`? Must be pinned." (Action Item 1, Venue Readiness).
- **Locked definitions**:

```
p_hat(v)         = argmax_y p_base(y | v)         # base-GNN predicted label
s(v, y)          = w_lbl · s_lbl(v, y) + w_tg · s_tg(v) + w_het · s_het(v) + w_rec · s_rec(v)   with  w_lbl = 1
s_lbl_min(v)     = min_y s_lbl(v, y) = 1 - max_y p_base(y | v)
s_composite(v)   = w_lbl · s_lbl_min(v) + w_tg · s_tg(v) + w_het · s_het(v) + w_rec · s_rec(v)    (== min_y s(v, y))
margin(v)        = q_hat - s_composite(v)          (signed distance to conformal threshold; positive ⇒ "easy")
set_size(v)      = |{y : s(v, y) ≤ q_hat}|
```

- `s_composite(v)` is used in three places and is the **same scalar everywhere**: (i) lexicographic trigger ordering, (ii) rewrite-amplitude factor `(1 - s_composite(v))`, (iii) accept/rollback comparison.
- `margin(v)` is used only in lexicographic ordering; never multiplied into the rewrite, never used in the gate.

### 2. `s_tg` representation-mismatch fix (Action 2)

- **Reviewer said**: "Cosine between `X_R[v]` and `h_GNN[v]` is undefined unless they're in the same space; specify the exact mapping (preferably zero-parameter), or switch to a disagreement in probability space." (Action Item 2).
- **Action**: switch `s_tg` to **probability-space Jensen-Shannon divergence**, using two predictive distributions already produced by the frozen pipeline. Zero new parameters. No learned projection.

```
p_LM(v)  = softmax(W_LM  · X_R[v] + b_LM)           # frozen LM-only classifier head from Stage-0 (already trained)
p_GNN(v) = softmax(W_GNN · h_GNN[v] + b_GNN)        # frozen GNN classifier head from Stage-0
s_tg(v)  = JSD(p_LM(v) || p_GNN(v)) / log 2         # normalized to [0, 1]
```

- Both heads exist in the current Stage-0 co-training pipeline (LM-only prediction is produced during LM finetuning; GNN head is the main classifier). This is strictly a re-use of existing outputs.
- JSD is bounded in `[0, log 2]` and symmetric, so the normalization is clean.
- Fallback if `p_LM(v)` is not exposed by a given backbone variant: use `p_LM(v) = softmax(MLP_stopgrad(X_R[v]))` where `MLP_stopgrad` is one layer fit on train_cal only — **still zero trainable parameters at inference**, counted as a calibration artifact like the auto-encoder.

### 3. Non-conformal baseline "T4-NS" (Action 3)

- **Reviewer said**: "Add a non-conformal baseline: same `s_composite`, same deterministic modifier, but no `q̂` / prediction set — just threshold `s_composite` for trigger and rollback. If EQC² beats this, the set-derived features are demonstrably load-bearing." (Action Item 3, CRITICAL).
- **Action**: add `T4-NS` to the Claim 1 trigger arm:

```
T1    entropy                          # scalar baseline
T2    temperature-scaled entropy       # scalar baseline
T3    label-only CP (s = s_lbl)        # CP without ego-quality composite
T4-NS EQC² composite, scalar gating    # same composite, threshold `s_composite ≥ θ` only; NO q_hat, NO set geometry
T4    EQC² composite + set-based gate  # v1 main: lex rule on (set_size, -margin, -s_composite)
```

Expected pattern if the set-based geometry is load-bearing: `T4 > T4-NS > T3 > T2 ≥ T1` on slice macro-F1. If `T4 ≈ T4-NS`, the paper's dominant-contribution claim contracts honestly: "the composite ego-quality score is the load-bearing piece; the set construction adds little on binary tasks." This is admitted up front and is the honest fallback reframe.

### 4. Locked tuning protocol (Action 4, CRITICAL)

- **Reviewer said**: "Make the tuning protocol airtight (train_cal for AE, valid_cal for the 7-scalar search, separate untouched test for final reporting); or lightweight nested split." (Action Item 4, CRITICAL).
- **Action**: define and lock the split protocol explicitly in the paper.

```
Fold                Purpose                                                  Touched by
----                -------                                                  ----------
train               Stage-0 base GNN + LM finetuning                         Stage-0 only
train_cal (~10%)    s_rec auto-encoder fit;
                    LM-only head fit if fallback needed                      Stage-A only
valid_cal (~10%)    6-scalar weight search; threshold τ* + q_hat             Stage-A only
valid               Stage-0 early-stopping + sanity checks                   Stage-0 only
test                ONLY final macro-F1, AURC, slice F1 for the paper        final reporting only
```

Invariants (enforced via unit tests):
- `q_hat` is fit on `valid_cal` non-conformity scores only; never recomputed on test.
- The 6-scalar weight search never accesses test labels, test logits, or test embeddings.
- The auto-encoder is trained on train_cal text embeddings only; never on valid_cal or test.
- Any hyperparameter set on valid (not valid_cal) is fixed *before* the composite score is defined.

Optional lightweight nested split: 5-fold rotation of `train_cal` / `valid_cal` within the train+valid pool; `q_hat` and τ* are re-fit per fold; test fold is untouched. Reported in appendix as a robustness check against tuning optimism.

### 5. Binary-set edge cases (Action 5)

- **Reviewer said**: "State explicitly what happens when the prediction set is empty or size=2, and how lexicographic ordering treats those cases." (Action Item 5).
- **Action**:

| `set_size(v)` | Interpretation | Lex-rule effect |
|---|---|---|
| 0 | All labels exceed `q_hat` — unusual when `q_hat` is the empirical quantile; indicates massive mis-calibration or distribution shift. Force `set_size := 2` for ordering (treat as maximally uncertain) and log a warning. | `g(v)` = (2, ...), ranked as hard. |
| 1 | Singleton set — model confident at the conformal level. | `g(v)` = (1, ...), typically easy unless margin is very small. Sorted by `-margin` and `-s_composite`. |
| 2 (binary task) | Both labels admissible — model abstains at the conformal level. | `g(v)` = (2, ...), top-priority hard. Sorted by `-s_composite`. |

The lexicographic ordering therefore retains discrimination even when `set_size` saturates at the binary minimum/maximum, because `margin` and `s_composite` continue to differentiate. This answers the reviewer's "set geometry can collapse" concern operationally: the rule degrades gracefully to a margin + composite ranking, while T4-NS (no-set baseline) removes both.

### 6. Simplification — fix `w_lbl = 1` (Simplification 2)

- **Reviewer said**: "Reduce the 7-scalar search degrees of freedom: fix `w_lbl = 1` and search only 3 weights + 3 modifier alphas." (Simplification 2, optional).
- **Action**: taken. Pins the scale of the composite to the label-non-conformity unit; all other weights are relative to it. The search is over `{w_tg, w_het, w_rec} ∈ [0, 2]³` and `{α_text, α_rel, α_tgd} ∈ [0, 1]³`, via coordinate-descent on valid_cal macro-F1 on the hard-node slice. Six scalars, not seven.
- **Impact**: the "hidden degrees of freedom" concern is reduced; the composite has an unambiguous interpretive unit (multiples of label non-conformity).

### 7. Simplification — `s_rec` stays gated, auto-encoder survives only if `w_rec > 0` on `valid_cal`

- **Reviewer said**: "Drop AE unless clearly additive beyond `s_tg + s_het`; keep only if it moves trigger ranking." (Simplification 1, optional).
- **Action**: the 6-scalar search can return `w_rec ≈ 0`. If it does on `valid_cal`, the paper's v1 main pipeline drops the auto-encoder and reports the 4-scalar variant; the AE-augmented variant becomes an appendix ablation. The decision is made *by the search on valid_cal*, not by us.
- **Impact**: if the auto-encoder is not load-bearing, the v1 main method becomes even simpler (4 scalars + one threshold + one quantile).

### 8. Language tightening (Simplification 3, selectively taken)

- **Reviewer said**: "Rename 'conformal' if needed: 'Quantile-Calibrated Ego-Quality Controller'." (Simplification 3, optional).
- **Action**: keep `EQC²` as the method short name (recognizable, already anchored internally), and use both names consistently — "**Ego-Quality Quantile-Calibrated Controller**" as the long form in the paper introduction; "**ego-quality conformalized selection**" in the method section; no "coverage" claim anywhere. Inductive-CP caveat retained.

---

## Revised Proposal

# Research Proposal: EQC² — Ego-Quality Quantile-Calibrated Controller for Post-Hoc Local Ego Refinement in Social Bot Detection

**Version**: Round 2 refinement
**Date**: 2026-05-09

## Problem Anchor

[verbatim from Round 0]

## Technical Gap

[unchanged from Round 1 — two residual-error classes; naive fixes fall short; binary-label-only CP collapses to near-scalar; missing primitive is a quantile-calibrated ego-quality selection rule that serves as both trigger and gate]

## Method Thesis

**One-sentence thesis**: An ego-quality-aware quantile-calibrated non-conformity score, reused as both selection trigger and accept/rollback gate, enables budgeted reversible local ego refinement on a frozen backbone — dominating scalar triggers, scalar-gated composites, and label-only CP triggers without duplicating binary edge-reliability literature, without requiring LLM calls, and without coverage claims.

## Contribution Focus

- **Dominant**: EQC² — ego-quality composite non-conformity score + quantile-calibrated set-based selection rule + same-primitive accept/rollback gate. A **single mechanism**, used three ways, with a non-conformal baseline (T4-NS) explicitly included so the set-based contribution is falsifiable.
- **Supporting**: deterministic text-graph modifier + soft reweight. Deliberately commodity; one-line modifier score; zero trainable parameters.
- **Explicit non-contributions**: no coverage theorem; no binary edge-reliability headline; no LLM-understands-graphs claim; no global structure learning; no causal identification; no SOTA detector.

## Complexity Budget

| Slot | v1 main | Appendix |
|---|---|---|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN/RGCN; ego adjacency; kNN semantic index; frozen LM-only head `p_LM`; frozen GNN head `p_GNN` | — |
| One-time fit (calibration artifacts) | (i) `s_rec` auto-encoder MLP on train_cal (5 min CPU, *only if `w_rec > 0`*); (ii) `p_LM` fallback MLP on train_cal (only if backbone does not expose LM-only head); (iii) 6-scalar weight vector by coordinate-descent on valid_cal; (iv) `q_hat` quantile and lex threshold τ* on valid_cal | Learned modifier (single-head, gradient-attribution-supervised) |
| Not trainable at all | Composite non-conformity score; lex selection rule; soft-reweight arithmetic; accept/rollback comparator | DAPS/NAPS diffusion; SNAPS semantic-kNN |
| Excluded from v1 main | LLM evidence prompt (v2); global structure learning; hard delete; iterated refinement | — |

**v1 main pipeline trainable parameters at inference**: zero.

## System Overview (locked)

```
Stage-0 (frozen)   RoBERTa X_R → BotRGCN → p_GNN(v), h_GNN(v); LM head → p_LM(v)

Stage-A (one-time, on train_cal / valid_cal only)
   a1. Fit s_rec auto-encoder on train_cal text embeddings
   a2. Fit p_LM fallback MLP on train_cal (only if backbone doesn't expose LM-only head)
   a3. Coordinate-descent search over (w_tg, w_het, w_rec, α_text, α_rel, α_tgd) ∈ [0,2]³×[0,1]³
       on valid_cal macro-F1 (hard-node slice)
   a4. Compute q_hat and τ* on valid_cal
   a5. Freeze all calibration artifacts; persist alongside Stage-0 checkpoint

Stage-1 (per-node, test time)
   s_lbl(v, y)      = 1 - p_base(y | v)
   s_tg(v)          = JSD(p_LM(v) || p_GNN(v)) / log 2           ← zero-param, probability-space
   s_het(v)         = 1 - #{u∈N(v): y_hat(u)=y_hat(v)} / max(deg(v), 1)
   s_rec(v)         = min(1, || X_R[v] - AE(X_R[v], Ego_v) ||_2 / δ_ref)
                      (omitted if w_rec = 0 from Stage-A)
   s(v, y)          = s_lbl(v, y) + w_tg·s_tg(v) + w_het·s_het(v) + w_rec·s_rec(v)   (w_lbl ≡ 1)
   s_composite(v)   = min_y s(v, y)
   prediction_set(v)= {y : s(v, y) ≤ q_hat}
   set_size(v), margin(v) = q_hat - s_composite(v)

Stage-2 (selection rule)
   g(v)     = (set_size(v), -margin(v), -s_composite(v))
   hard(v)  = g(v) ≥_lex τ*
   Edge-case handling:
     set_size = 0 → ordering treats as set_size = 2 + warning log
     set_size = 2 (binary) → top-priority hard; tie-break on -s_composite
     set_size = 1 → easy unless margin very small; tie-break on -s_composite

Stage-3 (retrieval, hard v only)
   propagation_edges(v)     = existing ego edges
   virtual_context_edges(v) = kNN_RoBERTa \ existing ego (evidence-only, never written back)

Stage-4 (deterministic modifier)
   modifier_score(e) = α_text·cos(X_R[v], X_R[u]) + α_rel·𝟙[rel(e)∈R_trusted] + α_tgd·sign(JSD(p_LM(v)||p_GNN(v)) - JSD(p_LM(u)||p_GNN(u)))

Stage-5 (soft rewrite, propagation_edges only)
   w_e' = w_e · clip(1 + λ · budget(v) · modifier_score(e) · (1 - s_composite(v)), 1-λ, 1+λ)
   budget(v) = min(B_max, κ · s_composite(v))

Stage-6 (accept / rollback)
   re-evaluate Stage-1 on Ego_v'
   accept iff   s_composite(v; Ego_v') < s_composite(v; Ego_v)
         AND   set_size(v; Ego_v')    ≤ set_size(v; Ego_v)
   else rollback.   max_iter = 1.
```

## Tuning Protocol (airtight)

- `train_cal` is a held-out 10% of `train`; used only in Stage-A a1 (AE fit) and a2 (LM-only fallback MLP fit).
- `valid_cal` is a held-out 10% of `train` (disjoint from `train_cal`); used only in a3 (weight search), a4 (`q_hat`, τ*).
- `valid` continues its existing Stage-0 role (early-stopping, sanity); never used in Stage-A.
- `test` is untouched during all of Stage-A; used only for final macro-F1, AURC, slice F1 reporting.
- Enforcement: unit tests assert that `test` indices do not appear in any `calibration_metadata.used_indices`.
- Nested split (appendix robustness): 5-fold rotation of `train_cal / valid_cal` within the train pool, test fold untouched.

## Calibrated Selection Rule Framing

Framed throughout as a **quantile-calibrated set-based selection rule**, not a coverage-guaranteed conformal predictor. Inductive CP caveat cited in Method Section 2: "under inductive message-passing, marginal coverage is not guaranteed; we report empirical coverage on `test_cal` as a diagnostic and do not claim it as a theorem." "Conformal" appears only as a method-heritage reference (CF-GNN, DAPS/NAPS, SNAPS); all claim language uses "quantile-calibrated".

## Deterministic Modifier (v1 main)

Single-line score as above. `R_trusted` dataset-declared: TwiBot-20/22 `{follow, reply}`; MGTAB top-3 densest relations. Coefficients fit in the same 6-scalar search.

## Modern Primitive Usage

Frozen RoBERTa → `X_R` used inside `s_tg` (via `p_LM`), `s_rec` (input), modifier (text cosine), retrieval (kNN). Frozen GNN head → `p_GNN` used inside `s_tg`. No LLM classifier. No graph-prompted LLM reasoning. No foundation-model-era primitive bolted on decoratively.

## Integration

- Extend `GraphConformalSetEstimator` at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) with composite score + AE fit hook + lex selection + edge-case handler; add `score_mode ∈ {label_only, composite_v1, composite_v1_no_rec}` for T3/T4/ablation switching.
- Add a 30-line Stage-A calibration script that writes all calibration artifacts deterministically.
- Extend `EgoRefinementRepairOperator` at [operators.py:194](LLMbot/baseline/core/operators.py#L194) with deterministic modifier + EQC² gate; add `gate_mode ∈ {off, eqc2, supervised_cheat}` for G1/G2/G3 switching.
- T4-NS (no-set baseline) reuses the same composite score, bypasses `q_hat`, thresholds `s_composite` scalar-wise.

## Failure Modes (updated)

| Mode | Detection | Fallback |
|---|---|---|
| AE overfits | Gap `s_rec` train_cal vs valid_cal | Shrink δ_ref; if unresolved, search already allows `w_rec = 0` |
| Weight search → w_tg=w_het=w_rec=0 | Inspect valid_cal search output | Signals EQC² composite offers no gain; paper claim contracts to "composite matters iff search assigns positive weights"; T4 collapses to T3 |
| `set_size` saturates (all = 1 or all = 2) | Distribution of `set_size(v)` | Lex rule falls back on `-margin, -s_composite`; T4 vs T4-NS comparison still diagnostic |
| Accept/rollback thrashing > 15% | Per-node rollback log | Lower `λ`, lower `B_max`, tighten `R_trusted` |
| Inductive CP violation (empirical coverage ≠ nominal) | Coverage diagnostic on `test_cal` | Already expected; we claim calibrated selection, not coverage |

## Novelty and Elegance (updated)

Closest prior: CF-GNN (conformal for GNN reporting), DAPS/NAPS/SNAPS (graph-diffused non-conformity, ablation-only), GATS/CaGCN (scalar GNN calibration), BotBR/BECE (binary edge reliability), GAugLLM (coupled text-graph modifier in contrastive learning), GLANCE/LOGIN (selective LLM consult).

Exact differences:
- We do **not** claim CP coverage; we claim a **quantile-calibrated selection rule**.
- Same primitive used three ways (trigger, rewrite-amplitude factor, rollback gate) — economy of mechanism.
- Non-conformal baseline T4-NS is in the main experiment to falsify the set-based contribution.
- Rewrite is soft, budgeted, rollbackable, and **preserves camouflage edges** as downgraded-but-retained evidence.
- v1 main pipeline has zero trainable inference parameters; 6 scalars are a calibration search.

## Claim-Driven Validation Sketch (unchanged except for T4-NS)

### Claim 1 — EQC² composite-with-set trigger dominates scalar and label-only CP triggers

Trigger swap under identical refinement.
- **T1** entropy · **T2** temperature-scaled entropy · **T3** label-only CP (s = s_lbl) · **T4-NS** EQC² composite, scalar gating (no q_hat, no set geometry) · **T4** EQC² composite + set-based gate (v1 main).
- Datasets: TwiBot-20 dev + TwiBot-22. Seeds: 3.
- Primary: macro-F1 on high-quality-risk slice. Secondary: global macro-F1 (non-degradation), AURC, ECE diagnostic, rollback rate.
- Expected evidence: **T4 > T4-NS > T3 > T2 ≥ T1** on slice; T4 does not hurt global; rollback < 15%.
- **Falsification**: if T4 ≈ T4-NS → dominant contribution contracts honestly to "composite ego-quality score is the load-bearing piece; set construction adds little on binary tasks."

### Claim 2 — EQC² accept/rollback gate is load-bearing

Trigger and modifier held identical (T4 + deterministic modifier); toggle gate.
- **G1** gate ON (v1 main) · **G2** gate OFF · **G3** gate uses delta-loss on valid labels (supervised cheat).
- Primary: global macro-F1 (G2 should degrade; G1 should not). Secondary: slice macro-F1 (G1 ≈ G3 ⇒ unsupervised gate suffices).

### Claim 3 — Cross-dataset robustness on multi-relational graph

T4 + G1 + deterministic modifier on MGTAB. Seeds: 3. Matched budget from TwiBot-20.
- Primary: global F1 no-degradation + slice F1 improvement.

### Appendix-only

Learned modifier vs deterministic; modifier output form (B0/B1/B2); DAPS-NAPS/SNAPS diffusion; `max_iter=2` with rollback; nested tuning split robustness.

## Compute & Timeline

- v1 main: ~ 40-80 GPU-hr across 3 datasets × 3 seeds × deterministic pipeline.
- Appendix: +30-50 GPU-hr.
- Timeline: W1 estimator + composite + AE + scalar search + T4-NS path; W2 operator + deterministic modifier + EQC² gate + edge cases; W3 Claim 1+2 on TwiBot-20/22; W4 Claim 3 on MGTAB + appendix; W5 analysis + paper draft.

## Remaining Risks and Planned Pushbacks

- If Round 3 reviewer asks to add a coverage theorem → reject, cite inductive-CP caveat and the explicit "we do not claim coverage" framing.
- If reviewer pushes for learned-modifier-in-main → reject, cite contribution-focus finding from Round 1.
- If reviewer asks for yet another dataset → acknowledge; point to MGTAB multi-relational stress test as sufficient secondary robustness.
- If reviewer flags the 6-scalar search as overfitting → point to the nested-split appendix and the auto-collapse mechanism (`w_rec → 0`).
