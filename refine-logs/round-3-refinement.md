# Round 3 Refinement

**Round**: 3
**Review source**: `refine-logs/round-3-review.md` (overall 8.8, REVISE; remaining blockers are empirical + one spec item)
**Headline revision**: three targeted spec fixes. Two empirical items (T4 > T4-NS, G1 vs G2) are acknowledged as the `/experiment-plan` handoff — they cannot land inside `/research-refine` by construction.

---

## Problem Anchor (verbatim from Round 0)

[unchanged]

## Anchor Check

- Original bottleneck still targeted: yes. Every change is a locality / wording tightening or a component deletion. No mechanism change.
- Reviewer suggestions rejected as drift: none. All three actionable items applied. The two empirical items are deferred to experiment execution (per skill boundary and reviewer's own framing).

## Simplicity Check

- Dominant contribution unchanged (EQC²). No new parallel contribution.
- Components removed / merged:
  - **`virtual_context_edges` removed from v1 main**. They had no role in the decision rule — removing them eliminates an entire retrieval pathway from the v1 main pipeline. Surviving use: `kNN_RoBERTa` lookup is still needed for the deterministic modifier's `α_text · cos(X_R[v], X_R[u])` term, but only on the **propagation-edge candidate set = existing ego edges**. No set-difference with the ego is computed, no "virtual" artifact is materialized.
- Simpler than Round 2: one fewer concept, no evidence-only artifact, clearer locality story (Stage-3 becomes trivial: "enumerate existing ego edges").

---

## Changes Made (Round 3)

### 1. Locality of recomputation — explicit subgraph and forward-pass spec (spec-level, CRITICAL)

**Reviewer said**: "Specify / verify the locality of the recomputation after reweighting (exact forward pass rerun, exact subgraph) so the 'post-hoc local intervention' is operationally real and budget-bounded, not implicitly a global rerun hidden inside 're-eval'."

**Action**: pin locality explicitly in the method spec.

```
Definitions
-----------
Ego_v^{(L)}     = L-hop ego subgraph around v (L = #message-passing layers of base GNN)
Ego_v^{(L)}_pre  = Ego_v^{(L)} with original edge weights
Ego_v^{(L)}_post = Ego_v^{(L)} with post-Stage-5 reweighted edge weights on propagation_edges(v)

Base GNN        = BotRGCN/RGCN with L ∈ {2, 3} (dataset-declared, matches Stage-0 config)

Stage-6 recomputation (local, bounded)
--------------------------------------
Input:   Ego_v^{(L)}_post; the frozen base-GNN parameters; node features X_R restricted to Ego_v^{(L)}
Output:  p_GNN_post(v), h_GNN_post(v)
Procedure:
  1. Extract Ego_v^{(L)}_post as a stand-alone subgraph with local node relabelling.
  2. Run L message-passing layers of the frozen base GNN restricted to this subgraph.
  3. Read p_GNN_post(v), h_GNN_post(v) from the root node.
Cost per hard node v: O(|Ego_v^{(L)}|) — identical to a standard L-hop subgraph forward pass, which is how inductive GNN eval already works for any test node.
Global cost: sum over hard(v) of |Ego_v^{(L)}|, bounded by budget(v) = min(B_max, κ · s_composite(v)) and the hard-node quota.
Explicitly NOT rerun:
  - Full-graph message passing on the entire adjacency.
  - Stage-0 LM/GNN training.
  - h_GNN of any node outside Ego_v^{(L)}.

Recomputed composite non-conformity score
-----------------------------------------
s_lbl_post(v, y) = 1 - p_GNN_post(y | v)
s_tg_post(v)     = JSD(p_LM(v) || p_GNN_post(v)) / log 2          (p_LM(v) unchanged — LM head is global and does not depend on Ego_v)
s_het_post(v)    = 1 - #{u ∈ N(v): ŷ_post(u) = ŷ_post(v)} / max(deg(v), 1)
                   where ŷ_post(u) = argmax_y p_GNN_post(y | u) for u ∈ propagation_edges(v) endpoints;
                         ŷ_post(u) = ŷ(u) for u ∈ N(v) outside the recomputed set.
s_rec_post(v)    = s_rec(v)    (AE input is X_R[v], untouched by rewrite)
s_post(v, y)     = s_lbl_post(v, y) + w_tg · s_tg_post(v) + w_het · s_het_post(v) + w_rec · s_rec_post(v)
s_composite_post(v) = min_y s_post(v, y)
prediction_set_post(v) = {y : s_post(v, y) ≤ q_hat}   (q_hat is frozen; never refit post-rewrite)
```

- **Accept/rollback gate** uses the locally recomputed `s_composite_post(v)` and `set_size_post(v)` vs the pre-rewrite `s_composite(v)` and `set_size(v)`. No global rerun.
- **Budget invariant**: total wall-clock per-dataset overhead = `|hard_nodes| × L-hop ego forward` + rollback tax (bounded by `r_max` rollback fraction). For TwiBot-20 / TwiBot-22 at ~20% hard-node quota, this is < 2× the base test-time cost.

### 2. Simplification — drop `virtual_context_edges` from v1 main

**Reviewer said**: "Drop `virtual_context_edges` from v1 main unless you can point to a concrete use (right now it's 'evidence-only' but not in the decision rule)."

**Action**: removed. Stage-3 in v1 main is now literally "use existing ego edges". The `kNN_RoBERTa` index remains in the Complexity Budget but is used only inside the modifier (`α_text · cos(...)` between `v` and existing-ego neighbors; no need to compute kNN at all for the modifier since the neighbor set is already the ego). Net effect: the kNN index is **removed** from the v1 main Complexity Budget and appears only in appendix L4 (LLM evidence prompt, v2).

Appendix use of virtual context edges stays as an optional future extension hook, not a v1 commitment.

### 3. Simplification — rename "zero trainable inference parameters"

**Reviewer said**: "Tighten wording around 'zero trainable inference parameters': if `s_rec` AE or `p_LM` fallback MLP is used at test time, they *are* trained artifacts. Safer phrasing: 'no end-to-end retraining; only calibration-time fits (unsupervised / distillation-style)'."

**Action**: replaced throughout the revised proposal. The phrasing is now:

> v1 main makes **no base-backbone retraining**. All learning is **calibration-time fits**: the `s_rec` auto-encoder is an unsupervised reconstruction fit on `train_cal` text embeddings; the `p_LM` fallback head is a supervised distillation-style fit on `train_cal` labels (used only when the backbone does not expose an LM-only head). The 6-scalar weight vector and the `q_hat / τ*` thresholds are a coordinate-descent / quantile fit on `valid_cal`. No gradient updates flow into the frozen RoBERTa or the frozen BotRGCN. This matches the anchor's "frozen Stage-1 RoBERTa + BotRGCN" constraint.

No "zero trainable parameters" language anywhere.

### 4. Two empirical items — explicit handoff to `/experiment-plan`

**Reviewer said**: "What still blocks a READY verdict is now empirical-risk, not spec-risk: (1) show T4 beats T4-NS, (2) show G1 prevents global degradation vs G2. You've set this up cleanly; you just can't claim it yet."

**Action**: the proposal's "Experiment Handoff Inputs" section now explicitly names these as the **two pilot gates** that the downstream `/experiment-plan` must schedule first. This is not a revision of the method; it is a contract with the execution phase. The method is complete; the pilots are its execution.

**Handoff clauses added to the revised proposal**:

> **Pilot Gate A** (Claim 1 first-order evidence): on TwiBot-20 dev, 1 seed, matched budget, run T1 / T3 / T4-NS / T4. Required for the dominant contribution to hold: T4 slice macro-F1 > T4-NS slice macro-F1 by ≥ 1 std of T4-NS across 3 seeds, or a ≥ 1pp gap at single-seed level. If this gate fails, the paper contracts to "composite ego-quality score is load-bearing; set-based geometry does not add on binary tasks", which is an honest fallback explicitly anticipated in Round 2 failure modes.

> **Pilot Gate B** (Claim 2 first-order evidence): on TwiBot-20 dev, 1 seed, matched budget, run G1 / G2. Required for the supporting-contribution narrative: G2 must degrade global macro-F1 by ≥ 0.3pp vs the no-refinement baseline, and G1 must preserve global macro-F1 within ≤ 0.1pp of the no-refinement baseline. If G2 does not degrade, the rollback gate is unneeded; if G1 does degrade, the rollback gate is ineffective. Either failure contracts the paper.

Both gates are cheap (single-seed pilots) and live inside the first week of the `/experiment-plan` execution.

---

## Revised Proposal (v3)

# Research Proposal: EQC² — Ego-Quality Quantile-Calibrated Controller for Post-Hoc Local Ego Refinement in Social Bot Detection

**Version**: Round 3 refinement
**Date**: 2026-05-09

## Problem Anchor

[verbatim from Round 0]

## Technical Gap

[unchanged]

## Method Thesis

**One-sentence thesis**: An ego-quality-aware quantile-calibrated non-conformity score, reused as both selection trigger and accept/rollback gate on strictly L-hop-local recomputation, enables budgeted reversible post-hoc ego refinement on a frozen backbone — dominating scalar triggers, scalar-gated composites, and label-only CP triggers without duplicating binary edge-reliability literature, without LLM calls, and without coverage claims.

## Contribution Focus

- **Dominant**: EQC² — ego-quality composite non-conformity score + quantile-calibrated set-based selection rule + same-primitive accept/rollback gate on L-hop-local recomputation. A single mechanism, used three ways, with the T4-NS non-conformal baseline in the main experiment so the set-based contribution is falsifiable.
- **Supporting**: deterministic text-graph modifier + soft reweight over existing ego edges. Deliberately commodity; one-line modifier score; no base-backbone retraining.
- **Explicit non-contributions**: no coverage theorem; no binary edge-reliability headline; no LLM-understands-graphs claim; no global structure learning; no causal identification; no SOTA detector; no global forward-pass recomputation.

## Complexity Budget

| Slot | v1 main | Appendix |
|---|---|---|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN/RGCN; existing ego adjacency; frozen `p_LM` head; frozen `p_GNN` head | — |
| Calibration-time fits | (i) `s_rec` AE on `train_cal` (only if `w_rec > 0`); (ii) `p_LM` fallback MLP on `train_cal` (only if backbone lacks native LM head); (iii) 6-scalar weight search on `valid_cal`; (iv) `q_hat`, `τ*` on `valid_cal` | Learned modifier (gradient-attribution supervision) |
| Not trainable at all | Composite score; lex selection rule; soft-reweight arithmetic; accept/rollback comparator; L-hop-local recomputation | DAPS/NAPS, SNAPS diffusion |
| Excluded from v1 main | LLM evidence prompt; learned modifier; global structure learning; hard delete; iterated refinement; **virtual context edges** (removed from v1 main this round) | — |

**No base-backbone retraining**. All learning is calibration-time; no gradient updates flow into the frozen RoBERTa or BotRGCN.

## System Overview (locked, with explicit locality)

```
Stage-0 (frozen)   RoBERTa → p_GNN(v), h_GNN(v); LM head → p_LM(v)
Stage-A (one-time, on train_cal / valid_cal)
   a1. Fit s_rec AE on train_cal text embeddings                    (skipped if w_rec = 0 in search)
   a2. Fit p_LM fallback MLP on train_cal labels                    (skipped if backbone exposes p_LM)
   a3. Coordinate-descent search {w_tg, w_het, w_rec, α_text, α_rel, α_tgd} on valid_cal
       maximizing slice macro-F1 at target intervention ratio; w_lbl ≡ 1
   a4. Compute q_hat, τ* on valid_cal
   a5. Persist artifacts alongside Stage-0 checkpoint

Stage-1 (per-node, test time, pre-rewrite)
   s_lbl(v, y)      = 1 - p_base(y | v)
   s_tg(v)          = JSD(p_LM(v) || p_GNN(v)) / log 2
   s_het(v)         = 1 - #{u ∈ N(v): ŷ(u) = ŷ(v)} / max(deg(v), 1)
   s_rec(v)         = min(1, || X_R[v] - AE(X_R[v], Ego_v) ||_2 / δ_ref)   (if enabled)
   s(v, y)          = s_lbl(v, y) + w_tg·s_tg(v) + w_het·s_het(v) + w_rec·s_rec(v)
   s_composite(v)   = min_y s(v, y)
   prediction_set(v) = {y : s(v, y) ≤ q_hat}
   set_size(v), margin(v) = q_hat - s_composite(v)

Stage-2 (selection rule)
   g(v) = (set_size(v), -margin(v), -s_composite(v));    hard(v) = g(v) ≥_lex τ*
   Edge cases: set_size=0 → treat as 2 + warning; set_size=2 top-priority; set_size=1 tie-break.

Stage-3 (candidate enumeration — trivial)
   propagation_edges(v) = existing ego edges of v
   (Virtual context edges removed from v1 main.)

Stage-4 (deterministic modifier, zero params at inference)
   modifier_score(e) = α_text · cos(X_R[v], X_R[u])
                     + α_rel  · 𝟙[rel(e) ∈ R_trusted]
                     + α_tgd  · sign(JSD(p_LM(v)||p_GNN(v)) - JSD(p_LM(u)||p_GNN(u)))

Stage-5 (soft rewrite, propagation_edges only)
   w_e' = w_e · clip(1 + λ · budget(v) · modifier_score(e) · (1 - s_composite(v)), 1-λ, 1+λ)
   budget(v) = min(B_max, κ · s_composite(v))

Stage-6 (LOCAL recomputation + accept / rollback)
   Ego_v^{(L)}_post = Ego_v^{(L)} with rewritten edge weights
   p_GNN_post(v), h_GNN_post(v) = L-layer forward pass of frozen base GNN on Ego_v^{(L)}_post
                                  (local subgraph only; no global rerun)
   s_composite_post(v), set_size_post(v) recomputed via Stage-1 formulas on post values
   accept iff   s_composite_post(v) < s_composite(v)
         AND   set_size_post(v)    ≤ set_size(v)
   else rollback edge weights to pre state; no global re-evaluation.
   max_iter = 1.
```

## Locality Invariants (new, explicit)

1. Stage-6 forward pass is restricted to `Ego_v^{(L)}` where `L = #message-passing layers of base GNN`. No node outside this subgraph has `h_GNN` recomputed.
2. `p_LM(v)` is global-free; it never needs recomputation post-rewrite.
3. `s_het_post(v)` uses `ŷ_post(u)` only for `u` whose ego edges with `v` were rewritten; all other neighbors keep their pre-rewrite labels.
4. `q_hat` is frozen after Stage-A a4; never refit.
5. Per-dataset wall-clock overhead is bounded by `|hard_nodes| × average |Ego^{(L)}|` for Stage-6 recomputation; test reported as "refinement wall-clock overhead ratio" in the paper.

## Tuning Protocol

[unchanged from Round 2 — train / train_cal / valid_cal / valid / test with leakage-assertion unit tests; 5-fold nested rotation in appendix]

## Calibrated Selection Rule Framing

[unchanged from Round 2 — "quantile-calibrated set-based selection rule", not coverage-guaranteed CP; inductive-CP caveat]

## Integration

- Extend `GraphConformalSetEstimator` at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) with composite score + AE hook + lex selection + edge-case handler + `score_mode ∈ {label_only, composite_v1, composite_v1_no_rec}` + `p_LM_source ∈ {native, fallback_mlp}`.
- Add 30-line Stage-A calibration script (`stage_a_calibrate.py`): produces all calibration artifacts deterministically.
- Extend `EgoRefinementRepairOperator` at [operators.py:194](LLMbot/baseline/core/operators.py#L194) with deterministic modifier + EQC² gate + `gate_mode ∈ {off, eqc2, supervised_cheat}` + explicit L-hop-local forward-pass method (not the existing full-graph eval).
- Add `t4_ns` path that reuses the composite score, bypasses `q_hat`, thresholds `s_composite` scalar-wise. Shares all other code with T4.

## Failure Modes

[unchanged from Round 2, plus:]

- **Non-local recomputation leakage**: unit test asserts that the Stage-6 forward pass only reads node features inside `Ego_v^{(L)}`. Leakage ⇒ fail-loud.

## Novelty and Elegance

[unchanged summary; explicitly add: "recomputation is L-hop-local, with a forward-pass-count invariant, so post-hoc locality is operationally defined, not just intent-signaled."]

## Claim-Driven Validation Sketch

[unchanged three claim blocks; in addition, **two pilot gates** gating the full run]

- **Pilot Gate A (Claim 1)**: TwiBot-20 dev, 1 seed, matched budget, T1 / T3 / T4-NS / T4. Pass iff T4 slice F1 exceeds T4-NS by ≥ 1pp at single-seed level (or ≥ 1 std after 3-seed expansion). Fail ⇒ paper contracts to "composite scalar gating" narrative.
- **Pilot Gate B (Claim 2)**: TwiBot-20 dev, 1 seed, matched budget, G1 / G2 / no-refinement. Pass iff G2 degrades global F1 by ≥ 0.3pp AND G1 is within 0.1pp of no-refinement. Fail ⇒ paper contracts or the gate is redesigned.

## Experiment Handoff Inputs

- Must-prove: C1 (trigger), C2 (gate), C3 (MGTAB robustness).
- Must-run pilots FIRST: Gate A + Gate B on TwiBot-20 dev, single seed.
- Must-run full matrix AFTER pilots pass: T1 / T2 / T3 / T4-NS / T4 × {TwiBot-20, TwiBot-22}; G1 / G2 / G3 × {TwiBot-20, TwiBot-22}; T4+G1 × MGTAB. 3 seeds.
- Critical metrics: global macro-F1 (non-degradation gate); high-quality-risk slice macro-F1; AURC diagnostic; empirical coverage on `test_cal` (diagnostic only); rollback rate; intervention ratio; **refinement wall-clock overhead ratio**; LLM call ratio = 0.
- Highest-risk assumptions: (i) 6-scalar search on valid_cal transfers; (ii) `s_rec` AE generalizes across TwiBot-20/22 without refit; (iii) unsupervised rollback gate does not starve useful rewrites; (iv) pilot gates pass.

## Compute & Timeline

- Pilot gates (Week 0, pre-commit): ~ 4 GPU-hr.
- v1 main (full): ~ 40-80 GPU-hr across 3 datasets × 3 seeds.
- Appendix: +30-50 GPU-hr.
- Timeline: W0 pilot gates + decision; W1 estimator + composite + AE + scalar search + T4-NS; W2 operator + deterministic modifier + EQC² gate + **locality unit tests**; W3 Claim 1+2 TwiBot-20/22; W4 Claim 3 MGTAB + appendix; W5 analysis + paper draft.

## Remaining Risks and Planned Pushbacks

- If Round 4 reviewer asks for a coverage theorem → reject, cite inductive-CP caveat and the explicit "we do not claim coverage" framing.
- If reviewer pushes for learned-modifier in main → reject, cite Round 1 contribution-focus finding.
- If reviewer asks for global-graph-rewrite comparison → acknowledge; point to FINAL_PROPOSAL.md (2026-04-19) empirical finding that global rewrite on TwiBot-20 is ±0.003 F1 within noise.
- If reviewer asks whether pilot gates are really "necessary or just defensive" → cite reviewer's own Round 3 directive that T4 > T4-NS and G1 > G2 are the two remaining load-bearing empirical claims.
