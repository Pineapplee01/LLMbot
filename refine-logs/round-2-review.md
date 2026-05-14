# Round 2 Review (Codex GPT-5.2, xhigh reasoning)

**Thread ID**: `019e0bbb-afa6-7900-ad35-12daef0539cd`
**Date**: 2026-05-09

---

## Scores

| Dimension | Round 1 | Round 2 | Δ |
|---|---|---|---|
| 1. Problem Fidelity | 8 | **9** | +1 |
| 2. Method Specificity | 6 | **8** | +2 |
| 3. Contribution Quality | 6 | **8** | +2 |
| 4. Frontier Leverage | 7 | **8** | +1 |
| 5. Feasibility | 6 | **8** | +2 |
| 6. Validation Focus | 6 | **8** | +2 |
| 7. Venue Readiness | 6 | **6** | 0 |
| **OVERALL (weighted)** | **6.5** | **8.1** | **+1.6** |

**Verdict**: REVISE (overall < 9; one blocking Venue Readiness issue).
**Drift Warning**: **NONE**. Anchor preserved.

---

## What Improved (Reviewer's words)

- Binary-CP thinness **materially addressed** by moving ego-quality into the score definition itself.
- Focus is **sharper**: removing the trainable router and demoting learned modifier/sensitivity supervision to appendix kills the biggest sprawl risk.
- Modernity is **appropriate**: frozen PLM as semantic prior, no forced LLM.
- Both flagged items signed off:
  - **Deterministic modifier as supporting contribution**: ACCEPTABLE (keep) — need a consumer to demonstrate controller → action → rollback as a method.
  - **`max_iter=1` v1 main**: ACCEPTABLE (keep) — aligned with anchor (reversible, auditable, avoid confirmation bias).

## The Single Blocking Issue — Venue Readiness (6/10, CRITICAL)

**Weakness**: The paper is still at risk of being read as "a well-tuned composite heuristic + quantile threshold" rather than a crisp mechanism contribution. Two concrete reasons:

1. We still call it "conformal" while explicitly not claiming coverage; reviewers will treat this as pseudo-novel branding unless we justify *why the conformalization step (q̂ / set events) is essential beyond scalar thresholding*. Also, in binary classification, set-size behavior can collapse depending on effective thresholds; we need to show the construction avoids reverting to always-singleton sets in practice.
2. The 7-scalar joint search (plus AE fit) is "training by another name". Raises (i) selection-bias concerns and (ii) hidden-degrees-of-freedom concerns unless the tuning protocol is locked down very explicitly (nested split or hard holdout).

**Concrete fix (method-level, minimal)**:
1. Add a **non-conformal baseline**: same `s_composite`, same deterministic modifier, but no `q̂` / prediction set — just threshold `s_composite` for trigger and rollback. If EQC² beats this, the set-derived features are demonstrably load-bearing, not decorative.
2. Make the tuning protocol airtight: (train_cal for AE), (valid_cal for the 7-scalar search), **separate untouched test for final reporting**; or a lightweight nested split to avoid optimistic bias from extensive search.
3. Tighten novelty language: "quantile-calibrated set-based controller" is safer than implying CP-style guarantees on graphs.

**Priority**: CRITICAL.

## Remaining Action Items (highest ROI, specificity-level)

1. **Define `s_composite(v)` and `margin(v)` formally** — is `s_composite = min_y s(v,y)`? `s(v, ŷ)`? Must be pinned because it appears in trigger ordering, rewrite amplitude, and rollback.
2. **Fix representation mismatch in `s_tg`**: cosine between `X_R[v]` and `h_GNN[v]` is undefined unless they're in the same space; specify exact mapping (preferably zero-parameter), or switch to a disagreement in probability space (divergence between two predictive distributions already available).
3. **Add the "no-set" baseline** (same composite, scalar gating) to defend the set-based controller necessity.
4. **Lock the tuning protocol** (splits + when `q̂` is fit).
5. **State binary-set edge cases** explicitly: what happens when prediction set is empty or size = 2, and how the lexicographic rule treats those cases.

If items 1–4 are addressed, reviewer expects Venue Readiness 6 → ~8 without adding modules.

## Simplification Opportunities (optional)

1. Drop AE (`s_rec`) unless it is clearly additive beyond `s_tg + s_het`; keep only if it moves trigger ranking. If `w_rec → 0` often in valid_cal search, cut it from v1.
2. Reduce 7-scalar search degrees of freedom: fix `w_lbl = 1` and search 3 composite weights + 3 modifier alphas (or constrain to simplex).
3. Rename "conformal" in the title if needed: "Quantile-Calibrated Ego-Quality Controller" preserves mechanism without inviting guarantee fights.

## Modernization Opportunities

**NONE** — already modern in the right way for this problem / constraint set.

---

<details>
<summary>Raw reviewer response (full)</summary>

Scores, Problem Anchor status, What Improved, dimension critique, drift status, signoff on flagged items, simplification opportunities, modernization opportunities, and five action items — all captured verbatim above. Key citations from reviewer: MAPIE conformal theoretical foundations, scikit-learn nested cross-validation, OpenReview graph-inductive-CP landmine paper. These are background references, not direct dependencies.

</details>
