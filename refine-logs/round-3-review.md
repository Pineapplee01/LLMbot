# Round 3 Review (Codex GPT-5.2, xhigh reasoning)

**Thread ID**: `019e0bbb-afa6-7900-ad35-12daef0539cd`
**Date**: 2026-05-09

---

## Scores

| Dimension | Round 1 | Round 2 | Round 3 | Δ (R2→R3) |
|---|---|---|---|---|
| 1. Problem Fidelity | 8 | 9 | **9** | 0 |
| 2. Method Specificity | 6 | 8 | **9** | +1 |
| 3. Contribution Quality | 6 | 8 | **9** | +1 |
| 4. Frontier Leverage | 7 | 8 | **8** | 0 |
| 5. Feasibility | 6 | 8 | **9** | +1 |
| 6. Validation Focus | 6 | 8 | **9** | +1 |
| 7. Venue Readiness | 6 | 6 | **8** | +2 |
| **OVERALL (weighted)** | **6.5** | **8.1** | **8.8** | **+0.7** |

**Verdict**: REVISE (overall < 9, but remaining blockers are explicitly empirical).
**Drift Warning**: **NONE**.

---

## Prior Blocker — Resolved at Spec Level

> Reviewer: "Mostly yes at the proposal/spec level. The previous blocker ('this might just be a composite heuristic with conformal branding / tuning ambiguity') is substantially addressed by (i) formal definitions, (ii) JSD fix, (iii) locked split/tuning protocol + leakage asserts, and (iv) the T4-NS falsifier."

## What Still Blocks READY — Empirical-Risk, Not Spec-Risk

> Reviewer: "A top-venue 'controller' paper needs the set-based quantile step and the unsupervised gate to be *measurably* load-bearing (T4 > T4-NS, and G1 prevents global harm vs G2). You've set this up cleanly; you just can't claim it yet."

Two of the three remaining action items are **pilot results** that by construction cannot land inside `/research-refine`:

1. **Empirical (Claim 1)**: Show T4 beats T4-NS by a non-trivial margin on the high-risk slice.
2. **Empirical (Claim 2)**: Show G1 (unsupervised rollback) prevents global degradation vs G2 at matched budgets.
3. **Spec (addressable now)**: Specify / verify the locality of the recomputation after reweighting (exact forward pass rerun, exact subgraph) so the "post-hoc local intervention" is operationally bounded, not an implicit global rerun hidden in "re-eval".

## Simplification Opportunities

1. **Drop `virtual_context_edges` from v1 main** unless there's a concrete use; currently "evidence-only" but not in any decision rule.
2. **Tighten the "zero trainable inference parameters" phrasing**: the `s_rec` AE and `p_LM` fallback MLP are trained artifacts even if used at test time. Safer: "no end-to-end retraining; only calibration-time fits (unsupervised / distillation-style)".

## Modernization Opportunities

**NONE** — foundation-model-era primitives used are the natural minimal ones, not forced.

---

<details>
<summary>Raw reviewer response (full)</summary>

Scores, anchor status, "prior blocker resolved at spec level" finding, two empirical items + one spec item, two simplifications, modernization NONE. Captured verbatim above.

</details>
