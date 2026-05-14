# Round 4 Review (Codex GPT-5.2, xhigh reasoning) — READY

**Thread ID**: `019e0bbb-afa6-7900-ad35-12daef0539cd`
**Date**: 2026-05-09

---

## Scores

| Dimension | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| 1. Problem Fidelity | 8 | 9 | 9 | **10** |
| 2. Method Specificity | 6 | 8 | 9 | **10** |
| 3. Contribution Quality | 6 | 8 | 9 | **9** |
| 4. Frontier Leverage | 7 | 8 | 8 | **8** |
| 5. Feasibility | 6 | 8 | 9 | **9** |
| 6. Validation Focus | 6 | 8 | 9 | **9** |
| 7. Venue Readiness | 6 | 6 | 8 | **8** |
| **OVERALL (weighted)** | **6.5** | **8.1** | **8.8** | **9.2** |

**Verdict**: **READY (spec-level; conditional on Pilot Gates A/B)**.
**Drift Warning**: **NONE**.

---

## Reviewer's Answer to the Framework Question

> "Is 'READY' allowed without the pilot numbers landing yet?"
>
> **Yes for a `/research-refine` stage proposal.** With the locality pinned, tuning protocol locked, and explicit falsifiable pilot gates + contraction plan, there is no longer a *spec-level* blocker.
>
> **Paper-level READY is conditional**: the paper's headline (set-based quantile controller adds beyond scalar composite gating; unsupervised rollback prevents harm) still depends on Pilot Gates A/B passing.

This is the correct stopping point for `/research-refine`. The skill is not supposed to run the experiments; it is supposed to produce a spec that is **executable and falsifiable**. Both are now true.

## Round 3 Blockers — Status

| Round 3 Action Item | Round 4 Status |
|---|---|
| Locality of recomputation spec | **Resolved and auditable**. L-hop-for-L-layer locality is standard message-passing dependency structure; reviewer cites DGL neighbor-sampling docs as baseline. |
| T4 > T4-NS empirical evidence | **Explicitly falsifiable** via Pilot Gate A. |
| G1 vs G2 empirical evidence | **Explicitly falsifiable** via Pilot Gate B. |

## Simplification Opportunities

**NONE (material)**. One optional presentational note: if `w_rec → 0` consistently in the weight search, drop `s_rec` entirely from the main text and keep it appendix-only. This is handled automatically by the Round 2 auto-collapse mechanism.

## Modernization Opportunities

**NONE**. The foundation-model-era primitive usage (frozen LM head + disagreement in probability space) is the natural minimal lever under the anchor's constraints.

## Remaining Action Items (for the execution phase, not the refinement phase)

1. Execute **Pilot Gate A** and **Pilot Gate B** exactly as specified (Week 0). If either fails, follow the pre-committed contraction; do not patch with extra modules.
2. Log the new locality / audit metrics (refinement wall-clock overhead ratio; any locality-test failures) alongside the pilot outcomes — part of what makes the method believable as "post-hoc local".
3. If pilots pass, freeze the tuning protocol as-written (no creeping degrees of freedom) and proceed to the 3-seed runs.
4. If pilots fail, rewrite claims and title to match the contracted contribution (composite scalar controller, or redesigned gate).

---

<details>
<summary>Raw reviewer response (full)</summary>

Scores 10/10/9/8/9/9/8, overall 9.2. "READY (spec-level; conditional on Pilot Gates A/B)". Anchor preserved. Locality of recomputation resolved. Set-based necessity + gate necessity both falsifiable. Simplifications NONE (material). Modernization NONE. Remaining action items are all execution-phase, not refinement-phase. Captured verbatim above.

</details>
