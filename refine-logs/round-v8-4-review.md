# Round v8-4 Review (Codex GPT-5.2 xhigh)

**Thread**: `019e1507-3c62-7651-98d7-4a9921fb8c62`
**Date**: 2026-05-09

## Scores

| Dimension | v8-1 | v8-3 | v8-4 |
|---|---|---|---|
| Problem Fidelity | 8.0 | 8.5 | **8.8** |
| Method Specificity | 6.5 | 7.3 | **8.0** |
| Contribution Quality | 6.0 | 7.1 | **7.6** |
| Frontier Leverage | 7.0 | 7.8 | **8.0** |
| Feasibility | 7.0 | 8.1 | **8.8** |
| Validation Focus | 6.5 | 7.4 | **8.1** |
| Venue Readiness | 6.5 | 7.2 | **7.7** |
| **OVERALL** | **6.675** | **7.585** | **8.09** |

**Verdict**: REVISE (CQ ceiling). **Drift**: NONE (explicitly "negligible" — this was a tightening pass, not a new story).

## What Holds READY Back (final blocker: Contribution Quality ceiling)

> "Remaining blocker is CQ ceiling: a Stage-5 serialization/budgeting protocol can still be perceived as 'prompt/schema engineering' unless you (a) show invariance/generalization beyond the current setting (starting via Block-CT), and/or (b) show the protocol is *the* minimal sufficient mechanism through your scramble/remove rules."
>
> "Expect the current shape to land as a strong REVISE, not a confident READY ≥ 9."

READY-shot lever (reviewer explicit): **promote edge-order/permutation invariance test from appendix to Block-E main**. This directly aligns with "protocol" as a contribution and bumps CQ/MS without new machinery.

## CRITICAL / IMPORTANT Action Items (3)

1. **CRITICAL**: Upgrade Claim C criterion from "≥ 1 seed-std on ≥ 1 dataset" to "**consistent direction on both datasets; magnitude threshold on at least one**". Current bar feels permissive for a centerpiece claim.
2. **IMPORTANT**: Add positioning sentence: "Stage-5 is a *protocol* whose key property is **predictable, testable sensitivity** (scramble tests) rather than 'best prompt'." Reduces prompt-engineering critique directly.
3. **IMPORTANT**: For ê_φ, pre-register guard against calibration overfit via **cross-fitting inside valid_cal** (or promote nested-split robustness into main text for Stage-4 accept).

## Simplifications (reviewer)

1. Make `retrieval_trail` appendix-only by default; promote to core only if Block-E shows it's load-bearing (reduces prompt-hacking risk).
2. Default `E-LLM-collapse` as the "minimal artifact" baseline; treat supportive/suspicious/uncertain as the incremental contribution (cleaner story).
3. Freeze Stage-3 ordering rule as part of the protocol (rank-by-PPR then sim); no secondary re-ranking.

## Modernizations (reviewer)

1. **Edge-order randomization within each bucket** (same tokens, permuted order) — READY-shot lever.
2. **LLM family robustness**: at least 2 LLMs, same Stage-5; single-model gains = "model-specific prompt fit" risk.
3. **Budget-256 vs Budget-512** (same schema) to show protocol benefit isn't just "more context".

## Direct Answers

- **ê_φ genuinely externalized**: YES (held-out label supervision is what makes it real). Limitation: still a meta-model over backbone features; if backbone is confidently wrong in a regime not represented in valid_cal, ê_φ misses it. Block-CT + nested-split is the right hedge.
- **Scrambled controls isolate protocol from token volume**: YES substantially. Identical 512-token budget enforced across variants removes "more tokens" explanation.
- **4-tuple Stage-5 definition independent of Stages 3/4**: conceptually independent (good enough), empirically still coupled (inevitable for a locked skeleton). OK because (i) minimal plumbing claims, (ii) Stage-5-centered falsification tests.
- **READY achievable Round 5**: reviewer explicitly says "**Still work left under your strict READY rule**" but offers READY-shot tweaks.

---

<details>
<summary>Raw reviewer response (truncated)</summary>

8.09/10 REVISE. NONE drift. CQ ceiling final blocker. 1 CRITICAL + 2 IMPORTANT + 3 simplifications + 3 modernizations. Expected verdict stability: strong REVISE current → potential READY if edge-order invariance + Claim-C tightening + LLM-family robustness + ê_φ cross-fit all land in Round 5.

</details>
