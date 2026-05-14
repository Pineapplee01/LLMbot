# Round v8-5 Review (Codex GPT-5.2 xhigh) — FINAL (MAX_ROUNDS reached)

**Thread**: `019e1507-3c62-7651-98d7-4a9921fb8c62`
**Date**: 2026-05-09

## Scores

| Dimension | v8-1 | v8-3 | v8-4 | **v8-5** |
|---|---|---|---|---|
| Problem Fidelity | 8.0 | 8.5 | 8.8 | **9.0** |
| Method Specificity | 6.5 | 7.3 | 8.0 | **8.5** |
| Contribution Quality | 6.0 | 7.1 | 7.6 | **8.6** |
| Frontier Leverage | 7.0 | 7.8 | 8.0 | **8.3** |
| Feasibility | 7.0 | 8.1 | 8.8 | **9.2** |
| Validation Focus | 6.5 | 7.4 | 8.1 | **8.6** |
| Venue Readiness | 6.5 | 7.2 | 7.7 | **8.2** |
| **OVERALL** | **6.675** | **7.585** | **8.09** | **8.63** |

**Verdict**: REVISE. **Drift**: NONE. "Clean monotone tightening pass with stronger falsification and less schema sprawl."

## Why Not READY at MAX_ROUNDS

Reviewer's verbatim diagnosis:
> "Not READY. Remaining gap is **primarily empirical, not structural**. Structurally, you now meet the 'one dominant contribution' and 'no obvious complexity bloat' bars, and drift is essentially zero. What blocks READY is that **CQ/MS at ≥ 9 depends on results**: the strengthened Block-E/CT suite must actually come out positive (and not fragile). If the experiments confirm your acceptance criteria cleanly on both datasets and both LLMs, the work could plausibly cross into READY territory; if they don't, no further proposal polishing will fix it."

"You've set up the latter (extremely convincing empirical dominance). You haven't demonstrated it yet."

## CRITICAL / IMPORTANT Remaining (all proposal-polish level)

1. **CRITICAL**: One sentence framing scientific novelty as "controlled representation protocol with falsifiable invariance/sensitivity tests" — make unmissable in intro + contributions list.
2. **IMPORTANT**: Minimal formalism — define Stage-5 as `f: (ego-subgraph + per-node signals) → token sequence under a budget` and state invariances (role / order / collapse) as properties of `f`.
3. **IMPORTANT**: Pre-register what happens if Block-E condition (iv) fails (Qwen ≠ Mistral): is protocol "works for Qwen-class" or negative result?

## Simplifications (reviewer)

1. Keep canonical ordering contingent (drop if shuffle-order ties); don't pre-commit as contribution unless survives both datasets.
2. If bucketization survives but ordering doesn't, present protocol as "schema + bucketization + budget" (3-tuple, not 4).
3. Limit Stage-5 v1.0 fields to those surviving scrambles (already enforced).

## Modernizations (reviewer)

1. 2-LLM robustness move is right; keep in main.
2. Promote Block-CT cross-dataset calibration transfer to **main-text secondary pillar** (best "not prompt-fit" defense).
3. If space allows, add "LLM temperature / decoding fixed" + "prompt strictly deterministic" as explicit reproducibility constraints.

## Direct Answers

- **(a) 4-part Claim C resolves permissive centerpiece?** Yes, largely. Direction on BOTH + scramble degradations + 2-LLM agreement directly targets main suspicions. Remaining mild permissiveness on magnitude is fine given directionality + invariance + robustness are real centerpiece.
- **(b) Edge-order + 2-LLM push CQ across 9?** "Pushes up meaningfully, but ceiling remains on proposal-only evaluation." Core object is still representation/serialization protocol; reviewers may label "schema engineering" unless results are strong AND consistent. Controls make it scientifically testable — but CQ ≥ 9 needs convincing empirical dominance.
- **(c) ê_φ cross-fit enough?** YES. Cross-fit addresses overfit concern. Remaining concern is expressivity limit (backbone-derived features can miss certain confident-wrong regimes) — inherent, mitigated by CT + robustness checks. Not a blocker.
- **(d) E-LLM-collapse narrative defensible?** Yes, much stronger. "Schema engineering done as a controlled method paper, not ad hoc prompt tuning."
- **(e) 9 Block-E variants: bloat?** "Still under control." Evaluation complexity, not method complexity. Principled (encoder swap, structure scrambles, collapse, 2 LLMs). Reads as "serious falsification".
- **(f) READY at this score?** "Not READY. Remaining gap is primarily empirical, not structural." No further proposal polishing will fix it. Need Block-E/CT results.

---

<details>
<summary>Raw reviewer response (truncated)</summary>

8.63/10 REVISE. NONE drift. "Clean monotone tightening." CQ/MS ≥ 9 depends on results; structurally complete. Reviewer offers "one-page Block-E + CT result interpretation rubric (what to claim under each pass/fail pattern)" for the paper-writing phase.

</details>
