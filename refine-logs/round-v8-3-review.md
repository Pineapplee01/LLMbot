# Round v8-3 Review (Codex GPT-5.2 xhigh)

**Thread**: `019e1507-3c62-7651-98d7-4a9921fb8c62`
**Date**: 2026-05-09

## Scores

| Dimension | R1 | R3 | Δ |
|---|---|---|---|
| Problem Fidelity | 8.0 | **8.5** | +0.5 |
| Method Specificity | 6.5 | **7.3** | +0.8 |
| Contribution Quality | 6.0 | **7.1** | +1.1 |
| Frontier Leverage | 7.0 | **7.8** | +0.8 |
| Feasibility | 7.0 | **8.1** | +1.1 |
| Validation Focus | 6.5 | **7.4** | +0.9 |
| Venue Readiness | 6.5 | **7.2** | +0.7 |
| **OVERALL** | **6.675** | **7.585** | **+0.91** |

**Verdict**: REVISE (was RETHINK). **Drift**: LOW (not "NONE strict" — Stage-5 semantics still partially depend on Stage-3/4 choices).

## CRITICAL Items (2)

1. **Scrambled-structure control for Stage-5 load-bearingness**: add Block-E variants with (a) shuffled role buckets, (b) shuffled retrieval_trail, (c) collapsed supportive/suspicious/uncertain into one list. Prove E-LLM degrades. Without this, reviewers will suspect "LLM just likes long text". This directly defends "protocol matters, not token volume".

2. **Stage-4 accept proxy is NOT actually external**: `(1 − p_GNN_post(ŷ|v))` with ŷ = argmax is model-self-referential and can preserve confidently wrong predictions. Replace with a **calibrated expected-error proxy**: fit `ê(confidence, entropy, q, set_size) → empirical error` on valid_cal; accept iff `ê(post) ≤ ê(pre)`. This is **actually externalized** via the valid-cal label mapping, and the key distinction the reviewer demands.

## IMPORTANT (2)

3. **Tighten dominant contribution**: Stage-5 = **(schema + deterministic bucketization + versioned serialization + pre-registered token budget)**. The object must be definable in abstraction from Stage-3/4. Stages 3/4 outputs are inputs, not part of the contribution.

4. **Pre-register Stage-5 token budget + truncation policy**. Otherwise "more tokens → better" is the trivial explanation.

## MINOR (1)

5. Clarify "linear probe ~300 params". With z_LLM dim ~3584 (Qwen-2.5-7B) + h_GNN_post ~256 + X_R ~768, concat is ~4600. 2-class probe is ~9200 params. State exact dimensionalities.

## Simplifications (reviewer)

- Stage-4: drop argmax-confidence proxy, use calibrated expected-error.
- Stage-5: cut any field not used in Block-E field-ablation.
- Stage-3: replace "sim>0" with explicit quantile gate (top-50% by calibrated sim).

## Modernizations (reviewer)

- **Same-LLM scrambled-structure control** in Block-E (CRITICAL 1).
- **Cross-dataset calibration carryover**: fit T_valid_cal / τ_bucket on TwiBot-20, apply to TwiBot-22 without refit (proves not dataset-tuned prompt engineering).
- **Field-level minimality as first-class claim** ("smallest sufficient evidence graph schema").

## Direct Answers

- Anchor preservation: **PASS**. All 7 stages intact; constraints respected.
- Dominant contribution: **much closer to single**, but Stage-5 needs explicit protocol definition independent of plumbing.
- "Pre-committed plumbing" framing: **acceptable if clean**; say "Stages 1/3/4 are fixed minimal instantiations required by the locked skeleton; we do not claim novelty there". Stage-necessity audit becomes transparency, not escape hatch.
- E-null/E-RoBERTa/E-LLM: **strong and reviewer-friendly — the best Round 2 change**. Plus scrambled-structure control.
- Temperature-calibrated similarity + Q66/Q33 bucketization: **defensible**, but rename "sim>0" to "top-50% quantile gate" explicitly.
- R_1-k* budget-matched: **right direction**. Also add single-pass multi-seed PPR (2 seeds) as additional control.
- 2-criterion accept rule: **reduces but does not eliminate self-confirmation**. Fix per CRITICAL 2.
- Remaining sprawl: **Stage-5 scope creep risk**. Defend via strict token budget + field minimality + cross-dataset calibration carryover.

---

<details>
<summary>Raw reviewer response (truncated)</summary>

7.585/10 REVISE. LOW drift. Stage-5 center clean but plumbing dependence partial. 2 CRITICAL + 2 IMPORTANT + 1 MINOR action items. "READY-shot path: scrambled-structure control + cross-dataset no-retune calibration + truly externalized accept proxy."

</details>
