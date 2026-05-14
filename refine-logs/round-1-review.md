# Round 1 Review (Codex GPT-5.2, xhigh reasoning)

**Thread ID**: `019e0bbb-afa6-7900-ad35-12daef0539cd`
**Reviewer model**: gpt-5.2 (ARIS reviewer spec says gpt-5.4; Codex harness resolved to gpt-5.2 with xhigh)
**Date**: 2026-05-09

---

## Scores

| Dimension | Score |
|---|---|
| 1. Problem Fidelity | **8** |
| 2. Method Specificity | **6** |
| 3. Contribution Quality | **6** |
| 4. Frontier Leverage | **7** |
| 5. Feasibility | **6** |
| 6. Validation Focus | **6** |
| 7. Venue Readiness | **6** |
| **OVERALL (weighted)** | **6.5 / 10** |

**Verdict**: REVISE
**Drift Warning**: Partial — the non-conformity score is mostly "conformalized label confidence", not an ego-graph-quality signal. In binary bot detection, prediction sets collapse to `{bot}`, `{human}`, or `{bot,human}`, so "set geometry" can degenerate into a near-scalar threshold. Without tying the non-conformity score more directly to ego-quality / text-graph conflict, the proposal risks re-introducing the anchor's criticized scalar-confidence primitive in a new wrapper.

---

## High-Level Take (verbatim)

- The proposal largely stays faithful to the anchor: post-hoc, local, budgeted, reversible ego refinement on a frozen RoBERTa→(Bot)RGCN backbone, with an explicit accept/rollback loop.
- The main technical risk is that "conformal ego-quality" is currently just conformalized label uncertainty from base logits. In **binary** bot detection, CP sets are usually just `{bot}`, `{human}`, or `{bot,human}`, so "set geometry" can collapse to a near-scalar threshold signal unless the score is tied more directly to ego-graph quality / text-graph conflict.
- The second risk is contribution sprawl by gravity: the counterfactual-supervised coupled modifier can easily become the real "thing", making the conformal controller feel like a wrapper unless the dominant claim is sharpened.

---

## Dimension Critiques

### 2. Method Specificity — 6 (CRITICAL)

**Weakness**: Interfaces underspecified at exactly the points that determine behavior.
- For binary CP, `set_size`, `coverage_margin`, and `abstain_risk` may be redundant unless defined precisely and shown to meaningfully differ from a calibrated scalar.
- `hard(v) = Router(...)` and `residual_risk_manifest` are named but not concretely specified.
- `Δ_down/Δ_drop/Δ_up` is conceptually clear but operationally ambiguous (what graph is re-run? which hops? cached embeddings?).

**Concrete fix**:
1. Define ONE trigger score `g(v)` deterministically from conformal outputs (e.g., `g(v) = 1[|C(v)|>1]` plus a single margin scalar), and REMOVE the trainable router in v1.
2. Replace `abstain_risk = 0.70·size_risk + 0.30·margin_risk` with either (a) a single monotone scalar tuned on `valid_cal` (1-parameter threshold), or (b) a lexicographic rule (minimize `|C(v)|`, break ties by margin).
3. Specify `Δ` supervision as one-step ego forward passes on a fixed-radius ego subgraph (explicit radius, edge-weight perturbation magnitude, batching plan).

### 3. Contribution Quality — 6 (CRITICAL)

**Weakness**: "Conformal-as-controller" is stated as dominant but coupled retrieval + 3-head modifier + counterfactual supervision + multiple output forms all join in; reviewers will see "a system of reasonable parts" rather than a single crisp mechanism.

**Concrete fix**:
- Main paper contribution = **conformal-triggered + conformal-rollbacked local refinement**.
- Demote the learned modifier to (i) single-head continuous scorer in v1, or (ii) ablation-only upgrade over a deterministic modifier.
- Provide a **deterministic modifier baseline** using only frozen signals (text similarity + relation type + simple text/graph disagreement) so the paper still stands if the learned modifier is cut.

### 5. Feasibility — 6 (IMPORTANT)

**Weakness**: Counterfactual edge supervision is the compute/method risk. Rerunning a GNN for many edges per node can burn the GPU budget and introduce complexity that dominates the paper.

**Concrete fix**:
- Cap supervision to top-K candidate edges per hard node (K small, fixed), with exact candidate selection rule.
- OR: replace counterfactual reruns with a **single backward-pass attribution proxy** (edge-weight gradient/sensitivity) as the training target or even as the modifier itself (no training). Keeps "post-hoc on a frozen backbone" in spirit.

### 6. Validation Focus — 6 (IMPORTANT)

**Weakness**: Ablation grid (E0..E5, R2..R5, B0..B2, S1..S4 + drop-conformal + drop-modifier + cross-dataset) reads like a mini-benchmark suite.

**Concrete fix**: keep only what proves the controller thesis:
1. Trigger swap (scalar vs conformal) under identical refinement.
2. Rollback gate on/off under identical trigger+refiner.
3. Cross-dataset once (TwiBot-22) with the final controller+refiner.

Everything else → optional appendix unless it changes the conclusion.

### 7. Venue Readiness — 6 (IMPORTANT)

**Weakness**: Novelty hinges on "prediction-set geometry as control". In binary classification the geometry can appear thin. Also, conformal validity language can draw scrutiny on graph exchangeability / inductive CP regimes where standard CP guarantees break due to score shifts from message passing.

**Concrete fix**: tighten the claim — either (a) make the non-conformity score **quality-aware** (see Drift Warning), OR (b) reframe the controller as a *calibrated selection rule* rather than making full CP-validity claims, and cite inductive-CP caveats explicitly.

---

## Answers to the 4 Open Questions

1. **E4 (DAPS/NAPS diffused conformal) in v1 main, or ablation-only?** — **Ablation-only.** Diffusion in main creates a second center of gravity and muddies the dominant contribution.
2. **Does counterfactual-intervention supervision look pseudo-causal?** — **Yes, it can be misread.** Rename and scope as *model-based sensitivity supervision* ("simulated edge-weight perturbation gain under the frozen backbone") and state explicitly: not causal identification of real-world influence.
3. **Is `0.70·size_risk + 0.30·margin_risk` too hand-tuned?** — **Yes (and unnecessary).** Pick a single monotone selection statistic, tune one threshold on `valid_cal`, keep the decomposition as reporting, not as a hand-mixed controller.
4. **MGTAB vs Cresci-17 as secondary robustness?** — **Prefer MGTAB.** Explicitly graph-based, multi-relational (7 relation types); stress-tests edge reweighting / heterophily-camouflage better than Cresci-17.

---

## Simplification Opportunities (Reviewer)

1. Delete trainable router in v1 — use a deterministic `g(v)` from conformal outputs.
2. Deterministic modifier baseline (frozen signals only) as v1 main; learned modifier moves to ablation.
3. Collapse `abstain_risk` to a single monotone scalar with one threshold on `valid_cal`.

## Modernization Opportunities (Reviewer)

1. Enrich the non-conformity score with a quality-specific signal (e.g., text-graph disagreement, local heterophily, ego-embedding reconstruction error) so the "ego-quality" framing holds in binary classification.
2. Use single-backward-pass gradient attribution (frozen backbone) in place of counterfactual reruns — cleaner, cheaper, still model-based.

---

<details>
<summary>Raw reviewer response (truncated at harness limit)</summary>

Scores plus dimension commentary captured above verbatim. The raw tool output was truncated at ~6KB; the key fragments ("Simplification", "Modernization", "Drift Warning", "Verdict") were recovered from the delivered content. No content extrapolation — every quoted sentence is from the reviewer.

</details>
