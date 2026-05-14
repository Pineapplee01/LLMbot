# Round 7 Review (Codex GPT-5.2 xhigh, NIGHTMARE difficulty)

**Thread ID**: `019e1147-4db9-7bc2-845c-fde19c29cbdd`
**Date**: 2026-05-09
**Trigger**: External-reviewer critique (different voice, same nightmare bar) → v6 package → Round 7 audit.

---

## Scores

| Dimension | R6 | R7 | Δ |
|---|---|---|---|
| 1. Problem Fidelity | 9.6 | **9.6** | 0 |
| 2. Method Specificity | 9.5 | **9.4** | -0.1 |
| 3. Contribution Quality | 9.1 | **8.7** | -0.4 |
| 4. Frontier Leverage | 8.9 | **8.6** | -0.3 |
| 5. Feasibility | 8.7 | **8.5** | -0.2 |
| 6. Validation Focus | 9.4 | **9.3** | -0.1 |
| 7. Venue Readiness | 9.1 | **8.6** | -0.5 |
| **OVERALL (weighted)** | **9.2** | **9.0** | **-0.2** |

**Verdict**: **REVISE** (despite overall ≥ 9, one IMPORTANT item blocks READY).
**Drift Warning**: **NONE**.

## The Single Remaining Blocker — C5 Regression (IMPORTANT)

**Reviewer verdict**: v6's softened C5 gate (`CI excludes 0` + `≥ 0.25·seed_std(M-rr - M-base)`) **regresses v5's sprawl-containment property**. The compound condition can pass when rewrite-only is a tiny fraction of total gain (especially when seed_std is small). That re-opens the Stage-8 mechanism-smuggling critique at nightmare bar.

**Reviewer's minimal fix**:
```
CI excludes 0 AND (M-ro - M-base) ≥ max(0.5pp, β × (M-rr - M-base))
for pre-registered β (even 0.25 is materially stronger than current 0.25·seed_std)
```

**My assessment**: the reviewer is right. My Round-7 compromise (switching fixed fraction 0.5 to `0.25 × seed_std`) converted an arbitrary threshold into a statistical threshold but **lost centrality**. The correct fix is a **compound gate**: both statistical significance AND an effect-size floor. This is an accepted partial-compromise, not a regression to v5's fixed 0.5.

## MINOR Item — u_edge component ablation

**Reviewer**: "Add a small u_edge leave-one-component-out (affinity / conflict / relation prior) mirroring Block-S2, to preempt 'heuristic stacking moved from q(v) to u_edge'."

**Accepted**: the same ablation discipline Block-S2 applies to `s_composite` must apply to `u_edge` for symmetry.

## Direct Answers to Round 7 Audit (1–9)

- **(1) EQC² → EQC terminology alignment**: **Materially improved, no novelty loss**. Novelty is "local, budgeted, reversible ego-controller on frozen backbone", not any conformal theorem. Reduces reviewer attack surface.
- **(2) 3-term composite + S2 promotion**: **Yes, largely closes "hand-crafted feature stacking"**. Leave-one-term-out + "drop if < 1 seed-std" rule converts composite from "feature soup" into a pre-registered, necessity-tested controller — consistent with prior graph-calibration/reliability work. One of v6's strongest upgrades.
- **(3) External prediction-stability guard**: **Mostly yes — defuses P0-3**. Still "internal" in that it uses frozen model outputs, but it's **external to q(v)** and blocks the exact failure mode (gaming q(v) by large output shifts). Key: it's an orthogonal constraint (margin + L∞ cap), not another monotone function of q(v).
- **(4) s_npi avoids heterophily sign trap?**: **Yes, cleanly** (no predicted-label reliance; directionality becomes "prediction neighborhood inconsistency"). Likely discriminative for low-quality egos. Mandatory camouflage-slice check + drop-if-harm is the right safeguard.
- **(5) u_edge propagation_conflict principled?**: "**More principled heuristic**." Interpretable proxy story (graph-vs-text residual direction agreement), not theory-backed. Acceptable because u_edge is explicitly commodity — but only with ablation discipline comparable to q(v) (→ MINOR fix).
- **(6) Pre-registered 768-cell grid + baselines**: **Yes, closes tuning concern at 2026 bar**. Exactly the kind of transparency reviewers accept.
- **(7) Statistical C5 kill gate**: **Addresses arbitrariness BUT regresses sprawl-containment**. See blocker above.
- **(8) C4 δ = max(0.005, 1.5·seed_std(L0))**: **Yes, much better grounded** than fixed 0.015. Factor 1.5 is convention but transparently tied to observed variability with a floor.
- **(9) Cumulative regression vs v5**: **Anchor + scope: no regression; simplicity: net improvement. ONE real regression**: weakening v5's "rewrite mechanism must be substantial share of benefit" guarantee via new C5 gate. Fixing C5 restores v5-level readiness.

---

<details>
<summary>Raw reviewer response (full verbatim)</summary>

Scores 9.6/9.4/8.7/8.6/8.5/9.3/8.6 = 9.0. Verdict REVISE (despite overall ≥ 9 because IMPORTANT item remains). Drift NONE. Simplifications NONE (net improvement). Modernizations: effect-size threshold alongside CI; u_edge component ablation. One IMPORTANT item: strengthen C5 back into centrality gate. Minimal fix: CI excludes 0 AND (M-ro − M-base) ≥ max(0.5pp, β × (M-rr − M-base)) for pre-registered β. One MINOR: u_edge leave-one-component-out. Captured above.

</details>
