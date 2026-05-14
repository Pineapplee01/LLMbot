# Round 5 Review (Codex GPT-5.2 xhigh, NIGHTMARE difficulty)

**Thread ID**: `019e1147-4db9-7bc2-845c-fde19c29cbdd` (fresh session; prior thread `019e0bbb-...` expired)
**Date**: 2026-05-09
**Difficulty**: beast / nightmare — strictest verdict rule (READY only if overall ≥ 9 AND zero drift AND every load-bearing claim has an explicit falsification path AND no IMPORTANT-or-higher item remains).

---

## Scores

| Dimension | R4 | R5 (nightmare) | Δ |
|---|---|---|---|
| 1. Problem Fidelity | 10 | **9.4** | -0.6 |
| 2. Method Specificity | 10 | **9.2** | -0.8 |
| 3. Contribution Quality | 9 | **8.1** | -0.9 |
| 4. Frontier Leverage | 8 | **8.4** | +0.4 |
| 5. Feasibility | 9 | **8.0** | -1.0 |
| 6. Validation Focus | 9 | **8.5** | -0.5 |
| 7. Venue Readiness | 8 | **6.8** | -1.2 |
| **OVERALL (weighted)** | **9.2** | **8.6** | **-0.6** |

**Verdict**: **REVISE**.
**Drift Warning**: **NONE** (anchor constraints preserved; scope expanded but still within "local reversible ego refinement on frozen backbone").

## Nightmare-Difficulty Interpretation

The rescope (adding Stage-7 artifact + Stage-8a artifact refiner + Stage-8b LLM refiner + Claim C4 + Block-R + Block-BR + R120 + R121–R123 + R125) cost ~0.6 points against the Round-4 9.2. Every dimension dropped except Frontier Leverage. The primary damage site is **Venue Readiness (8 → 6.8)** because Stage-7/8 introduces supervised-refiner components that a top-venue reviewer will credibly attribute gains to — rather than to the EQC² controller.

## The Single Dimension < 7 — Venue Readiness (6.8, IMPORTANT)

**Weakness**: Stage-7/8 risks reading as a second mechanism ("new supervised refiner head + artifact representation") competing with EQC²-as-the-primitive. Without explicit mechanism-isolation ablations, reviewers can credibly claim the gains come from the refiner rather than the budgeted reversible ego rewriting and its controller.

**Concrete fix** (do both, minimally):
1. Add a **mechanism-isolation block**: `NoRewrite+Refiner` vs `RewriteOnly(no refiner; p_GNN_post)` vs `Rewrite+Refiner`. Proves EQC²-controlled rewriting is the mechanism, not Stage-8.
2. Tighten paper spine to **one** dominant claim: "single composite uncertainty-under-ego-context score reused as trigger + amplitude + rollback gate". Treat artifact + refiner explicitly as *commodity interface* (or demote to appendix if isolation block fails).

**Priority**: IMPORTANT.

## Remaining Action Items

| # | Priority | Item |
|---|---|---|
| 1 | IMPORTANT | Add mechanism-isolation experiments (NoRewrite+Refiner vs RewriteOnly vs Rewrite+Refiner) |
| 2 | IMPORTANT | Strengthen BotBR/BECE differentiation — faithful public baseline if compatible, or explicitly justify non-reproduction with sensitivity checks |
| 3 | IMPORTANT | Upgrade C4 evaluation from ratio-only to non-inferiority (absolute margin + confidence intervals across seeds) |
| 4 | MINOR | Reduce 8-scalar tuning DOF perception — freeze τ+/τ- to fixed `valid_cal` quantiles (e.g. 75th / 25th percentile of `u_edge` distribution); keep search at 6 scalars |

## Simplification Opportunities

1. Collapse Stage-8 reporting to `L0` vs `L2` only as main; `L1` / `L3` strictly appendix/diagnostic; A-Prompt conditional.
2. CF-learned-modifier appendix fully optional: if ANY of R121/R122/R123 fires, DROP the entire learned-modifier story (not just "not recommended"). Strict gating.
3. Freeze `τ+/τ-` by fixed quantiles (not tuned); keep only `{w_tg, w_het, w_rec, α_text, α_rel, α_tgd}` in the 6-scalar search.

## Modernization Opportunities

1. Replace ratio-only C4 with **non-inferiority test** framing (absolute margin + CIs across seeds). F1 ratios are unstable when denominators move.
2. Report **risk–coverage / AURC curves** for all trigger variants (not just a single operating point) — aligns with modern selective prediction evaluation.
3. Add seeded uncertainty to main load-bearing comparisons (≥ 3 seeds for decisive blocks after Pilot A/B pass) — meets 2026 reviewer expectations.

## Direct Answers to Audit Questions (a–h)

- **(a) Sprawl / dominant primitive** — Moderate sprawl risk. EQC² can still be the single dominant primitive *iff* the paper explicitly demonstrates (i) rewriting+gate produces gains even without the refiner, and (ii) the refiner is not the primary mechanism. Without the isolation block, EQC² risks becoming "one-of-four" (score + modifier + artifact + supervised refiner).
- **(b) Stage-8a a smuggled contribution?** — Borderline. As written (MLP trained on train_cal labels), Stage-8a is a new supervised model component; it will be perceived as a contribution unless positioned as commodity readout AND empirically shown to not carry the core lift.
- **(c) Does the tightened estimator claim survive binary-CP thinness?** — Mostly yes. "Prediction-set uncertainty under current ego context" + "no coverage claims" + D-EstSem observational semantics study is the right repair. Residual risk: avoid CP-implying language that sounds distribution-free / theorem-backed; treat "quantile-calibrated" as heuristic calibration unless formal exchangeability is assumed (it is not).
- **(d) BR / BR-soft sufficient vs BotBR/BECE?** — Directionally good but not fully sufficient. Reviewers may still ask for a faithful reproduction OR a clearly justified "cannot under frozen-backbone constraint" statement. Min bar: BR/BR-soft as close as possible to BotBR/BECE within constraints + sensitivity checks; ideally one public implementation if compatible.
- **(e) C4 ratio 0.90 lenient/strict?** — Somewhat **lenient** and statistically awkward. Prefer `L2 - L0 ≤ δ` (absolute non-inferiority margin with CIs), or dual criteria (ratio + absolute).
- **(f) R121 + R122 + R123 sufficient for CF-learned-modifier honesty?** — Strong and close to sufficient. Add one more: **CF-signal permutation** (permute CF residual/targets across nodes within degree bins). If performance persists under permutation, learned modifier is not using CF information meaningfully and should be excluded.
- **(g) 8-scalar search tractable or hidden-DOF risk?** — Computationally tractable, but perception risk reappears. Mitigate via fixed search budget, coordinate-descent schedule disclosure, sensitivity/stability plots on valid_cal; no post-hoc per-dataset re-tuning.
- **(h) Block-R promoted to main — too wide or properly defends?** — Promoting Block-R is **justified**. Soft-vs-hard rewrite is central to differentiating from binary reliability and camouflage destruction. Keep it tightly scoped (two datasets, minimal seeds until pilots pass).

---

<details>
<summary>Raw reviewer response (full verbatim)</summary>

All scores, verdict, drift status, dimension critique, simplifications, modernizations, and answers (a)-(h) captured above verbatim. New thread ID `019e1147-4db9-7bc2-845c-fde19c29cbdd` replaces the expired `019e0bbb-...`.

</details>
