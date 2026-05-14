# Round 5 Refinement

**Round**: 5
**Review source**: `refine-logs/round-5-review.md` (overall 8.6, REVISE at nightmare difficulty)
**Headline revision**: add mechanism-isolation block (proves EQC² rewrite+gate carries the lift, not Stage-8); strengthen BotBR/BECE differentiation with faithful-baseline commitment + sensitivity checks; upgrade C4 to non-inferiority framing with CIs and AURC curves; freeze τ+/τ- to fixed valid_cal quantiles (search drops 8 → 6 scalars again); add CF-permutation falsification (R123b); enforce strict kill rule on learned-modifier appendix.

---

## Problem Anchor (verbatim from Round 0)

[unchanged]

## Anchor Check

- Original bottleneck still targeted: yes.
- All three IMPORTANT items from Round 5 are compatible with the anchor — they *tighten* EQC² as the dominant primitive by isolating it empirically.
- Reviewer suggestions rejected as drift: none.

## Simplicity Check

- Dominant contribution unchanged: EQC² single primitive. The mechanism-isolation block **reinforces** this framing by demonstrating empirically that the refiner is a commodity readout, not the mechanism.
- Components reduced:
  - 8 scalars → **6 scalars** (τ+ and τ- now fixed as the 75th and 25th quantiles of `u_edge` on `valid_cal` — a one-time computation, not a search variable).
- Components added (all falsifiers, not mechanism):
  - Mechanism-isolation block (proves EQC² is the mechanism).
  - BotBR faithful reproduction attempt + sensitivity table (differentiation evidence).
  - AURC curves + non-inferiority CI framing (modern evaluation).
  - CF-signal permutation control (R123b, additional CF-learned-modifier falsifier).

---

## Changes Made (Round 5)

### 1. Mechanism-isolation block (Round-5 IMPORTANT #1)

**Reviewer said**: "Add a mechanism isolation block: NoRewrite+Refiner vs RewriteOnly(no refiner; p_GNN_post) vs Rewrite+Refiner to show EQC²-controlled rewriting is necessary and not just a feature-engineering front-end for a new classifier."

**Action**: add **Block-M** (Mechanism Isolation) to the main experiment. Four variants, trigger = T4, gate = G1 (both the v1 main configuration):

| Variant | Ego rewrite (Stage-5) | Stage-6 gate | Refiner (Stage-8a) | Purpose |
|---|---|---|---|---|
| **M-base** | NO | — | NO (`p_final = p_GNN_base`) | Baseline (Stage-0 only) |
| **M-ro** | YES (soft reweight on hard v) | YES | NO (`p_final = p_GNN_post`) | **RewriteOnly** — pure EQC² mechanism, no Stage-8 |
| **M-re** | NO | — | YES (Stage-8a with `Artifact(v)` built on *unrewritten* ego) | **NoRewrite+Refiner** — refiner without the controller |
| **M-rr** | YES (soft reweight on hard v) | YES | YES (v1 main config) | **Rewrite+Refiner** — full v1 main |

**Claim**:
- `M-rr ≥ M-ro ≥ M-re ≥ M-base` on high-uncertainty ego slice F1, AND
- `M-ro - M-base ≥ 0.5 · (M-rr - M-base)` — EQC² rewrite+gate alone must carry **at least half** of the total lift (empirical necessity of the controller).

**Falsification**: if `M-ro - M-base < 0.5 · (M-rr - M-base)` on BOTH TwiBot-20 AND TwiBot-22 → EQC² rewrite+gate is not the mechanism; paper demotes Stage-5/6 to an enhancer and recenters on the artifact + refiner as the mechanism. This is a hard kill gate on the dominant contribution, and I accept it.

### 2. BotBR/BECE faithful reproduction (Round-5 IMPORTANT #2)

**Reviewer said**: "Either run a faithful public baseline (if compatible) or explicitly justify non-reproduction and make BR/BR-soft maximally faithful + sensitivity-checked."

**Action**: dual commitment.

- **BR-public**: attempt a faithful reproduction using the public BotBR implementation (checked: [https://doi.org/10.1145/3726302.3729908](https://doi.org/10.1145/3726302.3729908) — GitHub accessible). If compatible with our frozen RoBERTa + BotRGCN backbone, run as a top-tier prior-art baseline on TwiBot-20 + TwiBot-22. If *incompatible* (e.g. requires retraining the base GNN, needing architectures we do not use), document the incompatibility in the paper and fall back to the BR / BR-soft within-method baselines.
- **BR / BR-soft sensitivity**: for both BR and BR-soft, report a **sensitivity table** varying (i) the reliability threshold, (ii) the fraction of edges labelled unreliable, (iii) the soft downweight factor (for BR-soft). Three settings each. Demonstrates that the BR comparison is not cherry-picked.
- **Decision rule**: if BR-public or BR-soft-best-setting **beats** T4 on slice F1 for both datasets, paper **concedes** prior-art sufficiency and contracts to a *refinement-of-prior-art* framing. Honest. No patching.

### 3. C4 non-inferiority upgrade + AURC curves (Round-5 IMPORTANT #3)

**Reviewer said**: "Upgrade C4 evaluation from ratio-only to non-inferiority (absolute margin + confidence intervals across seeds). F1 ratios are unstable when denominators move."

**Action**: reframe C4 explicitly.

- **C4 (non-inferiority)**: `slice_F1(L2) - slice_F1(L0) ≤ δ = 0.015` across 3 seeds, reported with 95% paired-bootstrap CI. Dual criterion retained: additionally `slice_F1(L0) ≥ 0.90 · slice_F1(L2)`.
- **AURC curves**: for all trigger variants (T1, T2, T3, BR, BR-soft, T4-NS, T4), plot risk-coverage curves on TwiBot-20 dev, TwiBot-22 dev, MGTAB dev. AURC becomes a secondary metric in Block-T; the headline remains slice F1 at fixed intervention ratio. This is **modernization only** — the core claim is unchanged.
- **Seed floor** (reviewer Modernization #3): **3 seeds** for every decisive block (Block-T, Block-G, Block-M, Block-R, Block-X). Pilot gates remain single-seed by design but the full matrix is always ≥ 3 seeds.

### 4. τ+ / τ- freeze (Round-5 MINOR + Simplification #3)

**Reviewer said**: "Freeze τ+/τ- by fixed quantiles (not tuned) to reduce hidden-DOF perception; keep only {w_*, α_*} in the 8-scalar search if possible."

**Action**:
- `τ+ = Quantile_75(u_edge over valid_cal ego edges)`. One-time computation.
- `τ- = Quantile_25(u_edge over valid_cal ego edges)`. One-time computation.
- `τ0 = min(|τ+|, |τ-|)` (conservative symmetric neutral band).
- The calibration search drops back to **6 scalars**: `{w_tg, w_het, w_rec, α_text, α_rel, α_tgd}`. τ values are **published** in the calibration artifact alongside `q_hat` and `τ*`.

### 5. CF-signal permutation falsifier (Round-5 Audit (f))

**Reviewer said**: "Add CF-signal permutation (permute CF residual/targets across nodes within degree bins). If performance persists, learned modifier is not using CF information meaningfully and should be excluded."

**Action**: add **R123b** to the learned-modifier appendix. Procedure: bin nodes into degree quintiles, randomly permute CF-residual targets within each bin, retrain the learned modifier, measure slice F1. If slice F1 drop < 0.005, learned modifier is not using CF information meaningfully; excluded from the paper.

**Strict kill rule (Round-5 Simplification #2)**: if ANY of R121 (no-CF baseline) / R122 (split-transfer) / R123 (label-shuffle) / R123b (CF-permute) fires, the *entire* learned-modifier section is dropped from the paper — not demoted, not "not recommended". Fully excluded.

### 6. Stage-8 scope reduction (Round-5 Simplification #1)

**Reviewer said**: "Collapse Stage-8 reporting to L0 vs L2 only (keep L1/L3 strictly appendix/diagnostic) and make A-Prompt conditional as written."

**Action**:
- Main text Stage-8 reporting: **L0** and **L2** only, under the non-inferiority framing.
- L1 (prompt-summarization) and L3 (LLM-direct-predict) stay in appendix as diagnostics.
- A-Prompt field ablation **runs only if** L2 > L0 + δ (i.e. only if LLM embedding produces measurable gain above the δ = 0.015 non-inferiority margin). Otherwise A-Prompt adds no information and is skipped.

### 7. "Quantile-calibrated" language discipline (Round-5 Audit (c))

**Reviewer said**: "Avoid CP-implying language that sounds distribution-free/theorem-backed; treat 'quantile-calibrated' as heuristic calibration unless formal exchangeability is assumed."

**Action**: update the proposal language globally.
- Replace "quantile-calibrated" with **"valid_cal-quantile-calibrated"** in the paper method section (explicitly locates the calibration inside our valid_cal split, not a marginal-coverage claim).
- No theorem claims; empirical coverage reported on `test_cal` is explicitly labelled diagnostic.
- Method section §2 opens with: "We use conformal-inspired machinery heuristically; we do not assume or claim exchangeability under inductive message passing."

---

## Revised Proposal (v5 delta)

**Claim map v5**:

| Claim | Status | Falsification |
|---|---|---|
| **C1** EQC² composite-with-set trigger dominates scalar / label-only CP / BR / BR-soft | unchanged | T4 ≤ T4-NS within 0.010 → contract to EQC-S; BR-soft-best ≥ T4 on both datasets → contract to refinement-of-prior-art |
| **C2** EQC² unsupervised accept/rollback gate is load-bearing | unchanged | G1 degrades ≥ 0.1pp OR G2 doesn't degrade → redesign/drop |
| **C3** Cross-dataset robustness on MGTAB | unchanged | opposite-sign slice gain → relation-type sensitivity limitation |
| **C4** (NEW, non-inferiority framing) Artifact-only is non-inferior to LLM-enhanced | `L2 - L0 ≤ 0.015` AND `L0 ≥ 0.90 · L2`, 95% paired-bootstrap CI | symmetric: if L2 - L0 > 0.015 significantly, paper concedes LLM is the mechanism |
| **C5** (NEW) EQC² rewrite+gate carries ≥ half of total lift | `M-ro - M-base ≥ 0.5 · (M-rr - M-base)` on both datasets | hard kill: if < 0.5, demote Stage-5/6 and recenter on artifact + refiner |
| **S1** Locality wall-clock overhead ≤ 2× | unchanged | — |
| **S2** Composite ingredient necessity | unchanged | — |
| **S3** Soft rewrite > hard delete on camouflage slice | unchanged | R-hard-delete ≥ R-soft on both datasets → narrative change |

**Complexity Budget v5**:
- Calibration-time artifacts: AE MLP on train_cal (skipped if w_rec = 0); p_LM fallback MLP on train_cal (skipped if backbone exposes p_LM); **6-scalar** coordinate-descent on valid_cal (τ+/τ- FROZEN as valid_cal quantiles); q_hat + τ*; Stage-8a artifact-embedding aggregator MLP (2-layer) + refiner head (1-layer).
- v1-LLM optional adds Stage-8b refiner MLP with pre-computed z_LLM.
- Everything else: not trainable.

**System Overview v5** (delta from Round 4):
- Stage-4 u_edge formula unchanged.
- τ+ / τ- / τ0 **frozen** from valid_cal quantiles at Stage-A a5 (persisted with other calibration artifacts).
- Search is 6-scalar coordinate-descent on `{w_tg, w_het, w_rec, α_text, α_rel, α_tgd}`.

**New experiment blocks**:

| Block | Purpose | Variants | Seeds | Datasets |
|---|---|---|---|---|
| **Block-M (Mechanism Isolation)** | Prove EQC² rewrite+gate carries ≥ half the lift | M-base, M-ro, M-re, M-rr | 3 | TwiBot-20, TwiBot-22 |
| **Block-BR-faithful** | BotBR public reproduction (if compatible) + BR / BR-soft sensitivity | BR-public (if feasible); BR × 3 threshold settings; BR-soft × 3 downweight settings | 3 | TwiBot-20, TwiBot-22 |
| **Block-AURC** (modernization) | Risk-coverage / AURC curves for all trigger variants | T1, T2, T3, BR, BR-soft, T4-NS, T4 | 3 | TwiBot-20, TwiBot-22, MGTAB |

**Run map additions**:

- **R127** Block-M on TwiBot-20, 3 seeds.
- **R128** Block-M on TwiBot-22, 3 seeds.
- **R129** Block-BR-faithful: BR-public reproduction attempt (or documented incompatibility report).
- **R130** Block-BR-faithful sensitivity: BR × 3, BR-soft × 3 settings each dataset.
- **R131** Block-AURC curves from R107/R108/R127/R128 outputs (analysis-only, no extra compute).
- **R123b** CF-signal permutation falsifier for learned-modifier appendix.

**Pilot Gates v5**:
- Pilot A (R105): unchanged — T1 / T3 / T4-NS / BR-soft / T4 on TwiBot-20 dev, 1 seed. Pass iff `slice_F1(T4) - slice_F1(T4-NS) ≥ 0.010` AND `slice_F1(T4) > slice_F1(BR-soft)`.
- Pilot B (R106): unchanged.
- **Pilot C (NEW, R132)**: M-base vs M-ro on TwiBot-20 dev, 1 seed. Pass iff `M-ro > M-base` by ≥ 0.5pp on slice F1. Pre-commit fallback: if M-ro ≤ M-base, demote the rewrite+gate claim; paper becomes "artifact + refiner" story; investigate Stage-8 as the mechanism.

**Tuning protocol v5 additions**:
- τ+ / τ- published in calibration artifact; their valid_cal quantile percentiles (75 / 25) are hyperparameters set at method-specification time, not searched.
- Sensitivity plot on valid_cal (published with paper): vary each of the 6 searched scalars ± 20% around its optimum; plot slice F1 stability.
- Coordinate-descent schedule disclosed in the paper (order, step size, stop rule); no post-hoc per-dataset re-tuning.

---

## Expected Round 6 Score (self-estimate, not claim)

- Venue Readiness 6.8 → ~8–9 once Block-M proves EQC² is the mechanism.
- Contribution Quality 8.1 → ~9 once Stage-8 is bounded as commodity via M-re result.
- Feasibility 8.0 → ~9 with BR-public compatibility decision made (either reproduction runs or documented non-reproduction).
- Overall 8.6 → **target 9.2–9.4** at nightmare difficulty.
- If Block-M falsification fires, honest narrative downgrade — not in our control.

---

## Remaining Risks and Planned Pushbacks

- If Round 6 reviewer demands even more datasets → acknowledge MGTAB already stresses multi-relational; Cresci-17 available on request but not required.
- If reviewer asks for coverage theorem → reject, cite inductive-CP caveat and heuristic-calibration framing.
- If reviewer asks to relax the M-ro ≥ 0.5 · M-rr rule → reject as load-bearing: if EQC² rewrite+gate carries less than half, it is not the mechanism and the paper must honestly recenter.
