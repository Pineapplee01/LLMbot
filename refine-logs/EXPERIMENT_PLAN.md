# Experiment Plan (v7): EQC — Ego-Quality Controller

**Created**: 2026-05-09 (v7, Round-8 READY nightmare-difficulty)
**Method source**: `refine-logs/FINAL_PROPOSAL.md` (v7)
**Supersedes**: v5/v4 experiment plans. All prior plans in git history.
**v7 deltas**: EQC rename (drop "²"); 3-term composite `s_lbl + s_tg + s_npi` (dropped `s_rec` from main, `s_het` → `s_npi` posterior-based); principled three-role `u_edge` (affinity − conflict/2 + relation prior); external prediction-stability guard (posterior margin + L∞ cap from frozen classifier head); pre-registered 768-cell discrete grid (fixed + no-search baselines in main table); C4 δ grounded in seed-std; **C5 compound gate** (CI excludes 0 AND `(M-ro − M-base) ≥ max(0.5pp, 0.4·(M-rr − M-base))` with β=0.4 pre-registered, restored from v6 regression); **Block-S2 promoted to main** (composite term necessity); **Block-S4 NEW** (modifier component necessity, same pruning discipline); paper publishes post-pruning final `u_edge`; BR-public two-tier; citation language downgrade (SKETCH/GAugLLM/CTGL = "motivates"); unit tests → artifact documentation only.
**Method source**: `refine-logs/FINAL_PROPOSAL.md` (v5)
**Anchor source**: `research.md` (2026-05-09)
**Supersedes**: v4 experiment plan (2026-05-09 earlier in session) and FRMI v2 plan (2026-04-20). Prior plans remain in git history.

---

## Context

**Problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage. Scalar post-hoc calibration yields no operational ego-uncertainty signal for selective, budgeted, reversible refinement; binary edge-reliability methods (BotBR, BECE) collapse camouflage to `{reliable, unreliable}`; global structure learning (LDS, Pro-GNN, IDGL) is non-local and non-rollbackable.

**Method thesis**: A composite valid_cal-quantile-calibrated non-conformity score that estimates prediction-set uncertainty under current ego context, reused as selection trigger + rewrite-amplitude factor + L-hop-local accept/rollback gate, enables budgeted reversible local ego refinement on a frozen backbone — dominating scalar triggers, scalar-gated composites, label-only CP triggers, and BotBR-style binary-reliability triggers, without LLM calls in the main path, without coverage claims, and without base-backbone retraining.

**Compute**: existing project GPU budget (~140–220 GPU-hr total). Local RTX 4060 for analysis + remote GPU for training.

**Protocol constraint**: `test` split is untouched until final reporting. Calibration lives entirely on `train_cal ∪ valid_cal` (disjoint subsets of `train`). Leakage-asserting unit tests enforce the partition; a `test_virtual_context_isolation.py` test enforces that virtual-context edges never enter the main adjacency matrix.

---

## Claim Map (v5)

| Claim | Why it matters | Minimum convincing evidence | Falsification |
|---|---|---|---|
| **C1** EQC² composite-with-set trigger dominates scalar / label-only CP / BotBR-style binary-reliability | Core novelty of the dominant primitive | Expected slice macro-F1: T4 > T4-NS > BR-soft > T3 > BR > T2 ≥ T1 on TwiBot-20 + TwiBot-22, 3 seeds | T4 ≤ T4-NS within 0.010 → contract to EQC-S; BR-soft-best ≥ T4 on both datasets → concede prior-art sufficiency |
| **C2** EQC² unsupervised accept/rollback gate is load-bearing | Establishes "controller" framing; prevents global-F1 degradation | G2 degrades global F1 by ≥ 0.3pp vs no-refinement; G1 within 0.1pp; G1 ≈ G3 on slice F1 | G1 degrades ≥ 0.1pp OR G2 doesn't degrade → redesign / drop |
| **C3** Cross-dataset robustness on multi-relational graph | Generality; `R_trusted` transfer | Directional slice gain on MGTAB at matched budget; global F1 non-degraded | Opposite-sign slice gain → relation-type sensitivity limitation |
| **C4** Artifact-only refiner non-inferior to LLM-enhanced (LLM call is marginal) | Attribution: artifact is the mechanism; LLM is commodity | `slice_F1(L2) − slice_F1(L0) ≤ 0.015` 95% paired-bootstrap CI; AND `slice_F1(L0) ≥ 0.90 · slice_F1(L2)` | L2 − L0 > 0.015 significantly → concede LLM is the mechanism |
| **C5** **EQC² rewrite+gate is the mechanism** (hard kill gate) | Proves Stage-8 is commodity readout, not a second mechanism | `slice_F1(M-ro) − slice_F1(M-base) ≥ 0.5 · (slice_F1(M-rr) − slice_F1(M-base))` on **both** datasets, 3-seed paired-bootstrap CI on difference-of-differences | Hard kill: < 0.5 on either dataset → demote Stage-5/6; recenter paper on artifact+refiner |
| **S1** Locality operationally bounded | "Post-hoc local" is not rhetorical | Refinement wall-clock overhead ratio ≤ 2.0× base test-time cost; locality + virtual-context-isolation unit tests pass | — (correctness invariant) |
| **S2** Composite ingredients not wasted | Defends the composite | Drop-one of `s_tg / s_het / s_rec` does not strictly dominate the full composite on slice F1 | Auto-collapse `w_rec → 0` acceptable |
| **S3** Soft rewrite > hard delete on camouflage slice | Preserves the "camouflage is evidence" rationale | R-soft > R-hard-delete on camouflage slice on at least one of TwiBot-20 / TwiBot-22 | R-hard-delete ≥ R-soft on BOTH datasets → narrative change |

---

## Pre-Registered Slicing and Metric Protocol

**Slices** computed on `valid_cal`, thresholds frozen before test:
1. **High-uncertainty ego slice**: top-20% by `s_composite` (main headline slice).
2. **Text-graph conflict slice**: top-20% by `s_tg`.
3. **Heterophily slice**: top-20% by `s_het`.
4. **Low-degree slice**: bottom-20% by degree.
5. **Camouflage slice**: nodes where ≥ 50% of 1-hop neighbors have opposite predicted label (diagnostic-only; never conditions tuning).
6. **HCW slice**: `max_y p_base(y|v) > 0.9` but incorrect OOF prediction (diagnostic-only).

**Primary metrics**: global macro-F1 (non-degradation gate); slice macro-F1 on the high-uncertainty ego slice (improvement gate); rollback rate (< 15%); refinement wall-clock overhead ratio (≤ 2.0×).

**Secondary / diagnostic**: **AURC at pre-registered intervention ratios `{5%, 10%}`** (replaces prior single-operating-point reporting); ECE; empirical CP coverage on `test_cal` (diagnostic, not a theorem); intervention ratio (% hard nodes); edge-weight-change budget; LLM call ratio (= 0 in main).

**Statistical**: 3 seeds per condition for every decisive block (Block-T / G / M / R / X). **Paired-bootstrap CI on difference-of-differences for C5** `(M-ro − M-base) − 0.5 · (M-rr − M-base)`, 10k resamples. Paired-bootstrap CI on `L2 − L0` for C4. Matched-seed paired-t on global F1 across seeds.

---

## Intervention Budgets (matched across comparisons)

| Parameter | Default | Fit on | Notes |
|---|---|---|---|
| Intervention ratio (hard-node quota) | 15% | `valid_cal` | Main headline. AURC also reported at fixed `{5%, 10%}`. |
| `B_max` (per-node edge-weight-change budget) | 0.5 | `valid_cal` | Sum \|w_e' − w_e\| / w_e across `propagation_edges(v)`. |
| `κ` (budget-to-uncertainty coupling) | 1.0 | `valid_cal` | `budget(v) = min(B_max, κ · s_composite(v))`. |
| `λ` (soft-reweight multiplier cap) | 0.3 | `valid_cal` | Edge weights clamped to `[0.7·w_e, 1.3·w_e]`. |
| `τ+` | Quantile₇₅ | `valid_cal` u_edge dist | **Frozen quantile (one-time)**, not searched. |
| `τ-` | Quantile₂₅ | `valid_cal` u_edge dist | **Frozen quantile (one-time)**, not searched. |
| `τ₀` | `min(|τ+|, |τ-|)` | derived | Neutral band. |
| 6 search scalars | `{w_tg, w_het, w_rec, α_text, α_rel, α_tgd}` | `valid_cal` coordinate-descent | Coordinate-descent schedule disclosed; stability plot ± 20% reported. |

All budgets joined into the 6-scalar search; once fit, budgets are **frozen identically** across all Block-T / Block-G / Block-M / Block-R variants so the only knob changing is the block's focal variable.

---

## Ablation Matrix (5 claim blocks + 3 support-claim blocks + appendix)

### Block-T (Trigger Swap; Claim 1)

Trigger swap under identical refinement (deterministic modifier + soft reweight + EQC² accept/rollback = v1 main).

| Variant | Trigger | `q_hat`? | Set geometry? | Evidence source |
|---|---|---|---|---|
| T1 | entropy | No | No | Scalar baseline |
| T2 | temp-scaled entropy | No | No | Calibrated scalar baseline |
| T3 | label-only CP | Yes | Yes | CP without ego-quality |
| BR | BotBR-style hard binary reliability | n/a | n/a | Prior-art baseline |
| BR-soft | BotBR-style soft binary reliability | n/a | n/a | Soft variant of BR |
| T4-NS | EQC² composite + scalar gating | **No** | No | Falsifies set-based contribution |
| T4 | EQC² composite + set-based gate (v1 main) | Yes | Yes | Full mechanism |

### Block-G (Gate Toggle; Claim 2)

Trigger = T4, modifier identical.

| Variant | Gate |
|---|---|
| G1 | EQC² unsupervised (v1 main) |
| G2 | Gate OFF |
| G3 | Supervised-cheat (delta-loss on valid labels — upper-bound reference, NOT deployable) |

### Block-M (Mechanism Isolation; Claim 5 — hard kill gate) — **NEW**

Trigger = T4, gate (if on) = G1.

| Variant | Stage-5 rewrite + Stage-6 gate | Stage-8a refiner | Purpose |
|---|---|---|---|
| M-base | NO | NO (`p_final = p_GNN_base`) | Baseline |
| **M-ro** | **YES** | NO (`p_final = p_GNN_post`) | **RewriteOnly** — pure EQC² mechanism |
| M-re | NO | YES (artifact over unrewritten ego) | NoRewrite+Refiner |
| M-rr | YES | YES | v1 main |

C5 claim (hard kill gate): `slice_F1(M-ro) − slice_F1(M-base) ≥ 0.5 · (slice_F1(M-rr) − slice_F1(M-base))` on **both** TwiBot-20 **and** TwiBot-22, 3-seed paired-bootstrap CI on the difference-of-differences.

### Block-R (Soft vs Hard Rewrite; Claim S3) — promoted from appendix

Trigger = T4, gate = G1.

| Variant | Rewrite action |
|---|---|
| R-none | No rewrite |
| R-soft | Soft reweight (v1 main) |
| R-soft+virtual | Soft reweight + virtual context edges in artifact (== L0 main) |
| R-hard-delete | Delete edges with `u_edge(e) ≤ τ-` |
| R-hard-add | Add top-k virtual context edges as weight-1 to main graph (ablation only; violates v1 invariant) |

### Block-X (Cross-dataset; Claim 3)

T4 + G1 + deterministic modifier on MGTAB. 3 seeds. Matched budget from TwiBot-20. `R_trusted` re-selected on MGTAB `valid_cal` as top-3 densest relations.

### Block-LLM-Attrib (Claim 4)

Trigger = T4, gate = G1, modifier = deterministic. Main: **L0** (artifact-only refiner, v1 main) vs **L2** (LLM embedding → refiner). Appendix: **L1** (prompt-summarization) and **L3** (LLM-direct-predict).

Non-inferiority: `slice_F1(L2) − slice_F1(L0) ≤ 0.015` with 95% paired-bootstrap CI across 3 seeds, AND `slice_F1(L0) ≥ 0.90 · slice_F1(L2)`.

### Block-BR-sensitivity + BR-public (Claim 1 differentiation)

BR sensitivity: 3 threshold settings for BR; 3 downweight factors for BR-soft.
BR-public: faithful public reproduction attempt; documented compatibility report if not compatible with frozen RoBERTa + BotRGCN backbone.
Decision rule: if BR-public OR best-setting BR-soft beats T4 on slice F1 on **both** datasets, paper concedes prior-art sufficiency.

### Block-AURC (modernization)

Risk-coverage / AURC curves at **pre-registered intervention ratios `{5%, 10%}`** for all Block-T variants across TwiBot-20 + TwiBot-22 + MGTAB. Analysis-only (no extra GPU).

### Block-L (Locality Audit; Support claim S1)

Log `refinement_wall_clock_overhead_ratio` on every full-matrix run. Locality + virtual-context-isolation unit tests run in CI on every commit.

### D-EstSem (Diagnostic, not a claim) — NEW

Correlate `s_composite(v)` against: (a) ego heterophily, (b) degree deciles, (c) label-shuffle perturbation of ego labels, (d) edge-removal sensitivity. Observational only; published table. Answers the "what does the estimator measure?" question operationally.

### Appendix (conditional / diagnostic)

| Block | Run | Content |
|---|---|---|
| A-Composite | R113 | Drop-one of `s_tg / s_het / s_rec`. |
| **A-Modifier (strict-kill gated)** | R114 + R121 + R122 + R123 + R123b | Learned modifier (gradient-attribution supervised) vs deterministic, with four falsification controls. **If ANY of R121 / R122 / R123 / R123b fires, the entire A-Modifier section is DROPPED from the paper** — not demoted, not "not recommended". Fully excluded. |
| A-Diffusion | R115 | DAPS/NAPS / SNAPS inside `s_composite`. |
| A-Iter | R116 | `max_iter = 2` with strict per-iteration improvement + total-budget-respecting rollback. |
| A-NestedSplit | R117 | 5-fold rotation of `train_cal / valid_cal` within train pool. |
| A-Prompt (conditional) | R125 | 7-field drop-one on artifact. **Runs only if L2 > L0 + δ**; otherwise skipped. |

---

## Pre-Registered Pilot Gates (Week 0, ~6 GPU-hr total)

### Pilot Gate A (Claim 1, R105)

- TwiBot-20 dev, 1 seed. Variants: T1 / T3 / T4-NS / **BR-soft** / T4.
- Pass iff `slice_F1(T4) − slice_F1(T4-NS) ≥ 0.010` **AND** `slice_F1(T4) > slice_F1(BR-soft)`.
- Fail ⇒ paper contracts to **EQC-S** (composite scalar controller, no set geometry) OR to refinement-of-prior-art if BR-soft wins.

### Pilot Gate B (Claim 2, R106)

- TwiBot-20 dev, 1 seed. Variants: G1 / G2 / no-refinement.
- Pass iff G2 degrades global F1 by ≥ 0.3pp AND G1 is within 0.1pp of no-refinement.
- Fail ⇒ redesign gate (hysteresis / mini-quorum) and re-run; if still fail, drop contribution.

### Pilot Gate C (Claim 5, R132) — **NEW**

- TwiBot-20 dev, 1 seed. Variants: M-base vs M-ro.
- Pass iff `slice_F1(M-ro) − slice_F1(M-base) ≥ 0.005`.
- Fail ⇒ demote Stage-5/6; paper recenters on "artifact + refiner" story; treat EQC² rewrite+gate as no-op; investigate Stage-8 as the mechanism.

Pilots are **sequential**: A → decision → B → decision → C → decision → commit to full matrix only if all three pass (or the fallback narrative is adopted for whichever failed).

---

## Run Plan (Numbered)

| Run ID | Phase | Block | Purpose | Variants | Dataset | Seeds | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R101 | W1 | Core | Extend `GraphConformalSetEstimator` with composite score + AE hook + lex selection + edge-case handler + `score_mode` + `p_LM_source` | — | — | — | MUST | TODO | [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) |
| R102 | W1 | Core | Stage-A calibration script — AE fit + 6-scalar coordinate-descent + `q_hat` + `τ*` + **τ+/τ- quantile freeze** | — | TwiBot-20 | — | MUST | TODO | Deterministic, reproducible |
| R103 | W2 | Core | Extend `EgoRefinementRepairOperator` with `u_edge` modifier + EQC² gate + `gate_mode` + L-hop-local Stage-6 + locality + virtual-context unit tests | — | — | — | MUST | TODO | [operators.py:194](LLMbot/baseline/core/operators.py#L194) |
| R104 | W2 | Core | Add `t4_ns` path; add `rewrite_mode ∈ {off, soft, hard_delete, hard_add}`; add `refiner_mode ∈ {off, stage_8a, stage_8b}` for Block-M / R / LLM-Attrib | — | — | — | MUST | TODO | — |
| **R105** | **W0** | **Pilot A** | EQC² set-based necessity pilot | T1, T3, T4-NS, BR-soft, T4 | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| **R106** | **W0** | **Pilot B** | EQC² gate necessity pilot | G1, G2, no-refine | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| **R132** | **W0** | **Pilot C** | EQC² mechanism-isolation pilot (NEW) | M-base, M-ro | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| R107 | W3 | Block-T | Trigger swap full matrix | T1, T2, T3, BR, BR-soft, T4-NS, T4 | TwiBot-20 | 3 | MUST | BLOCKED on R105 pass | 7 × 3 = 21 runs |
| R108 | W3 | Block-T | Trigger swap full matrix | T1, T2, T3, BR, BR-soft, T4-NS, T4 | TwiBot-22 | 3 | MUST | BLOCKED on R107 finish | Primary robustness |
| R109 | W3 | Block-G | Gate toggle | G1, G2, G3 | TwiBot-20 | 3 | MUST | BLOCKED on R106 pass | 3 × 3 = 9 runs |
| R110 | W3 | Block-G | Gate toggle | G1, G2, G3 | TwiBot-22 | 3 | MUST | BLOCKED on R109 finish | — |
| **R127** | W4 | **Block-M** | Mechanism isolation (NEW) | M-base, M-ro, M-re, M-rr | TwiBot-20 | 3 | MUST | BLOCKED on R132 pass | 4 × 3 = 12 runs; **C5 hard kill gate** |
| **R128** | W4 | **Block-M** | Mechanism isolation (NEW) | M-base, M-ro, M-re, M-rr | TwiBot-22 | 3 | MUST | BLOCKED on R127 finish | — |
| **R126a** | W4 | **Block-R** | Soft vs hard rewrite (promoted) | R-none, R-soft, R-soft+virtual, R-hard-delete, R-hard-add | TwiBot-20 | 3 | MUST | BLOCKED on Pilots | 5 × 3 = 15 runs |
| **R126b** | W4 | **Block-R** | Soft vs hard rewrite (promoted) | R-none, R-soft, R-soft+virtual, R-hard-delete, R-hard-add | TwiBot-22 | 3 | MUST | BLOCKED on R126a finish | — |
| R111 | W4 | Block-X | Cross-dataset | T4 + G1 + deterministic modifier | MGTAB | 3 | MUST | BLOCKED on R108, R110, R128 | `R_trusted` = top-3 densest |
| R112 | W3–W4 | Block-L | Locality audit | Log wall-clock ratios; CI locality test + virtual-context-isolation test | — | — | MUST | CONTINUOUS | Defends S1 |
| **R129** | W4 | **Block-BR-public** | BotBR faithful reproduction | BR-public | TwiBot-20 + TwiBot-22 | 3 if feasible | MUST | TODO | Documented compatibility report if infeasible |
| **R130a** | W4 | **Block-BR-sensitivity** | BR × 3 thresholds + BR-soft × 3 downweight factors | 6 settings | TwiBot-20 | 3 | MUST | TODO | — |
| R130b | W4 | Block-BR-sensitivity | 6 settings | 6 settings | TwiBot-22 | 3 | MUST | BLOCKED on R130a finish | — |
| R131 | W4 | **Block-AURC** | AURC curves at `{5%, 10%}` | From R107/R108/R127/R128 outputs | TwiBot-20, 22, MGTAB | reuse | MUST | Analysis-only, no extra GPU | — |
| **R120** | W4 | **D-EstSem** (NEW) | Diagnostic correlation study | s_composite vs heterophily / degree / label-shuffle / edge-removal | TwiBot-20 + TwiBot-22 | 3 | SHOULD | TODO | Observational; answers U#1 |
| R124 | W5 | Block-LLM-Attrib | L0 vs L2 | 2 variants | TwiBot-20 + TwiBot-22 | 3 | MUST | BLOCKED on R101–R104 | Non-inferiority |
| R113 | W5 | A-Composite | Drop-one `s_tg`/`s_het`/`s_rec` | 3 variants | TwiBot-20 | 3 | SHOULD | TODO | Defends S2 |
| R114 | W5 | A-Modifier | Learned modifier vs deterministic | 1 variant | TwiBot-20 | 3 | NICE | **Strict-kill gated by R121–R123b** | — |
| R121 | W5 | A-CF-Falsifiers | No-CF baseline | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| R122 | W5 | A-CF-Falsifiers | Split-transfer | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| R123 | W5 | A-CF-Falsifiers | Label-shuffle | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| **R123b** | W5 | **A-CF-Falsifiers (NEW)** | CF-signal permutation within degree bins | 1 variant | TwiBot-20 | 3 | NICE | **Strict-kill gate** | Fire ⇒ drop A-Modifier |
| R115 | W5 | A-Diffusion | DAPS/NAPS, SNAPS | 2 variants | TwiBot-20 | 3 | NICE | TODO | — |
| R116 | W5 | A-Iter | max_iter=2 + rollback | 1 variant | TwiBot-20 | 3 | NICE | TODO | — |
| R117 | W5 | A-NestedSplit | 5-fold train_cal/valid_cal rotation | v1 main | TwiBot-20 | 3 | SHOULD | TODO | Defends tuning protocol |
| R125 | W5 | A-Prompt (conditional) | 7-field drop-one on artifact | 7 variants | TwiBot-20 | 3 | **Conditional on L2 > L0 + δ** | TODO | Skipped if C4 holds |
| R118 | W5 | Analysis | Tables + figures + paired-bootstrap tests + AURC curves + non-inferiority CIs | — | — | — | MUST | TODO | `/paper-figure` |
| R119 | W5 | Paper | `/paper-plan` → `/paper-write` | — | — | — | MUST | TODO | — |

**Dependency summary**:
- R101 → R102 → R103, R104 (core).
- R105 blocks R107; R107 blocks R108.
- R106 blocks R109; R109 blocks R110.
- **R132 blocks R127; R127 blocks R128. C5 hard kill gate evaluated after R127+R128.**
- R108, R110, R128 block R111 (MGTAB run only after primary findings stable).
- R124 runs independently (Claim 4 non-inferiority).
- A-Modifier (R114) is strict-kill-gated by R121 / R122 / R123 / R123b; if any fires, R114 is not reported.
- R125 is conditional on R124 outcome (L2 > L0 + δ).

---

## Decision Gates

| Gate | When | Pass | Fail |
|---|---|---|---|
| **Pilot A (R105)** | End W0 | Run R107/R108. | Contract to EQC-S or refinement-of-prior-art. |
| **Pilot B (R106)** | End W0 | Run R109/R110. | Redesign gate; re-run R106. If still fail, drop gate contribution. |
| **Pilot C (R132)** | End W0 | Run R127/R128. | **Demote Stage-5/6**; paper recenters on artifact+refiner; skip Block-M full matrix. |
| **Block-M hard kill (C5)** | End R127+R128 | Paper remains EQC²-as-mechanism. | Stage-5/6 demoted; paper centers on artifact+refiner; claims C1/C3 adjusted. |
| **Block-BR (R129+R130)** | End W4 | EQC² differentiation stands. | If BR-public OR best-BR-soft beats T4 on both datasets ⇒ concede prior-art sufficiency. |
| **C4 non-inferiority (R124)** | End W5 | Artifact is load-bearing. | Concede LLM is the mechanism; paper reframes Stage-8b as primary. |
| **Budget consumed > 2× plan** | End W3 | Continue. | Halt appendix blocks; report only what is completed. |
| **Rollback rate > 15% on TwiBot-22** | End R110 | Continue. | Single-step `λ` reduction on `valid_cal`; re-run once; if still > 15%, report as limitation. |
| **Locality / virtual-context unit test failure** | Any run | — | Fail-loud; fix before any claim. Correctness invariant. |
| **`test` leakage detection** | Any run | — | Fail-loud; discard run; audit Stage-A. |
| **A-Modifier strict kill (R121/R122/R123/R123b)** | End R121–R123b | Learned modifier appendix reported. | ANY fires ⇒ entire A-Modifier section dropped. Not demoted. Not "not recommended". Fully excluded. |

---

## Baselines (Required)

| Baseline | Block | Source |
|---|---|---|
| Frozen backbone (no refinement) | All | Stage-0 checkpoint |
| Entropy (T1) | Block-T | — |
| Temperature-scaled entropy (T2) | Block-T | Standard torch calibration |
| Label-only CP (T3) | Block-T | CF-GNN-style port, no diffusion |
| **BR** BotBR-style hard binary reliability | Block-T + Block-BR-sensitivity | Reproduced from BotBR description |
| **BR-soft** BotBR-style soft binary reliability | Block-T + Block-BR-sensitivity | Soft version of BR |
| **BR-public** BotBR faithful public reproduction | Block-BR-public | GitHub, if compatible |
| EQC² composite + scalar gate (T4-NS) | Block-T | Falsifies set-based contribution |
| Gate OFF (G2) | Block-G | — |
| Supervised-cheat gate (G3) | Block-G | Upper-bound reference using valid labels |
| **M-base** no rewrite, no refiner | Block-M | Mechanism isolation baseline |
| **M-re** no rewrite, refiner ON | Block-M | Isolates Stage-8 contribution |
| R-hard-delete, R-hard-add | Block-R | Hard-rewrite negative controls |
| L1, L3 (LLM variants) | Block-LLM-Attrib appendix | Diagnostic only |

---

## Risks and Mitigations (v5)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **T4 ≈ T4-NS** | Medium | High | Pre-committed contraction to EQC-S |
| **BR-soft ≥ T4 on both datasets** | Medium | High | Concede prior-art sufficiency; refinement-of-prior-art framing |
| **G1 ≈ G2** | Medium | High | Gate redesign; if still fail, drop supporting contribution |
| **C5 fails (M-ro < 0.5·M-rr)** | Medium | High | Hard kill: demote Stage-5/6; recenter on artifact+refiner. Pre-committed. |
| **C4 fails (L2 > L0 + δ)** | Medium | Medium | Honest reframing: LLM is the mechanism; paper reweights Stage-8b section. |
| AE fails to generalize TwiBot-20 → TwiBot-22 | Medium | Low | Auto-collapse `w_rec → 0` |
| MGTAB sign disagreement (Block-X) | Medium | Medium | Relation-type sensitivity limitation, honest report; `R_trusted` re-selected on MGTAB |
| Rollback rate > 15% | Low | Low | Gate-triggered `λ` reduction |
| Inductive CP violation | Certain | None | Expected; heuristic-calibration framing |
| Stage-6 wall-clock blows budget | Low | Medium | Bounded by locality invariant; monitored in Block-L |
| Stage-8a refiner overfits train_cal | Low | Medium | Fallback `p_final = p_GNN_post` |
| 6-scalar search overfits valid_cal | Low | Low | Stability plot ± 20% on valid_cal published; 5-fold nested split appendix |
| A-Modifier R121/R122/R123/R123b fires | High | None | Strict kill rule; A-Modifier dropped from paper; not a loss — a credibility instrument |
| LLM pass expensive | Low | Low | One-time per node; pre-computed; not in test inference budget |

---

## Out of Scope (Explicit)

- LLM-direct-predict as a v1 recommended configuration (L3 is diagnostic only).
- Binary edge-reliability headline (BotBR/BECE territory).
- Global graph structure learning (LDS/Pro-GNN/IDGL).
- Learned modifier in v1 main (appendix only, strict-kill-gated).
- `max_iter > 2` in any configuration.
- Causal identification language.
- CP coverage theorem claims.
- Cresci-17 as required dataset (MGTAB selected instead; Cresci-17 available on request).

---

## Paper Handoff

- **Narrative spine**: "one post-hoc controller primitive, used three ways, empirically load-bearing at the trigger step (T4 vs T4-NS), the gate step (G1 vs G2), and the mechanism step (M-ro vs M-base); cross-dataset robust on a multi-relational graph (MGTAB); artifact-only refiner is non-inferior to LLM-enhanced refiner (C4); soft rewrite preserves camouflage (S3)."
- **Headline figures**: Block-T slice-F1 table + AURC curves at `{5%, 10%}`; Block-G table; **Block-M difference-of-differences bar chart + CI (C5)**; Block-R camouflage-slice comparison; MGTAB cross-dataset row; Block-BR-sensitivity panel + BR-public row (or incompatibility statement); C4 L0-vs-L2 non-inferiority CI plot.
- **Key statistical tests**: paired-bootstrap on node-level F1 deltas; paired-bootstrap CI on difference-of-differences for C5; paired-bootstrap CI on `L2 − L0` for C4; matched-seed paired-t on global F1 across seeds.
