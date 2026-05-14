# Round 6 Critique Response — External Codex Reviewer Analysis

**Date**: 2026-05-09
**Source**: External Codex reviewer critique of v5 (Round-6 READY at nightmare 9.2/10)
**Purpose**: Rigorous triage of each P0/P1/P2 item before committing to v6 changes. Accept valid fixes; push back with evidence on overreach.

---

## Triage Summary

| # | Item | Verdict | Reason |
|---|---|---|---|
| P0-1 | "Conformal" naming w/o coverage theorem | **ACCEPT** | Terminology-reality mismatch; anchor tightening already wanted this |
| P0-2 | `s_composite` hand-crafted feature stacking | **PARTIAL** | Each term must have structural necessity + drop-one main; reject "every composite is engineered" framing |
| P0-3 | Single scalar 3 uses ⇒ self-confirmation loop | **ACCEPT** | Genuine methodological concern; external acceptance check needed |
| P0-4 | `s_het` heterophily sign wrong for social bot | **ACCEPT (strong)** | Direct contradiction with anchor §5 — anchor explicitly warns "异配边 ≠ 噪声" |
| P1-1 | `ego_rec` origin unclear | **ACCEPT** | Drop from v1 main; appendix-only until justified |
| P1-2 | `u_edge` `sign(ΔJSD)` too weak | **ACCEPT** | Replace with principled pairwise conflict penalty |
| P1-3 | 6-scalar search = tuning | **ACCEPT** | Pre-register small discrete grid + fixed-weight baseline |
| P1-4 | C5 threshold 0.5 arbitrary | **PARTIAL** | Drop the 0.5 fraction; keep CI-based kill gate (controller-only gain 95% CI excludes 0) |
| P1-5 | C4 δ=0.015 arbitrary | **ACCEPT** | Ground in baseline seed std |
| P1-6 | Pilot gates in paper text | **ACCEPT** | Protocol-only; paper presents final method + empirical characterization |
| P2-1 | SKETCH/GAugLLM/CTGL shallow citation | **ACCEPT** | Downgrade from "migrate" to "motivates / inspires" |
| P2-2 | BR-soft may be strawman | **ACCEPT** | Two-level comparison already planned (R129 BR-public + R130 sensitivity); make explicit |
| P2-3 | Unit tests not research contribution | **ACCEPT** | Move from method section to artifact documentation |
| P2-4 | "EQC²" over-theoretical | **ACCEPT** | Rename to **EQC** (Ego-Quality Controller); drop ² |
| Expert rec | "Three uses, one primitive" reformulation | **ADOPT** | Cleaner narrative spine |
| "Remove ALL kill gates" | — | **PUSH BACK** | Kill gates are the credibility instrument; soften thresholds, don't remove |

**Accepted**: 12 items fully + 2 partial + expert recommendation. Pushed back: 1 strong (don't abandon kill-gate framework) + 1 partial (composite is not inherently engineering).

---

## Detailed Analysis

### P0-1 — Conformal naming (ACCEPT)

**Reviewer's point**: Using "Conformal" without coverage theorem is rhetorical inflation.

**My assessment**: Correct. Round 3 evaluator already flagged "inductive CP language"; we added the heuristic-calibration disclaimer but retained the brand name. This is terminology debt.

**Fix**:
- Method name: **EQC — Ego-Quality Controller** (drop the ²).
- Long form in paper: "calibration-set quantile-gated ego-quality controller, inspired by graph conformal prediction but making no coverage claims under inductive message passing".
- `prediction_set(v)` remains an output diagnostic; not a theoretical vehicle.
- All "conformal" in method body → "calibrated" or "quantile-gated".

### P0-2 — Hand-crafted composite (PARTIAL)

**Reviewer's point**: `s_composite` is 4-term weighted sum + learned weights = feature engineering.

**My assessment**: Two parts to disentangle.
- **Valid**: each term must have unique structural role, and drop-one must decide what stays.
- **Pushback**: "every composite score in ML is engineered" is unfair — BotBR/BECE/GATS/CaGCN are ALL weighted compositions of signals with calibrated coefficients. The correct standard is not "avoid composition" but "each term has structural necessity and failure modes are disclosed".

**Fix (accepted portion)**:
- Define structural role of each term explicitly (see P0-3, P0-4 below for revised terms).
- **Promote A-Composite from appendix to main Block-S2** with `leave-one-term-out` decision rule: if dropping term X does not degrade slice F1 by `≥ 1 × seed_std`, term X is removed from final method.
- In main proposal, commit to reporting the pruned variant alongside full.

### P0-3 — Self-confirmation loop (ACCEPT, critical)

**Reviewer's point**: Same `s_composite` drives (i) which node to fix, (ii) how much to fix, (iii) whether to accept. System optimizes itself.

**My assessment**: Strong point. The acceptance gate is the most vulnerable — if the same signal that triggered the rewrite also validates it, there's no external ground truth.

**Fix**:

Add an **independent prediction-stability guard** to the accept condition:

```
accept iff  s_composite_post < s_composite_pre
       AND  set_size_post  ≤  set_size_pre
       AND  posterior_margin_post ≥ posterior_margin_pre - η       ← NEW
       AND  |p_GNN_post(v) - p_GNN_pre(v)|_∞ ≤ ρ                   ← NEW (stability cap)
```

- `posterior_margin = p_base(ŷ | v) - p_base(second-ŷ | v)` — purely from base GNN logits, independent of `s_composite`.
- `η`: margin-tolerance slack (fit on `valid_cal`, reported separately).
- `ρ`: prediction-drift cap (fixed a priori at ρ = 0.2, meaning post-rewrite posterior cannot move by more than 0.2 in L∞).

This gives Stage-6 a **signal the controller cannot optimize by construction** — posterior margin and drift depend on the frozen backbone's classifier head, not on the composite score.

### P0-4 — Heterophily sign direction (ACCEPT, strong)

**Reviewer's point**: `s_het = 1 − same-label ratio` penalizes heterophily, but anchor says "异配边不一定是噪声".

**My assessment**: Direct contradiction with anchor §5. This is actually the worst internal inconsistency in v5. The reviewer is catching a bug.

**Fix**:

Replace `s_het(v)` with **posterior-disagreement neighborhood inconsistency**:

```
s_npi(v) = mean over u ∈ N(v) of  JS_divergence(p_GNN(v) || p_GNN(u)) / log 2
```

- Does NOT use predicted labels (avoids "bot-human edge = error" bias).
- Uses full predictive distribution, so uncertainty-aware.
- **High `s_npi`** = target node's ego is in a region of posterior disagreement — which includes both "failure-prone heterophily" AND "legitimate bot-human camouflage evidence that the GNN cannot resolve".
- The controller triggers on both; Block-R (camouflage slice) and Block-X (MGTAB heterophily) empirically separate them.

Rename the term everywhere: `s_het` → `s_npi` (neighborhood posterior inconsistency).

**Mandatory camouflage-slice check**: if `s_npi` hurts bot-human-mixed-neighborhood slice on either TwiBot-20 or TwiBot-22, the term is dropped (part of new promoted Block-S2 leave-one-term-out).

### P1-1 — `ego_rec` origin unclear (ACCEPT)

**Reviewer's point**: AE-based reconstruction error is a new trained component; not commodity.

**My assessment**: Correct. The `w_rec → 0` auto-collapse was a safety net, not a justification. Cleaner to drop from v1 main.

**Fix**:
- v1 main composite: `s(v, y) = s_lbl(v, y) + w_tg · s_tg(v) + w_npi · s_npi(v)` — three terms only.
- `s_rec` moves to A-Composite appendix; only introduced if Block-S2 drop-one shows it beats all three-term alternatives.

### P1-2 — `u_edge` sign(ΔJSD) too weak (ACCEPT)

**Reviewer's point**: `α_tgd · sign(JSD_v − JSD_u)` has unclear direction and low information.

**My assessment**: Correct. I had defaulted to sign as a zero-parameter shortcut, but the direction ("higher target JSD than neighbor → what?") is not principled.

**Fix**:

Replace `u_edge` with a principled three-term formulation:

```
semantic_affinity(e = (v,u))  = cos(X_R[v], X_R[u])                  ∈ [-1, +1]
propagation_conflict(e)       = 1 - cos(p_GNN(v) - p_LM(v),
                                       p_GNN(u) - p_LM(u))           ∈ [0, 2]
relation_prior(e)             = 𝟙[rel(e) ∈ R_trusted]               ∈ {0, 1}

u_edge(e) = clip(  α_aff · semantic_affinity(e)
                 - α_conf · propagation_conflict(e) / 2
                 + α_rel · relation_prior(e),
                 -1, +1)
```

Three named, interpretable roles:
- **Semantic affinity**: do the texts agree?
- **Propagation conflict**: does this edge bring together two nodes whose LM-vs-GNN disagreement **points in different directions**? That is the principled "traversing this edge imports conflicting evidence" signal.
- **Relation prior**: dataset-declared trust for high-signal relations (e.g. `follow`).

This replaces the unprincipled `sign(ΔJSD)` with a direction-meaningful pairwise conflict (`cosine of residual posteriors`).

### P1-3 — 6-scalar search is tuning (ACCEPT)

**Reviewer's point**: valid_cal search could overfit; audit trail unclear.

**My assessment**: Correct. The "no training" defense needs a "no fishing" companion.

**Fix**:
- **Pre-registered discrete grid** (written in paper before running search):
  - `w_tg, w_npi ∈ {0, 0.25, 0.5, 1.0}` (4 values each)
  - `α_aff, α_conf ∈ {0.25, 0.5, 0.75, 1.0}` (4 values each)
  - `α_rel ∈ {0, 0.25, 0.5}` (3 values)
- **Grid size**: 4 × 4 × 4 × 4 × 3 = 768 combinations. Small enough to exhaust; large enough to cover reasonable weightings.
- **Selection rule**: pick combination maximizing `valid_cal` slice F1.
- **Required reporting** in paper:
  - Selected weights across 3 seeds (stability).
  - **Fixed-weights baseline** (all weights = 1.0 except `α_rel = 0.5`) as a must-report variant.
  - Sensitivity heatmap (top-5 vs bottom-5 grid cells by valid_cal slice F1).
  - No-search version: uniform weights as zero-DOF ablation.

### P1-4 — C5 threshold 0.5 arbitrary (PARTIAL)

**Reviewer's point**: Why 0.5? Looks post-hoc designed to prove "controller carries half".

**My assessment**: The specific fraction 0.5 **is** arbitrary. But removing the kill-gate framework entirely (reviewer's suggestion: "report-style decomposition") surrenders the self-honesty mechanism that makes EQC defensible against "Stage-8 is the real thing" attacks.

**Fix (compromise)**:

Convert C5 from fixed-fraction kill gate to **statistical kill gate** using paired bootstrap:

```
C5 (revised):  95% paired-bootstrap CI on (M-ro - M-base)  excludes 0
              AND  (M-ro - M-base) ≥ 0.25 × seed_std of (M-rr - M-base)

Paper reporting:  decomposition table
                  (M-ro - M-base),  (M-re - M-base),  (M-rr - M-base)
                  with 95% CIs
                  plus commentary on the relative magnitudes
```

- The **kill gate is statistical** (CI excludes 0) — this is defensible against "why 0.5".
- The fraction requirement is softened to "at least one seed-std-worth of controller-only effect" — grounded in data variability, not a round number.
- Main paper table is a **decomposition**, not a hard pass/fail.
- If CI includes 0: controller-only effect is not significant, and the paper honestly reports this; narrative shifts to "artifact+refiner-dominant" with the kill gate fired.

### P1-5 — C4 δ=0.015 arbitrary (ACCEPT)

**Reviewer's point**: δ should come from baseline variability, not a round number.

**Fix**:

```
δ = 1.5 × seed_std(slice_F1(L0) across 3 seeds)     ← computed from data
    (fallback floor δ ≥ 0.005 if variance is pathologically small)
```

Reported: estimated δ per dataset. Both criteria retained:
- `slice_F1(L2) - slice_F1(L0) ≤ δ_data` with 95% paired-bootstrap CI.
- `slice_F1(L0) ≥ 0.90 × slice_F1(L2)` (dual criterion).

### P1-6 — Pilot gates in paper text (ACCEPT)

**Reviewer's point**: Pre-committed fallbacks are project management, not paper content.

**Fix**:
- `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md` retain all pilot gates + fallback contractions (execution-phase protocol).
- Paper main text presents only: final method + ablation results + limitations.
- Failure-mode anticipations go into Discussion / Limitations section as observations, not process flow.

### P2-1 — SKETCH/GAugLLM/CTGL shallow (ACCEPT)

**Fix (paper language)**:
- ~~"We migrate SKETCH's decoupled aggregation..."~~
- "**These works motivate** why semantic and structural evidence should be separated and cross-checked; we adopt the **diagnostic** rather than the training objective."

Direct method support reattributed:
- Conformal / calibrated selection: CF-GNN, GATS, CaGCN.
- Graph edge weighting: GNNGuard, LDS (concept only).
- Graph prompt serialization: GraphText (with TMLR-2024 caveat).
- Social bot edge reliability comparisons: BotBR, BECE.

### P2-2 — BR-soft strawman risk (ACCEPT)

Already have R129 BR-public + R130 sensitivity in plan; make dual-level explicit in paper:

```
Baseline tier 1 (external): BR-public — faithful BotBR reproduction on
  frozen RoBERTa + BotRGCN, OR documented incompatibility report.
  Tests external method strength.

Baseline tier 2 (in-framework): BR, BR-soft — same trigger pipeline,
  same Stage-0, binary or soft-binary edge reliability.
  Isolates reliability-vs-utility design, not external method quality.
```

Paper discusses both explicitly. If tier 1 infeasible, paper justifies non-reproduction and relies on tier 2 + sensitivity.

### P2-3 — Unit tests not research contribution (ACCEPT)

**Fix**:
- Method section: one line per invariant ("Implementation enforces: Stage-6 locality; virtual-context isolation; calibration-test leakage safeguard. Details in artifact documentation").
- Detailed test list moves to supplementary material / code artifact.

### P2-4 — "EQC²" naming (ACCEPT)

**Fix**:
- **New name**: **EQC — Ego-Quality Controller**.
- Long form: "calibration-set quantile-gated ego-quality controller".
- No ² suffix. No "conformal" in method-name or theorem-like claims.

### Expert reformulation — three uses, one primitive (ADOPT)

This is the clean narrative spine. Adopting the exact framing:

```
Primitive:   calibrated ego-uncertainty score  q(v) = s_composite(v)
Use 1:       hard-node selection rule
Use 2:       soft-rewrite amplitude factor  (1 - q(v))
Use 3:       accept/rollback gate  (with external prediction-stability guard)
```

Edge utility `u_edge(e)` is explicitly **commodity** (semantic + conflict + relation prior), not a second contribution.

---

## Items I Push Back On

### Pushback 1: "Abandon ALL kill gates, use report-style decomposition" (on P1-4)

**Reviewer's suggestion**: replace C5 with "The controller accounts for a substantial fraction of the combined gain" report-style.

**Pushback**: kill gates are the **credibility instrument** that makes EQC defensible against "Stage-8 is the smuggled contribution" at venue bar — which was exactly why Block-M was added in Round 5. If I remove the kill-gate framework entirely, the argument regresses to "we report numbers, reader decides" — which is what a weak submission does. The correct fix is **statistical kill gate** (CI excludes 0), not **no kill gate**.

**Compromise adopted**: convert fixed-fraction thresholds to CI-based / seed-std-grounded thresholds. Keep the kill-gate framework. See P1-4 and P1-5 fixes above.

### Pushback 2: "Composite score is inherently engineering" (on P0-2 framing)

**Reviewer's framing**: "a composite score + hand-crafted edge utility + valid_cal search = hand-engineered scoring system".

**Pushback**: this framing is too absolute. BotBR, BECE, GATS, CaGCN, CF-GNN, and virtually every graph-calibration paper is a composite of signals with fit coefficients. The correct standard at venue is:
- Each term has **stated structural role**.
- Drop-one is **main-table**, not appendix.
- Search space is **pre-registered**.
- Fixed-weight and no-search baselines reported.

All four are in v6 fixes. "Don't use composites" would mean abandoning every competitive graph-calibration method; the right bar is rigor, not composition-freeness.

### Pushback 3 (minor): EQC² → EQC but keep "calibrated" in method name

**Reviewer's suggestion**: "calibrated ego-quality controller".

**Adopted**. Note: I keep "calibrated" because it is technically accurate (valid_cal quantile calibration is heuristic but real). I drop "conformal" because it implies coverage theorem. This is the correct middle ground.

---

## v6 Changes (Applied)

Summary of concrete changes from v5 → v6:

1. **Rename**: EQC² → **EQC** (drop ²). "Conformal" removed from method name; "calibration-set quantile-gated" in long form. C4/C5 framing updated.
2. **`s_composite` v1 main = 3 terms**: `s_lbl + w_tg · s_tg + w_npi · s_npi`. Drop `s_rec` from v1 main (appendix only).
3. **`s_het` → `s_npi`**: neighborhood posterior inconsistency via JSD of `p_GNN(v) || p_GNN(u)`; no longer uses predicted labels. Camouflage-slice sensitivity mandatory.
4. **`u_edge` redefined** with three named roles:
   - `semantic_affinity(e) = cos(X_R[v], X_R[u])`
   - `propagation_conflict(e) = 1 − cos(p_GNN(v) − p_LM(v), p_GNN(u) − p_LM(u))`
   - `relation_prior(e) = 𝟙[rel(e) ∈ R_trusted]`
   - `u_edge = clip(α_aff · affinity − α_conf · conflict/2 + α_rel · prior, −1, +1)`
5. **Accept gate external guard**: posterior_margin_post ≥ posterior_margin_pre − η AND |p_GNN_post − p_GNN_pre|_∞ ≤ ρ = 0.2.
6. **Pre-registered 768-cell grid search**; fixed-weight and no-search variants reported as main-table ablations.
7. **C4 δ grounded in seed std**: `δ = max(0.005, 1.5 × seed_std(L0))`.
8. **C5 kill gate statistical**: 95% paired-bootstrap CI on (M-ro − M-base) excludes 0 AND (M-ro − M-base) ≥ 0.25 × seed_std(M-rr − M-base). Main paper reports **decomposition table** with CIs.
9. **Block-S2 promoted to main**: leave-one-term-out for `s_composite`; remove terms that don't improve slice F1 by ≥ 1 seed-std.
10. **Pilot gates → protocol-only**; paper discusses limitations, not project fallbacks.
11. **Unit tests → artifact doc**; method section has one-line invariant mentions.
12. **Citation language downgrade**: SKETCH/GAugLLM/CTGL = "motivates", not "migrates".

---

## Self-Estimated Nightmare Re-Score (before Round 7 verification)

- Problem Fidelity: 9.6 → **9.8** (P0-4 removed an internal contradiction).
- Method Specificity: 9.5 → **9.5** (roles clearer; some added complexity).
- Contribution Quality: 9.1 → **9.4** (name and framing cleaner; commodity u_edge more principled).
- Frontier Leverage: 8.9 → **8.9** (unchanged).
- Feasibility: 8.7 → **8.5** (grid search + additional variants add ~ 15 GPU-hr).
- Validation Focus: 9.4 → **9.5** (leave-one-term-out promoted, decomposition reporting).
- Venue Readiness: 9.1 → **9.3** (naming + statistical thresholds + pilot-gate internalization).

**Self-estimate overall**: ~ 9.3. Sending to Round 7 verification.
