# Experiment Tracker (v7): EQC — Ego-Quality Controller

**Created**: 2026-05-09 (v7, Round-8 READY nightmare-difficulty)
**Method**: EQC single-primitive controller with compound C5 centrality gate + principled three-role u_edge + external prediction-stability guard + pruning discipline on both composite terms (Block-S2) and modifier components (Block-S4).
**v7 additions over v5**:
- **Rename**: EQC² → EQC (drop "²"); no "conformal" in method name.
- **3-term composite**: `s_lbl + w_tg·s_tg + w_npi·s_npi` (w_lbl ≡ 1); `s_rec` dropped from v1 main (appendix only); `s_het` replaced by `s_npi` (posterior-based, not label-based — avoids heterophily sign trap).
- **Three-role u_edge**: `α_aff · cos(X_R) − α_conf · (1 − cos(p_GNN − p_LM, ..)) / 2 + α_rel · 𝟙[rel ∈ R_trusted]`.
- **External stability guard in accept rule**: `posterior_margin_post ≥ posterior_margin_pre − η` AND `||p_GNN_post − p_GNN_pre||_∞ ≤ ρ = 0.2`.
- **Pre-registered 768-cell discrete grid** + fixed-weight + no-search uniform baselines.
- **C4 δ grounded**: `δ = max(0.005, 1.5 · seed_std(L0))`.
- **C5 compound kill gate** (restored from v6 regression): CI excludes 0 AND `(M-ro − M-base) ≥ max(0.5pp, 0.4·(M-rr − M-base))` on both datasets; β=0.4 pre-registered.
- **Block-S2 promoted to main**: composite term necessity (C6).
- **Block-S4 NEW**: `u_edge` component necessity (S4).
- **G1-NoGuard variant** added to Block-G: isolates external-stability-guard's contribution.
- **Ratio CI** reported for C5: `(M-ro − M-base) / (M-rr − M-base)` 95% paired-bootstrap CI.
- **Post-pruning final u_edge** published as actual method; full three-component form is pre-registered candidate set.

**Supersedes**: v5 tracker (2026-05-09 earlier in session); FRMI v2 (2026-04-20). All prior trackers in git history.
**Method**: EQC² single-primitive controller + deterministic continuous-utility modifier + evidence artifact + artifact-only refiner (v1 main, no LLM) or LLM-enhanced refiner (v1 optional ablation).
**Primary dataset**: TwiBot-20 dev. **Primary robustness**: TwiBot-22. **Secondary robustness**: MGTAB (multi-relational).
**Method source**: `refine-logs/FINAL_PROPOSAL.md` (v5)
**Supersedes**: v4 tracker (2026-05-09 earlier in session); FRMI v2 tracker (2026-04-20). All prior trackers remain in git history.

---

## Status (Week 0 entry, post-Round-6 READY)

| Run ID | Phase | Block | Purpose | Variants | Dataset | Seeds | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| R101 | W1 | Core | Extend `GraphConformalSetEstimator` with composite score + AE + lex selection + edge cases | — | — | — | MUST | TODO | [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) |
| R102 | W1 | Core | Stage-A script — AE + 6-scalar search + `q_hat` + `τ*` + **τ+/τ- quantile freeze** | — | TwiBot-20 | — | MUST | TODO | `train_cal`+`valid_cal` only |
| R103 | W2 | Core | Extend `EgoRefinementRepairOperator` with `u_edge` modifier + EQC² gate + L-hop-local Stage-6 + locality + virtual-context-isolation unit tests | — | — | — | MUST | TODO | [operators.py:194](LLMbot/baseline/core/operators.py#L194) |
| R104 | W2 | Core | Add `t4_ns` + `rewrite_mode` + `refiner_mode` + Stage-7 artifact serializer + Stage-8a refiner MLP | — | — | — | MUST | TODO | Falsification + refiner paths |
| **R105** | **W0** | **Pilot A** | Set-based necessity pilot | T1, T3, T4-NS, BR-soft, T4 | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| **R106** | **W0** | **Pilot B** | Gate necessity pilot | G1, G2, no-refine | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| **R132** | **W0** | **Pilot C (NEW)** | Mechanism-isolation pilot | M-base, M-ro | TwiBot-20 dev | 1 | **MUST** | **BLOCKED on R101–R104** | ~ 2 GPU-hr |
| R107 | W3 | Block-T | Trigger swap full matrix | T1, T2, T3, BR, BR-soft, T4-NS, T4 | TwiBot-20 | 3 | MUST | BLOCKED on R105 pass | 7 × 3 = 21 runs |
| R108 | W3 | Block-T | Trigger swap full matrix | T1, T2, T3, BR, BR-soft, T4-NS, T4 | TwiBot-22 | 3 | MUST | BLOCKED on R107 finish | Primary robustness |
| R109 | W3 | Block-G | Gate toggle | G1, G2, G3 | TwiBot-20 | 3 | MUST | BLOCKED on R106 pass | 3 × 3 = 9 runs |
| R110 | W3 | Block-G | Gate toggle | G1, G2, G3 | TwiBot-22 | 3 | MUST | BLOCKED on R109 finish | — |
| **R127** | W4 | **Block-M (NEW)** | Mechanism isolation — C5 hard kill | M-base, M-ro, M-re, M-rr | TwiBot-20 | 3 | MUST | BLOCKED on R132 pass | 4 × 3 = 12 runs |
| **R128** | W4 | **Block-M (NEW)** | Mechanism isolation — C5 hard kill | M-base, M-ro, M-re, M-rr | TwiBot-22 | 3 | MUST | BLOCKED on R127 finish | — |
| **R126a** | W4 | **Block-R (promoted)** | Soft vs hard rewrite | R-none, R-soft, R-soft+virtual, R-hard-delete, R-hard-add | TwiBot-20 | 3 | MUST | BLOCKED on Pilots | 5 × 3 = 15 runs |
| **R126b** | W4 | **Block-R (promoted)** | Soft vs hard rewrite | R-none, R-soft, R-soft+virtual, R-hard-delete, R-hard-add | TwiBot-22 | 3 | MUST | BLOCKED on R126a finish | — |
| R111 | W4 | Block-X | Cross-dataset | T4 + G1 + deterministic modifier | MGTAB | 3 | MUST | BLOCKED on R108, R110, R128 | `R_trusted` = top-3 densest |
| R112 | W3–W4 | Block-L | Locality audit (continuous) | Log wall-clock ratios; CI locality + virtual-context-isolation tests | — | — | MUST | CONTINUOUS | Defends S1 |
| **R129** | W4 | **Block-BR-public (NEW)** | BotBR faithful reproduction | BR-public | TwiBot-20 + TwiBot-22 | 3 if feasible | MUST | TODO | Documented compatibility report if infeasible |
| **R130a** | W4 | **Block-BR-sensitivity (NEW)** | BR × 3 + BR-soft × 3 | 6 settings | TwiBot-20 | 3 | MUST | TODO | — |
| R130b | W4 | Block-BR-sensitivity | 6 settings | 6 settings | TwiBot-22 | 3 | MUST | BLOCKED on R130a finish | — |
| **R131** | W4 | **Block-AURC (NEW)** | AURC curves at `{5%, 10%}` | From R107/R108/R127/R128 outputs | TwiBot-20, 22, MGTAB | reuse | MUST | Analysis-only, no extra GPU | — |
| **R120** | W4 | **D-EstSem (NEW)** | Diagnostic correlation study | s_composite vs heterophily / degree / label-shuffle / edge-removal | TwiBot-20 + TwiBot-22 | 3 | SHOULD | TODO | Observational; answers U#1 |
| R124 | W5 | Block-LLM-Attrib | L0 vs L2 non-inferiority | 2 variants | TwiBot-20 + TwiBot-22 | 3 | MUST | BLOCKED on R101–R104 | C4 |
| R113 | W5 | A-Composite | Drop-one `s_tg`/`s_het`/`s_rec` | 3 variants | TwiBot-20 | 3 | SHOULD | TODO | Defends S2 |
| R114 | W5 | A-Modifier | Learned modifier vs deterministic | 1 variant | TwiBot-20 | 3 | NICE | **Strict-kill gated by R121/R122/R123/R123b** | — |
| R121 | W5 | A-CF-Falsifiers | No-CF baseline | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| R122 | W5 | A-CF-Falsifiers | Split-transfer | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| R123 | W5 | A-CF-Falsifiers | Label-shuffle | 1 variant | TwiBot-20 | 3 | NICE | Strict-kill gate | Fire ⇒ drop A-Modifier |
| **R123b** | W5 | **A-CF-Falsifiers (NEW)** | CF-signal permutation within degree bins | 1 variant | TwiBot-20 | 3 | NICE | **Strict-kill gate** | Fire ⇒ drop A-Modifier |
| R115 | W5 | A-Diffusion | DAPS/NAPS, SNAPS | 2 variants | TwiBot-20 | 3 | NICE | TODO | — |
| R116 | W5 | A-Iter | `max_iter=2` + rollback | 1 variant | TwiBot-20 | 3 | NICE | TODO | — |
| R117 | W5 | A-NestedSplit | 5-fold `train_cal / valid_cal` rotation | v1 main | TwiBot-20 | 3 | SHOULD | TODO | Defends tuning protocol |
| R125 | W5 | A-Prompt | 7-field drop-one on artifact | 7 variants | TwiBot-20 | 3 | **Conditional on L2 > L0 + δ** | TODO | Skipped if C4 holds |
| R118 | W5 | Analysis | Tables/figures/bootstrap/CIs/AURC | — | — | — | MUST | TODO | Handoff to `/paper-figure` |
| R119 | W5 | Paper | `/paper-plan` → `/paper-write` | — | — | — | MUST | TODO | Final LaTeX draft |

---

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-09 | **Pivot to EQC²** (supersedes FRMI v2 and Embedding-Dominant) | `/research-refine` Round-4 9.2 READY spec-level; anchor preserved across 4 rounds. |
| 2026-05-09 | Dominant contribution = EQC² single-primitive controller | Round-1 reviewer forced consolidation. |
| 2026-05-09 | `s_tg` = JSD(p_LM ‖ p_GNN) / log 2 | Round-2 reviewer: probability-space fix. |
| 2026-05-09 | T4-NS falsification baseline added | Round-2 reviewer. |
| 2026-05-09 | `w_lbl ≡ 1` (6-scalar search) | Round-2 reviewer: reduce hidden DOF. |
| 2026-05-09 | Stage-6 L-hop-local recomputation + fail-loud locality test | Round-3 reviewer. |
| 2026-05-09 | `virtual_context_edges` dropped from v1 main (Round 3), RESTORED as artifact-only in v5 (user rescope) | Round-3 simplification; user-directed rescope for evidence artifact. |
| 2026-05-09 | Pilot Gates A/B pre-committed with thresholds | Round-3 reviewer: package empirical items as Week-0 gates. |
| 2026-05-09 | MGTAB replaces Cresci-17 as secondary robustness | Round-1 reviewer. |
| 2026-05-09 | **User rescope: 4 locked decisions + 9 uncertainties** | Principal-researcher directive; pivot to v5. |
| 2026-05-09 | Modifier locked to `continuous utility u_edge` only | User Decision 1 (BotBR/BECE collision avoidance). |
| 2026-05-09 | v1 main = artifact-only refiner; LLM branch = v1 optional ablation | User Decision 2. |
| 2026-05-09 | Rewrite boundary: soft reweight propagation edges only; virtual edges artifact-only | User Decision 3. |
| 2026-05-09 | Estimator claim language tightened | User Decision 4: "prediction-set uncertainty under ego context", not "graph quality". |
| 2026-05-09 | **Block-M + C5 hard kill gate added** | Round-5 nightmare reviewer: mechanism isolation required. |
| 2026-05-09 | **BR + BR-soft in Block-T; BR-public attempt + sensitivity** | Round-5 reviewer: BotBR/BECE differentiation must be experimental. |
| 2026-05-09 | **C4 upgraded to non-inferiority** (δ=0.015 + 95% CI + ratio 0.90 dual) | Round-5 reviewer: ratio-only unstable. |
| 2026-05-09 | **τ+/τ- frozen as valid_cal quantiles** | Round-5 reviewer: reduce hidden DOF; 8→6 scalars. |
| 2026-05-09 | **Strict kill rule on A-Modifier** (any of R121/R122/R123/R123b fires ⇒ dropped) | Round-5 reviewer: negative controls binding. |
| 2026-05-09 | **Pilot C (R132) added** | Round-5 reviewer: mechanism-isolation pilot before full Block-M. |
| 2026-05-09 | **Paired-bootstrap CI on C5 difference-of-differences** | Round-6 MINOR modernization. |
| 2026-05-09 | **AURC at pre-registered ratios {5%, 10%}** | Round-6 MINOR modernization. |

---

## v8 EQC Pipeline Results (2026-05-12, seed 1, TwiBot-20)

> Executed by `/experiment-bridge`. Single seed, single dataset. Proxy limitations noted per variant.
> Artifact root: `LLMbot/baseline/core/.tmp_base_rerun/seed_1/`

### Block-E: Encoder Variants (Stage-6 linear probe)

| Variant | val_F1 | test_F1 | hard_F1 | Notes |
|---|---|---|---|---|
| e_null | 0.8496 | 0.8518 | 0.7675 | Baseline: [h_gnn; x_roberta] only |
| e_roberta | 0.8488 | 0.8561 | 0.7574 | RoBERTa on canonical evidence graph text |
| e_llm_qwen | **0.8521** | **0.8569** | **0.7724** | Qwen3 PCA-768 proxy (raw user text, not evidence graph) |
| e_roberta_collapse | 0.8505 | 0.8569 | 0.7625 | RoBERTa on collapse schema |
| e_roberta_shuf_role | 0.8500 | 0.8577 | 0.7625 | RoBERTa on shuffle_role schema |
| e_roberta_shuf_order | 0.8488 | 0.8569 | 0.7625 | RoBERTa on shuffle_order schema |

Direction check (e_null < e_roberta < e_llm_qwen on test_F1): **PASS**
Scramble degradation (e_roberta_* < e_roberta): **FAIL** — scramble variants ≥ canonical on test_F1.
Interpretation: evidence graph role/order structure provides no signal for RoBERTa encoder; Qwen3 PCA proxy adds +0.005 F1 over e_null.

### Block-P: Necessity Audit (conformal trigger vs random vs all-nodes)

| Variant | val_F1 | test_F1 | hard_F1 | Notes |
|---|---|---|---|---|
| P0_conformal | 0.8476 | 0.8552 | 0.7724 | Conformal trigger → hard nodes only |
| P1_random | 0.8488 | 0.8543 | **0.7828** | Random 15% budget — hard_F1 > P0 |
| P3_all_nodes | 0.8488 | 0.8561 | 0.7574 | All nodes get evidence graph |
| P4_no_retrieval | 0.8484 | **0.8585** | 0.7777 | Hard nodes, empty retrieval (target text only) |

Conformal trigger necessity: **WEAK** — P1 (random) matches or exceeds P0 on hard_F1; P4 (no retrieval) has highest test_F1.
Interpretation: conformal trigger selects nodes but doesn't clearly outperform random selection; retrieval step adds noise rather than signal at this scale.

### Block-R: Retrieval Budget

| Variant | val_F1 | test_F1 | hard_F1 | Notes |
|---|---|---|---|---|
| R0_iter2_k8 | 0.8476 | 0.8552 | 0.7724 | Full retrieval (default) |
| R1_iter1_k8 | 0.8484 | **0.8569** | 0.7724 | Single-pass — slightly better |
| R2_iter1_k16 | 0.8484 | 0.8552 | 0.7673 | Wider k — no benefit |

Interpretation: single-pass retrieval (R1) is marginally better; wider k (R2) doesn't help. Retrieval budget has minimal impact.

### Block-L: Probe Architecture

| Variant | val_F1 | test_F1 | hard_F1 | Notes |
|---|---|---|---|---|
| L0_linear | 0.8476 | **0.8552** | **0.7724** | Linear probe (default) |
| L1_mlp256 | **0.8498** | 0.8463 | 0.7556 | 2-layer MLP-256 — worse on test |

Interpretation: linear probe dominates MLP-256 on test_F1 and hard_F1. Consistent with SimTeG finding.

### Summary and Fallback Assessment

| Claim | Status | Evidence |
|---|---|---|
| Conformal trigger > random trigger | **WEAK** | P1 hard_F1 (0.7828) > P0 (0.7724); test_F1 within 0.001 |
| Evidence graph structure matters | **FAIL** | Scramble variants ≥ canonical on test_F1 |
| Retrieval is load-bearing | **FAIL** | P4 (no retrieval) has highest test_F1 (0.8585) |
| LLM embedding adds signal | **WEAK PASS** | e_llm_qwen +0.005 over e_null (proxy, not actual evidence graph encoding) |
| Linear probe > MLP | **PASS** | L0 > L1 on test_F1 and hard_F1 |

**Fallback triggered**: Evidence graph structure and retrieval are not load-bearing at this scale (single seed, TwiBot-20). The dominant signal comes from the Qwen3 PCA embeddings (raw user text), not from the evidence graph construction. This activates the "Pilot-C / C5 fail" fallback: paper should recenter on the conformal trigger as a selection signal for an embedding-based refiner, not as a rewrite+gate controller.

**Next steps**: Run 3 seeds to confirm; run on TwiBot-22 for cross-dataset check; investigate whether actual evidence graph Qwen3 encodings (not PCA proxy) change the picture.

---

## Pre-committed Fallback Plans

| Trigger | Action | Paper impact |
|---|---|---|
| **Pilot A fail** (T4 ≤ T4-NS within 0.010) | Contract to **EQC-S** (composite scalar controller, no set geometry); T4-NS becomes v1 main; re-run pilot with contracted variants. | Dominant contribution → "composite ego-uncertainty score + calibrated scalar selection rule"; title/claim change. |
| **Pilot A fail via BR-soft** (BR-soft ≥ T4) | Concede prior-art sufficiency; refinement-of-prior-art framing. | Rework positioning against BotBR/BECE; honest novelty contraction. |
| **Pilot B fail** (G2 no degradation OR G1 degrades) | Redesign gate; re-run. If still fail, drop gate contribution; paper becomes trigger-only. | Supporting contribution removed. |
| **Pilot C / C5 fail** (M-ro < 0.5·M-rr) | **Demote Stage-5/6**; paper recenters on "artifact + refiner" story; investigate Stage-8 as mechanism. | Narrative reframe: EQC² becomes selection signal for an artifact-driven refiner, not a rewrite+gate controller. |
| **C4 fail** (L2 − L0 > 0.015 significantly) | Concede LLM is the mechanism; reframe Stage-8b as primary. | LLM-enhanced branch becomes core, artifact-only a baseline. |
| **Block-M C5 hard kill** (fails on either dataset) | As above (Pilot-C fallback). | — |
| **Block-BR-public beats T4 on both datasets** | Refinement-of-prior-art framing; EQC² positioned as structured-selection refinement. | Honest positioning. |
| **Rollback rate > 15% on TwiBot-22** | Single-step `λ` reduction; re-run once; if still > 15%, report as limitation. | — |
| **MGTAB sign disagreement (Block-X)** | Honest limitation report: "robust on TwiBot-20/22; MGTAB reveals relation-type sensitivity". | Minor narrative adjustment. |
| **Locality / virtual-context unit test failure** | Fail-loud; no claim until fix. | None — correctness invariant. |
| **`test` leakage detection** | Discard run; audit Stage-A. | None — correctness invariant. |
| **A-Modifier strict kill** (R121/R122/R123/R123b fires) | Entire A-Modifier section dropped from paper. | None — credibility instrument. |
| **A-Prompt conditional skip** (L2 ≤ L0 + δ) | A-Prompt not run. | None — conditional scope. |

## Compute Tally (Projected v5)

| Bucket | GPU-hr | Notes |
|---|---|---|
| Core machinery (R101–R104) | ~ 4 | Dev verification |
| Pilot A (R105) | ~ 2 | 1 seed × 5 variants × TwiBot-20 dev |
| Pilot B (R106) | ~ 2 | 1 seed × 3 variants × TwiBot-20 dev |
| Pilot C (R132, NEW) | ~ 2 | 1 seed × 2 variants × TwiBot-20 dev |
| Block-T (R107, R108) | ~ 20–30 | 7 × 3 × 2 datasets |
| Block-G (R109, R110) | ~ 12–18 | 3 × 3 × 2 datasets |
| Block-M (R127, R128, NEW) | ~ 16–24 | 4 × 3 × 2 datasets |
| Block-R promoted (R126a, R126b) | ~ 20–30 | 5 × 3 × 2 datasets |
| Block-X (R111) | ~ 10–15 | MGTAB |
| Block-BR-public (R129) | ~ 5–10 | If compatible; otherwise incompatibility report |
| Block-BR-sensitivity (R130a, R130b) | ~ 10–18 | 6 × 3 × 2 datasets |
| Block-AURC (R131) | 0 | Analysis-only |
| D-EstSem (R120, NEW) | ~ 3 | Observational |
| Block-LLM-Attrib (R124) | ~ 3 | L0 vs L2, 3 seeds, 2 datasets; LLM pre-computed |
| Appendix (R113/R115/R116/R117) | ~ 20–30 | — |
| A-Modifier + CF-Falsifiers (R114/R121/R122/R123/R123b) | ~ 15 | Conditional on strict kill |
| A-Prompt (R125) | ~ 3 | Conditional on C4 |
| Analysis + Paper (R118, R119) | ~ 5 | CPU-dominated |
| **Total** | **~ 140–220 GPU-hr** | Fits within existing project budget (previous Round-4 estimate was 85–125; +55–95 for Block-M, Block-R promotion, Block-BR-public, Block-BR-sensitivity, D-EstSem, Pilot C) |

## Links

- Method spec: `refine-logs/FINAL_PROPOSAL.md` (v5)
- Review history: `refine-logs/REVIEW_SUMMARY.md`, `refine-logs/REFINEMENT_REPORT.md`, `refine-logs/round-*.md`, `refine-logs/score-history.md`
- Research audit: `refine-logs/RESEARCH_AUDIT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (v5)
- Implementation entry points: [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922), [operators.py:194](LLMbot/baseline/core/operators.py#L194)
- Anchor: `research.md` (2026-05-09)
- Superseded charters (git history): `idea-stage/PAPER_CHARTER.md` (2026-04-22 FRMI v2); prior `refine-logs/FINAL_PROPOSAL.md` (2026-04-19 Embedding-Dominant)
