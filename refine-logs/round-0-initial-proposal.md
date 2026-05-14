# Research Proposal: Conformal Ego-Quality Triggered Local Refinement for Social Bot Detection

**Version**: Round 0 (initial proposal)
**Date**: 2026-05-09
**Anchor source**: `research.md` (2026-05-09, canonical)
**Supersedes**: FRMI v2 charter (2026-04-22), Embedding-Dominant proposal (2026-04-19)

---

## Problem Anchor

> Carried verbatim into every round. Do not edit.

- **Bottom-line problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage — not uniformly across the node population.
- **Must-solve bottleneck**:
  - Scalar post-hoc calibration (MSP / entropy / margin / temperature / GATS / CaGCN) produces a confidence number but no **operational ego-graph-quality signal** that can gate selective, budgeted, reversible ego refinement.
  - Hard edge-reliability methods already strong in this domain (BotBR SIGIR'25, BECE TNNLS'25) compress the bot-human heterophily signal into binary reliable/unreliable and thus destroy camouflage evidence that is itself diagnostic in social bot graphs.
  - Global graph structure learning (LDS, Pro-GNN, IDGL) is too expensive, non-local, and non-rollbackable to trust as a post-hoc correction layer on a frozen Stage-1 backbone.
- **Non-goals**: (i) global graph structure learning, (ii) LLM as final classifier, (iii) re-proving binary edge reliability, (iv) full causal-graph repair, (v) a new GNN architecture.
- **Constraints**:
  - Frozen Stage-1 RoBERTa + BotRGCN/RGCN backbone; no base-GNN retraining.
  - v1 main method makes **no LLM calls**; LLM evidence prompt is a v2 ablation only.
  - Soft-reweight existing ego edges only; virtual context edges are never written back to the main graph.
  - Compute ≤ existing project GPU budget (TwiBot-20 scale primary, TwiBot-22 scale validated).
  - Main method `max_iter = 1`; `max_iter = 2` allowed only as ablation with rollback.
- **Success condition**: On TwiBot-20 and TwiBot-22, conformal-triggered local ego refinement raises macro-F1 and/or reduces AURC / average set size on the high-`abstain_risk` slice without hurting global macro-F1 vs the frozen co-trained baseline, under a matched intervention budget. Ablations must separately isolate the contribution of (a) the conformal trigger, (b) coupled text-graph retrieval, and (c) the edge modifier output form.

---

## Technical Gap

The frozen Stage-1 RoBERTa → BotRGCN pipeline leaves two classes of residual error:

1. **Ego-quality errors**: ego-graph itself (edges, neighbor texts, propagation mass) is noisy or camouflaged; even a perfectly calibrated scalar confidence cannot say *which* local intervention is safe.
2. **Local text-structure conflict errors**: RoBERTa evidence and graph evidence disagree on the same node; a global fusion layer is both too blunt (touches all nodes) and too crude (cannot distinguish a supportive from a suspicious edge).

Naive fixes fall short:

- **Temperature scaling, GATS, CaGCN**: produce a better scalar but no refinement handle — no prediction set, no coverage margin, no per-edge role.
- **BotBR/BECE edge detector + reliability graph**: already occupy the "binary reliability" slot; duplicating them is not a contribution and also removes camouflage evidence useful for bot-human heterophily analysis.
- **LDS/Pro-GNN/IDGL global structure learning**: requires joint retraining, sacrifices audit trail, and drowns localized errors in global objectives.
- **LLM-as-predictor (GraphGPT, direct LLM classify)**: TMLR 2024 (graph-prompt analysis) shows LLMs treat graph prompts as keyword-augmented paragraphs, not graph reasoners. Scope + cost incompatible with a post-hoc correction layer.
- **LOGIN/GLANCE selective LLM consultation**: right spirit (only hard nodes) but triggers on scalar uncertainty / homophily and needs LLM in the loop — we first need a **more informative, LLM-free trigger**.

The missing primitive is a **post-hoc ego-graph-quality estimator** that (i) returns a prediction set and a scalar abstain risk on the calibration-fitted conformal scale, (ii) can route hard nodes to refinement, and (iii) provides the same quality signal before and after refinement so the pipeline can accept, rollback, or escalate. Everything else — retrieval, modifier, rewrite, LLM — sits downstream of that trigger.

---

## Method Thesis

**One-sentence thesis**: A post-hoc graph-conformal prediction-set estimator provides a calibrated, multi-signal (set-size, coverage-margin, abstain-risk) **trigger *and* accept/rollback gate** for budgeted local ego refinement, strictly dominating scalar uncertainty triggers while leaving the frozen backbone and the binary-edge-reliability literature untouched.

- **Why this is the smallest adequate intervention**: the estimator is fitted on train/valid only, consumes existing base logits, and outputs a structured quality signal without any parameter update to the backbone. All downstream refinement is a single-pass soft reweight over existing ego edges, with the same estimator closing the accept/rollback loop.
- **Why this route is timely in the foundation-model era**: RoBERTa-quality node embeddings make semantic retrieval cheap enough to couple with structural retrieval inside a modifier; we exploit that prior without making the LM the classifier (rejected by TMLR 2024 evidence) and without bolting on LLM calls in v1 (rejected by research.md §14).

---

## Contribution Focus

- **Dominant contribution**: **Conformal ego-graph-quality as a local-refinement trigger and accept/rollback gate**. We are the first to use graph-conformal prediction-set quality not as an end-of-pipe uncertainty report but as a structured, per-node control signal for budgeted, reversible ego-graph modification in social bot detection. Anchor reference: research.md §2, §12 Contribution 1.
- **Supporting contribution**: **Text-graph coupled local ego refinement with deliberately-open edge-modifier output form**. We instantiate a SKETCH-decoupled / GAugLLM-coupled modifier over existing ego edges, with binary / continuous / role-aware outputs left as an ablation axis (research.md §5 explicitly forbids prejudging this). Anchor reference: research.md §4, §5, §12 Contribution 2.
- **Explicit non-contributions** (enforced by Problem Anchor + BotBR/BECE prior art):
  - No "first graph rewrite for bot detection" claim (BotBR, BECE).
  - No "binary edge reliability" headline (covered by BotBR; would also destroy camouflage evidence).
  - No "LLM understands graphs" claim (TMLR 2024 counter-evidence).
  - No global structure learning (LDS / Pro-GNN / IDGL).
  - No "new SOTA detector" (we are a post-hoc layer).
  - No "causal repair" (we produce counterfactual-intervention *supervision*, not causal identification).

---

## Proposed Method

### Complexity Budget

| Slot | Choice | Rationale |
|------|--------|-----------|
| Frozen / reused | Stage-1 RoBERTa text encoder (`roberta-finetuned-20`), BotRGCN/RGCN base GNN, existing ego-graph adjacency | All pre-existing in `LLMbot/baseline/core/LM.py`, `model_building.py`. Reuse is the point. |
| New trainable (cap = 2) | (1) Coupled edge modifier head `M_phi: (z_text(e), z_graph(e), Q_phi(Ego_v)) -> (s_reliability, u_edge, p_role)`. (2) Optional router calibration head on top of `abstain_risk` + residual-risk manifest. | Router head may collapse into a deterministic thresholding rule if ablation shows it adds no gain. |
| Not trainable | Conformal estimator (threshold fit by quantile on calibration split), retrieval (kNN + PPR), rewrite arithmetic. | Deliberately kept as deterministic scaffolding for audit. |
| Tempting additions excluded | LLM evidence prompt (v2 only); global graph structure learning; binary hard delete; CF-GNN topology-aware correction (baseline only, not vendored); dual-model EM loop (GLEM — too expensive post-hoc). |

### System Overview

```
Stage-0 (existing, frozen)
  RoBERTa text embeddings X_R ──► BotRGCN base GNN ──► base_logits(v), base_probs(v)

Stage-1 Conformal ego-quality estimator Q_phi
  (fit on train/valid only)
  Q_phi(Ego_v, X_R, A_v) ─► {prediction_set(v), set_size(v),
                              coverage_margin(v), abstain_risk(v),
                              calibration_metadata}

Stage-2 Hard-node router
  hard(v) = Router(abstain_risk, set_size, coverage_margin,
                   base_uncertainty, existing residual_risk_manifest)

Stage-3 SKETCH-style retrieval (for hard v only)
  semantic_candidates(v)   = kNN_{RoBERTa}(v, k_sem)
  structural_candidates(v) = PPR / k-hop / degree-aware ranking over existing ego
  propagation_edges(v)     = candidates ∩ existing ego edges   (writable)
  virtual_context_edges(v) = candidates ∖ existing ego edges   (evidence-only, NOT writable)

Stage-4 Coupled modifier M_phi
  z_text(e)  = TextPairEncoder(X_R[v], X_R[u])
  z_graph(e) = LocalGraphPairEncoder(Ego_v, e, X_R, Q_phi)
  z_e        = CoupledGate(z_text(e), z_graph(e), Q_phi(Ego_v))
  outputs    = (s_reliability(e), u_edge(e), p_role(e))  # output form ablated

Stage-5 Soft rewrite (propagation_edges only)
  w_e' = w_e · clip(1 + λ · budget_v · modifier_score(e) · (1 - uncertainty_e),
                    1 - λ, 1 + λ)

Stage-6 Accept / rollback via Q_phi re-evaluation
  accept if set_size(v)↓ OR coverage_margin(v)↑ OR abstain_risk(v)↓
            AND budget(v) ≤ B_max
  else rollback Ego_v ← Ego_v_pre
```

### Core Mechanism — Conformal Ego-Quality Estimator `Q_phi`

- **Input**: base logits/probs from the frozen GNN; optional local adjacency `A_v` for graph-diffused scoring (E4 ablation).
- **Non-conformity score**: `s(v, y) = 1 - p_base(y | v)` (APS/THR variant per CF-GNN; exact choice ablated).
- **Threshold**: `q_hat = Quantile_{(n_cal+1)(1-alpha)/n_cal}({s(v_i, y_i) : v_i in cal_split})`.
- **Outputs**:
  - `prediction_set(v) = { y : s(v, y) ≤ q_hat }`.
  - `set_size(v) = |prediction_set(v)|`.
  - `coverage_margin(v) = q_hat - min_{y ∈ set} s(v, y)` (distance to the conformal threshold).
  - `abstain_risk(v) = 0.70 · size_risk(v) + 0.30 · margin_risk(v)` (matches existing implementation; weights are a hyperparameter, audited in ablation).
  - `calibration_metadata`: record of calibration split, alpha, quantile, which labels participated.
- **Critical invariant**: test labels never enter threshold fitting. Unit test already scaffolded at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922).
- **Why this is the main novelty**: the estimator is *used as a controller*, not a reporter. Three consequences downstream — routing hard nodes, parameterizing modifier uncertainty input (`1 - uncertainty_e`), and gating accept/rollback — all rely on the same calibrated structured signal. Scalar alternatives (MSP/entropy/GATS/CaGCN) cannot provide the triadic (size, margin, risk) handle.

### Supporting Component — Text-Graph Coupled Modifier `M_phi`

- **Input to `M_phi`** per candidate edge `e = (v, u)` on hard node `v`:
  - `z_text(e) = TextPairEncoder(X_R[v], X_R[u])` — e.g. concat + MLP over RoBERTa embeddings.
  - `z_graph(e) = LocalGraphPairEncoder(Ego_v, e, X_R, Q_phi)` — encodes degree, PPR score, role inside ego, `Q_phi` state.
  - `z_e = CoupledGate(z_text(e), z_graph(e), Q_phi(Ego_v))` — sigmoidal gate per GAugLLM; not a sum.
- **Output (all heads computed; selected by ablation)**:
  - `s_reliability(e) ∈ {0, 1}` — binary (B0, reproduces BotBR/BECE space).
  - `u_edge(e) ∈ [-1, 1]` — continuous utility (B1).
  - `p_role(e) ∈ Δ^4` — supportive / suspicious / neutral / uncertain (B2).
- **Supervision** (counterfactual edge intervention on train/valid only):
  - `Delta_down(e) = L_v(A_v[e ← w_e·(1-δ)]) - L_v(A_v)`.
  - `Delta_drop(e) = L_v(A_v \ e) - L_v(A_v)`.
  - `Delta_up(e)   = L_v(A_v[e ← w_e·(1+δ)]) - L_v(A_v)`.
  - Training targets: regression on normalized counterfactual gain (for `u_edge`), binarization with dead-zone (for `s_reliability`), soft-argmax over `{Delta_down, Delta_drop, Delta_up, |Delta| < τ}` (for `p_role`).
- **Why this is not "just another module"**: the modifier lives **only on hard nodes**, writes **only onto existing ego edges**, and is constrained by a budget derived from `abstain_risk(v)`. Without the conformal trigger, the modifier is just a scored edge head — the novelty sits in the trigger-bounded, rollback-gated usage, not in the head itself.

### Modern Primitive Usage

- **RoBERTa text encoder**: used as a frozen semantic prior for (i) `TextPairEncoder` input and (ii) semantic-kNN retrieval candidates. Not fine-tuned in v1. Follows SimTeG finding that frozen PEFT text features are already a strong backbone and TMLR-2024 warning against LLM-as-graph-reasoner.
- **Conformal prediction**: CF-GNN / DAPS-NAPS / SNAPS as method boundary, not direct dependency. Our v1 port is minimal (no topology-aware correction, no diffusion) so the estimator is easy to audit.
- **Foundation-model-era primitive deliberately *not* used in v1**: LLM evidence prompt. Justification is threefold: (i) research.md §14 locks it for v2 to keep the v1 cost and reproducibility envelope tight; (ii) TMLR-2024 shows LLMs read graph prompts as keyword-augmented paragraphs; (iii) the dominant claim is about the **trigger**, not the refiner content, so introducing an LLM refiner first would muddle attribution.

### Integration into Existing Pipeline

- Hook the estimator into the existing `GraphConformalSetEstimator` surface at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922); extend with (i) calibration-split recording, (ii) DAPS/NAPS diffusion switch, (iii) SNAPS semantic-kNN switch (both as ablation flags, off by default in v1 main method).
- Route hard nodes through the existing residual-risk selector at [estimators.py:58-67](LLMbot/baseline/core/estimators.py#L58-L67), adding a `conformal_filter_mode ∈ {off, risk_floor, rank_combo}` knob.
- Extend `EgoRefinementRepairOperator` at [operators.py:194](LLMbot/baseline/core/operators.py#L194) with the coupled modifier heads and the counterfactual-supervision objective; keep the existing `_edge_role()` deterministic scaffolding as an ablation baseline.
- No changes to Stage-0 (RoBERTa + base GNN).

### Training Plan

1. **Stage-0 is already trained and frozen.**
2. **Calibration fit**: on `train_cal ∪ valid_cal` split, compute non-conformity scores from frozen base logits, derive `q_hat`. Record `calibration_metadata`. No SGD.
3. **Modifier head pre-train** (only trainable step): SGD on train/valid counterfactual-supervision targets, with class-balanced sampling of edges and three heads (binary / continuous / role) trained jointly under a weighted loss that each ablation selects among.
4. **Inference**: single forward pass through Stages 1-5, then Stage 6 accept/rollback. `max_iter = 1`.

### Failure Modes and Diagnostics

| Failure mode | Detection | Fallback |
|---|---|---|
| Calibration leak (test labels in threshold) | CI check against `calibration_metadata.labels_used` | Fail loud; refuse to emit prediction sets. |
| Modifier overfits to counterfactual supervision | Gap between train-counterfactual loss and test global-F1 / AURC | Freeze modifier weights, fall back to deterministic `_edge_role` scaffolding. |
| Accept/rollback thrashing (> r% rollbacks) | Per-node rollback log | Tighten budget `B_max`, reduce `λ`, disable the refinement at `max_iter=1`. |
| Hard-node slice selected by residual-risk only (conformal filter inactive) | `conformal_filter_mode == off` in provenance | Raise warning; ablation E0 branch. |
| Confirmation bias in multi-iter | Coverage / set-size does not strictly improve across iters | `max_iter` capped at 2 with rollback; no iteration beyond 2 in main method. |

### Novelty and Elegance Argument

- **Closest prior art**: CF-GNN (conformal for GNN reporting), GLANCE (selective LLM consulting), BotBR/BECE (binary edge reliability), GAugLLM (collaborative edge modifier in contrastive learning).
- **Exact difference**: none of them use **conformal prediction-set geometry as a controller driving budgeted, reversible local ego refinement** in social bot detection. CF-GNN stops at reporting; GLANCE triggers on scalar homophily/uncertainty and needs LLM; BotBR/BECE collapse to binary reliability; GAugLLM is a global training-time augmentation, not a post-hoc single-node refiner.
- **Parsimony argument**: exactly one trigger primitive + one supporting modifier + one deterministic rewrite rule. No cascaded ML heads, no new backbone. The Complexity Budget has ≤ 2 trainable components (modifier, optional router calibration), satisfying `MAX_NEW_TRAINABLE_COMPONENTS`.

---

## Claim-Driven Validation Sketch

> Three core experiment blocks (cap `MAX_CORE_EXPERIMENTS = 3`). Full execution roadmap deferred to `/experiment-plan`.

### Claim 1: Conformal trigger dominates scalar triggers

- Minimal experiment: identical downstream refinement (default modifier = B1 continuous, fixed budget), swap only the trigger.
- Baselines / ablations: E0 (MSP/entropy/margin) · E1 (temperature scaling) · E2 (GATS/CaGCN) · E3 (conformal set, no graph diffusion) · E4 (+ DAPS/NAPS diffusion) · E5 (+ SNAPS semantic-kNN).
- Metric: macro-F1, AURC, ECE, singleton-hit ratio, hard-node-slice macro-F1. Primary: macro-F1 on high-`abstain_risk` slice.
- Expected evidence: E3 ≥ E0/E1/E2 on high-risk-slice F1 with tight budget; E4/E5 give additional gains only where graph diffusion is genuinely informative.

### Claim 2: Coupled text-graph modifier beats decoupled modifiers

- Minimal experiment: fix trigger = E3 and rewrite rule = soft reweight + virtual context (S4), swap only retrieval/modifier.
- Baselines / ablations: R2 (semantic-only retrieval) · R3 (structural-only retrieval) · R4 (uncoupled semantic+structural) · R5 (coupled `M_phi`).
- Modifier output form ablation (inside R5): B0 (binary reliability) · B1 (continuous utility) · B2 (evidence-aware role).
- Metric: macro-F1 on bot-human mixed-neighborhood slice and heterophily slice, average weight-change budget used, fraction of edges labeled suspicious but *retained* as evidence.
- Expected evidence: R5 ≥ R4 on heterophily slice; at least one of B1 / B2 strictly outperforms B0 on camouflage slice — validating the "don't collapse to binary" hypothesis.

### Claim 3: Soft local rewrite is necessary and sufficient; cross-dataset robustness

- Minimal experiment: fix trigger = E3, modifier = best of Claim-2, budget matched. Vary rewrite action and dataset.
- Baselines / ablations: S1 (hard delete) · S2 (hard add) · S3 (soft reweight, v1 main) · S4 (soft reweight + virtual context, v1 main).
- Cross-dataset: evaluate S3/S4 configs on TwiBot-22 (primary robustness) and MGTAB or Cresci-17 (secondary robustness). 3 seeds per condition, matched budget.
- Simplification check: drop the modifier (use deterministic `_edge_role`, keep everything else) — does the pipeline still move? **If yes, modifier contribution claim weakens.**
- Necessity check: drop the conformal estimator (replace with entropy trigger) — does the pipeline still move? **If yes, trigger contribution claim weakens.**
- Metric: global macro-F1 (no degradation gate), high-`abstain_risk`-slice macro-F1 (improvement gate), AURC, average budget, rollback rate.
- Expected evidence: S3/S4 preserve global F1 and improve slice F1; S1 hurts camouflage slice; on TwiBot-22 the gain persists; on MGTAB/Cresci the sign of the gain is consistent even if magnitude shrinks.

---

## Experiment Handoff Inputs

- **Must-prove claims**: Claim 1 (trigger), Claim 2 (coupled modifier), Claim 3 (soft rewrite + cross-dataset).
- **Must-run ablations**: E0 vs E3; R5 vs R4 vs R2/R3; B0 vs B1 vs B2; S1 vs S3 vs S4; simplification drop-modifier; necessity drop-conformal.
- **Critical datasets**: TwiBot-20 (primary), TwiBot-22 (primary robustness), MGTAB or Cresci-17 (secondary robustness).
- **Critical metrics**: macro-F1 (global + high-risk slice + heterophily slice), AURC, ECE, conformal coverage + set size, rollback rate, weight-change budget consumed, LLM call ratio = 0 (v1 sanity check).
- **Highest-risk assumptions**:
  - Counterfactual-intervention supervision on train/valid transfers to test without leakage.
  - `abstain_risk = 0.70·size_risk + 0.30·margin_risk` weighting is near-optimal; audit via ablation.
  - Frozen Stage-0 is not so weak that the hard-node slice is pathologically small.
  - Soft reweight `λ` and budget `B_max` can be fixed on valid_cal without per-seed drift.

---

## Compute & Timeline Estimate

- **GPU-hours (v1 only, 3 seeds × 3 datasets)**:
  - Stage-0: already trained (0h incremental).
  - Calibration fit: CPU-bound, < 5 minutes per dataset per seed.
  - Modifier head training: small MLP over pair features; ~ 1-2 GPU-hr per seed per dataset.
  - Full ablation matrix (E0..E5 × R2..R5 × B0..B2 × S1..S4, pruned to the ~ 30 leading cells per the three claim blocks): ~ 30-50 GPU-hr per dataset.
  - **Total estimate**: ~ 150-200 GPU-hr across 3 datasets × 3 seeds.
- **Data / annotation cost**: zero new human annotation. Counterfactual supervision is constructed from train/valid splits only.
- **Timeline** (once proposal is READY):
  - Week 1: extend conformal estimator with DAPS/NAPS + SNAPS switches; scaffold counterfactual supervisor.
  - Week 2: modifier heads + ablation scaffolding.
  - Week 3: run Claim 1 + Claim 2 blocks on TwiBot-20.
  - Week 4: Claim 3 block + TwiBot-22 + MGTAB/Cresci cross-dataset.
  - Week 5: analysis, figures, paper draft seeded by `/paper-plan`.

---

## Open Questions for the Round-1 Reviewer

We flag these so the reviewer can either confirm or steer:

1. Is E4 (DAPS/NAPS graph-diffused conformal scores) a legitimate v1 main-method ingredient, or must it stay as an ablation to keep novelty unambiguous?
2. Does the counterfactual-intervention supervision on train/valid edges cross into "pseudo-causal identification" territory we should rename or narrow?
3. Is the `abstain_risk = 0.70·size_risk + 0.30·margin_risk` weighting too hand-tuned to defend at top-venue review, and should it be replaced by a learned combiner fitted on valid_cal?
4. Is MGTAB or Cresci-17 a stronger choice for the cross-dataset robustness check, given that TwiBot-22 already covers a scale shift from TwiBot-20?
