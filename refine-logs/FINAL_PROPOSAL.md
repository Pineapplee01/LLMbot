# Final Proposal (v8): Evidence-Graph Construction for LLM-Consumable Ego Context in Social Bot Detection

**Version**: v8 (MAX_ROUNDS=5 reached at 8.63/10 REVISE; NONE drift; structurally complete; remaining gap is empirical not structural per Round-5 reviewer verdict)
**Date**: 2026-05-09
**Status**: **REVISE (structurally READY-adjacent)** — every structural blocker resolved; READY requires empirical confirmation of Block-E + Block-CT acceptance criteria.
**Anchor source**: `research.md` (2026-05-09) + user 7-stage-pipeline directive (2026-05-09)
**Supersedes**: v7 EQC cycle (Rounds 1-8 READY at nightmare 9.2/10 in a different scope; user rescoped LLM refiner back to main pipeline stage, necessitating v8 cycle). Prior v7 artifacts remain in git history.

---

## Problem Anchor (user-locked 7-stage pipeline, verbatim across all v8 rounds)

```
Stage-0 LM + GNN outputs
  ↓
Stage-1 conformal ego-quality estimator
  ↓
Stage-2 hard-node candidates
  ↓
Stage-3 Local Ego Retrieval (structural-first, text-confirm, iterative)
  ↓
Stage-4 text-confirmed local graph rewriter
  ↓
Stage-5 refined ego-evidence graph
  ↓
Stage-6 LLM embedding / refiner
```

- **Bottom-line problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage — not uniformly. A pure GNN post-hoc rewriter (v7 EQC) improves the ego-graph but does not exploit semantic evidence available to a modern LM. User directive requires the rewrite to be text-confirmed and the refined ego-evidence graph to be consumed by an LLM-embedding refiner.
- **Must-solve bottleneck**: structural retrieval alone is noisy; text-only loses topology; both must be coupled iteratively. Rewriter output must yield a refined ego-evidence graph that is structured enough to serialize into an LLM prompt. LLM embedding must measurably improve over GNN-only refinement.
- **Non-goals**: bypassing any of the 7 stages; global graph structure learning; LLM-as-classifier; binary edge-reliability headline; full causal graph repair.
- **Constraints**: frozen Stage-0 (RoBERTa + BotRGCN); soft-reweight existing ego edges only; `T_ret ≤ 3` (pinned to 2 in v8); LLM ≤ 1 call per hard node; max_iter on rewriter = 2 with rollback.
- **Success condition**: Block-E acceptance criteria (direction consistent on both datasets + ≥ 1 seed-std magnitude on ≥ 1 dataset + scramble-control degradations + 2-LLM robustness) pass on TwiBot-20 + TwiBot-22.

---

## Method Thesis (v8 final)

> We propose a procedure for converting frozen-backbone uncertain-ego contexts into **LLM-consumable graph-evidence artifacts**. The contribution is a 4-tuple protocol: **(i) a fixed 5-section schema, (ii) deterministic Q_66 / Q_33 bucketization on calibrated similarity, (iii) canonical intra-bucket rank order by `(PPR_rank, −sim)`, (iv) pre-registered 512-token budget with deterministic truncation**. Stage-5 is a protocol whose key property is **predictable, testable sensitivity**: its value is established by the pattern of degradations under scramble controls (role / order / collapse), not by any best-prompt search. We make no claims about prompt-optimality and run no prompt-optimization procedure. Upstream trigger / retrieval / rewrite stages are minimal instantiations required by the locked 7-stage pipeline; we make no novelty claims on them.

### Minimal Formalism

Stage-5 is a mapping `f : (ego-subgraph + per-node signals) → token sequence under a budget B`. Its properties are:

- **Role-sensitivity**: `f` degrades measurably when role labels are shuffled (Block-E E-LLM-shuffle-role).
- **Order-sensitivity**: `f` degrades when canonical ranks are permuted (Block-E E-LLM-shuffle-order).
- **Partition-sensitivity**: `f` degrades when buckets are collapsed (Block-E E-LLM-collapse).
- **Encoder-transferability**: `f`'s advantage persists across LLM families (Block-E E-LLM-Qwen ≈ E-LLM-Mistral).
- **Dataset-transferability**: `f`'s calibration transfers across datasets (Block-CT CT-xfer).

The protocol's scientific content is these 5 testable invariances.

## Contribution Focus (final)

- **Dominant**: **Stage-5 4-tuple protocol** for LLM-consumable graph-evidence artifacts. Defined abstractly; isolated empirically by 4 scramble controls + 2 encoder controls + 2 LLM families + 1 cross-dataset calibration transfer.
- **Supporting (explicitly non-novel plumbing)**: Stage-1 2-term composite `s = s_lbl + w_tg · JSD(p_LM ‖ p_GNN) / log 2`; Stage-3 2-iteration retrieval with top-50% quantile gate; Stage-4 ê_φ-gated 2-criterion accept (5-fold cross-fit on valid_cal); Stage-6 9218-param linear probe.
- **Rejected**: LLM-as-classifier; binary edge-reliability headline; global structure learning; CP coverage theorem; causal identification; `s_reliability` binary head; `p_role` 4-way training head; prompt-optimality claims; ordering-as-bonus claims; argmax-confidence accept proxy (replaced by calibrated expected-error ê_φ).

---

## Proposed Method

### Complexity Budget (v8 final)

| Slot | Value |
|---|---|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN; existing ego adjacency; `p_LM`, `p_GNN`, `h_GNN`; external LLM (Qwen-2.5-7B + Mistral-7B-Instruct for 2-LLM robustness) |
| Calibration-time params | ~ 9234 total (linear probe dominates); 16 non-probe scalars |
| New trainable inference params | **9218** (linear probe only: 4608 × 2 + 2) |
| Excluded | 2-layer MLP (A-L appendix only); learned retriever/gate/scorer; `s_npi`, `s_rec`; 3/4-criterion rollback; argmax-confidence proxy; hard-gate thresholds; `retrieval_trail` (demoted to appendix) |

**No base-backbone retraining**. All learning is calibration-time or linear-probe-on-calibration.

### System (v8 final)

```
Stage-0 (frozen)            p_GNN, h_GNN = BotRGCN(X_R, A);  p_LM = LM_head(X_R)

Stage-1 (plumbing, 2-term)  s(v,y) = s_lbl(v,y) + w_tg · JSD(p_LM || p_GNN) / log 2   (w_lbl ≡ 1)
                            q(v) = min_y s(v,y);   q_hat = Quantile_{1-α}(s on valid_cal)
                            prediction_set, set_size, margin = q_hat - q(v)

Stage-2 (plumbing, lex)     g(v) = (set_size, -margin, -q(v));   hard(v) = g ≥_lex τ*

Stage-3 (plumbing, T_ret=2) Init C_0(v) = ego propagation edges.
                            For t = 1, 2:
                              (a) top-k_struct u ∈ 2-hop(C_{t-1}) by PPR_from_v(u) · 𝟙[rel ∈ R_trusted]
                              (b) top-50% quantile gate: keep u iff cos(X_R[v], X_R[u]) / T_valid_cal > 0
                              (c) conformal stop: stop if q(v | C_t) improves ≤ ε or t = 2
                            C_final partitioned by calibrated sim:
                              supportive sim > Q_66;  suspicious sim < Q_33;  uncertain else
                            Canonical intra-bucket order: (PPR_rank(u), -calibrated_sim(u)) lex

Stage-4 (plumbing, ê_φ acc.) role_weight ∈ {+1, -1, 0}
                            w_e' = w_e · clip(1 + λ · budget(v) · role_weight(e) · (1 - q(v)), 1-λ, 1+λ)
                            ê_φ : (confidence, entropy, q, set_size) → [0,1]  (5-fold cross-fit on valid_cal)
                            accept iff  ê_φ(features_post) ≤ ê_φ(features_pre)   ← EXTERNAL
                                   AND  ||p_GNN_post - p_GNN_pre||_∞ ≤ ρ = 0.2

Stage-5 (INTELLECTUAL CENTER — 4-tuple protocol):
  Schema v1.0 (5 sections):
    target           : X_R[v] text + p_GNN(v) + p_LM(v) + q(v)
    supportive_edges : top-k_supp neighbors, canonical order, text + weight + role
    suspicious_edges : top-k_susp neighbors, canonical order, text + weight + role
    uncertain_edges  : top-k_unc neighbors, canonical order, text + weight
    quality_summary  : q_pre, q_post, set_size_pre/post, ê_φ_pre/post, rollback_flag
  Bucketization    : deterministic Q_66 / Q_33 on calibrated similarity (no trained head)
  Canonical order  : (PPR_rank, -sim) lex, frozen as protocol element
  Serialization    : versioned schema v1.0, fixed field ordering
  Token budget     : 512 total;  80 / 200 / 128 / 64 / 40 per section
  Truncation       : drop tail by rank if section exceeds budget; no padding

Stage-6 (plumbing, linear probe):
  Prompt(v) = serialize_v1.0(G_evidence(v))                   (deterministic, no sampling)
  z_LLM(v)  = LLM_embed(Prompt(v))                            (1 call per hard v, mean-pooled)
  h_out(v)  = concat(h_GNN_post(v), X_R[v], z_LLM(v))         [256 + 768 + 3584 = 4608]
  p_final(v) = softmax(W_out · h_out(v) + b)                  [9218 trainable params]
  W_out, b fit on train_cal labels (linear probe)
  Non-hard nodes:  p_final(v) = p_GNN(v)
```

### Reproducibility Constraints (per reviewer Modernization 3)

- LLM temperature = 0; deterministic decoding; no sampling.
- Prompt serialization is strictly deterministic given `G_evidence(v)` (no randomness in format).
- `ê_φ` 5-fold cross-fit: test-time value is average of 5 out-of-fold predictors.
- All random seeds fixed; PPR iteration counts fixed; no stochastic components except seed-over-seed replication for reporting.

### Calibration Fit Order (pre-registered)

1. Fit `p_LM` fallback MLP on `train_cal` (skipped if native head).
2. Fit `T_valid_cal` from median pairwise cosine on `valid_cal`.
3. Fit `ê_φ` 5-fold cross-fit on `valid_cal`.
4. 1-scalar grid search `w_tg ∈ {0, 0.25, 0.5, 0.75, 1.0}` on `valid_cal`.
5. Compute `q_hat`, `τ*` on `valid_cal` (α = 0.15, 15% hard-node budget).
6. Compute `τ_bucket = (Q_66, Q_33)` of calibrated `sim / T_valid_cal` on `valid_cal`.
7. Fit rewriter `{λ, B_max, κ}` on `valid_cal` small grid.
8. Freeze retriever `{k_struct, ε}` from valid_cal.
9. Precompute LLM embeddings on `train_cal ∪ valid_cal ∪ test` hard nodes (cached).
10. Fit linear probe `(W_out, b)` on `train_cal` labels.

---

## Validation (v8 final)

### CLAIM C (dominant) — Stage-5 Protocol Is the Mechanism

Block-E — **9 variants**, identical Stage-1..5, identical 512-token prompt, differ only in Stage-6 encoder / prompt-shuffle:

| Variant | Encoder | Prompt |
|---|---|---|
| E-null | none (linear probe on `h_GNN_post, X_R` only) | — |
| E-RoBERTa | frozen RoBERTa + structured field-aware mean-pool | v1.0 serialization |
| E-LLM-collapse | Qwen-2.5-7B | flat neighbor list, no role labels (minimal-artifact baseline) |
| E-LLM-Qwen | Qwen-2.5-7B | v1.0 serialization (canonical) |
| E-LLM-Mistral | Mistral-7B-Instruct-v0.3 | v1.0 serialization |
| E-LLM-shuffle-role | Qwen-2.5-7B | v1.0 tokens, role labels shuffled |
| E-LLM-shuffle-order | Qwen-2.5-7B | v1.0 tokens, intra-bucket order permuted |

**Appendix (budget control)**: E-LLM-Budget-256 (Qwen with halved budget).

**Claim C acceptance criterion** (all four must hold):

```
Over 3 seeds on BOTH TwiBot-20 AND TwiBot-22:

(i)  E-LLM-Qwen > E-RoBERTa > E-null
     with consistent sign(mean_diff) across BOTH datasets;

(ii) At least ONE of the two gaps exceeds 1 seed-std on at least ONE dataset;

(iii) E-LLM-Qwen > {E-LLM-shuffle-role, E-LLM-shuffle-order, E-LLM-collapse}
      with consistent direction across BOTH datasets;

(iv) |slice_F1(E-LLM-Qwen) - slice_F1(E-LLM-Mistral)| ≤ 1 seed-std
     on EACH dataset (LLM-family robustness).
```

### Pre-committed Block-E Failure Handling

| Which condition fails | Fallback narrative |
|---|---|
| (i) E-null ≥ E-RoBERTa | evidence-graph construction adds no signal; return to v7 EQC framing honestly. |
| (i) E-RoBERTa ≥ E-LLM-Qwen | LLM stage commodity; paper contracts to "evidence graph matters, LLM is commodity readout". |
| (ii) all gaps < 1 seed-std | effect too small for centerpiece; treat as marginal additional result. |
| (iii) E-LLM ≈ shuffle-role | role bucketization non-load-bearing; **drop role labels from Stage-5 schema**; protocol shrinks to 3-tuple. |
| (iii) E-LLM ≈ shuffle-order | canonical ordering non-load-bearing; **drop ordering from Stage-5 schema**. |
| (iii) E-LLM ≈ collapse | partitioning non-load-bearing entirely; dominant contribution collapses to "schema + budget only". |
| (iv) Qwen ≫ Mistral | protocol is model-specific; report as "works for Qwen-class, generalization open question". |

### SUPPORT S1 — Stage Necessity Audit (Block-P, transparency only)

`P-full` vs `{P-noConformal, P-noRetrieve, P-noRewrite, P-noLLM}`. Leave-one-stage-out. Stages within 1 seed-std of `P-full` are labeled "pre-committed plumbing, not measurably load-bearing".

### SUPPORT S2 — Retrieval Iteration Budget-Matched (Block-R)

`R_2` (v8 canonical) vs `R_1-k*` (single-pass, k_struct × 2 candidates, budget-matched) vs `R_2-seeds` (single-pass with 2 PPR seeds).

Expected: `R_2 > {R_1-k*, R_2-seeds}` ≥ 1 seed-std on at least one dataset; else T_ret dropped to 1 in final method.

### SUPPORT S3 — Linear Probe Sufficient (Block-L)

`L-linear` (v8 canonical) vs `L-MLP` (2-layer MLP upgrade). Expected ≈. If `L-MLP > L-linear` ≥ 1 seed-std on BOTH datasets, escalate to MLP and report.

### SUPPORT S4 — Cross-Dataset on MGTAB

`P-full + E-LLM-Qwen + L-linear` on MGTAB (7 relations), 3 seeds.

### SUPPORT S5 — Cross-Dataset Calibration Transfer (Block-CT, promoted to main secondary pillar)

```
CT-same: fit all calibration artifacts on TwiBot-22 valid_cal; evaluate on TwiBot-22 test.
CT-xfer: fit all calibration artifacts on TwiBot-20 valid_cal; evaluate on TwiBot-22 test WITHOUT REFIT.
```

Expected: `CT-xfer` F1 drops ≤ 2 seed-std from `CT-same`. Demonstrates protocol is dataset-agnostic, not prompt-engineering.

### Appendix

- A-Prompt: re-introduce `retrieval_trail` field; drop-one over all schema fields (conditional on Claim C main passing).
- A-Bucket: deterministic vs continuous role weights.
- A-Stop: ε sensitivity (T_ret_actual distribution vs ε).
- A-Budget: 256 vs 512 token comparison.
- A-BR-public / BR-soft: inherited from v7 Block-BR-sensitivity.
- A-Nested: 5-fold `train_cal / valid_cal` rotation with `ê_φ` cross-fit diagnostics.

---

## Failure Modes (v8 final)

| Mode | Detection | Fallback |
|---|---|---|
| `w_tg → 0` on valid_cal | 1-scalar search output | Stage-1 collapses to label non-conformity; honestly report JSD term unnecessary |
| `ê_φ` cross-fit fold disagreement | 5-fold Brier variance | Fall back to simpler entropy criterion |
| Rewriter rollback rate > 90% | Block-P logs | ρ grid extension pre-registered |
| Block-E condition (i) fails | Block-E main | Return to v7 EQC framing |
| Block-E condition (ii) fails | Gap magnitude | Treat protocol as marginal, not centerpiece |
| Block-E condition (iii) fails specific | Scramble delta | Drop that element from schema; protocol shrinks |
| Block-E condition (iv) fails | Qwen vs Mistral delta | Report as Qwen-specific; open future work |
| Block-CT fails | CT-xfer delta | Report calibration partial-dataset-specificity |
| Linear probe insufficient | Block-L | Upgrade to 2-layer MLP; report |
| Prompt overflow | Token count logs | Pre-registered truncation policy handles deterministically |

## Compute & Timeline

| Bucket | GPU-hr |
|---|---|
| Stage-A calibration | 4 |
| Stage-B probe training | 2 |
| LLM embeddings (Qwen, all hard nodes) | 10 |
| LLM embeddings (Mistral, all hard nodes) | 8 |
| Block-E (7 main variants × 3 seeds × 2 datasets; probes only, embeddings cached) | 36 |
| Block-P (5 variants × 3 × 2) | 20 |
| Block-R (3 variants × 3 × 2) | 18 |
| Block-L (2 variants × 3 × 2) | 6 |
| MGTAB (S4) | 10 |
| Block-CT (S5) | 4 |
| Appendix | 25 |
| **Total** | **~ 143 GPU-hr + 18 GPU-hr LLM = ~ 161 GPU-hr** |

Timeline (5 weeks): W1 Stage-1/2/3 + calibration; W2 Stage-4 (ê_φ cross-fit) + Stage-5 serialization v1.0; W3 Stage-6 embeddings (both LLMs) + probe fit; W4 Block-E main + Block-P + Block-R + Block-L; W5 MGTAB + Block-CT + appendix + paper draft.

---

## Round-5 Reviewer's Final Read

> "Not READY. Remaining gap is primarily empirical, not structural. Structurally, you now meet the 'one dominant contribution' and 'no obvious complexity bloat' bars, and drift is essentially zero. **What blocks READY is that CQ/MS at ≥ 9 depends on results**: the strengthened Block-E/CT suite must actually come out positive (and not fragile). If the experiments confirm your acceptance criteria cleanly on both datasets and both LLMs, the work could plausibly cross into READY territory; if they don't, no further proposal polishing will fix it."

**Interpretation**: the proposal is as tight as `/research-refine` can make it. Next step is `/experiment-plan` + execution; Round-5 reviewer offered to generate "a one-page Block-E + CT result interpretation rubric" for the paper-writing phase.
