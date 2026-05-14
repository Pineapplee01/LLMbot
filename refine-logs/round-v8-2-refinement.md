# Round v8-2 Refinement

**Round**: v8-2
**Review source**: `round-v8-1-review.md` (overall 6.675 RETHINK; drift = 3 parallel novelties)
**Headline revision**: pick ONE intellectual center (Stage-5 "LLM-consumable evidence graph construction"); demote Stages 1/3/4 to pre-committed plumbing; apply all 6 reviewer simplifications and 3 modernizations.

---

## Problem Anchor (verbatim from v8 Round 0; unchanged)

[user-locked 7-stage pipeline; LLM refiner in main path; constraints + success condition preserved]

## Anchor Check

- Original bottleneck still targeted: yes. User directive demanded LLM refiner in main path; v7 pre-rescope was zero-LLM which violated the user intent.
- All 7 stages preserved (user directive is hard-lock).
- Reviewer suggestions rejected as drift: none outright. Reviewer offered to "rewrite the contribution statement into single-dominant framing respecting the locked 7 stages" — that is exactly what Round 2 does.

## Simplicity Check

- **Intellectual center of v8**: Stage-5 "LLM-consumable evidence graph" construction. Everything upstream (Stages 1-4) is machinery that produces this artifact; Stage-6 LLM + refiner is how the artifact is consumed.
- Dominant contribution post-revision: **a principled protocol for turning hard-to-classify ego contexts into compact, trustworthy, LLM-consumable graph-evidence artifacts — and a load-bearing test (`Block-E`) that the LLM embedding genuinely exploits this artifact structure rather than its text content alone**.
- Demoted to pre-committed plumbing:
  - Stage-1 (estimator) — pinned to a **minimal 2-term composite**: `s_lbl + w_tg · JSD(p_LM || p_GNN) / log 2` (dropping `s_npi` per reviewer simplification 1). Published as fit hyperparameters, no method-level novelty.
  - Stage-3 (retriever) — pinned to **2-iteration max** (`T_ret = 2`, not 3; search collapses to binary). Core loop shape retained because the user directive requires "iterative". Hard gate replaced with **temperature-calibrated similarity** (reviewer modernization 1).
  - Stage-4 (rewriter) — pinned to **2 accept criteria** (error-proxy improvement + L∞ cap; drop set_size + margin per reviewer simplification 2).
- Components removed:
  - `s_npi` term from Stage-1.
  - Third rewriter accept criterion (set_size non-increase).
  - Fourth rewriter accept criterion (posterior margin guard, merged into error-proxy).
  - MLP refiner in Stage-6 replaced by **linear probe** default; MLP only if ablation proves need (reviewer simplification 3).
  - Partition into {supportive, suspicious, uncertain} becomes a **deterministic bucketization** of the (continuous) calibrated similarity, not a learned 3-way head.
- Why the remaining mechanism is the smallest adequate route: Stage-5 construction is where the actual novelty lives; the other stages are reduced to the minimum needed to produce a well-formed evidence artifact within the user's hard-locked pipeline.

---

## Changes Made

### 1. Intellectual center designation (addresses Drift + Contribution Quality)

- **Reviewer said**: "Three coupled novelties: retrieval loop, rewrite+rollback, LLM embedding+MLP. Demote two to plumbing; make only one the true intellectual center."
- **Action**: intellectual center = **Stage-5 evidence graph construction**. Reframed thesis:
  > "A conformal-triggered, calibrated text-confirmed construction procedure turns hard-to-classify ego contexts into compact, role-bucketed, LLM-consumable graph evidence such that an LLM embedding refiner strictly dominates (a) pure-GNN post-hoc correction, (b) LLM-direct prediction on raw ego text, (c) LLM embedding on the same evidence text using a frozen non-LLM encoder."
- **Reasoning**: the user's rescope identifies LLM refiner as mandatory-in-main; the natural dominant contribution is therefore not "LLM refiner itself" (that's just model selection) but the **protocol for what the LLM receives**. Stages 1-4 then serve a clear purpose: produce that artifact. The paper tells one story: "how to format a graph for an LLM to exploit".
- **Impact on core method**: claims reorder. Old C1 (full pipeline beats leave-one-out per stage) becomes an **ablation support claim**, not the headline. New headline claim: the **evidence artifact is the mechanism**.

### 2. Stage-1 minimization (reviewer simplification 1)

- **Reviewer**: "Drop either label-score term or one disagreement term; keep one + conformal machinery."
- **Action**: drop `s_npi`. Keep `s(v, y) = s_lbl(v, y) + w_tg · JSD(p_LM || p_GNN) / log 2`, `w_lbl ≡ 1`, 1 calibration-time scalar.
- **Reasoning**: JSD(p_LM || p_GNN) is the distinctive term — it directly measures text-graph disagreement at the node level. `s_npi` added neighborhood information but duplicates what Stage-3 retrieval already gathers; keeping both creates redundancy the reviewer flagged.
- **Impact**: Stage-1 becomes "1 scalar + 1 quantile + 1 lex threshold". Pinned pre-registered values derived from `valid_cal` statistics.

### 3. Stage-3 hard gate → calibrated similarity (modernization 1)

- **Reviewer**: "Replace cosine-threshold with calibrated semantic entailment / temperature-scaled similarity controls anisotropy."
- **Action**: replace `cos > τ_text AND |ΔJSD| ≤ τ_jsd` hard gate with a single **temperature-scaled RoBERTa similarity**:
  ```
  sim(v, u) = cos(X_R[v], X_R[u]) / T_valid_cal
  ```
  where `T_valid_cal` is set from valid-cal anisotropy: temperature matches the median pairwise similarity over `valid_cal` text pairs to 0.5. Gate = `sim(v, u) > 0` (i.e., above-median on calibration distribution).
- **Reasoning**: the cosine threshold was brittle and a tunable dataset artifact. Temperature calibration anchors the gate to valid-cal statistics, removing 1 hyperparameter and addressing reviewer's anisotropy concern without introducing a trained scorer.
- **Impact**: Stage-3 hyperparameters drop from `{T_ret, k_struct, k_text, τ_text, τ_jsd, ε}` to `{k_struct, ε}` (T_ret fixed at 2; τ_text replaced by valid-cal calibration; τ_jsd dropped; k_text absorbed into k_struct since gate is post-hoc).

### 4. Stage-3 iteration justified by budget-matched non-iterative baseline (modernization 2)

- **Reviewer**: "Budget-matched best non-iterative retriever (single-pass larger k / multi-seed PPR) to fairly isolate iteration value."
- **Action**: add to **Block-R** a variant `R_1-k*` = single-pass PPR with `k_struct · T_ret` candidates (equal compute to iterative R_2). Replace the parallel-scored baseline `R_parallel` with this budget-matched variant.
- **Reasoning**: isolates iteration's value from raw candidate count. If iteration beats equal-budget single-pass, it means "closed-loop frontier expansion" (not just compute) is necessary.

### 5. Stage-4 accept criteria reduced from 4 to 2 (reviewer simplification 2)

- **Reviewer**: "2 checks: externally calibrated error-proxy improvement + bounded L∞ change."
- **Action**: new Stage-4 accept rule:
  ```
  accept iff  error_proxy_post ≤ error_proxy_pre                      ← external calibrated error proxy
         AND  ||p_GNN_post(v) - p_GNN_pre(v)||_∞ ≤ ρ = 0.2              ← safety L∞ cap
  ```
  where `error_proxy(v)` = `1 − p_GNN(ŷ | v)` (frozen classifier's own error signal; no valid-cal labels at test time). Dropped: `q_post < q_pre` (self-confirming), `set_size_post ≤ set_size_pre` (correlated with error_proxy), and posterior margin (merged into error_proxy).
- **Reasoning**: error_proxy is the ideal external criterion — it uses the frozen backbone's own confidence change, orthogonal to q(v). L∞ cap remains the safety guard. Four overdetermined criteria was "tuned to never harm, becomes a no-op" per reviewer.
- **Impact**: rewriter becomes simpler and more analyzable; rollback semantics are cleaner.

### 6. Stage-6 linear probe default (reviewer simplification 3 + critique E)

- **Reviewer**: "Start Stage-6 as logistic regression / linear layer; upgrade only if ablation proves need."
- **Action**: Stage-6 default is a **linear probe**:
  ```
  h_out(v) = concat(h_GNN_post(v), X_R[v], z_LLM(v))
  p_final(v) = softmax(W_out · h_out(v) + b)
  ```
  where `W_out` is trained on `train_cal` labels (single linear layer, ~300 params vs previous ~20K MLP). Upgrade to 2-layer MLP only if **Block-L** (linear-vs-MLP) shows ≥ 1 seed-std improvement on both datasets.
- **Reasoning**: critique E said "only trainable is the MLP, so reviewers suspect gains come from a better head + more features". Linear probe minimizes this risk; if even the linear probe shows the pipeline wins, the argument is airtight. If MLP is needed, ablation justifies it.
- **Impact**: trainable inference parameter count drops ~70x. Stage-6 becomes "essentially a readout".

### 7. LLM load-bearing proof strengthened (critique C)

- **Reviewer**: "Same-serialization-different-encoder baseline — frozen RoBERTa with structured pooling over serialized evidence fields, not plain sentence embedding."
- **Action**: rewrite Block-E (was Block-LLM) with 3 variants sharing **identical evidence-graph serialization**:
  - **E-null**: no Stage-6; linear probe on `(h_GNN_post, X_R)` only.
  - **E-RoBERTa**: replace LLM with same-RoBERTa forward pass over serialized prompt text, structured mean-pool over evidence fields (target / supportive / suspicious / uncertain / quality_summary), then linear probe.
  - **E-LLM**: Stage-6 as specified with LLM embedding + linear probe.
- **Claim C (strengthened)**: `E-LLM > E-RoBERTa > E-null` on slice F1 by ≥ 1 seed-std each gap, on at least one dataset. If E-LLM ≈ E-RoBERTa, the LLM contributes nothing beyond serialization; paper contracts to "evidence graph construction matters; LLM is commodity readout".
- **Reasoning**: gives the LLM stage a falsifiable necessity condition. If the LLM loses to an RoBERTa-equivalent reading the same prompt, the paper is honest about it.

### 8. Budget-matched baseline preserved + seed-std pruning discipline inherited from v7

- All pruning thresholds use 1-seed-std (inherited from v7 Block-S2/S4 discipline).
- Block-P leave-one-stage-out still run, but reframed as **stage necessity audit** rather than dominant contribution. Non-load-bearing stages reported as "pre-committed plumbing necessary for pipeline completeness but not contributing measurable gain".

---

## Revised Proposal (v8 Round 2)

### Title
**Evidence-Graph Construction for LLM-Consumable Ego Context in Social Bot Detection**

### Method Thesis (one sentence)

> A conformal-triggered, temperature-calibrated text-confirmed procedure for turning a frozen backbone's uncertain ego contexts into compact, role-bucketed, LLM-consumable graph-evidence artifacts — such that an LLM embedding refiner strictly dominates both pure-GNN post-hoc correction and same-serialization frozen-encoder readout under matched budget on a frozen backbone.

### Contribution Focus

- **Dominant**: **Stage-5 evidence-graph construction protocol** — the procedural contribution is the full recipe: conformal trigger → PPR-structural → temperature-calibrated text-confirm → error-proxy-gated soft rewrite → role-bucketed evidence graph. Measured as a single artifact's quality by Block-E (LLM-vs-same-serialization-RoBERTa-vs-none) and Block-P (stage necessity).
- **Supporting (minimal plumbing, pre-committed)**: Stage-1 2-term composite; Stage-3 budget-matched iterative retrieval; Stage-4 2-criterion accept gate; Stage-6 linear probe.
- **Explicitly rejected**: LLM-as-classifier; binary edge-reliability headline; global structure learning; CP coverage theorem; 4-way role training head (bucketization is deterministic); learned retriever; learned gate; trained scorer.

### Complexity Budget (v8-2)

| Slot | Content |
|---|---|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN; existing ego adjacency; `p_LM`, `p_GNN`, `h_GNN`; external LLM (Qwen-2.5-7B or equivalent) for embedding only |
| Calibration-time (all on `train_cal` / `valid_cal`) | (i) `p_LM` fallback MLP if backbone lacks native LM head; (ii) 1-scalar search `{w_tg}` on valid_cal; (iii) `q_hat`, `τ*` on valid_cal; (iv) `T_valid_cal` temperature from median pairwise cosine; (v) retriever `{k_struct, ε}`; (vi) rewriter `{λ, B_max, κ}`; (vii) **linear probe `W_out, b`** on train_cal labels |
| New trainable inference params | Linear probe (~300 params, 2-class output over [h_GNN_post; X_R; z_LLM]) |
| Excluded | 2-layer refiner MLP (appendix only, gated on Block-L); learned retriever; learned gate; `s_npi`; `s_rec`; 3-criterion rollback; hard-gate thresholds |

### System

```
Stage-0 (frozen)  p_GNN, h_GNN = BotRGCN(X_R, A);  p_LM = LM_head(X_R)

Stage-1 (2-term composite, pre-committed plumbing)
  s(v, y)     = s_lbl(v, y) + w_tg · JSD(p_LM || p_GNN) / log 2    (w_lbl ≡ 1)
  q(v)         = min_y s(v, y)
  q_hat        = Quantile_{(1 − α)}(s on valid_cal)                (α = 0.15)
  prediction_set(v) = {y : s(v, y) ≤ q_hat};  set_size;  margin

Stage-2 (deterministic lex rule, pre-committed plumbing)
  g(v) = (set_size, -margin, -q(v))
  hard(v) = g(v) ≥_lex τ*                                          (τ* frozen from valid_cal 15% budget)

Stage-3 (budget-matched iterative retrieval, pre-committed plumbing, T_ret = 2)
  Init C_0(v) = existing ego propagation edges
  For t = 1, 2:
    (a) Structural proposal: top-k_struct u ∈ 2-hop(C_{t-1}) by PPR_from_v(u) · 𝟙[rel ∈ R_trusted]
    (b) Text confirmation (temperature-calibrated): keep u iff cos(X_R[v], X_R[u]) / T_valid_cal > 0
    (c) Conformal stop: stop if q(v | C_t) improves by ≤ ε or t = 2
  C_final(v): deterministic bucketization by `sim(v, u) / T_valid_cal`:
    supportive:  sim > +τ_bucket
    suspicious:  sim < -τ_bucket     (i.e. anti-aligned)
    uncertain:   |sim| ≤ τ_bucket
  τ_bucket pre-committed as Q_66 / Q_33 of calibrated sims on valid_cal (deterministic, not searched).

Stage-4 (rewriter, pre-committed plumbing, 2-criterion accept)
  role_weight(e) ∈ {+1, -1, 0}
  w_e' = w_e · clip(1 + λ · budget(v) · role_weight(e) · (1 - q(v)), 1-λ, 1+λ)
  max_iter = 2
  accept iff:
    (1 - p_GNN_post(ŷ|v)) ≤ (1 - p_GNN_pre(ŷ|v))        ← error proxy strictly non-worse
    AND  ||p_GNN_post(v) - p_GNN_pre(v)||_∞ ≤ ρ = 0.2     ← safety L∞ cap

Stage-5 (evidence graph construction — INTELLECTUAL CENTER)
  G_evidence(v) = {
    target           : X_R[v] text, p_GNN(v), p_LM(v), q(v),
    supportive_edges : top-k neighbors where sim > +τ_bucket, with text + weights + role,
    suspicious_edges : top-k neighbors where sim < -τ_bucket, with text + weights + role,
    uncertain_edges  : neighbors in neutral band, with text + weights,
    quality_summary  : q_pre, q_post, set_size_pre, set_size_post, rollback_flag,
    retrieval_trail  : T_ret_actual, |C_0|, |C_1|, |C_2|           ← enables retrieval auditability
  }
  Serialized to text with deterministic schema (versioned).

Stage-6 (LLM embedding + linear probe, pre-committed plumbing)
  Prompt(v) = serialize_to_text(G_evidence(v))
  z_LLM(v)  = LLM_embed(Prompt(v))                                  (1 LLM call per hard v)
  h_out(v)  = concat(h_GNN_post(v), X_R[v], z_LLM(v))
  p_final(v) = softmax(W_out · h_out(v) + b)
  (W_out, b) fit on train_cal labels (linear probe; MLP upgrade only if Block-L proves need)
  For non-hard nodes: p_final(v) = p_GNN(v).
```

### Core Mechanism (Stage-5)

The dominant claim is about the **evidence graph artifact** itself:

1. **Conformal trigger** ensures only ego contexts where the frozen backbone genuinely cannot decide receive evidence-graph construction — matches the user's "hard-node candidates" requirement.
2. **Iterative structural-first text-confirm retrieval** populates a candidate set that respects both topology and semantics. Iteration is budget-matched-ablation-defended.
3. **Error-proxy-gated soft rewrite** produces rewritten edge weights that never hurt the base classifier's confidence — delivering an artifact that is a strict improvement or null over the original ego.
4. **Deterministic role bucketization** on calibrated similarity labels every retained edge without introducing a trained classifier.
5. **Versioned evidence-graph schema** is the format published; its field ablation (appendix) answers "what does the LLM actually use?".

### Frontier Primitive Usage

- **LLM as enhancer, not predictor**: frozen Qwen-2.5-7B forward pass on the serialized evidence; one call per hard node; embedding fed into a linear probe. No LLM training, no LLM prompting tricks.
- **RoBERTa frozen semantic prior**: `X_R`, `p_LM`, text-confirm similarity.
- **Conformal calibration (heuristic)**: 1 scalar, no coverage claim.

### Validation (revised, 1 claim + 1 support + 3 necessity blocks)

**Claim C (dominant)** — The evidence graph is the mechanism, and the LLM genuinely reads it.

Block-E. Three variants sharing identical Stage-1..5:
- E-null: no Stage-6; linear probe on `(h_GNN_post, X_R)` only.
- E-RoBERTa: LLM replaced by same-RoBERTa forward pass + structured mean-pool over evidence fields; linear probe on `(h_GNN_post, X_R, z_RoBERTa)`.
- E-LLM: Stage-6 as specified.

Expected: `E-LLM > E-RoBERTa > E-null` on high-uncertainty slice F1; each gap ≥ 1 seed-std on at least one of TwiBot-20 / TwiBot-22.

**Falsification**: if `E-LLM ≈ E-RoBERTa`, the LLM is not exploiting semantic abstraction beyond RoBERTa — paper honestly contracts "evidence graph construction matters, LLM stage is commodity readout". If `E-RoBERTa ≈ E-null`, the evidence graph adds nothing — paper returns to v7 EQC framing honestly.

**Support S1 — Stage necessity audit (Block-P, demoted from headline)**

Leave-one-stage-out: `P-full` vs `P-noConformal` (Stage-1/2 → entropy trigger) vs `P-noRetrieve` (Stage-3 → identity) vs `P-noRewrite` (Stage-4 → identity) vs `P-noLLM` (Stage-6 → E-null equivalent).

Any stage whose leave-one-out falls within 1 seed-std is reported as "pre-committed plumbing, not measurably load-bearing — retained for pipeline completeness per anchor".

**Support S2 — Iteration budget-matched (Block-R)**

- R_1-k*: single-pass PPR, `k_struct × 2` candidates (budget-matched to T_ret = 2).
- R_2: iterative with text-confirm loop.

Expected: `R_2 > R_1-k*` by ≥ 1 seed-std on at least one dataset, OR iteration is dropped from plumbing claim and T_ret fixed to 1.

**Support S3 — Linear probe sufficient (Block-L)**

- L-linear: linear probe (main).
- L-MLP: 2-layer MLP.

Expected: `L-linear ≈ L-MLP` within 1 seed-std; if so, paper uses linear probe. If MLP > linear by ≥ 1 seed-std on both datasets, paper escalates to MLP and reports honestly.

**Support S4 — Cross-dataset (MGTAB)**

Full `P-full + E-LLM + L-linear` on MGTAB, 3 seeds, matched budget from TwiBot-20.

### Appendix (conditional / diagnostic)

- A-Prompt: evidence-graph field drop-one (conditional on E-LLM > E-RoBERTa).
- A-Bucket: deterministic bucketization vs continuous role weights.
- A-Stop: conformal stop rule (ε) effect on T_ret_actual distribution.
- A-PilotC / mechanism isolation (inherited from v7 Block-M, now under support S1).
- A-BR-public / BR-soft (inherited from v7 Block-BR-sensitivity).
- A-Nested split robustness (5-fold train_cal / valid_cal).

### Failure modes (updated)

| Mode | Detection | Fallback |
|---|---|---|
| `w_tg` → 0 on valid_cal | 1-scalar search output | Stage-1 collapses to pure label non-conformity; honestly report "JSD term unnecessary" |
| T_valid_cal makes gate too permissive / restrictive | % of candidates surviving gate on valid_cal | τ_bucket derived from valid_cal (Q_66 / Q_33) is self-calibrating; mitigation pre-registered |
| Rewriter accepts nothing | rollback rate > 90% | ρ too tight; increase from 0.2 to 0.3 (pre-registered grid) |
| `E-LLM ≈ E-RoBERTa` | Block-E | Paper narrative contracts; LLM stage becomes "interchangeable encoder" not "key primitive" |
| Stage-3 iteration not load-bearing | Block-R | T_ret = 1; paper omits iteration claim |
| Linear probe insufficient | Block-L | Upgrade to 2-layer MLP; paper reports |

### Novelty and Elegance

Closest prior:
- GraphRAG 2024 — retrieval-augmented LLM with graph-structured evidence. We differ by: (i) post-hoc on frozen GNN not generative, (ii) conformal-gated trigger, (iii) calibrated text-confirm gate, (iv) soft-rewrite feedback.
- LOGIN / GLANCE — selective LLM consultation. We differ by: constructing a **structured evidence graph artifact** before LLM consumption, not raw ego prompts.
- CF-GNN / DAPS/NAPS — conformal for GNNs. We use calibration, not coverage.
- BotBR / BECE — binary edge reliability. We reject binary reliability as headline; role bucketization is deterministic, not learned.
- v7 EQC — our own immediate predecessor. v8 restores LLM to main path per user directive and redesignates the intellectual center.

**Exact novelty**: first procedural protocol for constructing LLM-consumable graph-evidence artifacts from frozen-backbone ego contexts, with empirical falsification tests (E-null / E-RoBERTa / E-LLM) isolating which step adds signal. All other stages are bounded, calibrated, pre-registered plumbing.

### Compute

~ 150 GPU-hr + LLM embedding inference (Qwen-2.5-7B local or ~$30 API cost equivalent). 5-week timeline unchanged.

---

## Remaining Risks

- If Block-E shows `E-LLM ≤ E-RoBERTa`, dominant contribution collapses to "evidence graph construction" without LLM-specific novelty — still publishable but at a tier below the current framing.
- If Block-P shows all leave-one-out variants within 1 seed-std of `P-full`, the pipeline is not producing measurable gain — paper becomes a **method audit / honest null result** on ego-context refinement for bot detection.
- These are legitimate scientific risks; both are caught by pre-committed falsification and result in honest contracted claims rather than method abandonment.
