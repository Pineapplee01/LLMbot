# Round v8-4 Refinement

**Round**: v8-4
**Review source**: `round-v8-3-review.md` (overall 7.585 REVISE; 2 CRITICAL + 2 IMPORTANT + 1 MINOR)
**Headline revision**: accept all 5 items — scrambled-structure control in Block-E; **calibrated expected-error proxy** replaces argmax-confidence in Stage-4 (the key CRITICAL externalization fix); tighten dominant-contribution sentence to isolate Stage-5 as {schema + bucketization + serialization + token budget}; pre-register token budget; clarify param counts.

---

## Problem Anchor

[verbatim from Round 0; user-locked 7-stage pipeline; LLM refiner in main path]

## Anchor Check

- Original bottleneck still targeted: yes. Round-3 reviewer confirmed PASS.
- All 5 action items are pure sharpening or inside-stage modernization; none touch anchor.

## Simplicity Check

- Intellectual center unchanged: **Stage-5 evidence-graph artifact**.
- Stage-5 now explicitly defined as a 4-tuple, independent of Stages 3/4: **(schema, deterministic bucketization, versioned serialization, pre-registered token budget)**. Stages 3/4 outputs are inputs, not part of the contribution.
- Components added (all minimal; not new mechanisms):
  - Calibrated expected-error proxy `ê(confidence, entropy, q, set_size) → empirical error` — fit on valid_cal only; replaces argmax-confidence in Stage-4 accept. Mechanism: small logistic regression on 4 features → probability-of-error. ~10 params.
  - Scrambled-structure control variants in Block-E (shuffle role buckets / retrieval_trail / collapse partitions).
  - Cross-dataset calibration carryover (no retune) as an additional modernization claim.
- Components removed:
  - `(1 − p_GNN_post(ŷ|v))` argmax self-referential criterion.
  - Informal "sim > 0" gate language → explicit "top-50% quantile on calibrated cos" (no change in behavior, only clearer presentation).

---

## Changes Made

### 1. Calibrated Expected-Error Proxy (CRITICAL 2)

**Reviewer**: "`(1 − p_GNN(ŷ|v))` is the model's own confidence in its argmax. Replace with calibrated expected-error."

**Action**: define a 4-feature logistic calibrator fit on `valid_cal`:

```
Features per node v: [confidence(v), entropy(v), q(v), set_size(v)]
  confidence(v) = max_y p_GNN(y|v)
  entropy(v)    = -Σ_y p_GNN(y|v) log p_GNN(y|v)
  q(v)          = s_composite(v)        (from Stage-1)
  set_size(v)   = |prediction_set(v)|   (from Stage-1)

Calibrator ê_φ : ℝ⁴ → [0, 1]
  ê_φ(features) = σ(φ₀ + φ₁·conf + φ₂·entropy + φ₃·q + φ₄·set_size)
  Fit on valid_cal:  min Σ_{v∈valid_cal} BCE(ê_φ(features(v)), 𝟙[ŷ(v) ≠ y(v)])
```

Accept rule becomes:

```
accept iff  ê_φ(features_post(v))  ≤  ê_φ(features_pre(v))           ← EXTERNAL via valid_cal labels
       AND  ||p_GNN_post(v) - p_GNN_pre(v)||_∞ ≤ ρ = 0.2              ← safety L∞ cap
```

**Why this is actually external**: `ê_φ` is a calibration-set-supervised map. An increase in `ê_φ` is a predicted increase in the rewritten ego's empirical error on the calibration distribution, independent of the model's own argmax confidence. The rewrite can no longer be accepted just because the model's own top-1 confidence went up.

**Complexity**: 5 parameters fit on `valid_cal` once. Frozen at test time.

### 2. Block-E Scrambled-Structure Controls (CRITICAL 1)

**Reviewer**: "Same-LLM scrambled-structure control — shuffled role buckets, shuffled retrieval_trail, collapsed partitions. Test whether Stage-5 protocol matters beyond text volume."

**Action**: add 3 new Block-E variants (same LLM encoder, same token budget, identical underlying ego; differ only in how the evidence is serialized):

```
E-LLM              : v8-2 canonical serialization (role-bucketed, trail-included, quality_summary).
E-LLM-shuffle-role : same tokens as E-LLM, but role labels shuffled across supportive/suspicious/uncertain.
E-LLM-shuffle-trail: same tokens as E-LLM, but retrieval_trail order randomized.
E-LLM-collapse     : supportive + suspicious + uncertain collapsed into one flat neighbor list.
```

**Claim**: `E-LLM > {E-LLM-shuffle-role, E-LLM-shuffle-trail, E-LLM-collapse}` by ≥ 1 seed-std on at least one dataset. If any scrambled variant ties E-LLM, the corresponding protocol element is declared **non-load-bearing** and removed from Stage-5 (the schema shrinks).

**Why this test is right**: the reviewer's concern was "LLM just likes long text". If role labels don't matter, the role bucketization is prompt engineering not protocol. The scramble controls directly falsify that. If *all three* scrambles tie E-LLM, Stage-5 collapses to "any serialization of the rewritten ego"; the dominant contribution contracts to Block-P style stage necessity.

### 3. Cross-Dataset Calibration Carryover (Modernization 2)

**Reviewer**: "Calibrate T_valid_cal / τ_bucket on TwiBot-20 calibration, apply to TwiBot-22 without re-tuning — demonstrates the protocol is not dataset-tuned prompt engineering."

**Action**: add **Support S5 (Cross-Dataset Calibration Transfer)** — a two-cell experiment:

```
CT-same  : fit all calibration artifacts (T_valid_cal, τ_bucket, w_tg, q_hat, τ*, λ, B_max, κ, ρ, ê_φ, W_out)
           on TwiBot-22 valid_cal; evaluate on TwiBot-22 test.
CT-xfer  : fit all calibration artifacts on TwiBot-20 valid_cal; evaluate on TwiBot-22 test WITHOUT REFIT.
```

Claim: `CT-xfer` F1 drops by ≤ 2 seed-std from `CT-same` F1. A successful carryover is a strong novelty statement: the protocol is **dataset-agnostic**, not prompt-engineering.

**Where this lives**: Support S5 appendix (new). Runs once per direction; cheap.

### 4. Tightened Dominant Contribution Sentence (IMPORTANT 3)

**Reviewer**: "Stage-5 must be definable as a protocol that could be fed by alternative plumbing."

**Action**: the new one-line definition of Stage-5:

> **Stage-5 is a 4-tuple protocol: (i) a fixed-schema evidence graph (target node, three role buckets of ego neighbors with per-edge weight + role, per-node quality summary, per-node retrieval trail); (ii) deterministic bucketization by calibrated similarity quantiles Q₆₆ / Q₃₃ fit once on any valid_cal; (iii) a versioned deterministic serialization; (iv) a pre-registered token budget with a deterministic truncation policy. The protocol is defined independently of which trigger, which retriever, or which rewriter produces its inputs.**

Paper abstract-level claim becomes:

> "We propose a protocol for converting frozen-backbone uncertain-ego contexts into LLM-consumable graph-evidence artifacts. The protocol is defined as a 4-tuple; its value is isolated empirically by same-LLM scrambled-structure controls, same-serialization frozen-encoder controls, and cross-dataset calibration transfer. Upstream trigger / retrieval / rewrite stages are the minimal anchored instantiations required for the locked pipeline; we make no novelty claims on them."

Stages 3/4 are now truly **plumbing in the paper's narrative**. Block-P stage-necessity audit becomes transparency about which plumbing actually helps — not the dominant claim.

### 5. Pre-Registered Token Budget and Truncation Policy (IMPORTANT 4)

**Reviewer**: "Pre-register Stage-5 token budget + truncation policy, otherwise 'more tokens → better' is the trivial explanation."

**Action**: pin the following into Stage-5's definition:

```
Stage-5 token budget per hard node (pre-registered for all experiments):
  TOKEN_BUDGET = 512 tokens total

Per-section allocation (fixed):
  target                :  64 tokens (truncate X_R[v] text from the start; keep last 64)
  supportive_edges      : 192 tokens (top-k_supp neighbors by calibrated sim, texts truncated to fit)
  suspicious_edges      : 128 tokens (top-k_susp neighbors)
  uncertain_edges       :  64 tokens (top-k_unc neighbors)
  quality_summary       :  32 tokens (compact numeric summary)
  retrieval_trail       :  32 tokens (compact numeric summary)

Truncation policy:
  If any section exceeds its budget, truncate neighbor list by rank (drop tail).
  If below budget, do not pad; allow prompt to be shorter.
```

**Budget-matched controls**: E-RoBERTa and all E-LLM-shuffle-* variants use the **identical 512-token prompt**, so the reviewer's "long text" explanation is structurally disallowed.

### 6. Param-Count Clarification (MINOR 5)

**Reviewer**: "~300 params depends on z_LLM dim."

**Action**: state exact dimensionalities:

```
Input dimensions:
  h_GNN_post(v)       : 256  (BotRGCN hidden dim, pre-frozen)
  X_R[v]              : 768  (RoBERTa pooled output)
  z_LLM(v)            : 3584 (Qwen-2.5-7B last hidden layer, mean-pooled)

Concatenation:
  h_out(v) ∈ ℝ^4608

Linear probe:
  W_out : 4608 × 2
  b     : 2
  Total trainable inference params: 9218
```

**Main-paper phrasing**: "single linear layer (9218 params) trained on `train_cal` labels; upgrade to a 2-layer MLP (~600K params) only if Block-L demonstrates ≥ 1 seed-std improvement on both datasets."

### 7. Explicit Quantile-Gate Language in Stage-3

**Reviewer simplification**: "Rename `sim > 0` to explicit quantile gate."

**Action**: replace Stage-3 text confirmation wording:

```
Text confirm (top-50% quantile gate):
  Accept candidate u iff  cos(X_R[v], X_R[u]) / T_valid_cal  > 0
  Equivalently: u survives iff its raw cosine exceeds Q₅₀ of valid_cal pairwise cosines
               (i.e., above-median on the calibration distribution).
```

No behavior change; reviewer-friendly phrasing.

### 8. Param-Count Note for Calibrated Error Proxy and Bucketizer

```
Calibration-time parameters (fit on valid_cal):
  w_tg                  : 1
  T_valid_cal           : 1
  τ_bucket (Q_66 / Q_33): 2  (2-d quantile cuts)
  q_hat                 : 1
  τ* (lex threshold)    : 1
  k_struct              : 1
  ε (stop tolerance)    : 1
  λ, B_max, κ, ρ        : 4
  ê_φ logistic          : 5  (4 feature weights + bias)
  Linear probe W_out, b : 9218
  -----
  Total calibration-time fit: 9234 params, ~9.2K, all fit on calibration folds (not test).
```

## Revised Proposal (v8-4)

### Title

**Evidence-Graph Construction for LLM-Consumable Ego Context in Social Bot Detection**

### Method Thesis (final, v8-4)

> We propose a procedure for converting frozen-backbone uncertain-ego contexts into **LLM-consumable graph-evidence artifacts**. The contribution is a 4-tuple protocol (**fixed schema + deterministic calibrated-quantile bucketization + versioned serialization + pre-registered token budget**), *isolated empirically* by same-LLM scrambled-structure controls, same-serialization frozen-encoder controls (E-null / E-RoBERTa / E-LLM), and cross-dataset calibration transfer. Upstream trigger / retrieval / rewrite stages are **minimal instantiations required by the locked pipeline**; we make no novelty claims on them.

### Contribution Focus (v8-4)

- **Dominant**: **Stage-5 4-tuple protocol** for LLM-consumable graph-evidence artifacts. Defined independently of plumbing. Isolated empirically by Block-E (plus 3 scramble controls) + Block-CT (cross-dataset carryover).
- **Supporting plumbing (explicitly non-novel)**: Stage-1 2-term composite; Stage-3 2-iteration retrieval with top-50% quantile gate; Stage-4 calibrated-error-proxy 2-criterion accept; Stage-6 9218-param linear probe.
- **Rejected**: LLM-as-classifier; binary edge-reliability headline; global structure learning; CP coverage theorem; 4-way role training head; learned retriever/gate/scorer; argmax-confidence accept proxy (replaced by calibrated expected-error ê_φ).

### Complexity Budget (v8-4)

| Slot | Value |
|---|---|
| Frozen / reused | RoBERTa, BotRGCN, ego adjacency, p_LM, p_GNN, h_GNN, external LLM (Qwen-2.5-7B or equiv) |
| Calibration-time params | 9234 total (linear probe dominates; 16 non-probe) |
| New trainable inference params | 9218 (linear probe only) |
| Excluded | 2-layer refiner MLP (A-L appendix only); learned retriever/gate/scorer; s_npi, s_rec; 3/4-criterion rollback; argmax-confidence proxy; hard-gate thresholds |

### System (v8-4)

```
Stage-0 (frozen)                     p_GNN, h_GNN = BotRGCN(X_R, A);  p_LM = LM_head(X_R)

Stage-1 (plumbing, 2-term composite) s(v,y) = s_lbl(v,y) + w_tg · JSD(p_LM || p_GNN) / log 2
                                     q(v) = min_y s(v,y);  q_hat = Quantile_{1-α}(s on valid_cal)

Stage-2 (plumbing, lex rule)         g(v) = (set_size, -margin, -q(v));  hard(v) = g ≥_lex τ*

Stage-3 (plumbing, T_ret = 2)        For t = 1, 2:
                                       (a) top-k_struct u ∈ 2-hop(C_{t-1}) by PPR·𝟙[rel ∈ R_trusted]
                                       (b) top-50% quantile gate: keep u iff cos(X_R[v], X_R[u]) / T_valid_cal > 0
                                       (c) conformal stop: stop if q(v | C_t) improves ≤ ε or t = 2
                                     C_final partitioned by calibrated sim:
                                       supportive > Q_66;  suspicious < Q_33;  uncertain else

Stage-4 (plumbing, 2-criterion accept using calibrated expected-error ê_φ):
  role_weight(e) ∈ {+1, -1, 0}
  w_e' = w_e · clip(1 + λ · budget(v) · role_weight · (1 - q(v)), 1-λ, 1+λ)
  max_iter = 2
  accept iff:
    ê_φ(conf_post, entropy_post, q_post, set_size_post)
      ≤  ê_φ(conf_pre, entropy_pre, q_pre, set_size_pre)       ← EXTERNAL, fit on valid_cal
    AND  ||p_GNN_post(v) - p_GNN_pre(v)||_∞ ≤ 0.2

Stage-5 (INTELLECTUAL CENTER — 4-tuple protocol):
  Schema:
    {target: X_R[v] text + p_GNN + p_LM + q,
     supportive_edges : [top-k_supp texts + weights + role],
     suspicious_edges : [top-k_susp texts + weights + role],
     uncertain_edges  : [top-k_unc texts + weights],
     quality_summary  : q_pre, q_post, set_size_pre/post, rollback_flag, ê_φ_pre/post,
     retrieval_trail  : T_ret_actual, |C_0|, |C_1|, |C_2|}
  Bucketization       : deterministic Q_66 / Q_33 on calibrated similarity (no learned head)
  Serialization       : versioned schema v1.0 with fixed field ordering
  Token budget        : 512 total;  64/192/128/64/32/32 per section (pre-registered)
  Truncation          : drop tail by rank if section exceeds budget; no padding

Stage-6 (plumbing, linear probe):
  Prompt(v) = serialize_v1.0(G_evidence(v))
  z_LLM(v)  = LLM_embed(Prompt(v))                               (mean-pool final hidden layer)
  h_out(v)  = concat(h_GNN_post(v), X_R[v], z_LLM(v))
              [256 + 768 + 3584 = 4608]
  p_final(v) = softmax(W_out · h_out(v) + b)                      [W_out: 4608×2, b: 2; total 9218]
  Non-hard nodes:  p_final(v) = p_GNN(v)
```

### Calibration-Fit Order (pre-registered)

1. Fit `p_LM` fallback MLP on `train_cal` (skipped if native head).
2. Fit `T_valid_cal` from median pairwise cosine on `valid_cal`.
3. Fit `ê_φ` logistic calibrator on `valid_cal` node features (4 inputs → 1 output).
4. 1-scalar grid search over `w_tg` on `valid_cal` (e.g., `{0, 0.25, 0.5, 0.75, 1.0}`).
5. Compute `q_hat`, `τ*` on `valid_cal` (target α = 0.15, 15% hard-node budget).
6. Compute `τ_bucket` = (Q_66, Q_33) of calibrated `sim / T_valid_cal` on `valid_cal`.
7. Fit rewriter `{λ, B_max, κ}` on `valid_cal` small grid.
8. Freeze retriever `{k_struct, ε}` from valid_cal (small grid, then frozen).
9. Fit linear probe `(W_out, b)` on `train_cal` labels using triples `(h_GNN_post, X_R, z_LLM)`.

Total calibration-time compute: minutes (all small fits). No gradient updates to any frozen component.

### Validation (v8-4)

**CLAIM C (dominant)**: Stage-5 protocol is the mechanism, and the LLM genuinely reads structured evidence — not just text volume.

Block-E (6 variants; identical Stage-1..4; identical 512-token budget; differ only in Stage-6 encoder and serialization):

```
E-null             : no Stage-6; linear probe on (h_GNN_post, X_R).
E-RoBERTa          : same-serialization, frozen RoBERTa forward pass + structured field-aware mean-pool,
                     linear probe on (h_GNN_post, X_R, z_RoBERTa).
E-LLM              : canonical Stage-6 with LLM embedding of v1.0 serialization.
E-LLM-shuffle-role : same tokens as E-LLM, role labels shuffled.         ← structure control
E-LLM-shuffle-trail: same tokens as E-LLM, retrieval_trail randomized.   ← structure control
E-LLM-collapse     : supportive+suspicious+uncertain collapsed; no role labels.  ← structure control
```

Expected: `E-LLM > E-RoBERTa > E-null` (load-bearingness gradient) AND `E-LLM > {shuffle-role, shuffle-trail, collapse}` by ≥ 1 seed-std on at least one dataset.

**Falsification paths**:
- `E-LLM ≈ E-RoBERTa` ⇒ LLM commodity; dominant contracts to "evidence graph construction matters; LLM is commodity readout".
- `E-RoBERTa ≈ E-null` ⇒ evidence graph adds nothing; paper returns to v7 EQC framing honestly.
- `E-LLM ≈ any shuffle variant` ⇒ the specific structural element (roles / trail / partitions) is non-load-bearing; that field is REMOVED from Stage-5 schema in the final method.
- `E-LLM ≈ E-LLM-collapse` ⇒ role bucketization entirely non-load-bearing; Stage-5 collapses to flat serialization; dominant contribution contracts.

**SUPPORT S1** — Stage necessity audit (Block-P): `P-full` vs `P-noConformal / P-noRetrieve / P-noRewrite / P-noLLM`. Transparent pre-committed plumbing audit. Stages within 1 seed-std of P-full labeled "pre-committed plumbing, not measurably load-bearing". No contribution claim on any single-stage success.

**SUPPORT S2** — Iteration budget-matched (Block-R): `R_2` vs `R_1-k*` (single-pass with `k_struct × 2` candidates) vs `R_2-seeds` (single-pass multi-seed PPR, 2 seeds). Expected `R_2 ≥ {R_1-k*, R_2-seeds}` ≥ 1 seed-std on at least one dataset, else retriever drops to single-pass.

**SUPPORT S3** — Linear probe sufficient (Block-L): `L-linear` vs `L-MLP`. Expected ≈. If MLP > linear ≥ 1 seed-std both datasets, escalate.

**SUPPORT S4** — Cross-dataset MGTAB: `P-full + E-LLM + L-linear` on MGTAB, 3 seeds.

**SUPPORT S5 (NEW)** — Cross-dataset calibration transfer (Block-CT): `CT-same` vs `CT-xfer`. Fit calibration on TwiBot-20; evaluate on TwiBot-22 test without refit. Drop ≤ 2 seed-std from `CT-same`. Strong dataset-agnostic protocol claim.

**Appendix**: A-Prompt field drop-one (conditional on E-LLM > E-RoBERTa); A-Bucket deterministic vs continuous; A-Stop ε sensitivity; A-BR-public / BR-soft (inherited from v7 Block-BR-sensitivity); A-Nested split robustness.

### Failure Modes (v8-4)

| Mode | Detection | Fallback |
|---|---|---|
| `w_tg → 0` on valid_cal | 1-scalar search output | Stage-1 collapses to label non-conformity; honestly report JSD term unnecessary |
| `ê_φ` poorly calibrated | Brier score on valid_cal | Fall back to simpler proxy entropy_post ≤ entropy_pre |
| Rewriter rollback rate > 90% | Block-P pipeline logs | ρ too tight; pre-registered grid alternative ρ ∈ {0.2, 0.3} |
| E-LLM ≈ E-RoBERTa | Block-E | Paper contracts LLM to commodity readout |
| E-LLM ≈ E-LLM-collapse | Block-E structure control | Role bucketization removed; Stage-5 simplifies |
| CT-xfer degrades > 2 seed-std | Block-CT | Report as "calibration partially dataset-specific"; protocol still reusable with refit |
| Linear probe underperforms | Block-L | Upgrade to 2-layer MLP |
| Prompt overflow | Token count logs | Pre-registered truncation policy handles deterministically |

### Novelty and Elegance

Exact novelty claim (v8-4):

> "First pre-registered **4-tuple protocol** (schema + calibrated-quantile bucketization + versioned serialization + fixed token budget) for constructing LLM-consumable graph-evidence artifacts from frozen-backbone ego contexts. Novelty is isolated by (i) same-LLM scrambled-structure controls that falsify token-volume explanations, (ii) same-serialization frozen-encoder controls that falsify LLM-identity explanations, (iii) cross-dataset calibration transfer that falsifies prompt-engineering explanations. Upstream trigger / retriever / rewriter stages are minimal anchored instantiations; no novelty claims."

### Compute

~ 160 GPU-hr + LLM inference (Qwen-2.5-7B local or equiv API). Adds Block-CT (~5 GPU-hr) and 3 shuffle-variant runs (~15 GPU-hr, reuses LLM embeddings). Total ~ 175 GPU-hr. 5-week timeline unchanged.

---

## Remaining Risks

- `ê_φ` calibrator generalization TwiBot-20 → TwiBot-22: directly tested by CT-xfer.
- Scrambled variants tying E-LLM on some dataset: pre-committed to schema shrinkage, not narrative defense.
- Pre-registered token budget too small: budget ablation in appendix (`{256, 512, 1024}`).
- 9218-param linear probe overfits train_cal: detected by train_cal vs valid_cal gap; fall back to p_GNN_post.
