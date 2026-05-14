# Round v8-5 Refinement (final refinement; MAX_ROUNDS cap reached after next review)

**Round**: v8-5
**Review source**: `round-v8-4-review.md` (overall 8.09 REVISE NONE drift; CQ ceiling blocks READY)
**Headline revision**: apply **every** READY-shot lever the reviewer offered. No pushback.

---

## Problem Anchor

[verbatim from v8 Round 0; user-locked 7-stage pipeline unchanged]

## Anchor Check

- Original bottleneck targeted: yes.
- Round-4 reviewer: "NONE / negligible drift. Actual tightening pass."
- All five round-4 items are pure CQ/MS elevation without new stages or structural changes.

## Simplicity Check

- Intellectual center still Stage-5 4-tuple protocol.
- Components changed (all simplifications, not additions):
  - `retrieval_trail` moved to appendix-only (reviewer simplification 1).
  - Claim C criterion tightened to "consistent direction on both datasets; magnitude ≥ 1 seed-std on at least one".
  - `ê_φ` fitted with 5-fold cross-fitting inside valid_cal (prevents calibration overfit).
- Components added (all invariance tests, not mechanisms):
  - `E-LLM-shuffle-order` (edge-order randomization within each bucket) promoted to Block-E main.
  - `E-LLM-Qwen` and `E-LLM-Mistral` (or equivalent) two-model robustness check in Block-E.
  - `E-LLM-Budget-256` vs `E-LLM-Budget-512` comparison in Block-E.
- Net: Stage-5 schema shrinks (retrieval_trail demoted); Block-E expands by 4 variants; accept-rule robustness improves.

---

## Changes Made

### 1. Tighten Claim C Acceptance Bar (CRITICAL)

**Reviewer**: "≥ 1 seed-std on ≥ 1 dataset is permissive for a centerpiece claim. Require consistent direction on both datasets."

**Action**: final Claim C acceptance criterion:

```
Claim C passes iff, over 3 seeds on BOTH TwiBot-20 and TwiBot-22:
  (i)  E-LLM > E-RoBERTa > E-null,  sign(mean_diff) consistent across BOTH datasets;
  AND
  (ii) Magnitude: at least ONE of the two gaps exceeds 1 seed-std on at least ONE dataset;
  AND
  (iii) E-LLM > {E-LLM-shuffle-role, E-LLM-shuffle-trail, E-LLM-shuffle-order, E-LLM-collapse}
        with consistent direction across BOTH datasets.
```

This makes `P(Claim C passes | method works)` demand cross-dataset consistency — directly addresses "permissive centerpiece" critique.

### 2. Promote Edge-Order Invariance to Block-E Main (READY-shot lever)

**Reviewer modernization 1**: "Edge-order randomization within each bucket — same tokens, permuted order — to show the protocol is not just sorted adjacency list luck."

**Action**: add `E-LLM-shuffle-order` to Block-E main (not appendix). The variant keeps all role labels intact but randomizes the intra-bucket order. Canonical `E-LLM` ordering: rank-by-PPR-then-sim within each bucket (now explicitly locked as part of Stage-5 protocol per reviewer simplification 3).

Expected: `E-LLM > E-LLM-shuffle-order` ≥ 1 seed-std on ≥ 1 dataset. If tie, canonical ordering is **non-load-bearing** and Stage-5 protocol drops the ordering requirement (schema shrinks).

**Why this is the key CQ lever**: demonstrates the protocol is **testably sensitive to structural invariants**, not just "a long prompt that works". Reviewer called this "directly aligns with 'protocol' as a contribution and can bump CQ/MS perception without new machinery".

### 3. LLM-Family Robustness in Block-E (READY-shot lever)

**Reviewer modernization 2**: "At least 2 LLMs, same Stage-5; single-model gains = 'model-specific prompt fit'."

**Action**: run Block-E E-LLM variant with **two LLMs**:
- `E-LLM-Qwen` (Qwen-2.5-7B-Instruct, v8 canonical).
- `E-LLM-Mistral` (Mistral-7B-Instruct-v0.3) — different family, similar size.

Claim: `E-LLM-Qwen ≈ E-LLM-Mistral` (within 1 seed-std on each dataset) AND both > E-RoBERTa. If a **single LLM** dominates by a large margin, protocol is model-specific; honest reporting.

**Feasibility**: reuses Stage-1/2/3/4/5 output (identical serialized evidence); only Stage-6 embedding call differs. Adds ~ 6 GPU-hr (one extra LLM inference pass over hard nodes in train_cal ∪ valid_cal ∪ test).

### 4. Token-Budget Invariance (Modernization 3)

**Reviewer modernization 3**: "Budget-256 vs Budget-512 (same schema) to show protocol's benefit isn't just 'more context'."

**Action**: add `E-LLM-Budget-256` to Block-E appendix (not main). Same schema + deterministic truncation per section (halved). If `E-LLM-Budget-256 ≈ E-LLM-Budget-512`, the 512-token choice is not critical; if smaller budget much worse, budget-matching is important.

Lives in appendix to avoid bloating Block-E main.

### 5. ê_φ Cross-Fitting (IMPORTANT 3)

**Reviewer**: "Pre-register cross-fitting inside valid_cal to prevent calibration overfit."

**Action**: fit `ê_φ` with 5-fold cross-validation inside valid_cal:

```
Split valid_cal into 5 folds.
For each fold k:
  Fit ê_φ^{(k)} on valid_cal \ fold_k.
  Predict ê_φ^{(k)}(features) for valid_cal ∩ fold_k.
Aggregate: the "ê_φ used at test time" is the average of the 5 out-of-fold predictors.
```

Prevents the calibrator from seeing its own fitting data. Adds ~ 0 compute (still a 5-param logistic).

### 6. Positioning Sentence for Prompt-Engineering Defense (IMPORTANT 2)

**Reviewer**: "Add one explicit positioning sentence: 'Stage-5 is a protocol whose key property is predictable, testable sensitivity (scramble tests) rather than best prompt'."

**Action**: adopt verbatim in method thesis paragraph. The line now reads:

> "**Stage-5 is a protocol whose key property is predictable, testable sensitivity**: the protocol's value is established by the pattern of *degradations* under scramble controls (role, trail, order, collapse), not by any best-prompt search. We make no claims about prompt-optimality and run no prompt-optimization procedure."

### 7. `retrieval_trail` Demoted to Appendix (Reviewer Simplification 1)

**Reviewer**: "Metadata that might help the LLM is prompt-hacking risk."

**Action**: remove `retrieval_trail` from canonical `G_evidence` schema v1.0. Appendix A-Prompt retains the "with-trail" variant as an optional field-ablation.

Stage-5 schema v1.0 (final):

```
G_evidence(v) = {
  target           : X_R[v] text + p_GNN(v) + p_LM(v) + q(v)
  supportive_edges : top-k_supp neighbors, canonical rank order, with text + weight + role
  suspicious_edges : top-k_susp neighbors, canonical rank order, with text + weight + role
  uncertain_edges  : top-k_unc neighbors, canonical rank order, with text + weight
  quality_summary  : q_pre, q_post, set_size_pre, set_size_post, ê_φ_pre, ê_φ_post, rollback_flag
}
```

5 sections, not 6. Token budget reallocated:

```
TOKEN_BUDGET = 512 total
  target            :  80 tokens  (+16 from freed retrieval_trail)
  supportive_edges  : 200 tokens  (+8)
  suspicious_edges  : 128 tokens
  uncertain_edges   :  64 tokens
  quality_summary   :  40 tokens  (+8)
```

Reallocation is deterministic and pre-registered.

### 8. `E-LLM-collapse` as Minimal-Artifact Baseline (Reviewer Simplification 2)

**Reviewer**: "Default E-LLM-collapse as the minimal artifact baseline; treat supportive/suspicious/uncertain as the incremental contribution."

**Action**: reframe Block-E narrative:

```
E-null            : no Stage-6 (frozen-GNN + RoBERTa linear probe).
E-RoBERTa         : same-serialization + frozen-RoBERTa + structured mean-pool.
E-LLM-collapse    : minimal-artifact baseline (same 512 tokens, flat neighbor list, no role labels).
E-LLM-shuffle-role: shuffled role labels (tests role semantics).
E-LLM-shuffle-order: intra-bucket order permuted (tests canonical ordering).
E-LLM (canonical) : v1.0 schema + role labels + canonical ordering.
```

Narrative claim becomes:
> "Relative to E-LLM-collapse (flat neighbor list of same 512 tokens), the full Stage-5 protocol adds (a) role bucketization, (b) canonical rank ordering, (c) quality summary fields. We isolate the contribution of each via scramble controls that null the corresponding structural element."

This is the cleanest possible presentation: the contribution is what you add on top of a minimal flat artifact.

### 9. Stage-3 Ordering Rule Frozen as Protocol Element (Reviewer Simplification 3)

**Reviewer**: "Freeze Stage-3 ordering rule as part of the protocol (rank-by-PPR then sim); no secondary re-ranking."

**Action**: canonical ordering within each bucket:

```
canonical_rank(u) = (PPR_rank(u), -calibrated_sim(u))  lexicographic
  (i.e. primary sort by PPR descending, tie-break by calibrated similarity descending)
```

This ordering is now Stage-5 protocol element (iii'): "versioned serialization with canonical rank order". `E-LLM-shuffle-order` is the corresponding scramble test.

---

## Revised Proposal (v8-5 final)

### Title

**Evidence-Graph Construction for LLM-Consumable Ego Context in Social Bot Detection**

### Method Thesis (final, v8-5)

> We propose a procedure for converting frozen-backbone uncertain-ego contexts into **LLM-consumable graph-evidence artifacts**. The contribution is a 4-tuple protocol: (i) a fixed 5-section schema, (ii) deterministic Q_66 / Q_33 bucketization on calibrated similarity, (iii) canonical intra-bucket rank order by `(PPR_rank, -sim)`, (iv) pre-registered 512-token budget with deterministic truncation. **Stage-5 is a protocol whose key property is predictable, testable sensitivity**: its value is established by the pattern of degradations under scramble controls (role, trail, order, collapse), not by any best-prompt search. We make no claims about prompt-optimality and run no prompt-optimization procedure. Upstream trigger / retrieval / rewrite stages are minimal instantiations required by the locked 7-stage pipeline; we make no novelty claims on them.

### Contribution Focus (final)

- **Dominant**: Stage-5 4-tuple protocol (schema + bucketization + canonical ordering + fixed budget). Defined abstractly; instantiated with cross-dataset calibration carryover. Value isolated by 4 scramble controls + 2 encoder controls + 2 LLM families + 1 budget control.
- **Supporting plumbing (explicitly non-novel)**: Stage-1 2-term composite; Stage-3 2-iteration retrieval with top-50% quantile gate; Stage-4 ê_φ-gated accept (5-fold cross-fit on valid_cal); Stage-6 9218-param linear probe.
- **Rejected**: everything previously listed + prompt-optimality claims + ordering-as-bonus claims.

### Validation (v8-5 final)

**CLAIM C** (dominant, acceptance criterion upgraded):

```
Block-E (9 variants, all sharing Stage-1..5; all at TOKEN_BUDGET=512):
  Load-bearingness ladder (3 encoder variants):
    E-null:     no Stage-6.
    E-RoBERTa:  same-serialization + frozen RoBERTa + structured mean-pool.
    E-LLM-Qwen: canonical Stage-6 with Qwen-2.5-7B.
    E-LLM-Mistral: canonical Stage-6 with Mistral-7B-Instruct-v0.3.
  
  Scramble controls (all use Qwen; same tokens as E-LLM-Qwen):
    E-LLM-shuffle-role  : role labels shuffled.
    E-LLM-shuffle-order : intra-bucket order randomized.
    E-LLM-collapse      : minimal-artifact baseline (flat list, no roles).
  
  Appendix budget control:
    E-LLM-Budget-256 (Qwen, reduced budget).

Claim C passes iff, across 3 seeds on BOTH TwiBot-20 AND TwiBot-22:
  (i)   E-LLM-Qwen > E-RoBERTa > E-null  with consistent direction on BOTH datasets;
  (ii)  At least ONE of the two gaps exceeds 1 seed-std on at least ONE dataset;
  (iii) E-LLM-Qwen > {shuffle-role, shuffle-order, collapse} with consistent direction on BOTH datasets;
  (iv)  E-LLM-Qwen ≈ E-LLM-Mistral within 1 seed-std on EACH dataset (model robustness).
```

Falsification: any of (i)-(iv) fail ⇒ narrative contracts:
- (i) fails: evidence graph adds nothing; return to v7 EQC framing.
- (ii) fails: gap too small; treat protocol as marginal not centerpiece.
- (iii) fails for a specific scramble: that structural element (role / order) is dropped from Stage-5 schema; protocol shrinks.
- (iv) fails: protocol is model-specific; report as "Qwen-specific for now; generalization open question".

**SUPPORT S1** — Stage necessity audit (Block-P): `P-full` vs leave-one-out, transparency only.

**SUPPORT S2** — Iteration budget-matched (Block-R): `R_2` vs `R_1-k*` (single-pass, 2·k_struct) vs `R_2-seeds` (multi-seed PPR).

**SUPPORT S3** — Linear probe sufficient (Block-L): `L-linear` vs `L-MLP`.

**SUPPORT S4** — Cross-dataset MGTAB: `P-full + E-LLM-Qwen + L-linear`, 3 seeds.

**SUPPORT S5** — Cross-dataset calibration transfer (Block-CT): `CT-same` vs `CT-xfer` (fit on TwiBot-20 → evaluate on TwiBot-22).

**Appendix**: A-Prompt field drop-one with retrieval_trail re-introduction (conditional on E-LLM > E-RoBERTa); A-Bucket deterministic vs continuous; A-Stop ε sensitivity; A-Budget-256; A-BR-public / BR-soft (inherited); A-Nested split robustness with ê_φ cross-fit diagnostics.

### Claim-to-Block mapping

| Claim | Block | Acceptance |
|---|---|---|
| C dominant | Block-E | 4-part criterion (direction + magnitude + scrambles + LLM-robust) |
| S1 plumbing transparency | Block-P | within-1-seed-std stages labeled non-load-bearing |
| S2 iteration | Block-R | R_2 > budget-matched controls |
| S3 linear probe | Block-L | L-linear ≈ L-MLP |
| S4 cross-dataset | MGTAB | directional + no-degradation |
| S5 calibration transfer | Block-CT | CT-xfer ≤ 2 seed-std drop |

### Failure Modes (v8-5)

| Mode | Detection | Fallback |
|---|---|---|
| `w_tg → 0` on valid_cal | 1-scalar search output | Stage-1 collapses to label non-conformity |
| `ê_φ` cross-fit fold disagreement | 5-fold Brier variance | Fall back to simpler entropy criterion |
| Rewriter rollback > 90% | Block-P logs | ρ grid extension |
| Block-E condition (i) fails | Block-E main | Return to v7 EQC framing |
| Block-E condition (iii) fails specific | Scramble delta | Drop that schema element; report honestly |
| Block-E condition (iv) fails | Qwen vs Mistral delta | Report as Qwen-specific; open future work |
| Block-CT fails | CT-xfer delta | Report calibration partial-dataset-specificity |
| Linear probe underperforms | Block-L | Upgrade to MLP, report |

### Compute (final estimate)

```
Stage-A calibration         :   4 GPU-hr
Stage-B probe training      :   2 GPU-hr
LLM embeddings (Qwen)       :  10 GPU-hr  (cached, all hard nodes in train_cal ∪ valid_cal ∪ test)
LLM embeddings (Mistral)    :   8 GPU-hr  (new, for Block-E LLM robustness)
Block-E (9 variants × 3 seeds × 2 datasets) : 36 GPU-hr  (probes only; embeddings cached)
Block-P                     :  20 GPU-hr
Block-R                     :  18 GPU-hr
Block-L                     :   6 GPU-hr
MGTAB (S4)                  :  10 GPU-hr
Block-CT (S5)               :   4 GPU-hr
Appendix                    :  25 GPU-hr
  ──────────────────────────
  Total                     : ~143 GPU-hr + 18 GPU-hr LLM = ~161 GPU-hr
```

5-week timeline unchanged:
- W1: Stage-1/2/3 + calibration fits.
- W2: Stage-4 (ê_φ cross-fit) + Stage-5 serialization v1.0.
- W3: Stage-6 embeddings (both LLMs) + linear probe fit.
- W4: Block-E main + Block-P + Block-R + Block-L.
- W5: MGTAB + Block-CT + appendix + paper draft.

---

## Dominant-contribution single-line summary

> "A pre-registered 4-tuple protocol (schema + calibrated-quantile bucketization + canonical ordering + fixed token budget) for converting frozen-backbone uncertain-ego contexts into LLM-consumable graph-evidence artifacts; its load-bearingness is established by a 9-variant evaluation isolating encoder choice, schema elements, ordering, role semantics, and LLM family."

This single sentence is what the paper's abstract claims. Every other stage is acknowledged plumbing.
