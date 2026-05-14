# Research Proposal (v8 Round 0): Conformal-Triggered, Text-Confirmed Iterative Local Graph Rewriting with LLM Evidence Refiner

**Cycle**: v8 Round 0 (initial proposal; supersedes v7 EQC READY scope decision on LLM optionality)
**Date**: 2026-05-09
**Anchor source**: `research.md` (2026-05-09) + user directive 2026-05-09 locking 7-stage pipeline with LLM as a main stage

---

## Problem Anchor (user-locked 7-stage pipeline, verbatim into every round)

**The user has directly fixed the pipeline boundary. This is not negotiable.** Research must not change the pipeline skeleton below; only internal mechanism design is open.

```
Stage-0 LM + GNN outputs                    (frozen RoBERTa + BotRGCN)
  ↓
Stage-1 conformal ego-quality estimator
  ↓
Stage-2 hard-node candidates
  ↓
Stage-3 Local Ego Retrieval                  (structural-first, text-confirm)
  ↓
Stage-4 text-confirmed local graph rewriter  (iterative, refines candidates into edits)
  ↓
Stage-5 refined ego-evidence graph
  ↓
Stage-6 LLM embedding / refiner              (MAIN stage, not optional)
```

- **Bottom-line problem**: Residual errors in GNN-based social bot detection concentrate on nodes with locally poor graph quality, text-structure conflict, or heterophilic camouflage. A pure post-hoc local rewriter (v7 EQC) improves the ego-graph but does not fully exploit the semantic evidence available to a modern LM. The user-mandated pipeline requires the rewrite to be **text-confirmed** and the refined ego-evidence graph to be **consumed by an LLM-embedding refiner**, not left as a pure GNN artifact.
- **Must-solve bottleneck**:
  - Structural retrieval alone produces noisy candidates; text-only retrieval alone loses graph topology; both must be coupled iteratively.
  - Rewriter output cannot just be a weight vector — it must yield a **refined ego-evidence graph** that is structured enough to serialize into an LLM prompt.
  - LLM embedding of the refined graph must measurably improve over the GNN-only refinement, otherwise the LLM stage is decoration.
- **Non-goals**:
  - Bypassing any of the 7 stages (all are mandatory).
  - Global graph structure learning (LDS / Pro-GNN / IDGL).
  - LLM as final classifier (TMLR 2024 counter-evidence; LLM is enhancer).
  - Binary edge-reliability headline (BotBR/BECE territory).
  - Full causal graph repair.
- **Constraints**:
  - Frozen Stage-0 backbone (RoBERTa `roberta-finetuned-20` + BotRGCN/RGCN).
  - Soft-reweight / edit existing ego edges only; no virtual edges written to main graph.
  - Retrieval must run **structural-first then text-confirm**, iteratively up to `T_ret ≤ 3` iterations.
  - LLM embedding pass is **one-time per hard node** (not per token / per iteration); cost bounded by `|hard_nodes| × 1`.
  - Existing project GPU budget; max_iter on rewriter = 2 with rollback.
- **Success condition**: on TwiBot-20 + TwiBot-22, the full 7-stage pipeline raises macro-F1 on high-uncertainty ego slice compared to (a) frozen baseline, (b) ablation with LLM stage removed, (c) ablation with iterative retrieval replaced by single-pass retrieval, (d) ablation with rewriter removed (retrieval → LLM direct). Every stage must show ≥ 1-seed-std incremental slice-F1 contribution on at least one dataset, otherwise the redundant stage is reported as non-load-bearing.

---

## Technical Gap

Three gaps the 7-stage pipeline addresses (in order along the pipeline):

1. **Conformal estimator needs to localize "ego unreliability" precisely enough that only a small hard-node slice is routed to downstream refinement**. v7 solved this with a composite calibration-set-quantile-gated score and a lexicographic selection rule with external stability guard. v8 preserves this core.

2. **Local ego retrieval must produce candidates that a text-confirmed rewriter can actually use**. A one-pass kNN-over-RoBERTa-embeddings retrieval (v7) is too lossy: text-similar nodes may be structurally irrelevant, and structure-similar nodes may be text-contradictory. User directive is **structural-first, text-confirm, iterative**: generate structural candidates cheaply, filter/confirm by text evidence, then re-generate with the confirmed set as seed. This is an IR-style retrieve-and-rerank loop, but adapted to graphs.

3. **Refined ego-evidence graph must be consumable by an LLM refiner as a main contribution**. v7 dropped the LLM refiner to an optional branch with a non-inferiority gate — the user explicitly reverses this. The LLM is now **in the main path**; its value must be demonstrated empirically (not hidden behind a compatibility argument).

### Why naive fixes fail

- **Pure structural retrieval (v7 u_edge)**: α_text + α_rel + α_conf is a weighted sum, not a coupled iterative loop. Misses the "candidates are confirmed → new candidates are proposed" dynamic.
- **Pure text retrieval** (RoBERTa kNN only): ignores graph topology; TMLR 2024 shows LLMs treat graph prompts as keyword bags, so text-only alignment is insufficient for graph-reasoning tasks.
- **Single-pass retrieve-then-rewrite**: throws away the possibility that a confirmed edge unlocks new retrieval candidates (e.g. confirmed trusted neighbor → its 1-hop textually-coherent neighbor becomes a new candidate).
- **LLM-direct-predict** (L3 in v7): TMLR 2024 disconfirmed.
- **GNN-only refined graph** (v7 Stage-8a): useful but leaves LM semantic signal unused for the hardest nodes.

---

## Method Thesis

**One-sentence thesis**: A post-hoc pipeline where (i) a conformal ego-quality estimator selects hard nodes, (ii) an iterative structural-first-text-confirm retriever builds a refined candidate set, (iii) a text-confirmed local rewriter converts the set into a soft-edited ego-evidence graph, and (iv) an LLM refiner consumes the evidence graph to emit final predictions — strictly dominates both pure-GNN refinement (no LLM) and pure-LLM prediction (no refinement), under matched budget and frozen backbone.

### Why smallest adequate intervention

- Every stage is **user-mandated**; the method's degrees of freedom are only in how each stage is implemented. Skill principle "smallest adequate mechanism" is redefined per-stage, not pipeline-global.
- Each stage's implementation is pinned to its minimum sufficient form:
  - Stage-1 estimator: 3-term composite from v7 (preserved).
  - Stage-3 retrieval: iterative structural-first-text-confirm with ≤ 3 iterations (NEW v8).
  - Stage-4 rewriter: soft edge reweighting driven by text-confirmed candidates (refined from v7 `u_edge`).
  - Stage-6 LLM: one embedding call per hard node; output consumed by a small trainable refiner MLP.

### Why timely in foundation-model era

- Retrieve-confirm-retrieve is the modern RAG loop adapted to **graphs**; aligns with GraphRAG / GraphReasoner 2025 era work.
- LLM-as-enhancer (not predictor) matches the dominant paradigm (LOGIN, GLANCE, GraphText).
- Conformal calibration for graph uncertainty is a 2023–2026 frontier (CF-GNN, DAPS/NAPS, SNAPS).

---

## Contribution Focus

- **Dominant contribution**: **Iterative text-confirmed local graph rewriting with conformal gating for LLM-consumable ego-evidence**. The novelty is the **closed loop** — conformal triggering → structural retrieval → text confirmation → rewrite → LLM embedding — all inside a bounded budget per hard node. No prior work combines graph conformal gating, iterative structural-semantic retrieval, text-confirmed rewriter, and LLM evidence refiner in a single post-hoc pipeline on a frozen backbone for social bot detection.
- **Supporting contribution**: **`T_ret`-iteration structural-first-text-confirm retriever** — shows that iterative coupling of structure-then-text beats both single-pass retrieval and parallel-scored retrieval on high-uncertainty ego slice.
- **Explicitly rejected**:
  - LLM as final classifier.
  - Binary edge-reliability headline.
  - Global structure learning.
  - CP coverage theorem.
  - LLM-understands-graphs claim.
  - Any stage being skippable in the main pipeline (all 7 stages are mandated).

---

## Proposed Method

### Complexity Budget

| Slot | Content |
|---|---|
| Frozen / reused | RoBERTa `roberta-finetuned-20`; BotRGCN/RGCN; existing ego adjacency; frozen `p_LM`, `p_GNN`; external LLM (e.g. Qwen-2.5-7B-Instruct or GPT-4-grade) for embedding only |
| Calibration-time artifacts (not backbone retraining) | (i) 3-term composite weight search on `valid_cal` (inherited from v7); (ii) `q_hat` + `τ*`; (iii) retriever iteration hyperparameters (`T_ret`, `k_struct`, `k_text`) on `valid_cal`; (iv) rewriter `λ`, `B_max`, `κ`; (v) Stage-6 refiner MLP (2-layer) trained on `train_cal` labels consuming LLM embeddings |
| New trainable inference components | Stage-6 refiner MLP (≤ 2 × 10⁴ params) |
| Excluded | LLM-as-classifier; `s_rec` ego AE; `s_het` label-based heterophily; `sign(ΔJSD)`; hard delete/add; iterated refinement > 2 on rewriter; global graph rewrite |

### System Overview

```
Stage-0 (frozen)  X_R = RoBERTa(node_text);  p_GNN, h_GNN = BotRGCN(X_R, A);  p_LM = LM_head(X_R)

Stage-1 conformal ego-quality estimator (inherited from v7)
  s(v, y)      = s_lbl(v, y) + w_tg · JSD(p_LM || p_GNN) / log 2 + w_npi · s_npi(v)
  s_npi(v)     = mean over u ∈ N(v) of JSD(p_GNN(v) || p_GNN(u)) / log 2
  q(v) = s_composite(v) = min_y s(v, y)
  prediction_set(v) = {y : s(v, y) ≤ q_hat},  set_size, margin = q_hat - q(v)

Stage-2 hard-node selection (deterministic lex rule)
  g(v) = (set_size(v), -margin(v), -q(v));  hard(v) = g(v) ≥_lex τ*

Stage-3 Local Ego Retrieval — iterative (NEW v8 core mechanism)
  Initialize: C_0(v) = existing ego edges (propagation edges only; virtual edges never written)
  For t = 1, …, T_ret (T_ret ≤ 3, pre-registered; chosen on valid_cal):
    (a) Structural proposal: rank u ∈ 2-hop neighborhood of C_{t-1}(v) by
          struct_score(u; C_{t-1}, v) = PPR-from-v(u) · 𝟙[rel(v,u) ∈ R_trusted]
        Top-k_struct structural candidates.
    (b) Text confirmation: filter candidates by
          text_confirm(u, v) = cos(X_R[v], X_R[u]) > τ_text
                              AND |JSD(p_LM(u)) - JSD(p_LM(v))| ≤ τ_jsd
        Surviving set = C_t(v).
    (c) Stopping rule (conformal): re-evaluate q(v | C_t(v)) using Stage-1 formulas
          on the tentatively-augmented ego; stop if q improves by ≤ ε or if t = T_ret.
  Output: C_final(v) — text-confirmed local candidate set partitioned into:
    {supportive_ego_edges, suspicious_ego_edges, uncertain_ego_edges}.

Stage-4 text-confirmed local graph rewriter
  For each e = (v, u) ∈ C_final(v):
    role(e) = supportive  if text_confirm(u, v) AND u yields positive GNN margin lift
            = suspicious  if text_confirm fails BUT e ∈ existing ego
            = uncertain   else
  Soft reweight propagation-edge weights on existing ego edges only:
    w_e' = w_e · clip(1 + λ · budget(v) · role_weight(e) · (1 - q(v)), 1-λ, 1+λ)
    role_weight(supportive)=+1, role_weight(suspicious)=-1, role_weight(uncertain)=0
  max_iter on rewrite = 2; accept/rollback via:
    q_post(v) < q_pre(v)  AND  set_size_post ≤ set_size_pre
    AND posterior_margin_post ≥ posterior_margin_pre - η       ← from frozen head (v7 external guard)
    AND ||p_GNN_post - p_GNN_pre||_∞ ≤ ρ = 0.2

Stage-5 refined ego-evidence graph construction
  G_evidence(v) = (target=v, supportive_edges, suspicious_edges, uncertain_edges, post-rewrite weights,
                   quality_summary={q_pre, q_post, set_size_pre, set_size_post, rollback_flag})

Stage-6 LLM embedding / refiner
  Prompt(v) = serialize_to_text(G_evidence(v))   — target node profile + roled neighbors + weights
  z_LLM(v)  = LLM_embed(Prompt(v))               — one call per hard node
  h_out(v)  = MLP_refiner(concat(h_GNN_post(v), X_R[v], z_LLM(v)))   — 2-layer MLP, trained on train_cal
  p_final(v) = softmax(W_out · h_out(v))
  For non-hard nodes, p_final(v) = p_GNN(v) (no LLM call, no refiner).
```

### Core Mechanism — Iterative Structural-First-Text-Confirm Retrieval

**Why "structural-first, text-confirm" not "text-first, structural-confirm"**:
- Structural candidates are cheap (PPR over ego + local 2-hop is `O(|E_ego|)`), and are naturally topology-aware.
- Text confirmation is the **trust filter** — it answers "does the semantic evidence support connecting to this candidate?"
- Reverse order (text-first) either blows up kNN search over all nodes or ignores topology entirely.

**Why iterative (`T_ret` > 1)**:
- A confirmed candidate unlocks its 1-hop neighborhood for the next iteration's structural proposal. This is the key mechanism that a single-pass retriever cannot replicate.
- Bounded at `T_ret ≤ 3` by compute budget + confirmation-bias concern.
- Conformal stopping rule ensures we stop when q(v) no longer improves — external to the retriever itself.

**Why text confirmation, not parallel scoring**:
- In v7 `u_edge = α_aff · cos − α_conf · conflict + α_rel · prior` is a **weighted sum at scoring time** — nothing forces text to act as a filter. In v8, text confirmation is a **hard gate** (pass/fail per edge), separating structural recall from semantic precision.

### Stage-6 LLM Refiner — One-Call-Per-Node Evidence Consumer

- **Prompt format** (serialize `G_evidence(v)`):
  ```
  Target user: <X_R[v] text summary + p_GNN(v) + q(v)>
  Supportive neighbors (top-k by text_confirm): <list of neighbors with texts and weights>
  Suspicious neighbors (failed text_confirm, in ego): <list>
  Uncertain neighbors: <list>
  Ego quality: q_pre=X, q_post=Y, set_size change=Z, rollback=N/Y
  ```
- **Embedding extraction**: mean-pool final layer hidden states of the prompt tokens. One forward pass; one LLM call per hard node per test-time inference.
- **Refiner MLP**: concatenates `h_GNN_post(v)` (post-rewrite GNN state), `X_R[v]` (text embedding), `z_LLM(v)` (LLM embedding of evidence prompt). Two linear layers with ReLU; output head to 2-class softmax. Trained on `train_cal` with LLM embeddings pre-computed offline.
- **Why not LLM-direct-predict**: we tested this in v7 diagnostic (L3) as a reference only; TMLR 2024 disconfirms graph-prompt-as-reasoning. Embedding + refiner preserves the LLM's strong semantic abstractions without requiring structural-reasoning capability.

### Modern Primitive Usage

- **LLM as enhancer** (Stage-6): embedding refiner; not predictor; aligns with LOGIN / GLANCE / Graph Meets LLM Survey.
- **RoBERTa frozen embedding prior**: inside Stage-1 (`s_tg` JSD), Stage-3 (`cos`, text confirmation), Stage-4 (role assignment).
- **Conformal quantile calibration** (inspired, not theorem-backed): CF-GNN / DAPS/NAPS / SNAPS.
- **Iterative retrieve-and-rerank**: classic IR pattern; modern LLM equivalent is the RAG loop. We adapt it to graph candidates.

### Integration into Existing Pipeline

- Extend `GraphConformalSetEstimator` at [estimators.py:1922](LLMbot/baseline/core/estimators.py#L1922) with Stage-1 composite score + lex selection.
- Extend `EgoRefinementRepairOperator` at [operators.py:194](LLMbot/baseline/core/operators.py#L194) with the iterative retriever (Stage-3), the text-confirmed rewriter (Stage-4), and the evidence-graph serializer (Stage-5).
- New `LLMEvidenceRefiner` module at `LLMbot/baseline/core/llm_refiner.py` with one-call-per-node LLM inference + 2-layer MLP.
- Unit tests enforce (i) virtual edges never written to main adjacency, (ii) LLM called only for hard nodes, (iii) calibration indices disjoint from test indices, (iv) retriever halts at T_ret regardless.

### Training Plan

- Stage-0: already trained and frozen (no change).
- Stage-A (calibration-time):
  - Fit `p_LM` fallback MLP on `train_cal` (skipped if backbone has native LM head).
  - Pre-registered search on `valid_cal` for `{w_tg, w_npi}`, `{T_ret, k_struct, k_text, τ_text, τ_jsd, ε}`, `{λ, B_max, κ, η}`. Grid size capped at < 1000 cells.
  - Compute `q_hat`, `τ*` on `valid_cal`.
- Stage-B (Stage-6 refiner only; NEW v8):
  - Pre-compute LLM embeddings on `train_cal ∪ valid_cal` hard nodes (one-time; cached).
  - Fit 2-layer `MLP_refiner` on `train_cal` labels using `(h_GNN_post, X_R, z_LLM)` triples.
- Inference: 7-stage pipeline one-pass per test node.

### Failure Modes and Diagnostics

| Mode | Detection | Fallback |
|---|---|---|
| Retriever collapses: `T_ret` converges to 1 on valid_cal | `valid_cal` search output | Accept T_ret=1 as main; paper reports "iterative retrieval not necessary" |
| Text confirmation rejects all candidates | Empty C_t(v) | Fall back to single-pass structural candidates; log as warning |
| LLM embedding uninformative (Stage-6 adds ≤ ε slice F1) | Block-LLM ablation | Report honestly; paper claim on LLM stage contracts |
| Evidence-graph serialization truncated by LLM context | Prompt length > model limit | Prompt-compression: keep only top-k supportive + top-k suspicious by role_weight, drop uncertain |
| Rewriter confirmation bias | Second rewrite iteration approves worse ego | Strict `q_post < q_pre` + external stability guard (inherited from v7) catches this |
| LLM cost explosion | LLM call budget exceeds plan | Enforce `|hard_nodes| × 1` cap; downsample hard-node set if needed |
| Refiner overfits train_cal LLM embeddings | Gap train_cal vs valid_cal | Freeze refiner, fall back to GNN-only `p_GNN_post` |

### Novelty and Elegance

Closest prior:
- CF-GNN / DAPS/NAPS: conformal on graphs — we use calibration, not coverage claims.
- GAugLLM / CTGL: text-graph coupled modifier in contrastive training — we do post-hoc, no training, iterative retrieval.
- GLANCE / LOGIN: selective LLM consultation — we extend with text-confirmed iterative retrieval + evidence-graph formatting.
- BotBR / BECE: edge reliability — we do not claim binary reliability; v5 Block-T will directly compare.
- GraphRAG / HippoRAG: retrieval-augmented graph generation — we do retrieval-augmented **bot classification** with a frozen backbone and conformal gating.

Exact novelty claim: "We are the first to combine conformal ego-uncertainty gating, iterative structural-first-text-confirm retrieval, text-confirmed local graph rewriting, and LLM evidence embedding in a single post-hoc pipeline on a frozen backbone for social bot detection, while honoring all anchor constraints (no backbone retraining, no virtual edges written, no LLM-as-classifier)."

---

## Claim-Driven Validation Sketch

Three main claim blocks + three ablation blocks, each pinned to one user-mandated stage.

### Claim 1 — Full 7-stage pipeline beats frozen baseline and each "single-stage-removed" ablation

Block-P (Pipeline). Variants:
- P0 frozen baseline (Stage-0 only, no refinement).
- P-noLLM full pipeline minus Stage-6 LLM (GNN-only; v7 EQC equivalent).
- P-noRewrite retrieval only; Stage-4 replaced by identity.
- P-noRetrieve Stage-3 replaced by one-pass kNN (`T_ret = 1`, no text confirmation).
- P-noConformal Stage-1/2 replaced by entropy trigger.
- P-full 7-stage pipeline (v8 main).

Datasets: TwiBot-20, TwiBot-22. 3 seeds. Primary metric: macro-F1 on high-uncertainty ego slice (top 20% by `q`). Secondary: global macro-F1 (non-degradation gate).

Expected: `P-full > P-noLLM > P-noRewrite > P-noRetrieve ≥ P-noConformal > P0`, with each gap ≥ 1 seed-std on at least one dataset.

Falsification rule: **if any stage's leave-one-out variant is within 1 seed-std of `P-full`, that stage's claim is downgraded to "not load-bearing" — paper reports honestly**. This mirrors v7 Block-S2/S4 pruning discipline extended pipeline-wide.

### Claim 2 — Iterative retrieval beats single-pass retrieval (T_ret > 1 is necessary)

Block-R (Retrieval). Variants: R_t=1, R_t=2, R_t=3, R_parallel (structural + text scored jointly, single pass).

Expected: `R_t=3 ≥ R_t=2 > R_t=1 ≈ R_parallel` on slice F1. Falsification: if `R_t=2 ≤ R_t=1` or `R_parallel ≥ R_t=3`, `T_ret > 1` claim dies and method contracts to single-pass retrieval.

### Claim 3 — Text confirmation (hard gate) beats text scoring (soft weight)

Block-TC (Text-Confirm). Variants:
- TC-gate (v8 main): text confirmation as hard filter.
- TC-weight: text similarity enters as soft weight in structural score (v7 `u_edge` style).
- TC-none: structural only, no text involvement in retrieval.

Expected: `TC-gate > TC-weight > TC-none` on slice F1. This is the "coupling matters" claim.

### Support S1 — LLM embedding (Stage-6) meaningful ≠ cosmetic

Block-LLM. Variants:
- LLM-full (Stage-6 as specified).
- LLM-ablate-prompt: use frozen RoBERTa sentence embedding of the prompt text instead of LLM (tests whether LLM semantic capability is needed vs plain text encoder).
- LLM-direct: LLM predicts label directly from prompt (TMLR 2024 diagnostic reference).

Expected: `LLM-full > LLM-ablate-prompt` by ≥ 1 seed-std. If not, LLM embedding is not adding semantic value over frozen RoBERTa, and the paper contracts Claim 1 accordingly.

### Support S2 — Cross-dataset robustness

MGTAB (7-relation multi-relational). Full `P-full` pipeline. 3 seeds. Matched budget from TwiBot-20.

### Support S3 — Rewriter external stability guard (inherited v7)

Block-G. G1 with external guard (v8 main) vs G2 no guard. Expected G1 ≥ G2 on global F1 non-degradation.

### Appendix-only

- Retriever component ablation: structural-only / text-only / coupled within single iteration.
- A-Prompt: evidence-graph prompt field ablation (conditional on LLM-full passing).
- BR-public / BR-soft comparison (inherited from v7).
- `max_iter = 2` rewriter appendix.
- Nested 5-fold `train_cal / valid_cal` rotation robustness.

---

## Experiment Handoff Inputs

- Must-prove claims: C1 (7-stage load-bearing), C2 (iterative retrieval), C3 (text confirmation).
- Must-run ablations: Block-P (5 variants), Block-R (4 variants), Block-TC (3 variants), Block-LLM (3 variants), Block-G (2 variants), cross-dataset MGTAB.
- Datasets: TwiBot-20 dev, TwiBot-20 full, TwiBot-22, MGTAB.
- Critical metrics: global macro-F1, slice macro-F1 (top-20% by q), rollback rate, retriever iteration count average, LLM call count, refinement wall-clock overhead ratio, refiner train-vs-valid gap.
- Highest-risk assumptions:
  - LLM embedding provides > 1 seed-std slice F1 improvement over RoBERTa sentence embedding on prompt (Claim 1 load-bearing LLM stage).
  - Iterative retrieval provides > 1 seed-std improvement over single-pass (Claim 2).
  - Text confirmation as hard gate beats soft weighting (Claim 3).
  - LLM embeddings generalize train_cal → test without refitting.
  - Evidence-graph prompt formatting is informative to LLM (not just padding).

---

## Compute & Timeline

| Bucket | GPU-hr |
|---|---|
| Stage-A calibration grid search | ~ 6 |
| Stage-B refiner training (once LLM embeddings cached) | ~ 2 |
| Block-P 5 variants × 2 datasets × 3 seeds | ~ 45 |
| Block-R 4 variants × 2 × 3 | ~ 24 |
| Block-TC 3 variants × 2 × 3 | ~ 18 |
| Block-LLM 3 variants × 2 × 3 | ~ 18 |
| Block-G 2 × 2 × 3 | ~ 12 |
| MGTAB | ~ 10 |
| Appendix | ~ 30 |
| LLM embedding inference (train_cal + valid_cal + test hard nodes) | ~ 15 GPU-hr (Qwen-2.5-7B local) or equivalent API cost |
| **Total** | **~ 180 GPU-hr + LLM inference** |

Timeline (5 weeks after core machinery ready):
- W1: Stage-1/2/3 implementation + Stage-A search.
- W2: Stage-4/5 implementation + evidence-graph serializer.
- W3: Stage-6 LLM pipeline + refiner training on cached embeddings.
- W4: Block-P, Block-R, Block-TC on TwiBot-20/22.
- W5: Block-LLM, Block-G, MGTAB, appendix, analysis, paper draft.
