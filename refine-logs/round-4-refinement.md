# Round 4 Refinement — User-Directed Rescope

**Round**: 4
**Trigger**: User message 2026-05-09 with 9 uncertainties + 4 locked decisions + beast/nightmare execution directive.
**Review source of record up to this point**: `refine-logs/round-4-review.md` (Codex GPT-5.2 xhigh, 9.2 READY spec-level).
**Purpose**: apply the four user-locked decisions, address the nine user-flagged uncertainties, and submit the revised proposal to Round 5 review at maximum reasoning effort.

---

## Problem Anchor (verbatim from Round 0)

[unchanged — carried verbatim; the four user decisions are rescoping decisions inside the anchor, not a new anchor]

---

## Anchor Check

- **Original bottleneck still targeted**: yes. The four user decisions tighten scope (modifier form locked, claim language tightened) and restore one branch deferred in Round 3 (optional LLM embedding).
- **Drift risk assessment**: decision #2 "v1 main = artifact-only + optional hard-node LLM embedding" is a **rescope, not drift**. It matches `research.md` §7 ("v1 允许实现 LLM-enhanced branch，但需要把它作为可选分支和消融项") and §14 ("LLM: no LLM in v1 main method; hard-node evidence ego embedding in v2"). My Round-3 dropping of evidence artifact + virtual context edges was a reviewer-approved simplification under a stricter "no LLM anywhere in v1" interpretation. The user's rescope is looser: artifact stays in v1 main (LLM-free); LLM embedding is an optional v1 ablation branch (not deferred to v2). This is consistent with research.md.
- **Reviewer suggestions rejected as drift**: none in the current message (user is adjusting scope, not the anchor).

## Simplicity Check

- **Dominant contribution unchanged**: EQC² — single-primitive quantile-calibrated selection controller. The user's decision #4 **tightens the contribution claim language**, which is a sharpening, not a widening.
- **Components added back** (from Round-3 removals):
  - **Evidence artifact**: a read-only serialization of the refined ego subgraph per hard node (target summary + supportive/suspicious/uncertain neighbors + semantic + structural context + quality summary). Contains no learned parameters.
  - **Virtual context edges**: kNN-retrieved neighbors that do NOT exist in the original ego. They enter the artifact and (optionally) the LLM prompt. They **never** enter the main graph, the edge modifier scoring, or the Stage-5 soft-reweight rule.
- **Components locked** (resolving user uncertainty #2):
  - **Modifier output form**: `u_edge(e) ∈ [-1, +1]` continuous utility as v1 main. The `s_reliability` binary head is deleted (would duplicate BotBR/BECE). The `p_role` 4-way distribution is moved to appendix only (diagnostic, never v1 main).
- **Components added for falsification** (resolving user uncertainties #1, #3, #4, #5, #6, #9):
  - Estimator semantics diagnostic study (uncertainty #1).
  - No-counterfactual baseline + split-transfer test + label-shuffle diagnostic for the learned modifier appendix (uncertainty #3).
  - LLM attribution ablation: artifact-only vs prompt-summarization vs embedding vs direct-predict (uncertainty #4).
  - Evidence prompt field ablation (uncertainty #5).
  - Hard-delete / hard-add rewrite variants as appendix negative controls (uncertainty #6).
  - BotBR/BECE-style binary-reliability baseline in Block-T (uncertainty #9).

## User-Locked Decisions — Applied

### Decision 1 — Modifier output form locked to `continuous utility`

- **Rationale (user)**: "`binary reliability` 容易撞 BotBR/BECE；四类 role 更适合 evidence prompt 和诊断，不宜作为第一版主标签空间."
- **Applied**:
  - v1 main edge modifier outputs a single continuous utility `u_edge(e) ∈ [-1, +1]`. The deterministic form locked in Round 2 was already continuous (`α_text · cos + α_rel · 𝟙 + α_tgd · sign`); it is relabelled explicitly as "`u_edge(e)` continuous utility" and clipped to `[-1, +1]`.
  - `s_reliability` binary head: **removed entirely** (including from Block-B modifier-output-form appendix). Reason: user explicitly flags BotBR/BECE collision.
  - `p_role` 4-way distribution: **moved to evidence-prompt diagnostic only**. Still computable from the continuous `u_edge` via a deterministic bucketization (`u_edge ≥ τ+ → supportive`, `u_edge ≤ τ- → suspicious`, `|u_edge| ≤ τ0 → neutral`, else → uncertain). Used only in the artifact / LLM prompt, never as a training head.
- **Impact on plan**: `A-Modifier` appendix now compares deterministic continuous modifier vs learned continuous modifier vs the *deterministic + 4-way bucketization* artifact variant. Binary-reliability is deleted from that appendix; a separate **BotBR/BECE-style binary-reliability baseline** is added to Block-T for honest comparison with prior art (user uncertainty #9).

### Decision 2 — v1 main branch = `artifact-only + optional hard-node LLM embedding`

- **Rationale (user + research.md §7/§14)**: v1 allows LLM-enhanced branch as an ablation but does not require it as the main path.
- **Applied**: v1 is now split into two nested branches with a clear contribution map.

```
v1-Main-Artifact  (NO LLM call anywhere):
  Stage-0 → Stage-1 EQC² score → Stage-2 lex selection → Stage-3 retrieval (propagation + virtual) →
  Stage-4 continuous modifier (u_edge) → Stage-5 soft reweight over propagation edges →
  Stage-6 L-hop-local recomputation + accept/rollback →
  Stage-7 emit evidence artifact per accepted hard node (serialization only; no LLM).
  Optional final: Stage-8a Artifact-Only Refiner h_artifact = MLP(h_GNN_post, artifact_features) → p_final.

v1-LLM (optional ablation branch, feeds on the artifact):
  Stage-8b Evidence ego prompt constructed from artifact →
  z_LLM(v) = LLM_embed(EvidencePrompt_v) →
  p_final(v) = Refiner(h_GNN_post(v), X_R[v], z_LLM(v), Q_phi_post(v))
```

- **Contribution framing (sharp)**:
  - Dominant contribution (unchanged) = EQC² single primitive as trigger + rewrite-amplitude factor + L-hop-local accept/rollback gate.
  - Supporting contribution = deterministic continuous modifier over existing ego edges, emitting the refined ego + artifact.
  - The optional LLM branch is framed **not as a new contribution** but as a compatibility check: "does the controller-produced artifact carry usable signal for downstream LLM enhancement?" The LLM branch exists to answer uncertainty #4 (attribution of gains).
- **Claim additions**:
  - **C4 (claim)**: the refined artifact alone (without any LLM call) recovers most of the slice F1 gain; any additional gain from LLM embedding is attributable to the refiner, not the artifact. This is an *attribution* claim, not a performance claim.
- **Claim boundary (hard)**: LLM direct prediction (L5 in research.md §9.6) is reported only as a diagnostic baseline, never as a recommended configuration.

### Decision 3 — Rewrite boundary = soft reweight existing ego edges only; virtual edges not written back

- **Applied**: unchanged from prior rounds. Virtual context edges enter (a) the evidence artifact and (b) the LLM prompt if the LLM branch is active. They are **never** scored by the edge modifier and **never** written to the main graph. This is now written into the **Invariants** section with a unit test.

### Decision 4 — Estimator claim language tightened

- **Rationale (user)**: "不要说'估计图质量'，更精确地说'估计当前 ego context 下的 prediction-set uncertainty，并将其作为 low-quality ego candidate signal'."
- **Applied globally** in the revised proposal:
  - Method thesis reworded.
  - Contribution Focus reworded.
  - Novelty-and-Elegance section reworded.
  - Experiment plan metric names reworded (high-quality-risk slice → **high-uncertainty ego slice**; "ego-quality score" → **ego-uncertainty score**).
  - Every remaining use of "ego-quality" in documentation either (a) changes to "ego-uncertainty" or (b) explicitly rewords as "low-quality ego candidate (operationalized as prediction-set uncertainty under ego context)".

## Responses to the Nine User-Flagged Uncertainties

### U#1 — Estimator → ego-quality mapping validity

- **Response**: accept that `s_composite` measures prediction-set uncertainty conditional on ego context; do *not* claim it measures graph-structural quality directly. Add a diagnostic study (**R120, Block-D**) that correlates `s_composite(v)` against: (a) ego heterophily, (b) degree deciles, (c) label-shuffle perturbation of ego labels, (d) edge-removal sensitivity. Report the strength of each correlation; do not infer causation. This diagnostic study is *required for the paper* per the reframed claim (it is now an attribution study, not a "graph quality estimator" claim).

### U#2 — Modifier output form

- **Locked by Decision 1**: `u_edge(e)` continuous utility in v1 main. See above.

### U#3 — Counterfactual supervision reliability

- **Response**: the learned modifier (gradient-attribution supervised) is already appendix-only (Round 2). I now add three falsification controls to that appendix (**R121, R122, R123**):
  - **R121** (no-counterfactual baseline): learned modifier with zero supervision (random-init, scores = 0). If the v1-main deterministic modifier ≈ no-counterfactual learned modifier ≈ trained learned modifier, the "training signal helps" claim dies.
  - **R122** (split-transfer): train modifier on train_cal counterfactuals only; test on valid_cal counterfactuals. Report gap.
  - **R123** (label-shuffle diagnostic): shuffle training labels → retrain modifier → report slice F1 drop. If learned modifier is largely insensitive to label shuffling, it is learning base-model idiosyncrasies, not real edge utility.
- **Interpretation rule**: if any of R121/R122/R123 fires, the learned modifier stays out of the paper. Only the deterministic modifier survives.

### U#4 — LLM branch attribution

- **Response**: define the **LLM attribution table** explicitly (**R124, Block-LLM-Attrib**). Four conditions:

| Variant | Artifact | LLM called | How LLM signal enters | Measures |
|---|---|---|---|---|
| L0 | Yes | No | — | Baseline: refined-artifact-only gain |
| L1 (summ) | Yes | Yes | Prompt-summarization only; LLM output text discarded except as human-readable diagnostic | Prompt-formatting gain |
| L2 (embed) | Yes | Yes | `z_LLM(v)` → Refiner | LLM-embedding gain via refiner |
| L3 (direct) | Yes | Yes | LLM directly emits label; refiner = identity | Upper-bound diagnostic; never recommended |

- Paper claim (C4) = `acc(L0) ≥ 0.90 · acc(L2)` on high-uncertainty slice ⇒ artifact is the load-bearing piece; LLM call is a marginal enhancer, not the mechanism.

### U#5 — Evidence prompt field ablation

- **Response**: add **Block-Prompt** appendix (**R125**). The artifact has 7 fields: target summary, supportive evidence, suspicious evidence, uncertain evidence, semantic context, structural context, quality summary. Drop-one ablation on the L2 (embed) variant reports which fields are load-bearing. If any single field dominates, the artifact contracts to that field.

### U#6 — Soft vs hard rewrite

- **Response**: move the hard-rewrite comparison from appendix-only to **Block-R** (Claim-supporting). Variants:
  - **R-none** no rewrite (baseline).
  - **R-soft** soft reweight (v1 main).
  - **R-soft+virtual** soft reweight + virtual context edges in artifact (L0-equivalent).
  - **R-hard-delete** delete edges with `u_edge(e) ≤ -τ`.
  - **R-hard-add** add top-k kNN candidates as weight-1 edges.
- Claim: `R-soft ≥ R-soft+virtual ≥ R-hard-delete, R-hard-add` on slice F1, **AND R-soft > R-hard-delete on camouflage slice** (because camouflage edges are preserved by soft reweight).
- **Falsification**: if R-hard-delete > R-soft on slice F1 on *both* TwiBot-20 and TwiBot-22, the "preserve camouflage" argument weakens and the paper must report honestly.

### U#7 — Virtual context edges boundary

- **Response**: explicit usage matrix. Locked invariants written into the Invariants section.

| Surface | Propagation edges (∈ ego) | Virtual context edges (kNN \ ego) |
|---|---|---|
| `s_composite(v)` input (Stage-1) | Yes | No (stays out of ego for quality score) |
| Edge modifier scoring (Stage-4) | Yes (`u_edge(e)`) | No |
| Soft reweight (Stage-5) | Yes | No (never written to main graph) |
| Stage-6 local recomputation | Yes | No |
| Evidence artifact (Stage-7) | Yes (with final `w_e'` and `u_edge(e)`) | Yes (as "semantic context" + "structural context" fields) |
| LLM prompt (Stage-8b) | Yes (serialized from artifact) | Yes (same) |
| Refiner input (Stage-8a or 8b) | Via `h_GNN_post(v)` only | Never as structural input; only as `z_LLM(v)` context in 8b |

- **Unit test**: `tests/test_virtual_context_isolation.py` asserts the main-graph adjacency matrix is unchanged after Stage-5 for any node `u` not in the original ego of any hard node.

### U#8 — Iterative estimator loop

- **Response**: hold Round-3 position. `max_iter = 1` in v1 main. `max_iter = 2` is appendix-only (**R116**) with two additional guards: (a) the second iteration's accept rule uses `s_composite_post(iter=2) < s_composite_post(iter=1)` (strict), and (b) total budget is budgeted across iterations (not per-iteration). If the second iteration cannot find strict improvement under the same total budget, paper reports "iteration offers no gain; single-shot is the right configuration".

### U#9 — BotBR/BECE differentiation experiment

- **Response**: add **Block-BR** (Block-Baseline-Reliability). A BotBR/BECE-style binary-reliability baseline is added to Block-T:
  - **BR**: binary reliability classifier trained on the same train_cal counterfactuals; edges labeled reliable/unreliable; reliable edges kept, unreliable edges deleted.
  - **BR-soft**: binary reliability classifier; reliable edges kept at weight 1, unreliable edges at weight 0.3 (soft version of BR for apples-to-apples soft vs hard comparison).
- Claim: `u_edge (T4) > BR-soft > BR` on camouflage slice (soft-continuous > soft-binary > hard-binary). This directly addresses the user's concern that the paper must *experimentally* show differentiation from BotBR/BECE, not only charter it.

---

## Revised Proposal (v4 diff summary; full method body unchanged except the items below)

**Contribution Focus v4**:

- **Dominant**: **EQC²** — single primitive (composite quantile-calibrated non-conformity score) used as trigger + rewrite-amplitude factor + L-hop-local accept/rollback gate. Claim is framed as "estimating prediction-set uncertainty under current ego context, used as a low-quality ego candidate signal" (not "estimating graph quality").
- **Supporting 1**: deterministic continuous-utility modifier `u_edge(e) ∈ [-1, +1]` over existing ego edges; emits the refined ego + evidence artifact.
- **Supporting 2** (new): evidence artifact format, serving *both* the artifact-only refiner (v1 main) and the optional LLM refiner (v1 ablation branch).
- **Explicitly rejected**:
  - Binary edge-reliability headline (BotBR / BECE territory).
  - LLM-understands-graphs claim (TMLR 2024).
  - Global structure learning.
  - CP coverage theorem claim.
  - Causal identification language.
  - `s_reliability` binary head in v1 main modifier.
  - `p_role` 4-way distribution as a training head.
  - Hard delete / hard add as main rewrite action.

**Complexity Budget v4**:

- v1-Main-Artifact adds Stage-7 (artifact serialization, zero parameters) and Stage-8a (artifact-only refiner, a 2-layer MLP trained on `train_cal` counterfactuals as a calibration-time artifact).
- v1-LLM ablation adds Stage-8b (prompt construction + LLM embedding + refiner MLP) as an optional branch, never on the critical path for the dominant contribution.
- `s_reliability` head deleted; `p_role` head deleted (replaced by deterministic bucketization inside the artifact).

**System Overview v4** (delta from Round-3 Stage-3, Stage-4, plus new Stage-7 and Stage-8):

```
Stage-3 (retrieval on hard v)
  propagation_edges(v)     = existing ego edges                            (writable)
  virtual_context_edges(v) = kNN_RoBERTa \ existing ego                    (artifact-only)

Stage-4 (deterministic continuous-utility modifier)
  u_edge(e) = clip(α_text · cos(X_R[v], X_R[u]) + α_rel · 𝟙[rel(e) ∈ R_trusted]
                   + α_tgd · sign(JSD(p_LM(v) || p_GNN(v)) - JSD(p_LM(u) || p_GNN(u))),
                   -1, +1)

Stage-5 (soft rewrite, propagation edges only)
  w_e' = w_e · clip(1 + λ · budget(v) · u_edge(e) · (1 - s_composite(v)), 1-λ, 1+λ)

Stage-6 (L-hop-local recomputation + accept/rollback)
  [unchanged from Round 3]

Stage-7 (evidence artifact, zero params)
  Artifact(v) = {
     target_summary:       {X_R[v], p_GNN_post(v), s_composite_post(v), set_size_post(v)},
     supportive_evidence:  top-k propagation edges by u_edge(e) > τ+       (from ego; with w_e')
     suspicious_evidence:  top-k propagation edges by u_edge(e) < τ-       (from ego; with w_e')
     uncertain_evidence:   top-k propagation edges by |u_edge(e)| ≤ τ0     (from ego; with w_e')
     semantic_context:     top-k virtual_context_edges by cos similarity   (not in ego)
     structural_context:   top-k structural neighbors (ego k-hop, PPR)     (from ego context)
     quality_summary:      {s_composite(v), s_composite_post(v), budget(v), rollback?}
  }

Stage-8 (refiner; two mutually exclusive variants)
  8a (v1 main, no LLM):
     h_artifact(v) = MLP(concat(h_GNN_post(v), artifact_embedding(Artifact(v))))
     p_final(v) = softmax(W_refiner · h_artifact(v))
     MLP fit on train_cal (calibration-time only)
  8b (v1 optional ablation, LLM branch):
     EvidencePrompt(v) = serialize_to_text(Artifact(v))
     z_LLM(v) = LLM_embed(EvidencePrompt(v))
     h_refined(v) = MLP(concat(h_GNN_post(v), X_R[v], z_LLM(v), artifact_embedding(Artifact(v))))
     p_final(v) = softmax(W_refiner · h_refined(v))
```

**Failure Modes v4 additions**:

- **Artifact refiner overfits** → detect via train_cal vs valid_cal gap on `p_final`; fallback: skip refiner, use `p_GNN_post` directly.
- **LLM branch adds no attribution gain (C4 fails)** → report honestly; paper still valid because C4 is an attribution claim, not a performance claim.
- **`u_edge` clipping saturates** → inspect distribution; adjust `α_text, α_rel, α_tgd` weight search bounds.
- **Virtual-context-edge leakage to main graph** → `test_virtual_context_isolation.py` fail-loud.

**Tuning Protocol v4**: unchanged from Round 3, plus:
- `train_cal` is additionally used to fit the Stage-8a refiner MLP (supervised distillation from ground-truth labels on `train_cal`; unsupervised with respect to test).
- If the LLM branch is enabled, the 8b refiner MLP is fit on `train_cal` with LLM embeddings pre-computed (LLM inference on train_cal is a one-time cost, not part of test inference budget).

---

## Claim-Driven Validation Sketch v4 (three claim blocks + five appendix blocks)

**Pilot Gates** (Week 0; unchanged from Round 3 and binding).

### Claim 1 — EQC² composite-with-set trigger dominates

Block-T, with BotBR/BECE-style baseline added (Block-BR).

Variants: T1 entropy · T2 temp-entropy · T3 label-only CP · **BR** hard-binary-reliability · **BR-soft** soft-binary-reliability · T4-NS · T4.

Expected: **T4 > T4-NS > BR-soft > T3 > BR > T2 ≥ T1** on high-uncertainty ego slice.

### Claim 2 — EQC² accept/rollback gate is load-bearing

Block-G unchanged (G1 / G2 / G3).

### Claim 3 — Cross-dataset robustness on multi-relational graph

Block-X unchanged (MGTAB).

### Claim 4 (new) — Artifact is the load-bearing piece; LLM call is marginal

Block-LLM-Attrib. L0 / L1 / L2 / L3. Pass iff `slice_F1(L0) ≥ 0.90 · slice_F1(L2)`.

### Appendix blocks

- **A-Composite**: drop-one of `s_tg / s_het / s_rec` (R113).
- **A-Modifier**: deterministic continuous vs learned continuous vs deterministic + 4-way bucketized artifact (R114) — with R121/R122/R123 falsification controls.
- **A-Diffusion**: DAPS/NAPS, SNAPS inside `s_composite` (R115).
- **A-Iter**: `max_iter=2` with budget-respecting rollback (R116).
- **A-NestedSplit**: 5-fold `train_cal / valid_cal` rotation (R117).
- **A-Prompt**: 7-field drop-one ablation on L2 (R125).
- **A-Rewrite** (new, from uncertainty #6): **Block-R** — `R-none / R-soft / R-soft+virtual / R-hard-delete / R-hard-add` on TwiBot-20 + TwiBot-22.
- **D-EstSem** (new, from uncertainty #1): diagnostic correlation study of `s_composite` vs heterophily / degree / label-shuffle / edge-removal (R120). Observational only.

---

## Remaining Risks v4

- Scope widens by one ablation branch (LLM). Mitigation: strictly labeled *optional* in every artifact; the dominant contribution stands without any LLM call; paper structured so the LLM branch is a single subsection, not a top-level claim.
- Artifact refiner (Stage-8a) introduces a trainable component not present in prior Round-4 v1 main. Mitigation: same `train_cal` discipline; if overfitting is detected, fallback path is `p_final = p_GNN_post` (no refiner).
- Claim 4 (LLM attribution) is a *new* paper claim. Mitigation: it is framed as attribution, not performance — falsification via L0/L2 ratio is symmetric (failing reads "LLM is the load-bearing piece after all", which the user already flagged as undesirable and the paper can report honestly).

---

## Output of Round 4 refinement

- Round 5 review request is the immediate next step.
- Revised proposal body to submit: the Round-3 proposal body, with the above changes folded in.
